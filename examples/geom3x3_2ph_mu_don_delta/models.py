from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, jit, grad, vmap
from jax.tree_util import tree_map
from jax.numpy.linalg import inv as invert

import optax

from jaxpi import archs
from jaxpi.models import ForwardBVP, ForwardIVP
from jaxpi.utils import ntk_fn
from jaxpi.evaluator import BaseEvaluator

from flax import linen as nn


class NavierStokes2DwSat(ForwardIVP):

    def __init__(self, config, p_in, temporal_dom, U_max, L_max, fluid_params):   
        super().__init__(config)

        # self.inflow_fn = inflow_fn
        self.p_in = p_in
        self.temporal_dom = temporal_dom
        self.U_max = U_max
        self.L_max = L_max
        # self.Re = Re  # Reynolds number
        self.fluid_params = fluid_params

        self.delta_matrices = None

        # Predict functions over batch
        self.u0_pred_fn = vmap(self.u_net, (None, None, 0, 0))
        self.v0_pred_fn = vmap(self.v_net, (None, None, 0, 0))
        self.p0_pred_fn = vmap(self.p_net, (None, None, 0, 0))
        self.s0_pred_fn = vmap(self.s_net, (None, None, 0, 0))
              
        # self.u_pred_1_fn = vmap(self.u_net, (None, 0, 0, 0, None))
        # self.v_pred_1_fn = vmap(self.v_net, (None, 0, 0, 0, None))
        
        self.ufem_pred_fn = vmap(self.u_net, (None, 0, 0, None))
        self.vfem_pred_fn = vmap(self.v_net, (None, 0, 0, None))
        self.pfem_pred_fn = vmap(self.p_net, (None, 0, 0, None))
        self.sfem_pred_fn = vmap(self.s_net, (None, 0, 0, None))

        self.u_pred_fn = vmap(self.u_net, (None, 0, 0, 0))
        self.v_pred_fn = vmap(self.v_net, (None, 0, 0, 0))
        self.p_pred_fn = vmap(self.p_net, (None, 0, 0, 0))
        self.s_pred_fn = vmap(self.s_net, (None, 0, 0, 0))
        self.r_pred_fn = vmap(self.r_net, (None,  0, 0, 0, 0, 0, 0, 0 ))
        self.r_pred_fn_mu = vmap(self.r_net, (None,  0, 0, None, 0, 0, 0, 0 ))
        self.r_pred_fn_t = vmap(self.r_net, (None,  None, 0, 0, 0, 0, 0, 0 ))
        
    def update_delta_matrices(self, delta_matrices):
        self.delta_matrices = delta_matrices

    def neural_net(self, params, t, X, mu):
                
        t = t / (self.temporal_dom[1])  # rescale t into [0, 1]
        mu = 2* ((mu - .0025) / (.1 - .0025)) -1
        ones = jnp.stack([jnp.ones(t.shape)])
        # X = jnp.stack([X]) # branch
        t = jnp.stack([ t ] )
        mu = jnp.stack([ mu ] )
        outputs = self.state.apply_fn(params, t, X, mu, ones)

        # Start with an initial state of the channel flow
        u = outputs[0]
        v = outputs[1]
        p = outputs[2]
        s = outputs[3] # nn.softplus( outputs[3] )
        D = nn.sigmoid(outputs[4]) *5*10**(-3)
        u_scaler = 0.04 #0.00174
        v_scaler = 0.007 # 0.00027
        
        u = u *u_scaler
        v = v *v_scaler

        return u, v, p, s, D

    def u_net(self, params, t, X, mu):
        u, _, _, _, _ = self.neural_net(params, t, X, mu)
        return u

    def v_net(self, params, t, X, mu):
        _, v, _, _, _ = self.neural_net(params, t, X, mu)
        return v

    def p_net(self, params, t, X, mu):
        _, _, p, _, _ = self.neural_net(params, t, X, mu)
        return p

    def s_net(self, params, t, X, mu):
        _, _, _, s, _ = self.neural_net(params, t, X, mu)
        return s
    
    def D_net(self, params, t, X, mu):
        _, _, _, _, D = self.neural_net(params, t, X, mu)
        return D

    def r_net(self, params, t, eigenvecs_element, mu0, B, A, M, N):
        # Re = jnp.ones(x.shape)
        ( _, mu1, rho0, rho1) = self.fluid_params
        
        Minv = invert(M)
               
        u1 , v1 , p1, s1, D = self.neural_net(params, t, jnp.squeeze(jnp.take(eigenvecs_element, jnp.array([0]), axis=0)) , mu0)
        u2 , v2 , p2, s2, _ = self.neural_net(params, t, jnp.squeeze(jnp.take(eigenvecs_element, jnp.array([1]), axis=0)) , mu0)
        u3 , v3 , p3, s3, _ = self.neural_net(params, t, jnp.squeeze(jnp.take(eigenvecs_element, jnp.array([2]), axis=0)) , mu0)
        
        u_e = jnp.array([u1, u2, u3])[:, jnp.newaxis]
        v_e = jnp.array([v1, v2, v3])[:, jnp.newaxis]
        p_e = jnp.array([p1, p2, p3])[:, jnp.newaxis]
        s_e = jnp.array([s1, s2, s3])[:, jnp.newaxis]
        
        N = N.mT
        
        u = N @ u_e
        v = N @ v_e
        p = N @ p_e
        s = N @ s_e
        u, v, p, s = u[0][0], v[0][0], p[0][0], s[0][0] 
        
        u_t1 = grad(self.u_net, argnums=1)(params, t, eigenvecs_element[0], mu0) 
        u_t2 = grad(self.u_net, argnums=1)(params, t, eigenvecs_element[1], mu0) 
        u_t3 = grad(self.u_net, argnums=1)(params, t, eigenvecs_element[2], mu0) 
        u_t  = N @ jnp.array([u_t1, u_t2, u_t3])[:, jnp.newaxis]
        v_t1 = grad(self.v_net, argnums=1)(params, t, eigenvecs_element[0], mu0)
        v_t2 = grad(self.v_net, argnums=1)(params, t, eigenvecs_element[1], mu0)
        v_t3 = grad(self.v_net, argnums=1)(params, t, eigenvecs_element[2], mu0)
        v_t  = N @ jnp.array([v_t1, v_t2, v_t3])[:, jnp.newaxis]
        s_t1 = grad(self.s_net, argnums=1)(params, t, eigenvecs_element[0], mu0)
        s_t2 = grad(self.s_net, argnums=1)(params, t, eigenvecs_element[1], mu0)
        s_t3 = grad(self.s_net, argnums=1)(params, t, eigenvecs_element[2], mu0)
        s_t  = N @ jnp.array([s_t1, s_t2, s_t3])[:, jnp.newaxis]

        u_x = B @ u_e
        v_x = B @ v_e
        p_x = B @ p_e
        s_x = B @ s_e

        u_y = u_x[1][0] *self.L_max ; u_x = u_x[0][0] *self.L_max 
        v_y = v_x[1][0] *self.L_max ; v_x = v_x[0][0] *self.L_max 
        p_y = p_x[1][0] *self.L_max ; p_x = p_x[0][0] *self.L_max 
        s_y = s_x[1][0] *self.L_max ; s_x = s_x[0][0] *self.L_max 
        
        u_xx = Minv @ A @ u_e
        v_xx = Minv @ A @ v_e
        s_xx = Minv @ A @ s_e

        u_yy = u_xx[2][0] *(self.L_max**2) ; u_xx = u_xx[0][0] *(self.L_max**2) 
        v_yy = v_xx[2][0] *(self.L_max**2) ; v_xx = v_xx[0][0] *(self.L_max**2) 
        s_yy = s_xx[2][0] *(self.L_max**2) ; s_xx = s_xx[0][0] *(self.L_max**2) 

        Re = rho0*self.U_max*(self.L_max)/mu1  
        # Re = 1
        mu = (1-s)*mu1 + s*mu0
        mu_ratio = mu/mu1
                
        # PDE residual
        ru = u_t + u * u_x + v * u_y + (p_x - mu_ratio*(u_xx + u_yy)) / Re #  
        rv = v_t + u * v_x + v * v_y + (p_y - mu_ratio*(v_xx + v_yy)) / Re #
        rc = u_x + v_y
        rs = s_t + u * s_x + v * s_y - D*(s_xx + s_yy)

        return ru, rv, rc, rs

    def ru_net(self, params, t, X, mu):
        ru, _, _, _ = self.r_net(params, t, X, mu)
        return ru

    def rv_net(self, params, t, X, mu):
        _, rv, _, _ = self.r_net(params, t, X, mu)
        return rv

    def rc_net(self, params, t, X, mu):
        _, _, rc, _ = self.r_net(params, t, X, mu)
        return rc

    def rs_net(self, params, t, x, y, mu):
        _, _, _, rs = self.r_net(params, t, X, mu)
        return rs

    @partial(jit, static_argnums=(0,))
    def res_and_w(self, params, batch):
        # Sort temporal coordinates
        t_sorted = batch[:, 0].sort()
        ru_pred, rv_pred, rc_pred, rs_pred = self.r_pred_fn(
            params, t_sorted, batch[:, 1], batch[:, 2], batch[:,3]
        )

        ru_pred = ru_pred.reshape(self.num_chunks, -1)
        rv_pred = rv_pred.reshape(self.num_chunks, -1)
        rc_pred = rc_pred.reshape(self.num_chunks, -1)
        rs_pred = rs_pred.reshape(self.num_chunks, -1)

        ru_l = jnp.mean(ru_pred**2, axis=1)
        rv_l = jnp.mean(rv_pred**2, axis=1)
        rc_l = jnp.mean(rc_pred**2, axis=1)
        rs_l = jnp.mean(rs_pred**2, axis=1)

        ru_gamma = lax.stop_gradient(jnp.exp(-self.tol * (self.M @ ru_l)))
        rv_gamma = lax.stop_gradient(jnp.exp(-self.tol * (self.M @ rv_l)))
        rc_gamma = lax.stop_gradient(jnp.exp(-self.tol * (self.M @ rc_l)))
        rs_gamma = lax.stop_gradient(jnp.exp(-self.tol * (self.M @ rs_l)))

        # Take minimum of the causal weights
        gamma = jnp.vstack([ru_gamma, rv_gamma, rc_gamma, rs_gamma])
        gamma = gamma.min(0)

        return ru_l, rv_l, rc_l, rs_l, gamma

    @partial(jit, static_argnums=(0,))
    def compute_diag_ntk(self, params, batch):
        # Unpack batch
        res_batch = batch["res"]
        
        (t, X, X_bc, mu_batch, delta_matrices, fields, _) = res_batch
        Xin, Xout, Xnoslip, mu_inlet, t_inlet, mu_noslip, t_noslip = X_bc
        (X_fem, t_fem, mu_fem, _, _, _, _, _, _, _, _) = fields
        ( _, N, B, A, M) = delta_matrices

        u_data_ntk = vmap(ntk_fn, (None, None, 0, 0, None))(
            self.u_net, params, t_fem, X_fem, .0025)
        v_data_ntk = vmap(ntk_fn, (None, None, 0, 0, None))(
            self.v_net, params, t_fem, X_fem, .0025)
        p_data_ntk = vmap(ntk_fn, (None, None, 0, 0, None))(
            self.p_net, params, t_fem, X_fem, .0025)
        s_data_ntk = vmap(ntk_fn, (None, None, 0, 0, None))(
            self.s_net, params, t_fem, X_fem, .0025)
        

        u_ic_ntk = vmap(ntk_fn, (None, None, None, 0, 0))(
            self.u_net, params, 0.0, X_fem, mu_fem
        )
        v_ic_ntk = vmap(ntk_fn, (None, None, None, 0, 0))(
            self.v_net, params, 0.0, X_fem, mu_fem
        )
        p_ic_ntk = vmap(ntk_fn, (None, None, None, 0, 0))(
            self.p_net, params, 0.0, X_fem, mu_fem
        )
        s_ic_ntk = vmap(ntk_fn, (None, None, None, 0, 0))(
            self.s_net, params, 0.0, X_fem, mu_fem
        )

        noslip_ntk = vmap(ntk_fn, (None, None, 0, 0, 0))(
            self.u_net, params, t_noslip, Xnoslip, mu_noslip
        )
        sin_ntk = vmap(ntk_fn, (None, None, 0, 0, 0))(
            self.s_net, params, t_inlet, Xin, mu_inlet
        )
        dp_ntk = vmap(ntk_fn, (None, None, 0, 0, 0))(
            self.p_net, params, t_inlet, Xin, mu_inlet
        )

        ru_ntk = vmap(ntk_fn, (None, None, 0, 0, 0, 0, 0, 0, 0))(
            self.ru_net, params, t, X, mu_batch, B, A, M, N 
        )
        rv_ntk = vmap(ntk_fn, (None, None, 0, 0, 0, 0, 0, 0, 0))(
            self.rv_net, params, t, X, mu_batch, B, A, M, N 
        )
        rc_ntk = vmap(ntk_fn, (None, None, 0, 0, 0, 0, 0, 0, 0))(
            self.rc_net, params, t, X, mu_batch, B, A, M, N 
        )
        rs_ntk = vmap(ntk_fn, (None, None, 0, 0, 0, 0, 0, 0, 0))(
            self.rs_net, params, t, X, mu_batch, B, A, M, N 
        )

        ntk_dict = {
            "u_data": u_data_ntk,
            "v_data": v_data_ntk,
            "p_data": p_data_ntk,
            "s_data": s_data_ntk,
            # "u_ic": u_ic_ntk,
            # "v_ic": v_ic_ntk,
            # "p_ic": p_ic_ntk,
            # # "s_ic": s_ic_ntk,
            "noslip": noslip_ntk,
            "sin": sin_ntk,
            "dp": dp_ntk,
            # "ru": ru_ntk, #
            # "rv": rv_ntk, #
            # "rc": rc_ntk,
            # "rs": rs_ntk,
        }

        return ntk_dict
    
    @partial(jit, static_argnums=(0,))
    def losses(self, params, batch):
        # Unpack batch

        res_batch = batch["res"]
        
        (t, X, X_bc, mu_batch, delta_matrices, fields, fields_ic) = res_batch
        Xin, Xout, Xnoslip, mu_inlet, t_inlet, mu_noslip, t_noslip = X_bc
        ( _, N, B, A, M) = delta_matrices
        (X_fem, t_fem, mu_fem, u_fem_q, v_fem_q, p_fem_q, s_fem_q, u_fem_s, v_fem_s, p_fem_s, s_fem_s) = fields
        (u_ic, v_ic, p_ic, s_ic) = fields_ic
        
        # u_fem_q_pred = self.ufem_pred_fn(params, jnp.repeat(t, 3), jnp.reshape(X, (-1,52)), .0025)
        # v_fem_q_pred = self.vfem_pred_fn(params, jnp.repeat(t, 3), jnp.reshape(X, (-1,52)), .0025)
        # p_fem_q_pred = self.pfem_pred_fn(params, jnp.repeat(t, 3), jnp.reshape(X, (-1,52)), .0025)
        # s_fem_q_pred = self.sfem_pred_fn(params, jnp.repeat(t, 3), jnp.reshape(X, (-1,52)), .0025)
        
        # u_fem_s_pred = self.ufem_pred_fn(params, jnp.repeat(t, 3), jnp.reshape(X, (-1,52)), .1)
        # v_fem_s_pred = self.vfem_pred_fn(params, jnp.repeat(t, 3), jnp.reshape(X, (-1,52)), .1)
        # p_fem_s_pred = self.pfem_pred_fn(params, jnp.repeat(t, 3), jnp.reshape(X, (-1,52)), .1)
        # s_fem_s_pred = self.sfem_pred_fn(params, jnp.repeat(t, 3), jnp.reshape(X, (-1,52)), .1)
        
        # u_data = jnp.mean( jnp.mean((u_fem_q_pred -  jnp.reshape(u_fem_q, (-1,)) ) ** 2) + jnp.mean((u_fem_s_pred -  jnp.reshape(u_fem_s, (-1,)) ) ** 2) ) 
        # v_data = jnp.mean( jnp.mean((v_fem_q_pred -  jnp.reshape(v_fem_q, (-1,)) ) ** 2) + jnp.mean((v_fem_s_pred -  jnp.reshape(v_fem_s, (-1,)) ) ** 2) ) 
        # p_data = jnp.mean( jnp.mean((p_fem_q_pred -  jnp.reshape(p_fem_q, (-1,)) ) ** 2) + jnp.mean((p_fem_s_pred -  jnp.reshape(p_fem_s, (-1,)) ) ** 2) ) 
        # s_data = jnp.mean( jnp.mean((s_fem_q_pred -  jnp.reshape(s_fem_q, (-1,)) ) ** 2) + jnp.mean((s_fem_s_pred -  jnp.reshape(s_fem_s, (-1,)) ) ** 2) ) 

        u_fem_q_pred = self.ufem_pred_fn(params, t_fem, X_fem, .0025)
        v_fem_q_pred = self.vfem_pred_fn(params, t_fem, X_fem, .0025)
        p_fem_q_pred = self.pfem_pred_fn(params, t_fem, X_fem, .0025)
        s_fem_q_pred = self.sfem_pred_fn(params, t_fem, X_fem, .0025)
        
        u_fem_s_pred = self.ufem_pred_fn(params, t_fem, X_fem, .1)
        v_fem_s_pred = self.vfem_pred_fn(params, t_fem, X_fem, .1)
        p_fem_s_pred = self.pfem_pred_fn(params, t_fem, X_fem, .1)
        s_fem_s_pred = self.sfem_pred_fn(params, t_fem, X_fem, .1)
        
        u_data = jnp.mean( jnp.mean((u_fem_q_pred - u_fem_q  ) ** 2) + jnp.mean((u_fem_s_pred - u_fem_s  ) ** 2) ) 
        v_data = jnp.mean( jnp.mean((v_fem_q_pred - v_fem_q  ) ** 2) + jnp.mean((v_fem_s_pred - v_fem_s  ) ** 2) ) 
        p_data = jnp.mean( jnp.mean((p_fem_q_pred - p_fem_q  ) ** 2) + jnp.mean((p_fem_s_pred - p_fem_s  ) ** 2) ) 
        s_data = jnp.mean( jnp.mean((s_fem_q_pred - s_fem_q  ) ** 2) + jnp.mean((s_fem_s_pred - s_fem_s  ) ** 2) ) 
        
        # print(X.shape)  (512, 3, 52)
        # print(t.shape)  (512,)
        # print(u_fem_q.shape) (512, 3)
        # print(u0.shape)      (512, 3)
        
        # Initial condition loss
        # u_ic_pred = self.u0_pred_fn(params, 0.0, jnp.reshape(X, (-1,52)), jnp.repeat(mu_batch, 3))
        # v_ic_pred = self.v0_pred_fn(params, 0.0, jnp.reshape(X, (-1,52)), jnp.repeat(mu_batch, 3))
        # p_ic_pred = self.p0_pred_fn(params, 0.0, jnp.reshape(X, (-1,52)), jnp.repeat(mu_batch, 3))
        # s_ic_pred = self.s0_pred_fn(params, 0.0, jnp.reshape(X, (-1,52)), jnp.repeat(mu_batch, 3))
        # u_ic_loss = jnp.mean((u_ic_pred - jnp.reshape(u_ic, (-1,)) ) ** 2)
        # v_ic_loss = jnp.mean((v_ic_pred - jnp.reshape(v_ic, (-1,)) ) ** 2)
        # p_ic_loss = jnp.mean((p_ic_pred - jnp.reshape(p_ic, (-1,)) ) ** 2)
        # s_ic_loss = jnp.mean((s_ic_pred - jnp.reshape(s_ic, (-1,)) ) ** 2)
        
        u_ic_pred = self.u0_pred_fn(params, 0.0, X_fem, mu_fem)
        v_ic_pred = self.v0_pred_fn(params, 0.0, X_fem, mu_fem)
        p_ic_pred = self.p0_pred_fn(params, 0.0, X_fem, mu_fem)
        s_ic_pred = self.s0_pred_fn(params, 0.0, X_fem, mu_fem)
        u_ic_loss = jnp.mean((u_ic_pred - u_ic ) ** 2)
        v_ic_loss = jnp.mean((v_ic_pred - v_ic ) ** 2)
        p_ic_loss = jnp.mean((p_ic_pred - p_ic ) ** 2)
        s_ic_loss = jnp.mean((s_ic_pred - s_ic ) ** 2)
        
        sin_pred = self.s_pred_fn( params, t_inlet, Xin, mu_inlet)
        sin_loss = jnp.mean((1.0 - sin_pred)**2)
        
        pin_pred = self.p_pred_fn( params, t_inlet, Xin, mu_inlet)
        pout_pred = self.p_pred_fn( params, t_inlet, Xout, mu_inlet)
        dp_loss = jnp.mean(jnp.mean((self.p_in - pin_pred)**2) + jnp.mean((0 - pout_pred)**2))
        
        u_nos_pred = self.u_pred_fn( params, t_noslip, Xnoslip, mu_noslip)
        v_nos_pred = self.v_pred_fn( params, t_noslip, Xnoslip, mu_noslip)
        noslip_loss = jnp.mean(jnp.mean((0 - u_nos_pred)**2) + jnp.mean((0 - v_nos_pred)**2))

        ru_pred, rv_pred, rc_pred, rs_pred = self.r_pred_fn( params, t, X, mu_batch, B, A, M, N )
        ru1 =  jnp.mean(ru_pred**2)
        rv1 =  jnp.mean(rv_pred**2)
        rc1 =  jnp.mean(rc_pred**2)
        rs1 =  jnp.mean(rs_pred**2)
        
        ru_pred, rv_pred, rc_pred, rs_pred = self.r_pred_fn_mu( params, t, X, .0025, B, A, M, N )
        ru2 =  jnp.mean(ru_pred**2)
        rv2 =  jnp.mean(rv_pred**2)
        rc2 =  jnp.mean(rc_pred**2)
        rs2 =  jnp.mean(rs_pred**2)
        
        ru_pred, rv_pred, rc_pred, rs_pred = self.r_pred_fn_mu( params, t, X, .1, B, A, M, N )
        ru3 =  jnp.mean(ru_pred**2)
        rv3 =  jnp.mean(rv_pred**2)
        rc3 =  jnp.mean(rc_pred**2)
        rs3 =  jnp.mean(rs_pred**2)
        
        ru_pred, rv_pred, rc_pred, rs_pred = self.r_pred_fn_t( params, 0.0, X, mu_batch, B, A, M, N )
        ru4 =  jnp.mean(ru_pred**2)
        rv4 =  jnp.mean(rv_pred**2)
        rc4 =  jnp.mean(rc_pred**2)
        rs4 =  jnp.mean(rs_pred**2)

        ru_loss = jnp.mean( ru1 + ru2 + ru3 + ru4 ) 
        rv_loss = jnp.mean( rv1 + rv2 + rv3 + rv4 )
        rc_loss = jnp.mean( rc1 + rc2 + rc3 + rc4 )
        rs_loss = jnp.mean( rs1 + rs2 + rs3 + rs4 )

        loss_dict = {
            "u_data": u_data,
            "v_data": v_data,
            "p_data": p_data,
            "s_data": s_data,
            # "u_ic": u_ic_loss,
            # "v_ic": v_ic_loss,
            # "p_ic": p_ic_loss,
            # "s_ic": s_ic_loss,
            "noslip": noslip_loss,
            "sin": sin_loss,
            "dp": dp_loss,
            # "ru": ru_loss,
            # "rv": rv_loss,
            # "rc": rc_loss,
            # "rs": rs_loss,
        }

        return loss_dict


class NavierStokesEvaluator(BaseEvaluator):
    def __init__(self, config, model):
        super().__init__(config, model)

    # def log_preds(self, params, x_star, y_star):
    #     u_pred = vmap(vmap(model.u_net, (None, None, 0)), (None, 0, None))(params, x_star, y_star)
    #     v_pred = vmap(vmap(model.v_net, (None, None, 0)), (None, 0, None))(params, x_star, y_star)
    #     U_pred = jnp.sqrt(u_pred ** 2 + v_pred ** 2)
    #
    #     fig = plt.figure()
    #     plt.pcolor(U_pred.T, cmap='jet')
    #     log_dict['U_pred'] = fig
    #     fig.close()

    def __call__(self, state, batch):
        self.log_dict = super().__call__(state, batch)

        if self.config.weighting.use_causal:
            # _, _, _, causal_weight = self.model.res_and_w(state.params, batch["res"])
            _, _, _, _, causal_weight = self.model.res_and_w(state.params, batch["res"])
            self.log_dict["cas_weight"] = causal_weight.min()

        # if self.config.logging.log_errors:
        #     self.log_errors(state.params, coords, u_ref, v_ref)
        #
        # if self.config.logging.log_preds:
        #     self.log_preds(state.params, coords)

        return self.log_dict
