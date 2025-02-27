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
        
        self.p_in = p_in
        self.temporal_dom = temporal_dom
        self.U_max = U_max
        self.L_max = L_max
        self.fluid_params = fluid_params

        self.delta_matrices = None
        self.epoch = 0

        # Predict functions over batch
        self.u0_pred_fn = vmap(self.u_net, (None, None, 0, 0, 0, 0))
        self.v0_pred_fn = vmap(self.v_net, (None, None, 0, 0, 0, 0))
        self.p0_pred_fn = vmap(self.p_net, (None, None, 0, 0, 0, 0))
        self.s0_pred_fn = vmap(self.s_net, (None, None, 0, 0, 0, 0))
        
        self.ufem_pred_fn = vmap(self.u_net, (None, 0, 0, 0, 0, 0))
        self.vfem_pred_fn = vmap(self.v_net, (None, 0, 0, 0, 0, 0))
        self.pfem_pred_fn = vmap(self.p_net, (None, 0, 0, 0, 0, 0))
        self.sfem_pred_fn = vmap(self.s_net, (None, 0, 0, 0, 0, 0))

        self.r_pred_fn = vmap(self.r_net, (None, 0, 0, 0, 0, 0))
        self.r_pred_fn_t = vmap(self.r_net, (None,  None, 0, 0, 0, 0))
        
    def update_delta_matrices(self, delta_matrices):
        self.delta_matrices = delta_matrices
        
    def update_epoch(self):
        self.epoch = self.epoch+1

    def neural_net(self, params, t, x, y, X, mu):
                
        t = t / (self.temporal_dom[1])  # rescale t into [0, 1]
        mu = 2* ((mu - .0025) / (.1 - .0025)) -1

        inputs = jnp.stack([t, x, y])
        ones = jnp.ones(mu.shape)
        ones = jnp.stack([ ones ] )
        mu = jnp.stack([ mu ] )
        outputs = self.state.apply_fn(params, inputs, X, mu, ones)

        # Start with an initial state of the channel flow
        u = outputs[0]
        v = outputs[1]
        p = outputs[2]
        s = outputs[3] # nn.softplus( outputs[3] )
        # u_scaler = 0.00174  
        # v_scaler = 0.00027 
        u_scaler = .01669+.0249*.45
        v_scaler = .000908+.00699*.5
        
        u = u *u_scaler
        v = v *v_scaler

        return u, v, p, s

    def u_net(self, params, t, x, y, X, mu):
        u, _, _, _ = self.neural_net(params, t, x, y, X, mu)
        return u

    def v_net(self, params, t, x, y, X, mu):
        _, v, _, _ = self.neural_net(params, t, x, y, X, mu)
        return v

    def p_net(self, params, t, x, y, X, mu):
        _, _, p, _ = self.neural_net(params, t, x, y, X, mu)
        return p

    def s_net(self, params, t, x, y, X, mu):
        _, _, _, s = self.neural_net(params, t, x, y, X, mu)
        return s

    def r_net(self, params, t, x, y, X, mu0):
        # Re = jnp.ones(x.shape)
        ( _, mu1, rho0, rho1) = self.fluid_params
        
        # Minv = invert(M)
               
        u , v, _, s = self.neural_net(params, t, x, y, X, mu0)
        
        u_t = grad(self.u_net, argnums=1)(params, t, x, y, X, mu0) 
        v_t = grad(self.v_net, argnums=1)(params, t, x, y, X, mu0) 
        s_t = grad(self.s_net, argnums=1)(params, t, x, y, X, mu0) 

        u_x = grad(self.u_net, argnums=2)(params, t, x, y, X, mu0) 
        v_x = grad(self.v_net, argnums=2)(params, t, x, y, X, mu0) 
        p_x = grad(self.p_net, argnums=2)(params, t, x, y, X, mu0) 
        s_x = grad(self.s_net, argnums=2)(params, t, x, y, X, mu0) 

        u_y = grad(self.u_net, argnums=3)(params, t, x, y, X, mu0) 
        v_y = grad(self.v_net, argnums=3)(params, t, x, y, X, mu0) 
        p_y = grad(self.p_net, argnums=3)(params, t, x, y, X, mu0) 
        s_y = grad(self.s_net, argnums=3)(params, t, x, y, X, mu0)
        
        u_xx = grad(grad(self.u_net, argnums=2), argnums=2)(params, t, x, y, X, mu0) 
        v_xx = grad(grad(self.v_net, argnums=2), argnums=2)(params, t, x, y, X, mu0) 

        u_yy = grad(grad(self.u_net, argnums=3), argnums=3)(params, t, x, y, X, mu0) 
        v_yy = grad(grad(self.v_net, argnums=3), argnums=3)(params, t, x, y, X, mu0)  

        Re = rho0*self.U_max*(self.L_max)/mu1  
        mu = (1-s)*mu1 + s*mu0
        mu_ratio = mu/mu1
                
        # PDE residual
        ru = u_t + u * u_x + v * u_y + (p_x - mu_ratio*(u_xx + u_yy)) / Re #  
        rv = v_t + u * v_x + v * v_y + (p_y - mu_ratio*(v_xx + v_yy)) / Re #
        rc = u_x + v_y
        rs = s_t + u * s_x + v * s_y 
        
        return ru, rv, rc, rs

    def ru_net(self, params, t, x, y, X, mu):
        ru, _, _, _ = self.r_net(params, t, x, y, X, mu)
        return ru

    def rv_net(self, params, t, x, y, X, mu):
        _, rv, _, _ = self.r_net(params, t, x, y, X, mu)
        return rv

    def rc_net(self, params, t, x, y, X, mu):
        _, _, rc, _ = self.r_net(params, t, x, y, X, mu)
        return rc

    def rs_net(self, params, t, x, y, mu):
        _, _, _, rs = self.r_net(params, t, x, y, X, mu)
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
        
        ( fields, _) = res_batch
        (X_fem, t_fem, mu_fem, _, _, _, _) = fields

        u_data_ntk = vmap(ntk_fn, (None, None, 0, 0, 0, 0, 0))(
            self.u_net, params, t_fem, X_fem[:,0], X_fem[:,1], X_fem[:,2:], mu_fem)
        v_data_ntk = vmap(ntk_fn, (None, None, 0, 0, 0, 0, 0))(
            self.v_net, params, t_fem, X_fem[:,0], X_fem[:,1], X_fem[:,2:], mu_fem)
        p_data_ntk = vmap(ntk_fn, (None, None, 0, 0, 0, 0, 0))(
            self.p_net, params, t_fem, X_fem[:,0], X_fem[:,1], X_fem[:,2:], mu_fem)
        s_data_ntk = vmap(ntk_fn, (None, None, 0, 0, 0, 0, 0))(
            self.s_net, params, t_fem, X_fem[:,0], X_fem[:,1], X_fem[:,2:], mu_fem)
        

        u_ic_ntk = vmap(ntk_fn, (None, None, None, 0, 0, 0, 0))(
            self.u_net, params, 0.0, X_fem[:,0], X_fem[:,1], X_fem[:,2:], mu_fem
        )
        v_ic_ntk = vmap(ntk_fn, (None, None, None, 0, 0, 0, 0))(
            self.v_net, params, 0.0, X_fem[:,0], X_fem[:,1], X_fem[:,2:], mu_fem
        )
        p_ic_ntk = vmap(ntk_fn, (None, None, None, 0, 0, 0, 0))(
            self.p_net, params, 0.0, X_fem[:,0], X_fem[:,1], X_fem[:,2:], mu_fem
        )
        s_ic_ntk = vmap(ntk_fn, (None, None, None, 0, 0, 0, 0))(
            self.s_net, params, 0.0, X_fem[:,0], X_fem[:,1], X_fem[:,2:], mu_fem
        )

        # ru_ntk = vmap(ntk_fn, (None, None, 0, 0, 0, 0, 0))(
        #     self.ru_net, params, t_fem, X_fem[:,0], X_fem[:,1], X_fem[:,2:], mu_fem
        # )
        # rv_ntk = vmap(ntk_fn, (None, None, 0, 0, 0, 0, 0))(
        #     self.rv_net, params, t_fem, X_fem[:,0], X_fem[:,1], X_fem[:,2:], mu_fem
        # )
        # rc_ntk = vmap(ntk_fn, (None, None, 0, 0, 0, 0, 0))(
        #     self.rc_net, params, t_fem, X_fem[:,0], X_fem[:,1], X_fem[:,2:], mu_fem
        # )
        # rs_ntk = vmap(ntk_fn, (None, None, 0, 0, 0, 0, 0))(
        #     self.rs_net, params, t_fem, X_fem[:,0], X_fem[:,1], X_fem[:,2:], mu_fem
        # )

        ntk_dict = {
            "u_data": u_data_ntk,
            "v_data": v_data_ntk,
            "p_data": p_data_ntk,
            "s_data": s_data_ntk,
            "u_ic": u_ic_ntk,
            "v_ic": v_ic_ntk,
            "p_ic": p_ic_ntk,
            "s_ic": s_ic_ntk,
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
        
        (fields, fields_ic) = res_batch
        (X_fem, t_fem, mu_fem, u_fem_q, v_fem_q, p_fem_q, s_fem_q) = fields
        (u_ic, v_ic, p_ic, s_ic) = fields_ic

        u_fem_q_pred = self.ufem_pred_fn(params, t_fem, X_fem[:,0], X_fem[:,1], X_fem[:,2:], mu_fem)
        v_fem_q_pred = self.vfem_pred_fn(params, t_fem, X_fem[:,0], X_fem[:,1], X_fem[:,2:], mu_fem)
        p_fem_q_pred = self.pfem_pred_fn(params, t_fem, X_fem[:,0], X_fem[:,1], X_fem[:,2:], mu_fem)
        s_fem_q_pred = self.sfem_pred_fn(params, t_fem, X_fem[:,0], X_fem[:,1], X_fem[:,2:], mu_fem)
        u_data = jnp.mean(jnp.abs(u_fem_q_pred - u_fem_q  ) ** 1)
        v_data = jnp.mean(jnp.abs(v_fem_q_pred - v_fem_q  ) ** 1)
        p_data = jnp.mean(jnp.abs(p_fem_q_pred - p_fem_q  ) ** 1)
        s_data = jnp.mean(jnp.abs(s_fem_q_pred - s_fem_q  ) ** 1)
        
        u_ic_pred = self.u0_pred_fn(params, 0.0, X_fem[:,0], X_fem[:,1], X_fem[:,2:], mu_fem)
        v_ic_pred = self.v0_pred_fn(params, 0.0, X_fem[:,0], X_fem[:,1], X_fem[:,2:], mu_fem)
        p_ic_pred = self.p0_pred_fn(params, 0.0, X_fem[:,0], X_fem[:,1], X_fem[:,2:], mu_fem)
        s_ic_pred = self.s0_pred_fn(params, 0.0, X_fem[:,0], X_fem[:,1], X_fem[:,2:], mu_fem)
        u_ic_loss = jnp.mean(jnp.abs(u_ic_pred - u_ic ) ** 1)
        v_ic_loss = jnp.mean(jnp.abs(v_ic_pred - v_ic ) ** 1) 
        p_ic_loss = jnp.mean(jnp.abs(p_ic_pred - p_ic ) ** 1)
        s_ic_loss = jnp.mean(jnp.abs(s_ic_pred - s_ic ) ** 1)
        
        # ru = 0
        # rv = 0
        # rc = 0
        # rs = 0
        # ru_pred, rv_pred, rc_pred, rs_pred = self.r_pred_fn( params, t_fem, X_fem[:,0], X_fem[:,1], X_fem[:,2:], mu_fem)
        # ru += jnp.mean(ru_pred**2)
        # rv += jnp.mean(rv_pred**2)
        # rc += jnp.mean(rc_pred**2)
        # rs += jnp.mean(rs_pred**2)  

        # ru_pred, rv_pred, rc_pred, rs_pred = self.r_pred_fn_t( params, 0.0, X_fem[:,0], X_fem[:,1], X_fem[:,2:], mu_fem)
        # ru += jnp.mean(ru_pred**2)
        # rv += jnp.mean(rv_pred**2)
        # rc += jnp.mean(rc_pred**2)
        # rs += jnp.mean(rs_pred**2)
        
        # ru_loss = jnp.mean( jnp.array(ru)) 
        # rv_loss = jnp.mean( jnp.array(rv))
        # rc_loss = jnp.mean( jnp.array(rc))
        # rs_loss = jnp.mean( jnp.array(rs))

        loss_dict = {
            "u_data": u_data,
            "v_data": v_data,
            "p_data": p_data,
            "s_data": s_data,
            "u_ic": u_ic_loss,
            "v_ic": v_ic_loss,
            "p_ic": p_ic_loss,
            "s_ic": s_ic_loss,
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
