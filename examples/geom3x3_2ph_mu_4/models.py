from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, jit, grad, vmap
from jax.tree_util import tree_map

import optax

from jaxpi import archs
from jaxpi.models import ForwardBVP, ForwardIVP
from jaxpi.utils import ntk_fn
from jaxpi.evaluator import BaseEvaluator

from flax import linen as nn


class NavierStokes2DwSat(ForwardIVP):

    def __init__(self, config, p_inflow, temporal_dom, coords, U_max, L_max, fluid_params, uv_max):   
        super().__init__(config)

        # self.inflow_fn = inflow_fn
        self.epoch = 0
        self.p_in = p_inflow 
        self.temporal_dom = temporal_dom
        self.coords = coords
        self.U_max = U_max
        self.L_max = L_max
        # self.Re = Re  # Reynolds number
        self.fluid_params = fluid_params

        self.uv_max = uv_max

        # Non-dimensionalized domain length and width
        self.L, self.W = self.coords.max(axis=0) - self.coords.min(axis=0)


        # Predict functions over batch
        self.u0_pred_fn = vmap(self.u_net, (None, None, 0, 0, 0))
        self.v0_pred_fn = vmap(self.v_net, (None, None, 0, 0, 0))
        self.p0_pred_fn = vmap(self.p_net, (None, None, 0, 0, 0))
        self.s0_pred_fn = vmap(self.s_net, (None, None, 0, 0, 0))
              
        self.u_pred_1_fn = vmap(self.u_net, (None, 0, 0, 0, None))
        self.v_pred_1_fn = vmap(self.v_net, (None, 0, 0, 0, None))
        
        self.ufem_pred_fn = vmap(self.u_net, (None, 0, 0, 0, None))
        self.vfem_pred_fn = vmap(self.v_net, (None, 0, 0, 0, None))
        self.pfem_pred_fn = vmap(self.p_net, (None, 0, 0, 0, None))
        self.sfem_pred_fn = vmap(self.s_net, (None, 0, 0, 0, None))

        self.u_pred_fn = vmap(self.u_net, (None, 0, 0, 0, 0))
        self.v_pred_fn = vmap(self.v_net, (None, 0, 0, 0, 0))
        self.p_pred_fn = vmap(self.p_net, (None, 0, 0, 0, 0))
        self.s_pred_fn = vmap(self.s_net, (None, 0, 0, 0, 0))
        self.r_pred_fn = vmap(self.r_net, (None, 0, 0, 0, 0))
        self.r_pred_fem_fn = vmap(self.r_net, (None, 0, 0, 0, None))
    
    def update_step(self, step):
        self.epoch = step
        
    def __nonlinear_scaler(self, mu, umax_q, umax_s):
        mus = .1
        muq = .0025
        
        y_scaler =  self.__nonlinear_scaler_equation(mu, mus, muq, umax_q, umax_s)                   
        return y_scaler

    def __nonlinear_scaler_equation(self, mu, mus, muq, umax_q, umax_s):
        
        a = (umax_s - umax_q) / ( (mus**(-2) - muq**(-2)) )
        b = umax_q - a * muq**(-2)
        
        return a*mu**(-2) +b

    def neural_net(self, params, t, x, y, mu):
        
        (u_maxq, v_maxq, u_maxs, v_maxs) = self.uv_max
        #  def __nonlinear_scaler(self, mu, umax_q, umax_s)
        # u_scaler = self.__nonlinear_scaler(mu, u_maxq, u_maxs)
        # v_scaler = self.__nonlinear_scaler(mu, v_maxq, v_maxs)
        
        t = t / (self.temporal_dom[1])  # rescale t into [0, 1]
        x = x / self.L  # rescale x into [0, 1]
        y = y / self.W  # rescale y into [0, 1]   mu = (mu-.045)/(.1 - .045)
        mu = 2* ((mu - .0025) / (.1 - .0025)) -1
        ones = jnp.stack([jnp.ones(t.shape)])
        inputs = jnp.stack([t, x, y]) # branch
        mu = jnp.stack([ mu ] )
        # mu = jnp.stack([ mu, jnp.exp(mu), jnp.exp(2*mu)])  # trunk
        outputs = self.state.apply_fn(params, inputs, mu, ones)

        # Start with an initial state of the channel flow
        u = outputs[0]
        v = outputs[1]
        p = outputs[2]
        s = outputs[3] # nn.softplus( outputs[3] )
        D = nn.sigmoid(outputs[4]) *5*10**(-3)
        u_scaler = outputs[5] * (u_maxq - u_maxs) + u_maxs
        v_scaler = outputs[6] * (v_maxq - v_maxs) + v_maxs
        
        u = u *u_scaler
        v = v *v_scaler

        return u, v, p, s, D

    def u_net(self, params, t, x, y, mu):
        u, _, _, _, _ = self.neural_net(params, t, x, y, mu)
        return u

    def v_net(self, params, t, x, y, mu):
        _, v, _, _, _ = self.neural_net(params, t, x, y, mu)
        return v

    def p_net(self, params, t, x, y, mu):
        _, _, p, _, _ = self.neural_net(params, t, x, y, mu)
        return p

    def s_net(self, params, t, x, y, mu):
        _, _, _, s, _ = self.neural_net(params, t, x, y, mu)
        return s
    
    def D_net(self, params, t, x, y, mu):
        _, _, _, _, D = self.neural_net(params, t, x, y, mu)
        return D


    def r_net(self, params, t, x, y, mu0):
        # Re = jnp.ones(x.shape)
        ( _, mu1, rho0, rho1) = self.fluid_params

        u , v , p, s, D = self.neural_net(params, t, x, y, mu0)

        u_t = grad(self.u_net, argnums=1)(params, t, x, y, mu0) 
        v_t = grad(self.v_net, argnums=1)(params, t, x, y, mu0) 
        s_t = grad(self.s_net, argnums=1)(params, t, x, y, mu0)

        u_x = grad(self.u_net, argnums=2)(params, t, x, y, mu0)
        v_x = grad(self.v_net, argnums=2)(params, t, x, y, mu0)
        p_x = grad(self.p_net, argnums=2)(params, t, x, y, mu0)
        s_x = grad(self.s_net, argnums=2)(params, t, x, y, mu0)

        u_y = grad(self.u_net, argnums=3)(params, t, x, y, mu0)
        v_y = grad(self.v_net, argnums=3)(params, t, x, y, mu0)
        p_y = grad(self.p_net, argnums=3)(params, t, x, y, mu0)
        s_y = grad(self.s_net, argnums=3)(params, t, x, y, mu0)

        u_xx = grad(grad(self.u_net, argnums=2), argnums=2)(params, t, x, y, mu0)
        v_xx = grad(grad(self.v_net, argnums=2), argnums=2)(params, t, x, y, mu0)
        s_xx = grad(grad(self.s_net, argnums=2), argnums=2)(params, t, x, y, mu0)

        u_yy = grad(grad(self.u_net, argnums=3), argnums=3)(params, t, x, y, mu0)
        v_yy = grad(grad(self.v_net, argnums=3), argnums=3)(params, t, x, y, mu0)
        s_yy = grad(grad(self.s_net, argnums=3), argnums=3)(params, t, x, y, mu0)

        Re = rho0*self.U_max*(self.L_max)/mu1  
        Re = .1
        mu = (1-s)*mu1 + s*mu0
        mu_ratio = mu/mu1
                
        # PDE residual
        ru = u_t + u * u_x + v * u_y + (p_x - mu_ratio*(u_xx + u_yy)) / Re #  
        rv = v_t + u * v_x + v * v_y + (p_y - mu_ratio*(v_xx + v_yy)) / Re #
        rc = u_x + v_y
        rs = s_t + u * s_x + v * s_y - D*(s_xx + s_yy)  #self.D*(s_xx + s_yy)  #.0001*(s_xx + s_yy) 

        return ru, rv, rc, rs

    def ru_net(self, params, t, x, y, mu):
        ru, _, _, _ = self.r_net(params, t, x, y, mu)
        return ru

    def rv_net(self, params, t, x, y, mu):
        _, rv, _, _ = self.r_net(params, t, x, y, mu)
        return rv

    def rc_net(self, params, t, x, y, mu):
        _, _, rc, _ = self.r_net(params, t, x, y, mu)
        return rc

    def rs_net(self, params, t, x, y, mu):
        _, _, _, rs = self.r_net(params, t, x, y, mu)
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
        ic_batch = batch["ic"]
        ic_batch_qs = batch["ic_qs"]
        # ic_batch_s = batch["ic_s"]
        inflow_batch = batch["inflow"]
        outflow_batch = batch["outflow"]
        noslip_batch = batch["noslip"]
        res_batch = batch["res"]
        

        (coords_fem, t_fem, u_fem_q, v_fem_q, p_fem_q, s_fem_q, u_fem_s, v_fem_s, p_fem_s, s_fem_s) = ic_batch_qs
        # (coords_fem, t_fem, u_fem_s, v_fem_s, p_fem_s, s_fem_s) = ic_batch_s

        u_data_ntk = vmap(ntk_fn, (None, None, 0, 0, 0, None))(
            self.u_net, params, t_fem, coords_fem[:, 0], coords_fem[:, 1], .0025)
        v_data_ntk = vmap(ntk_fn, (None, None, 0, 0, 0, None))(
            self.v_net, params, t_fem, coords_fem[:, 0], coords_fem[:, 1], .0025)
        p_data_ntk = vmap(ntk_fn, (None, None, 0, 0, 0, None))(
            self.p_net, params, t_fem, coords_fem[:, 0], coords_fem[:, 1], .0025)
        s_data_ntk = vmap(ntk_fn, (None, None, 0, 0, 0, None))(
            self.s_net, params, t_fem, coords_fem[:, 0], coords_fem[:, 1], .0025)
        
        coords_batch, u_batch, v_batch, p_batch, s_batch, mu_batch = ic_batch

        u_ic_ntk = vmap(ntk_fn, (None, None, None, 0, 0, 0))(
            self.u_net, params, 0.0, coords_batch[:, 0], coords_batch[:, 1], mu_batch
        )
        v_ic_ntk = vmap(ntk_fn, (None, None, None, 0, 0, 0))(
            self.v_net, params, 0.0, coords_batch[:, 0], coords_batch[:, 1], mu_batch
        )
        p_ic_ntk = vmap(ntk_fn, (None, None, None, 0, 0, 0))(
            self.p_net, params, 0.0, coords_batch[:, 0], coords_batch[:, 1], mu_batch
        )
        s_ic_ntk = vmap(ntk_fn, (None, None, None, 0, 0, 0))(
            self.s_net, params, 0.0, coords_batch[:, 0], coords_batch[:, 1], mu_batch
        )

        p_in_ntk = vmap(ntk_fn, (None, None, 0, 0, 0, 0))(
            self.p_net,
            params,
            inflow_batch[:, 0],
            inflow_batch[:, 1],
            inflow_batch[:, 2],
            inflow_batch[:, 3],
        )
        v_in_ntk = vmap(ntk_fn, (None, None, 0, 0, 0, 0))(
            self.v_net,
            params,
            inflow_batch[:, 0],
            inflow_batch[:, 1],
            inflow_batch[:, 2],
            inflow_batch[:, 3],
        )
        s_in_ntk = vmap(ntk_fn, (None, None, 0, 0, 0, 0))(
            self.s_net,
            params,
            inflow_batch[:, 0],
            inflow_batch[:, 1],
            inflow_batch[:, 2],
            inflow_batch[:, 3]
        )

        p_out_ntk = vmap(ntk_fn, (None, None, 0, 0, 0, 0))(
            self.p_net,
            params,
            outflow_batch[:, 0],
            outflow_batch[:, 1],
            outflow_batch[:, 2],
            outflow_batch[:, 3]
        )
        v_out_ntk = vmap(ntk_fn, (None, None, 0, 0, 0, 0))(
            self.v_net,
            params,
            outflow_batch[:, 0],
            outflow_batch[:, 1],
            outflow_batch[:, 2],
            outflow_batch[:, 3]
        )

        u_noslip_ntk = vmap(ntk_fn, (None, None, 0, 0, 0, 0))(
            self.u_net,
            params,
            noslip_batch[:, 0],
            noslip_batch[:, 1],
            noslip_batch[:, 2],
            noslip_batch[:, 3]
        )
        v_noslip_ntk = vmap(ntk_fn, (None, None, 0, 0, 0, 0))(
            self.v_net,
            params,
            noslip_batch[:, 0],
            noslip_batch[:, 1],
            noslip_batch[:, 2],
            noslip_batch[:, 3]
        )
        u_noslip_ntk = jnp.mean(u_noslip_ntk + v_noslip_ntk)
        # Consider the effect of causal weights
        if self.config.weighting.use_causal:
            res_batch = jnp.array(
                [res_batch[:, 0].sort(), res_batch[:, 1], res_batch[:, 2], res_batch[:, 3]]
            ).T
            ru_ntk = vmap(ntk_fn, (None, None, 0, 0, 0, 0))(
                self.ru_net, params, res_batch[:, 0], res_batch[:, 1], res_batch[:, 2], res_batch[:, 3]
            )
            rv_ntk = vmap(ntk_fn, (None, None, 0, 0, 0, 0))(
                self.rv_net, params, res_batch[:, 0], res_batch[:, 1], res_batch[:, 2], res_batch[:, 3]
            )
            rc_ntk = vmap(ntk_fn, (None, None, 0, 0, 0, 0))(
                self.rc_net, params, res_batch[:, 0], res_batch[:, 1], res_batch[:, 2], res_batch[:, 3]
            )
            rs_ntk = vmap(ntk_fn, (None, None, 0, 0, 0, 0))(
                self.rs_net, params, res_batch[:, 0], res_batch[:, 1], res_batch[:, 2], res_batch[:, 3]
            )

            ru_ntk = ru_ntk.reshape(self.num_chunks, -1)  # shape: (num_chunks, -1)
            rv_ntk = rv_ntk.reshape(self.num_chunks, -1)
            rc_ntk = rc_ntk.reshape(self.num_chunks, -1)
            rs_ntk = rs_ntk.reshape(self.num_chunks, -1)

            ru_ntk = jnp.mean(
                ru_ntk, axis=1
            )  # average convergence rate over each chunk
            rv_ntk = jnp.mean(rv_ntk, axis=1)
            rc_ntk = jnp.mean(rc_ntk, axis=1)
            rs_ntk = jnp.mean(rs_ntk, axis=1)

            _, _, _, _, causal_weights = self.res_and_w(params, res_batch)
            ru_ntk = ru_ntk * causal_weights  # multiply by causal weights
            rv_ntk = rv_ntk * causal_weights
            rc_ntk = rc_ntk * causal_weights
            rs_ntk = rs_ntk * causal_weights
        else:
            ru_ntk = vmap(ntk_fn, (None, None, 0, 0, 0, 0))(
                self.ru_net, params, res_batch[0][:, 0], res_batch[0][:, 1], res_batch[0][:, 2], res_batch[0][:, 3]
            )
            rv_ntk = vmap(ntk_fn, (None, None, 0, 0, 0, 0))(
                self.rv_net, params, res_batch[0][:, 0], res_batch[0][:, 1], res_batch[0][:, 2], res_batch[0][:, 3]
            )
            rc_ntk = vmap(ntk_fn, (None, None, 0, 0, 0, 0))(
                self.rc_net, params, res_batch[0][:, 0], res_batch[0][:, 1], res_batch[0][:, 2], res_batch[0][:, 3]
            )
            rs_ntk = vmap(ntk_fn, (None, None, 0, 0, 0, 0))(
                self.rs_net, params, res_batch[0][:, 0], res_batch[0][:, 1], res_batch[0][:, 2], res_batch[0][:, 3]
            )

        ntk_dict = {
            "u_data": u_data_ntk,
            "v_data": v_data_ntk,
            "p_data": p_data_ntk,
            "s_data": s_data_ntk,
            "u_ic": u_ic_ntk,
            "v_ic": v_ic_ntk,
            "p_ic": p_ic_ntk,
            "s_ic": s_ic_ntk,
            "p_in": p_in_ntk,
            "p_out": p_out_ntk,
            # "s_in": s_in_ntk,
            # "v_in": v_in_ntk,
            # "v_out": v_out_ntk,
            "u_noslip": u_noslip_ntk,
            # "v_noslip": v_noslip_ntk,
            "ru": ru_ntk, #
            "rv": rv_ntk, #
            "rc": rc_ntk,
            "rs": rs_ntk,
        }

        return ntk_dict
    
    @partial(jit, static_argnums=(0,))
    def losses(self, params, batch):
        # Unpack batch
        ic_batch = batch["ic"]
        ic_batch_qs = batch["ic_qs"]
        # ic_batch_s = batch["ic_s"]
        inflow_batch = batch["inflow"]
        outflow_batch = batch["outflow"]
        noslip_batch = batch["noslip"]
        res_batch = batch["res"]
        
        (u_maxq, v_maxq, u_maxs, v_maxs) = self.uv_max
        (coords_fem, t_fem, u_fem_q, v_fem_q, p_fem_q, s_fem_q, u_fem_s, v_fem_s, p_fem_s, s_fem_s) = ic_batch_qs
        # (coords_fem, t_fem, u_fem_s, v_fem_s, p_fem_s, s_fem_s) = ic_batch_s
                
        coords_batch, u_batch, v_batch, p_batch, s_batch, mu_batch = ic_batch
        
            
        u_fem_q_pred = self.ufem_pred_fn(params, t_fem, coords_fem[:, 0], coords_fem[:, 1], .0025)
        v_fem_q_pred = self.vfem_pred_fn(params, t_fem, coords_fem[:, 0], coords_fem[:, 1], .0025)
        p_fem_q_pred = self.pfem_pred_fn(params, t_fem, coords_fem[:, 0], coords_fem[:, 1], .0025)
        s_fem_q_pred = self.sfem_pred_fn(params, t_fem, coords_fem[:, 0], coords_fem[:, 1], .0025)
        
        u_fem_s_pred = self.ufem_pred_fn(params, t_fem, coords_fem[:, 0], coords_fem[:, 1], .1)
        v_fem_s_pred = self.vfem_pred_fn(params, t_fem, coords_fem[:, 0], coords_fem[:, 1], .1)
        p_fem_s_pred = self.pfem_pred_fn(params, t_fem, coords_fem[:, 0], coords_fem[:, 1], .1)
        s_fem_s_pred = self.sfem_pred_fn(params, t_fem, coords_fem[:, 0], coords_fem[:, 1], .1)
        
        # u_fem_res_pred = self.u_pred_fn(params, t_fem, coords_fem[:, 0], coords_fem[:, 1], mu_batch)
        # v_fem_res_pred = self.v_pred_fn(params, t_fem, coords_fem[:, 0], coords_fem[:, 1], mu_batch)
        # s_fem_res_pred = self.s_pred_fn(params, t_fem, coords_fem[:, 0], coords_fem[:, 1], mu_batch)
        # u_fem_res_pred = jnp.mean((u_fem_res_pred - (u_fem_q/u_maxq)*((self.__nonlinear_scaler(mu_batch, u_maxq, u_maxs)-u_maxs))+u_fem_s) ** 2)
        # v_fem_res_pred = jnp.mean((v_fem_res_pred - (v_fem_q/v_maxq)*((self.__nonlinear_scaler(mu_batch, v_maxq, v_maxs)-v_maxs))+v_fem_s) ** 2)
        # s_slider = (1/u_maxq)*((self.__nonlinear_scaler(mu_batch, u_maxq, u_maxs)-u_maxs))
        # s_fem_interpol = (s_fem_q*s_slider + s_fem_s*(1-s_slider))
        # s_fem_res_pred = jnp.mean( (s_fem_res_pred - s_fem_interpol) **2 )
         
        # mult = jax.lax.cond( # True for selected , False for not selected
        #     self.epoch>100000,
        #     lambda _: 0.0,  # True case
        #     lambda _: 1.-(self.epoch/100000)*.9,   # False case
        #     operand=None
        # )
        
        u_data = jnp.mean( jnp.mean((u_fem_q_pred - u_fem_q) ** 2) + jnp.mean((u_fem_s_pred - u_fem_s) ** 2) )# + u_fem_res_pred*mult ) ##
        v_data = jnp.mean( jnp.mean((v_fem_q_pred - v_fem_q) ** 2) + jnp.mean((v_fem_s_pred - v_fem_s) ** 2) )# + v_fem_res_pred*mult ) ##
        p_data = jnp.mean( jnp.mean((p_fem_q_pred - p_fem_q) ** 2) + jnp.mean((p_fem_s_pred - p_fem_s) ** 2) )  ##
        s_data = jnp.mean( jnp.mean((s_fem_q_pred - s_fem_q) ** 2) + jnp.mean((s_fem_s_pred - s_fem_s) ** 2) )# + s_fem_res_pred*mult )  ##
        
        # Initial condition loss
        u_ic_pred = self.u0_pred_fn(params, 0.0, coords_batch[:, 0], coords_batch[:, 1], mu_batch)
        v_ic_pred = self.v0_pred_fn(params, 0.0, coords_batch[:, 0], coords_batch[:, 1], mu_batch)
        p_ic_pred = self.p0_pred_fn(params, 0.0, coords_batch[:, 0], coords_batch[:, 1], mu_batch)
        s_ic_pred = self.s0_pred_fn(params, 0.0, coords_batch[:, 0], coords_batch[:, 1], mu_batch)
        u_ic_loss = jnp.mean((u_ic_pred - u_batch) ** 2)
        v_ic_loss = jnp.mean((v_ic_pred - v_batch) ** 2)
        p_ic_loss = jnp.mean((p_ic_pred - p_batch) ** 2)
        s_ic_loss = jnp.mean((s_ic_pred - s_batch) ** 2)
        u_ic_pred = self.ufem_pred_fn(params, 
                                    jax.random.uniform(jax.random.PRNGKey(0), shape=(coords_batch.shape[0],),
                                       minval=0, maxval=500),
                                    coords_batch[:, 0], coords_batch[:, 1], .05)
        v_ic_pred = self.vfem_pred_fn(params, 
                                    jax.random.uniform(jax.random.PRNGKey(0), shape=(coords_batch.shape[0],),
                                       minval=0, maxval=500),
                                    coords_batch[:, 0], coords_batch[:, 1], .05)
        p_ic_pred = self.pfem_pred_fn(params, 
                                    jax.random.uniform(jax.random.PRNGKey(0), shape=(coords_batch.shape[0],),
                                       minval=0, maxval=500),
                                    coords_batch[:, 0], coords_batch[:, 1], .05)
        u_ic_loss = jnp.mean( u_ic_loss + jnp.mean((u_ic_pred - u_batch) ** 2) )
        v_ic_loss = jnp.mean( v_ic_loss + jnp.mean((v_ic_pred - v_batch) ** 2) )
        p_ic_loss = jnp.mean( p_ic_loss + jnp.mean((p_ic_pred - p_batch) ** 2) )

        # inflow outflow loss
        p_in_pred = self.p_pred_fn(
            params, inflow_batch[:, 0], inflow_batch[:, 1], inflow_batch[:, 2], inflow_batch[:, 3]
        )

        p_in_loss = jnp.mean((p_in_pred - self.p_in) ** 2)
        #
        p_out_pred = self.p_pred_fn(
            params, outflow_batch[:, 0], outflow_batch[:, 1], outflow_batch[:, 2], outflow_batch[:, 3]
        )
        p_out_loss = jnp.mean((p_out_pred) ** 2)
        #
        s_in_pred = self.s_pred_fn(
            params, inflow_batch[:, 0], inflow_batch[:, 1], inflow_batch[:, 2], inflow_batch[:, 3]
        )
        s_in_loss = jnp.mean((s_in_pred - 1.0) ** 2)
        s_ic_loss = jnp.mean(s_ic_loss + s_in_loss)
        #
        v_in_pred = self.v_pred_fn(
            params, inflow_batch[:, 0], inflow_batch[:, 1], inflow_batch[:, 2], inflow_batch[:, 3]
        )
        v_in_loss = jnp.mean((v_in_pred) ** 2)
        #
        v_out_pred = self.v_pred_fn(
            params, outflow_batch[:, 0], outflow_batch[:, 1], outflow_batch[:, 2], outflow_batch[:, 3]
        )
        v_out_loss = jnp.mean((v_out_pred) ** 2)
        #
        # noslip loss
        u_noslip_pred = self.u_pred_fn(
            params, noslip_batch[:, 0], noslip_batch[:, 1], noslip_batch[:, 2], noslip_batch[:, 3]
        )
        v_noslip_pred = self.v_pred_fn(
            params, noslip_batch[:, 0], noslip_batch[:, 1], noslip_batch[:, 2], noslip_batch[:, 3]
        )

        u_noslip_loss = jnp.mean(u_noslip_pred**2)
        v_noslip_loss = jnp.mean(v_noslip_pred**2)
        u_noslip_loss = jnp.mean(u_noslip_loss + v_noslip_loss)

        # residual loss
        if self.config.weighting.use_causal == True:
            ru_l, rv_l, rc_l, rs_l, gamma = self.res_and_w(params, jnp.concatenate((res_batch, noslip_batch), axis=0))
            ru1 = jnp.mean(gamma * ru_l)
            rv1 = jnp.mean(gamma * rv_l)
            rc1 = jnp.mean(gamma * rc_l)
            rs1 = jnp.mean(gamma * rs_l)
            
            ru_l, rv_l, rc_l, rs_l, gamma = self.res_and_w(params, 
                    jnp.concatenate(
                        (                    
                            jnp.concatenate((t_fem[:,jnp.newaxis], coords_fem, jnp.ones(t_fem.shape)[:,jnp.newaxis]*.0025), axis=1),
                            jnp.concatenate((t_fem[:,jnp.newaxis], coords_fem, jnp.ones(t_fem.shape)[:,jnp.newaxis]*.1), axis=1),
                            jnp.concatenate((
                                             jax.random.uniform(jax.random.PRNGKey(0), shape=(coords_batch.shape[0],),
                                       minval=0, maxval=500)[:,jnp.newaxis],
                                                      coords_batch, jnp.ones(coords_batch.shape[0])[:,jnp.newaxis]*.05), axis=1)
                        )
                    , axis=0)
            )
            
            ru2 = jnp.mean(gamma * ru_l)
            rv2 = jnp.mean(gamma * rv_l)
            rc2 = jnp.mean(gamma * rc_l)
            rs2 = jnp.mean(gamma * rs_l)
            
            ru_loss = jnp.mean( ru1 + ru2 )
            rv_loss = jnp.mean( rv1 + rv2 )
            rc_loss = jnp.mean( rc1 + rc2 )
            rs_loss = jnp.mean( rs1 + rs2 ) 
            
        else:
            ru_pred, rv_pred, rc_pred, rs_pred = self.r_pred_fn(
                params, 
                jnp.concatenate( ( res_batch[0][:, 0], res_batch[1][:, 0], t_fem            ), axis=0),
                jnp.concatenate( ( res_batch[0][:, 1], res_batch[1][:, 1], coords_fem[:, 0] ), axis=0),
                jnp.concatenate( ( res_batch[0][:, 2], res_batch[1][:, 2], coords_fem[:, 1] ), axis=0),
                jnp.concatenate( ( res_batch[0][:, 3], res_batch[1][:, 3], mu_batch         ), axis=0),
            )
            ru1 =  jnp.mean(ru_pred**2)
            rv1 =  jnp.mean(rv_pred**2)
            rc1 =  jnp.mean(rc_pred**2)
            rs1 =  jnp.mean(rs_pred**2)

            ru_pred, rv_pred, rc_pred, rs_pred = self.r_pred_fem_fn(  params,
             jax.random.uniform(jax.random.PRNGKey(0), shape=(coords_batch.shape[0],),
                                       minval=0, maxval=500),
                                    coords_batch[:, 0], coords_batch[:, 1], .05
            )
            ru6 =  jnp.mean(ru_pred**2)
            rv6 =  jnp.mean(rv_pred**2)
            rc6 =  jnp.mean(rc_pred**2)
            rs6 =  jnp.mean(rs_pred**2)

            ru_pred, rv_pred, rc_pred, rs_pred = self.r_pred_fem_fn(  params,
                res_batch[0][:, 0], res_batch[0][:, 1], res_batch[0][:, 2], res_batch[0][0, 3]
            )
            ru7 =  jnp.mean(ru_pred**2)
            rv7 =  jnp.mean(rv_pred**2)
            rc7 =  jnp.mean(rc_pred**2)
            rs7 =  jnp.mean(rs_pred**2)
            
            ru_pred, rv_pred, rc_pred, rs_pred = self.r_pred_fem_fn(  params,
                res_batch[1][:, 0], res_batch[1][:, 1], res_batch[1][:, 2], res_batch[1][0, 3]
            )
            ru8 =  jnp.mean(ru_pred**2)
            rv8 =  jnp.mean(rv_pred**2)
            rc8 =  jnp.mean(rc_pred**2)
            rs8 =  jnp.mean(rs_pred**2)

            ru_loss = jnp.mean( ru1 + ru6 + ru7 + ru8 ) 
            rv_loss = jnp.mean( rv1 + rv6 + rv7 + rv8 )
            rc_loss = jnp.mean( rc1 + rc6 + rc7 + rc8 )
            rs_loss = jnp.mean( rs1 + rs6 + rs7 + rs8 )       

        loss_dict = {
            "u_data": u_data,
            "v_data": v_data,
            "p_data": p_data,
            "s_data": s_data,
            "u_ic": u_ic_loss,
            "v_ic": v_ic_loss,
            "p_ic": p_ic_loss,
            "s_ic": s_ic_loss,
            "p_in": p_in_loss,
            "p_out": p_out_loss,
            # "s_in": s_in_loss,
            # "v_in": v_in_loss,
            # "v_out": v_out_loss,
            "u_noslip": u_noslip_loss,
            # "v_noslip": v_noslip_loss,
            "ru": ru_loss,
            "rv": rv_loss,
            "rc": rc_loss,
            "rs": rs_loss,
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
