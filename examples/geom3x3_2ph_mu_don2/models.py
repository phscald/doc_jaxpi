from functools import partial

import jax
import jax.numpy as jnp
from jax import lax, jit, grad, vmap
from jax.tree_util import tree_map

import optax

from jaxpi import archs
from jaxpi.models import ForwardBVP, ForwardIVP
from jaxpi.utils import ntk_fn
from jaxpi.evaluator import BaseEvaluator

from flax import linen as nn


class NavierStokes2DwSat(ForwardIVP):

    def __init__(self, config, p_inflow, temporal_dom, coords, U_max, L_max, fluid_params, D):   
        super().__init__(config)

        # self.inflow_fn = inflow_fn
        self.p_in = p_inflow 
        self.temporal_dom = temporal_dom
        self.coords = coords
        self.U_max = U_max
        self.L_max = L_max
        # self.Re = Re  # Reynolds number
        self.fluid_params = fluid_params
        self.D = D

        # Non-dimensionalized domain length and width
        self.L, self.W = self.coords.max(axis=0) - self.coords.min(axis=0)


        # Predict functions over batch
        self.u0_pred_fn = vmap(self.u_net, (None, None, 0, 0, None))
        self.v0_pred_fn = vmap(self.v_net, (None, None, 0, 0, None))
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

    def neural_net(self, params, t, x, y, mu):
        t = t / (self.temporal_dom[1])  # rescale t into [0, 1]
        x = x / self.L  # rescale x into [0, 1]
        y = y / self.W  # rescale y into [0, 1]
        mu = (mu-.01)/.05 
        inputs = jnp.stack([t, x, y]) # branch
        mu = jnp.stack([mu])  # trunk
        outputs = self.state.apply_fn(params, inputs, mu)

        # Start with an initial state of the channel flow
        u = outputs[0]
        v = outputs[1]
        p = outputs[2]
        s = outputs[3]
        # return u, v, p, s
        return u*0.005, v*0.0005, p, s
    # u*0.01, v*0.001, p
    # lembrar de copiar as condiçoes iniciais no folder geom1x2

    def u_net(self, params, t, x, y, mu):
        u, _, _, _ = self.neural_net(params, t, x, y, mu)
        return u

    def v_net(self, params, t, x, y, mu):
        _, v, _, _ = self.neural_net(params, t, x, y, mu)
        return v

    def p_net(self, params, t, x, y, mu):
        _, _, p, _ = self.neural_net(params, t, x, y, mu)
        return p

    def s_net(self, params, t, x, y, mu):
        _, _, _, s = self.neural_net(params, t, x, y, mu)
        return s


    def r_net(self, params, t, x, y, mu1):
        # Re = jnp.ones(x.shape)
        (mu0, _, rho0, rho1) = self.fluid_params

        u, v, p, s = self.neural_net(params, t, x, y, mu1)

        u_t = grad(self.u_net, argnums=1)(params, t, x, y, mu1)
        v_t = grad(self.v_net, argnums=1)(params, t, x, y, mu1)
        s_t = grad(self.s_net, argnums=1)(params, t, x, y, mu1)

        u_x = grad(self.u_net, argnums=2)(params, t, x, y, mu1)
        v_x = grad(self.v_net, argnums=2)(params, t, x, y, mu1)
        p_x = grad(self.p_net, argnums=2)(params, t, x, y, mu1)
        s_x = grad(self.s_net, argnums=2)(params, t, x, y, mu1)

        u_y = grad(self.u_net, argnums=3)(params, t, x, y, mu1)
        v_y = grad(self.v_net, argnums=3)(params, t, x, y, mu1)
        p_y = grad(self.p_net, argnums=3)(params, t, x, y, mu1)
        s_y = grad(self.s_net, argnums=3)(params, t, x, y, mu1)

        u_xx = grad(grad(self.u_net, argnums=2), argnums=2)(params, t, x, y, mu1)
        v_xx = grad(grad(self.v_net, argnums=2), argnums=2)(params, t, x, y, mu1)
        s_xx = grad(grad(self.s_net, argnums=2), argnums=2)(params, t, x, y, mu1)

        u_yy = grad(grad(self.u_net, argnums=3), argnums=3)(params, t, x, y, mu1)
        v_yy = grad(grad(self.v_net, argnums=3), argnums=3)(params, t, x, y, mu1)
        s_yy = grad(grad(self.s_net, argnums=3), argnums=3)(params, t, x, y, mu1)


        Re = rho0*self.U_max*(self.L_max*.112**2)/mu0
        mu = (1-s)*mu1 + s*mu0
        mu_ratio = mu/mu0
                
        # PDE residual
        ru = u_t + u * u_x + v * u_y + (p_x - mu_ratio*(u_xx + u_yy)) / Re
        rv = v_t + u * v_x + v * v_y + (p_y - mu_ratio*(v_xx + v_yy)) / Re
        rc = u_x + v_y
        rs = s_t + u * s_x + v * s_y - self.D*(s_xx + s_yy) 

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
        ic_batch1 = batch["ic1"]
        ic_batch5 = batch["ic5"]
        inflow_batch = batch["inflow"]
        outflow_batch = batch["outflow"]
        noslip_batch = batch["noslip"]
        res_batch = batch["res"]
        
        (coords_fem1, t_fem1, u_fem1, v_fem1, p_fem1, s_fem1) = ic_batch1
        (coords_fem5, t_fem5, u_fem5, v_fem5, p_fem5, s_fem5) = ic_batch5

        u_data_ntk = vmap(ntk_fn, (None, None, 0, 0, 0, None))(
            self.u_net, params, t_fem1, coords_fem1[:, 0], coords_fem1[:, 1], .01)
        v_data_ntk = vmap(ntk_fn, (None, None, 0, 0, 0, None))(
            self.v_net, params, t_fem1, coords_fem1[:, 0], coords_fem1[:, 1], .01)
        p_data_ntk = vmap(ntk_fn, (None, None, 0, 0, 0, None))(
            self.p_net, params, t_fem1, coords_fem1[:, 0], coords_fem1[:, 1], .01)
        s_data_ntk = vmap(ntk_fn, (None, None, 0, 0, 0, None))(
            self.s_net, params, t_fem1, coords_fem1[:, 0], coords_fem1[:, 1], .01)
        
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
            "s_in": s_in_ntk,
            # "v_in": v_in_ntk,
            # "v_out": v_out_ntk,
            "u_noslip": u_noslip_ntk,
            "v_noslip": v_noslip_ntk,
            "ru": ru_ntk,
            "rv": rv_ntk,
            "rc": rc_ntk,
            "rs": rs_ntk,
        }

        return ntk_dict
    
    @partial(jit, static_argnums=(0,))
    def losses(self, params, batch):
        # Unpack batch
        ic_batch = batch["ic"]
        ic_batch1 = batch["ic1"]
        ic_batch2 = batch["ic2"]
        ic_batch5 = batch["ic5"]
        inflow_batch = batch["inflow"]
        outflow_batch = batch["outflow"]
        noslip_batch = batch["noslip"]
        res_batch = batch["res"]
        
        (coords_fem1, t_fem1, u_fem1, v_fem1, p_fem1, s_fem1) = ic_batch1
        (coords_fem2, t_fem2, u_fem2, v_fem2, p_fem2, s_fem2) = ic_batch2
        (coords_fem5, t_fem5, u_fem5, v_fem5, p_fem5, s_fem5) = ic_batch5
        
        # coords_batch, u_batch, v_batch, p_batch, s_batch, u05_batch, v05_batch, mu_batch = ic_batch
        # coords_fem1 = coords_batch
        # u_fem1 = u_batch
        # v_fem1 = v_batch
            
        u_fem1_pred = self.ufem_pred_fn(params, t_fem1, coords_fem1[:, 0], coords_fem1[:, 1], .01)
        v_fem1_pred = self.vfem_pred_fn(params, t_fem1, coords_fem1[:, 0], coords_fem1[:, 1], .01)
        p_fem1_pred = self.pfem_pred_fn(params, t_fem1, coords_fem1[:, 0], coords_fem1[:, 1], .01)
        s_fem1_pred = self.sfem_pred_fn(params, t_fem1, coords_fem1[:, 0], coords_fem1[:, 1], .01)
        
        u_fem5_pred = self.ufem_pred_fn(params, t_fem5, coords_fem5[:, 0], coords_fem5[:, 1], .05)
        v_fem5_pred = self.vfem_pred_fn(params, t_fem5, coords_fem5[:, 0], coords_fem5[:, 1], .05)
        p_fem5_pred = self.pfem_pred_fn(params, t_fem5, coords_fem5[:, 0], coords_fem5[:, 1], .05)
        s_fem5_pred = self.sfem_pred_fn(params, t_fem5, coords_fem5[:, 0], coords_fem5[:, 1], .05)
        
        u_fem2_pred = self.ufem_pred_fn(params, t_fem2, coords_fem2[:, 0], coords_fem2[:, 1], .02)
        v_fem2_pred = self.vfem_pred_fn(params, t_fem2, coords_fem2[:, 0], coords_fem2[:, 1], .02)
        p_fem2_pred = self.pfem_pred_fn(params, t_fem2, coords_fem2[:, 0], coords_fem2[:, 1], .02)
        s_fem2_pred = self.sfem_pred_fn(params, t_fem2, coords_fem2[:, 0], coords_fem2[:, 1], .02)
        
        u_data = jnp.mean((u_fem1_pred - u_fem1) ** 2) + jnp.mean((u_fem5_pred - u_fem5) ** 2) + jnp.mean((u_fem2_pred - u_fem2) ** 2)  ##
        v_data = jnp.mean((v_fem1_pred - v_fem1) ** 2) + jnp.mean((v_fem5_pred - v_fem5) ** 2) + jnp.mean((v_fem2_pred - v_fem2) ** 2)  ##
        p_data = jnp.mean((p_fem1_pred - p_fem1) ** 2) + jnp.mean((p_fem5_pred - p_fem5) ** 2) + jnp.mean((p_fem2_pred - p_fem2) ** 2)  ##
        s_data = jnp.mean((s_fem1_pred - s_fem1) ** 2) + jnp.mean((s_fem5_pred - s_fem5) ** 2) + jnp.mean((s_fem2_pred - s_fem2) ** 2)  ##
        
        # Initial condition loss
        coords_batch, u_batch, v_batch, p_batch, s_batch, u05_batch, v05_batch, mu_batch = ic_batch

        u_ic_pred = self.u0_pred_fn(params, 0.0, coords_batch[:, 0], coords_batch[:, 1], .01)
        v_ic_pred = self.v0_pred_fn(params, 0.0, coords_batch[:, 0], coords_batch[:, 1], .01)
        u_ic_loss = jnp.mean((u_ic_pred - u_batch) ** 2)
        v_ic_loss = jnp.mean((v_ic_pred - v_batch) ** 2)
        u_ic_pred = self.u0_pred_fn(params, 0.0, coords_batch[:, 0], coords_batch[:, 1], .05)
        v_ic_pred = self.v0_pred_fn(params, 0.0, coords_batch[:, 0], coords_batch[:, 1], .05)
        u_ic_loss = u_ic_loss + jnp.mean((u_ic_pred - u05_batch) ** 2) ##
        v_ic_loss = v_ic_loss + jnp.mean((v_ic_pred - v05_batch) ** 2) ##
               
        u_ic_pred2 = self.u_pred_1_fn(params,
                                    jax.random.uniform(jax.random.PRNGKey(0), shape=(coords_batch.shape[0],),
                                    minval=0.0, maxval=self.temporal_dom[1]), coords_batch[:, 0], coords_batch[:, 1], .01) #
        v_ic_pred2 = self.v_pred_1_fn(params, 
                                    jax.random.uniform(jax.random.PRNGKey(1), shape=(coords_batch.shape[0],), 
                                    minval=0.0, maxval=self.temporal_dom[1]), coords_batch[:, 0], coords_batch[:, 1], .01) #
        u_ic_loss2 = jnp.mean((u_ic_pred2 - u_batch) ** 2) #
        v_ic_loss2 = jnp.mean((v_ic_pred2 - v_batch) ** 2) #
        u_ic_loss = u_ic_loss+u_ic_loss2
        v_ic_loss = v_ic_loss+v_ic_loss2
        
        p_ic_pred = self.p0_pred_fn(params, 0.0, coords_batch[:, 0], coords_batch[:, 1], mu_batch)
        s_ic_pred = self.s0_pred_fn(params, 0.0, coords_batch[:, 0], coords_batch[:, 1], mu_batch)
        p_ic_loss = jnp.mean((p_ic_pred - p_batch) ** 2)
        s_ic_loss = jnp.mean((s_ic_pred - s_batch) ** 2)
        

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

        # residual loss
        if self.config.weighting.use_causal == True:
            ru_l, rv_l, rc_l, rs_l, gamma = self.res_and_w(params, res_batch)
            ru_loss = jnp.mean(gamma * ru_l)
            rv_loss = jnp.mean(gamma * rv_l)
            rc_loss = jnp.mean(gamma * rc_l)
            rs_loss = jnp.mean(gamma * rs_l)

        else:
            ru_pred, rv_pred, rc_pred, rs_pred = self.r_pred_fn(
                params, res_batch[:, 0], res_batch[:, 1], res_batch[:, 2], res_batch[:, 3]
            )
            ru_loss = jnp.mean(ru_pred**2)
            rv_loss = jnp.mean(rv_pred**2)
            rc_loss = jnp.mean(rc_pred**2)
            rs_loss = jnp.mean(rs_pred**2)
            
        u_pred = self.u_pred_fn( params, t_fem5, coords_fem5[:, 0], coords_fem5[:, 1], 
                                jax.random.uniform(jax.random.PRNGKey(0), shape=(coords_fem1.shape[0],),
                                minval=0.01, maxval=0.05))
        ru_add = jnp.mean(nn.relu(u_fem5 - u_pred)) # u_pred has to be greater than u_fem5
        ru_loss = ru_loss + 100*ru_add
        
        v_pred = self.v_pred_fn( params, t_fem5, coords_fem5[:, 0], coords_fem5[:, 1], 
                                jax.random.uniform(jax.random.PRNGKey(0), shape=(coords_fem1.shape[0],),
                                minval=0.01, maxval=0.05))
        rv_add = jnp.mean(nn.relu(v_fem5 - v_pred)) # v_pred has to be greater than v_fem5
        rv_loss = rv_loss + 100*rv_add

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
            "s_in": s_in_loss,
            # "v_in": v_in_loss,
            # "v_out": v_out_loss,
            "u_noslip": u_noslip_loss,
            "v_noslip": v_noslip_loss,
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
