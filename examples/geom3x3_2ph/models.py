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


class NavierStokes2DwSat(ForwardIVP):

    def __init__(self, config, p_inflow, p_factor, temporal_dom, coords, U_max, L_max, fluid_params, D):   
        super().__init__(config)

        # self.inflow_fn = inflow_fn
        self.p_in = p_inflow 
        self.temporal_dom = temporal_dom
        self.coords = coords
        self.U_max = U_max
        self.L_max = L_max
        self.fluid_params = fluid_params
        self.D = D
        self.p_factor = p_factor

        # Non-dimensionalized domain length and width
        self.L, self.W = self.coords.max(axis=0) - self.coords.min(axis=0)

        if config.nondim == True:
            self.U_star = 1.0
            self.L_star = 0.1
        else:
            self.U_star = 1.0
            self.L_star = 1.0

        # Predict functions over batch
        self.u0_pred_fn = vmap(self.u_net, (None, None, 0, 0))
        self.v0_pred_fn = vmap(self.v_net, (None, None, 0, 0))
        self.p0_pred_fn = vmap(self.p_net, (None, None, 0, 0))
        self.s0_pred_fn = vmap(self.s_net, (None, None, 0, 0))

        self.u_pred_fn = vmap(self.u_net, (None, 0, 0, 0))
        self.v_pred_fn = vmap(self.v_net, (None, 0, 0, 0))
        self.p_pred_fn = vmap(self.p_net, (None, 0, 0, 0))
        self.s_pred_fn = vmap(self.s_net, (None, 0, 0, 0))
        self.r_pred_fn = vmap(self.r_net, (None, 0, 0, 0))

    def neural_net(self, params, t, x, y):
        t = t / self.temporal_dom[1]  # rescale t into [0, 1]
        x = x / self.L  # rescale x into [0, 1]
        y = y / self.W  # rescale y into [0, 1]
        inputs = jnp.stack([t, x, y])
        outputs = self.state.apply_fn(params, inputs)

        # Start with an initial state of the channel flow
        y_hat = y * self.L_star * self.W
        u = outputs[0]
        v = outputs[1]
        p = outputs[2]
        s = outputs[3]
        return u, v, p, s

    def u_net(self, params, t, x, y):
        u, _, _, _ = self.neural_net(params, t, x, y)
        return u

    def v_net(self, params, t, x, y):
        _, v, _, _ = self.neural_net(params, t, x, y)
        return v

    def p_net(self, params, t, x, y):
        _, _, p, _ = self.neural_net(params, t, x, y)
        return p

    def s_net(self, params, t, x, y):
        _, _, _, s = self.neural_net(params, t, x, y)
        return s



    def r_net(self, params, t, x, y):
        (mu0, mu1, rho0, rho1) = self.fluid_params

        u, v, p, s = self.neural_net(params, t, x, y)

        u_t = grad(self.u_net, argnums=1)(params, t, x, y)
        v_t = grad(self.v_net, argnums=1)(params, t, x, y)
        s_t = grad(self.s_net, argnums=1)(params, t, x, y)

        u_x = grad(self.u_net, argnums=2)(params, t, x, y)
        v_x = grad(self.v_net, argnums=2)(params, t, x, y)
        p_x = grad(self.p_net, argnums=2)(params, t, x, y)
        s_x = grad(self.s_net, argnums=2)(params, t, x, y)

        u_y = grad(self.u_net, argnums=3)(params, t, x, y)
        v_y = grad(self.v_net, argnums=3)(params, t, x, y)
        p_y = grad(self.p_net, argnums=3)(params, t, x, y)
        s_y = grad(self.s_net, argnums=3)(params, t, x, y)

        u_xx = grad(grad(self.u_net, argnums=2), argnums=2)(params, t, x, y)
        v_xx = grad(grad(self.v_net, argnums=2), argnums=2)(params, t, x, y)
        s_xx = grad(grad(self.s_net, argnums=2), argnums=2)(params, t, x, y)

        u_yy = grad(grad(self.u_net, argnums=3), argnums=3)(params, t, x, y)
        v_yy = grad(grad(self.v_net, argnums=3), argnums=3)(params, t, x, y)
        s_yy = grad(grad(self.s_net, argnums=3), argnums=3)(params, t, x, y)

        # compute Reynolds, initialized as Re=1
        # mu = (1-s)*mu1+s*mu0
        # rho = (1-s)*rho1+s*rho0
        # Re = Re*rho*self.U_max*self.L_max/mu
        Re = rho0*self.U_max*self.L_max/mu0
        mu = (1-s)*mu1 + s*mu0
        mu_ratio = mu/mu0
                
        # PDE residual
        ru = u_t + u * u_x + v * u_y + (1/self.p_factor * p_x - mu_ratio*(u_xx + u_yy)) / Re
        rv = v_t + u * v_x + v * v_y + (1/self.p_factor * p_y - mu_ratio*(v_xx + v_yy)) / Re
        rc = u_x + v_y
        rs = s_t + u * s_x + v * s_y - self.D*(s_xx + s_yy) 

        return ru/(10**4.8), rv/(10**4.8), rc, rs

    def ru_net(self, params, t, x, y):
        ru, _, _, _ = self.r_net(params, t, x, y)
        return ru

    def rv_net(self, params, t, x, y):
        _, rv, _, _ = self.r_net(params, t, x, y)
        return rv

    def rc_net(self, params, t, x, y):
        _, _, rc, _ = self.r_net(params, t, x, y)
        return rc

    def rs_net(self, params, t, x, y):
        _, _, _, rs = self.r_net(params, t, x, y)
        return rs



    @partial(jit, static_argnums=(0,))
    def res_and_w(self, params, batch):
        # Sort temporal coordinates
        t_sorted = batch[:, 0].sort()
        ru_pred, rv_pred, rc_pred, rs_pred = self.r_pred_fn(
            params, t_sorted, batch[:, 1], batch[:, 2]
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
        inflow_batch = batch["inflow"]
        outflow_batch = batch["outflow"]
        noslip_batch = batch["noslip"]
        res_batch = batch["res"]

        coords_batch, u_batch, v_batch, p_batch, s_batch, coords_initial = ic_batch # coords_initial for u v p

        u_ic_ntk = vmap(ntk_fn, (None, None, None, 0, 0))(
            self.u_net, params, 0.0, coords_initial[:, 0], coords_initial[:, 1]
        )
        v_ic_ntk = vmap(ntk_fn, (None, None, None, 0, 0))(
            self.v_net, params, 0.0, coords_initial[:, 0], coords_initial[:, 1]
        )
        p_ic_ntk = vmap(ntk_fn, (None, None, None, 0, 0))(
            self.p_net, params, 0.0, coords_initial[:, 0], coords_initial[:, 1]
        )
        s_ic_ntk = vmap(ntk_fn, (None, None, None, 0, 0))(
            self.s_net, params, 0.0, coords_batch[:, 0], coords_batch[:, 1]
        )

        p_in_ntk = vmap(ntk_fn, (None, None, 0, 0, 0))(
            self.p_net,
            params,
            inflow_batch[:, 0],
            inflow_batch[:, 1],
            inflow_batch[:, 2],
        )
        v_in_ntk = vmap(ntk_fn, (None, None, 0, 0, 0))(
            self.v_net,
            params,
            inflow_batch[:, 0],
            inflow_batch[:, 1],
            inflow_batch[:, 2],
        )
        s_in_ntk = vmap(ntk_fn, (None, None, 0, 0, 0))(
            self.s_net,
            params,
            inflow_batch[:, 0],
            inflow_batch[:, 1],
            inflow_batch[:, 2],
        )

        p_out_ntk = vmap(ntk_fn, (None, None, 0, 0, 0))(
            self.p_net,
            params,
            outflow_batch[:, 0],
            outflow_batch[:, 1],
            outflow_batch[:, 2],
        )
        v_out_ntk = vmap(ntk_fn, (None, None, 0, 0, 0))(
            self.v_net,
            params,
            outflow_batch[:, 0],
            outflow_batch[:, 1],
            outflow_batch[:, 2],
        )

        u_noslip_ntk = vmap(ntk_fn, (None, None, 0, 0, 0))(
            self.u_net,
            params,
            noslip_batch[:, 0],
            noslip_batch[:, 1],
            noslip_batch[:, 2],
        )
        v_noslip_ntk = vmap(ntk_fn, (None, None, 0, 0, 0))(
            self.v_net,
            params,
            noslip_batch[:, 0],
            noslip_batch[:, 1],
            noslip_batch[:, 2],
        )

        # Consider the effect of causal weights
        if self.config.weighting.use_causal:
            res_batch = jnp.array(
                [res_batch[:, 0].sort(), res_batch[:, 1], res_batch[:, 2]]
            ).T
            ru_ntk = vmap(ntk_fn, (None, None, 0, 0, 0))(
                self.ru_net, params, res_batch[:, 0], res_batch[:, 1], res_batch[:, 2]
            )
            rv_ntk = vmap(ntk_fn, (None, None, 0, 0, 0))(
                self.rv_net, params, res_batch[:, 0], res_batch[:, 1], res_batch[:, 2]
            )
            rc_ntk = vmap(ntk_fn, (None, None, 0, 0, 0))(
                self.rc_net, params, res_batch[:, 0], res_batch[:, 1], res_batch[:, 2]
            )
            rs_ntk = vmap(ntk_fn, (None, None, 0, 0, 0))(
                self.rs_net, params, res_batch[:, 0], res_batch[:, 1], res_batch[:, 2]
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
            ru_ntk = vmap(ntk_fn, (None, None, 0, 0, 0))(
                self.ru_net, params, res_batch[:, 0], res_batch[:, 1], res_batch[:, 2]
            )
            rv_ntk = vmap(ntk_fn, (None, None, 0, 0, 0))(
                self.rv_net, params, res_batch[:, 0], res_batch[:, 1], res_batch[:, 2]
            )
            rc_ntk = vmap(ntk_fn, (None, None, 0, 0, 0))(
                self.rc_net, params, res_batch[:, 0], res_batch[:, 1], res_batch[:, 2]
            )
            rs_ntk = vmap(ntk_fn, (None, None, 0, 0, 0))(
                self.rs_net, params, res_batch[:, 0], res_batch[:, 1], res_batch[:, 2]
            )

        ntk_dict = {
            "u_ic": u_ic_ntk,
            "v_ic": v_ic_ntk,
            "p_ic": p_ic_ntk,
            "s_ic": s_ic_ntk,
            "p_in": p_in_ntk,
            "p_out": p_out_ntk,
            "s_in": s_in_ntk,
            "v_in": v_in_ntk,
            "v_out": v_out_ntk,
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
        inflow_batch = batch["inflow"]
        outflow_batch = batch["outflow"]
        noslip_batch = batch["noslip"]
        res_batch = batch["res"]

        # Initial condition loss
        coords_batch, u_batch, v_batch, p_batch, s_batch, coords_initial = ic_batch # coords_initial for u v p
        # s_batch = ic_batch

        u_ic_pred = self.u0_pred_fn(params, 0.0, coords_initial[:, 0], coords_initial[:, 1])
        v_ic_pred = self.v0_pred_fn(params, 0.0, coords_initial[:, 0], coords_initial[:, 1])
        p_ic_pred = self.p0_pred_fn(params, 0.0, coords_initial[:, 0], coords_initial[:, 1])
        s_ic_pred = self.s0_pred_fn(params, 0.0, coords_batch[:, 0], coords_batch[:, 1])

        u_ic_loss = jnp.mean((u_ic_pred - u_batch) ** 2)
        v_ic_loss = jnp.mean((v_ic_pred - v_batch) ** 2)
        p_ic_loss = jnp.mean((p_ic_pred - p_batch) ** 2)
        # p_ic_loss = p_ic_loss*0
        s_ic_loss = jnp.mean((s_ic_pred - s_batch) ** 2)

        # inflow outflow loss
        p_in_pred = self.p_pred_fn(
            params, inflow_batch[:, 0], inflow_batch[:, 1], inflow_batch[:, 2]
        )

        p_in_loss = jnp.mean((p_in_pred - self.p_in) ** 2)
        #
        p_out_pred = self.p_pred_fn(
            params, outflow_batch[:, 0], outflow_batch[:, 1], outflow_batch[:, 2]
        )
        p_out_loss = jnp.mean((p_out_pred) ** 2)
        #
        s_in_pred = self.s_pred_fn(
            params, inflow_batch[:, 0], inflow_batch[:, 1], inflow_batch[:, 2]
        )
        s_in_loss = jnp.mean((s_in_pred - 1.0) ** 2)
        #
        v_in_pred = self.v_pred_fn(
            params, inflow_batch[:, 0], inflow_batch[:, 1], inflow_batch[:, 2]
        )
        v_in_loss = jnp.mean((v_in_pred) ** 2)
        #
        v_out_pred = self.v_pred_fn(
            params, outflow_batch[:, 0], outflow_batch[:, 1], outflow_batch[:, 2]
        )
        v_out_loss = jnp.mean((v_out_pred) ** 2)
        #
        # noslip loss
        u_noslip_pred = self.u_pred_fn(
            params, noslip_batch[:, 0], noslip_batch[:, 1], noslip_batch[:, 2]
        )
        v_noslip_pred = self.v_pred_fn(
            params, noslip_batch[:, 0], noslip_batch[:, 1], noslip_batch[:, 2]
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
                params, res_batch[:, 0], res_batch[:, 1], res_batch[:, 2]
            )
            ru_loss = jnp.mean(ru_pred**2)
            rv_loss = jnp.mean(rv_pred**2)
            rc_loss = jnp.mean(rc_pred**2)
            rs_loss = jnp.mean(rs_pred**2)

        loss_dict = {
            "u_ic": u_ic_loss,
            "v_ic": v_ic_loss,
            "p_ic": p_ic_loss,
            "s_ic": s_ic_loss,
            "p_in": p_in_loss,
            "p_out": p_out_loss,
            "s_in": s_in_loss,
            "v_in": v_in_loss,
            "v_out": v_out_loss,
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
