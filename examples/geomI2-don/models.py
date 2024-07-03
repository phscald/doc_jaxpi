# from functools import partial

# import jax
# import jax.numpy as jnp
# import numpy as np
# from jax import random
# from jax import lax, jit, grad, vmap, pmap, jacrev, hessian
# from jax.tree_util import tree_map
# from jax.flatten_util import ravel_pytree

# from jaxpi.models import ForwardBVP
# from jaxpi.evaluator import BaseEvaluator
# from jaxpi.utils import ntk_fn



# class NavierStokes2D(ForwardBVP):
#     def __init__(
#         self,
#         config,
#         # u_inflow,
#         # p_inflow,
#         # inflow_coords,
#         # outflow_coords,
#         wall_coords,
#         # cylinder_coords,
#         #mu, 
#         # U_max, pmax
#         # Re,
#     ):
#         super().__init__(config)

#         # self.u_in = u_inflow  # inflow profile
#         # self.p_in = p_inflow * jnp.ones((inflow_coords.shape[0]))
#         # self.Re = Re  # Reynolds number
#         #self.mu = mu; 
#         # self.U_max = 1/U_max; self.pmax = pmax

#         # Initialize coordinates
#         # self.inflow_coords = inflow_coords
#         # self.outflow_coords = outflow_coords
#         self.wall_coords = wall_coords
#         # self.cylinder_coords = cylinder_coords
#         # self.noslip_coords = jnp.vstack((self.wall_coords, self.cylinder_coords))

#         # Non-dimensionalized domain length and width
#         self.L, self.W = self.wall_coords.max(axis=0) - self.wall_coords.min(axis=0)

#         # Predict functions over batch
#         self.u_pred_fn = vmap(self.u_net, (None, 0, 0, 0, 0))
#         self.v_pred_fn = vmap(self.v_net, (None, 0, 0, 0, 0))
#         self.p_pred_fn = vmap(self.p_net, (None, 0, 0, 0, 0))
#         self.r_pred_fn = vmap(self.r_net, (None, 0, 0, 0, 0))

#     def neural_net(self, params, x, y, mu, pin):
#         # x = x / self.L  # rescale x into [0, 1]
#         # y = y / self.W  # rescale y into [0, 1]
#         # z = jnp.stack([x, y])
#         # z_1 = jnp.stack([mu, pin])
#         z = jnp.stack([x, y, mu, pin])

#         #TODO alterar aqui 
#         outputs = self.state.apply_fn(params, z)
#         u = outputs[0]
#         v = outputs[1]
#         p = outputs[2]
#         return u, v, p

#     def u_net(self, params, x, y, mu, pin):
#         u, _, _ = self.neural_net(params, x, y, mu, pin)
#         return u

#     def v_net(self, params, x, y, mu, pin):
#         _, v, _ = self.neural_net(params, x, y, mu, pin)
#         return v

#     def p_net(self, params, x, y, mu, pin):
#         _, _, p = self.neural_net(params, x, y, mu, pin)
#         return p

#     def r_net(self, params, x, y, mu, pin):
#         u, v, p = self.neural_net(params, x, y, mu, pin)
#         (u_x, u_y), (v_x, v_y), (p_x, p_y) = jacrev(self.neural_net, argnums=(1, 2))(params, x, y, mu, pin)

#         u_hessian = hessian(self.u_net, argnums=(1, 2))(params, x, y, mu, pin)
#         v_hessian = hessian(self.v_net, argnums=(1, 2))(params, x, y, mu, pin)

#         u_xx = u_hessian[0][0]
#         u_yy = u_hessian[1][1]

#         v_xx = v_hessian[0][0]
#         v_yy = v_hessian[1][1]

#         # PDE residual
#         # ru = u * u_x + v * u_y + p_x - (u_xx + u_yy) / self.Re
#         # rv = u * v_x + v * v_y + p_y - (v_xx + v_yy) / self.Re
#         ru = p_x - (u_xx + u_yy)
#         rv = p_y - (v_xx + v_yy)
#         rc = u_x + v_y

#         # outflow boundary residual
#         # u_out = u_x - p
#         v_out = v

#         return ru, rv, rc, v_out#, u_out, v_out

#     def ru_net(self, params, x, y, mu, pin):
#         ru, _, _, _ = self.r_net(params, x, y, mu, pin)
#         return ru

#     def rv_net(self, params, x, y, mu, pin):
#         _, rv, _, _ = self.r_net(params, x, y, mu, pin)
#         return rv

#     def rc_net(self, params, x, y, mu, pin):
#         _, _, rc, _ = self.r_net(params, x, y, mu, pin)
#         return rc

#     def v_out_net(self, params, x, y, mu, pin):
#         _, _, _, v_out = self.r_net(params, x, y, mu, pin)
#         return v_out
    


#     @partial(jit, static_argnums=(0,))
#     def losses(self, params, batch):

#         (mu_batch, pin_batch, inflow_batch, outflow_batch, batch) = batch

#         noslip_coords = self.wall_coords

#         # Inflow boundary conditions

#         p_in_pred = self.p_pred_fn(
#             params, inflow_batch[:, 0], inflow_batch[:, 1], 
#             mu_batch, pin_batch
#         )
#         p_in_loss = jnp.mean((p_in_pred - pin_batch) ** 2)

#         # Outflow boundary conditions

#         _, _, _, v_out_pred = self.r_pred_fn(
#             params, outflow_batch[:, 0], outflow_batch[:, 1], 
#             mu_batch, pin_batch
#         )

#         _, _, _, v_in_pred = self.r_pred_fn(
#             params, inflow_batch[:, 0], inflow_batch[:, 1], 
#             mu_batch, pin_batch
#         )
#         # u_out_loss = jnp.mean(u_out_pred**2)
#         v_out_loss = jnp.mean(v_out_pred**2)
#         v_in_loss = jnp.mean(v_in_pred**2)

#         p_out_pred = self.p_pred_fn(
#             params, outflow_batch[:, 0], outflow_batch[:, 1], 
#             mu_batch, pin_batch
#         )
#         p_out_loss = jnp.mean((p_out_pred) ** 2)

#         # No-slip boundary conditions
#         # key1, key2, key3, key4 = random.split(random.PRNGKey(1234), 4)
#         u_noslip_pred = self.u_pred_fn(
#             params, noslip_coords[:, 0], noslip_coords[:, 1], 
#             # random.uniform(key1, shape=(noslip_coords.shape[0],)), random.uniform(key2, shape=(noslip_coords.shape[0],))
#             jax.device_put(np.ones(noslip_coords.shape[0])*mu_batch[1]), jax.device_put(np.ones(noslip_coords.shape[0])*pin_batch[1])
#         )

#         v_noslip_pred = self.v_pred_fn(
#             params, noslip_coords[:, 0], noslip_coords[:, 1], 
#             # random.uniform(key3, shape=(noslip_coords.shape[0],)), random.uniform(key4, shape=(noslip_coords.shape[0],))
#             jax.device_put(np.ones(noslip_coords.shape[0])*mu_batch[1]), jax.device_put(np.ones(noslip_coords.shape[0])*pin_batch[1])
#         )

#         u_noslip_loss = jnp.mean(u_noslip_pred**2)
#         v_noslip_loss = jnp.mean(v_noslip_pred**2)

#         # Residual losses
#         ru_pred, rv_pred, rc_pred, _  = self.r_pred_fn(
#             params, batch[:, 0], batch[:, 1], mu_batch, pin_batch
#         )

#         ru_loss = jnp.mean(ru_pred**2)
#         rv_loss = jnp.mean(rv_pred**2)
#         rc_loss = jnp.mean(rc_pred**2)

#         loss_dict = {
#             "p_in": p_in_loss,
#             "v_in": v_in_loss,
#             "p_out": p_out_loss,
#             "v_out": v_out_loss,
#             "u_noslip": u_noslip_loss,
#             "v_noslip": v_noslip_loss,
#             "ru": ru_loss,
#             "rv": rv_loss,
#             "rc": rc_loss,
#         }

#         return loss_dict

#     @partial(jit, static_argnums=(0,))
#     def compute_diag_ntk(self, params, batch):

#         (mu_batch, pin_batch, inflow_batch, outflow_batch, batch) = batch

#         noslip_coords = self.wall_coords
#         v_out_ntk = vmap(ntk_fn, (None, None, 0, 0, 0, 0))(
#             self.v_out_net, params, outflow_batch[:, 0], outflow_batch[:, 1],
#             mu_batch, pin_batch
#         )
#         v_in_ntk = vmap(ntk_fn, (None, None, 0, 0, 0, 0))(
#             self.v_out_net, params, inflow_batch[:, 0], inflow_batch[:, 1],   
#             mu_batch, pin_batch
#         )

#         p_in_ntk = vmap(ntk_fn, (None, None, 0, 0, 0, 0))(
#             self.p_net, params, inflow_batch[:, 0], inflow_batch[:, 1],  
#             mu_batch, pin_batch
#         )
#         p_out_ntk = vmap(ntk_fn, (None, None, 0, 0, 0, 0))(
#             self.p_net, params, outflow_batch[:, 0], outflow_batch[:, 1], 
#             mu_batch, pin_batch
#         )

#         key1, key2, key3, key4 = random.split(random.PRNGKey(1234), 4)
#         u_noslip_ntk = vmap(ntk_fn, (None, None, 0, 0, 0, 0))(
#             self.u_net, params, noslip_coords[:, 0], noslip_coords[:, 1], 
#             # random.uniform(key1, shape=(noslip_coords.shape[0],)), random.uniform(key2, shape=(noslip_coords.shape[0],))
#             jax.device_put(np.ones(noslip_coords.shape[0])*mu_batch[1]), jax.device_put(np.ones(noslip_coords.shape[0])*pin_batch[1])
#         )
#         v_noslip_ntk = vmap(ntk_fn, (None, None, 0, 0, 0, 0))(
#             self.v_net, params, noslip_coords[:, 0], noslip_coords[:, 1], 
#             # random.uniform(key3, shape=(noslip_coords.shape[0],)), random.uniform(key4, shape=(noslip_coords.shape[0],))
#             jax.device_put(np.ones(noslip_coords.shape[0])*mu_batch[1]), jax.device_put(np.ones(noslip_coords.shape[0])*pin_batch[1])
#         )

#         ru_ntk = vmap(ntk_fn, (None, None, 0, 0, 0, 0))(
#             self.ru_net, params, batch[:, 0], batch[:, 1], mu_batch, pin_batch
#         )
#         rv_ntk = vmap(ntk_fn, (None, None, 0, 0, 0, 0))(
#             self.rv_net, params, batch[:, 0], batch[:, 1], mu_batch, pin_batch
#         )
#         rc_ntk = vmap(ntk_fn, (None, None, 0, 0, 0, 0))(
#             self.rc_net, params, batch[:, 0], batch[:, 1], mu_batch, pin_batch
#         )

#         ntk_dict = {
#             # "u_in": u_in_ntk,
#             "v_in": v_in_ntk,
#             # "u_out": u_out_ntk,
#             "v_out": v_out_ntk,
#             'p_in': p_in_ntk,
#             'p_out': p_out_ntk,
#             "u_noslip": u_noslip_ntk,
#             "v_noslip": v_noslip_ntk,
#             "ru": ru_ntk,
#             "rv": rv_ntk,
#             "rc": rc_ntk,
#         }

#         return ntk_dict

#     @partial(jit, static_argnums=(0,))
#     def compute_l2_error(self, params, coords, u_test, v_test):
#         u_pred = self.u_pred_fn(params, coords[:, 0], coords[:, 1])
#         v_pred = self.v_pred_fn(params, coords[:, 0], coords[:, 1])

#         u_error = jnp.linalg.norm(u_pred - u_test) / jnp.linalg.norm(u_test)
#         v_error = jnp.linalg.norm(v_pred - v_test) / jnp.linalg.norm(v_test)

#         return u_error, v_error

# class NavierStokesEvaluator(BaseEvaluator):
#     def __init__(self, config, model):
#         super().__init__(config, model)

#     def log_errors(self, params, coords, u_ref, v_ref):
#         u_error, v_error = self.model.compute_l2_error(params, coords, u_ref, v_ref)
#         self.log_dict["u_error"] = u_error
#         self.log_dict["v_error"] = v_error

#     # def log_preds(self, params, x_star, y_star):
#     #     u_pred = vmap(vmap(model.u_net, (None, None, 0)), (None, 0, None))(params, x_star, y_star)
#     #     v_pred = vmap(vmap(model.v_net, (None, None, 0)), (None, 0, None))(params, x_star, y_star)
#     #     U_pred = jnp.sqrt(u_pred ** 2 + v_pred ** 2)
#     #
#     #     fig = plt.figure()
#     #     plt.pcolor(U_pred.T, cmap='jet')
#     #     log_dict['U_pred'] = fig
#     #     fig.close()

#     def __call__(self, state, batch, coords, u_ref, v_ref):
#         self.log_dict = super().__call__(state, batch)

#         if self.config.logging.log_errors:
#             self.log_errors(state.params, coords, u_ref, v_ref)

#         if self.config.logging.log_preds:
#             self.log_preds(state.params, coords)

#         return self.log_dict






from functools import partial

import jax.numpy as jnp
from jax import lax, jit, grad, vmap, pmap, jacrev, hessian
from jax.tree_util import tree_map
from jax.flatten_util import ravel_pytree

from jaxpi.models import ForwardBVP
from jaxpi.evaluator import BaseEvaluator
from jaxpi.utils import ntk_fn


class NavierStokes2D(ForwardBVP):
    def __init__(
        self,
        config,
        # u_inflow,
        p_inflow,
        # inflow_coords,
        # outflow_coords,
        wall_coords,
        #mu, U_max, pmax
        # Re,
    ):
        super().__init__(config)

        # self.u_in = u_inflow  # inflow profile
        self.p_in = p_inflow#p_inflow * jnp.ones((inflow_coords.shape[0]))
        # self.Re = Re  # Reynolds number
        #self.mu = mu; self.U_max = U_max; self.pmax = pmax

        # Initialize coordinates
        # self.inflow_coords = inflow_coords
        # self.outflow_coords = outflow_coords
        self.wall_coords = wall_coords
        # self.noslip_coords = self.wall_coords
        # self.noslip_coords = jnp.vstack((self.wall_coords, self.cylinder_coords))

        # Non-dimensionalized domain length and width
        self.L, self.W = self.wall_coords.max(axis=0) - self.wall_coords.min(axis=0)

        # Predict functions over batch
        self.u_pred_fn = vmap(self.u_net, (None, 0, 0, 0, 0))
        self.v_pred_fn = vmap(self.v_net, (None, 0, 0, 0, 0))
        self.p_pred_fn = vmap(self.p_net, (None, 0, 0, 0, 0))
        self.r_pred_fn = vmap(self.r_net, (None, 0, 0, 0, 0))

    def neural_net(self, params, x, y, pin, mu):
        # x = x / self.L  # rescale x into [0, 1]
        # y = y / self.W  # rescale y into [0, 1]
        z = jnp.stack([x, y])
        z1 = jnp.stack([pin, mu])
        outputs = self.state.apply_fn(params, z, z1) # u branch, x trunk (primeira entrada u, segunda)
        u = outputs[0]
        v = outputs[1]
        p = outputs[2]
        return u, v, p

    def u_net(self, params, x, y, pin, mu):
        u, _, _ = self.neural_net(params, x, y, pin, mu)
        return u

    def v_net(self, params, x, y, pin, mu):
        _, v, _ = self.neural_net(params, x, y, pin, mu)
        return v

    def p_net(self, params, x, y, pin, mu):
        _, _, p = self.neural_net(params, x, y, pin, mu)
        return p

    def r_net(self, params, x, y, pin, mu):
        u, v, p = self.neural_net(params, x, y, pin, mu)

        (u_x, u_y), (v_x, v_y), (p_x, p_y) = jacrev(self.neural_net, argnums=(1, 2))(params, x, y, pin, mu)

        u_hessian = hessian(self.u_net, argnums=(1, 2))(params, x, y, pin, mu)
        v_hessian = hessian(self.v_net, argnums=(1, 2))(params, x, y, pin, mu)

        u_xx = u_hessian[0][0]
        u_yy = u_hessian[1][1]

        v_xx = v_hessian[0][0]
        v_yy = v_hessian[1][1]

        # PDE residual
        # ru = u * u_x + v * u_y + p_x - (u_xx + u_yy) / self.Re
        # rv = u * v_x + v * v_y + p_y - (v_xx + v_yy) / self.Re
        ru = p_x - mu/.1*(u_xx + u_yy)
        rv = p_y - mu/.1*(v_xx + v_yy)
        rc = u_x + v_y

        # outflow boundary residual
        # u_out = u_x - p
        v_out = v

        return ru, rv, rc, v_out#, u_out, v_out

    def ru_net(self, params, x, y, pin, mu):
        ru, _, _, _ = self.r_net(params, x, y, pin, mu)
        return ru

    def rv_net(self, params, x, y, pin, mu):
        _, rv, _, _ = self.r_net(params, x, y, pin, mu)
        return rv

    def rc_net(self, params, x, y, pin, mu):
        _, _, rc, _ = self.r_net(params, x, y, pin, mu)
        return rc

    # def u_out_net(self, params, x, y):
    #     _, _, _, u_out, _ = self.r_net(params, x, y)
    #     return u_out

    def v_out_net(self, params, x, y, pin, mu):
        _, _, _, v_out = self.r_net(params, x, y, pin, mu)
        return v_out

    @partial(jit, static_argnums=(0,))
    def losses(self, params, batch):
        (mu_batch, pin_batch, inflow_batch, outflow_batch, batch) = batch

        noslip_coords = self.wall_coords

        p_in_pred = self.p_pred_fn(
            params, inflow_batch[:, 0], inflow_batch[:, 1], pin_batch, mu_batch
        )
        p_in_loss = jnp.mean((p_in_pred - pin_batch) ** 2)

        # Outflow boundary conditions
        # _, _, _, u_out_pred, v_out_pred = self.r_pred_fn(
        #     params, self.outflow_coords[:, 0], self.outflow_coords[:, 1]
        # )
        _, _, _, v_out_pred = self.r_pred_fn(
            params, outflow_batch[:, 0], outflow_batch[:, 0], pin_batch, mu_batch
        )
        _, _, _, v_in_pred = self.r_pred_fn(
            params, inflow_batch[:, 0], inflow_batch[:, 1], pin_batch, mu_batch
        )
        # u_out_loss = jnp.mean(u_out_pred**2)
        v_out_loss = jnp.mean(v_out_pred**2)
        v_in_loss = jnp.mean(v_in_pred**2)
        p_out_pred = self.p_pred_fn(
            params, outflow_batch[:, 0], outflow_batch[:, 1], pin_batch, mu_batch
        )
        p_out_loss = jnp.mean((p_out_pred) ** 2)

        # No-slip boundary conditions
        u_noslip_pred = self.u_pred_fn(
            params, noslip_coords[:, 0], noslip_coords[:, 1], 
            jnp.ones(noslip_coords.shape[0])*pin_batch[0],
            jnp.ones(noslip_coords.shape[0])*mu_batch[0]
        )
        v_noslip_pred = self.v_pred_fn(
            params, noslip_coords[:, 0], noslip_coords[:, 1], 
            jnp.ones(noslip_coords.shape[0])*pin_batch[0],
            jnp.ones(noslip_coords.shape[0])*mu_batch[0]
        )

        u_noslip_loss = jnp.mean(u_noslip_pred**2)
        v_noslip_loss = jnp.mean(v_noslip_pred**2)

        # Residual losses
        ru_pred, rv_pred, rc_pred, _  = self.r_pred_fn(
            params, batch[:, 0], batch[:, 1], pin_batch, mu_batch
        )

        ru_loss = jnp.mean(ru_pred**2)
        rv_loss = jnp.mean(rv_pred**2)
        rc_loss = jnp.mean(rc_pred**2)

        loss_dict = {
            "p_in": p_in_loss,
            "v_in": v_in_loss,
            "p_out": p_out_loss,
            "v_out": v_out_loss,
            "u_noslip": u_noslip_loss,
            "v_noslip": v_noslip_loss,
            "ru": ru_loss,
            "rv": rv_loss,
            "rc": rc_loss,
        }

        return loss_dict

    @partial(jit, static_argnums=(0,))
    def compute_diag_ntk(self, params, batch):

        (mu_batch, pin_batch, inflow_batch, outflow_batch, batch) = batch

        noslip_coords = self.wall_coords
        v_out_ntk = vmap(ntk_fn, (None, None, 0, 0, 0, 0))(
            self.v_out_net, params, outflow_batch[:, 0], outflow_batch[:, 1], pin_batch, mu_batch
        )
        v_in_ntk = vmap(ntk_fn, (None, None, 0, 0, 0, 0))(
            self.v_out_net, params, inflow_batch[:, 0], inflow_batch[:, 1], pin_batch, mu_batch
        )

        p_in_ntk = vmap(ntk_fn, (None, None, 0, 0, 0, 0))(
            self.p_net, params, inflow_batch[:, 0], inflow_batch[:, 1], pin_batch, mu_batch
        )
        p_out_ntk = vmap(ntk_fn, (None, None, 0, 0, 0, 0))(
            self.p_net, params, outflow_batch[:, 0], outflow_batch[:, 1], pin_batch, mu_batch
        )

        u_noslip_ntk = vmap(ntk_fn, (None, None, 0, 0, 0, 0))(
            self.u_net, params, noslip_coords[:, 0], noslip_coords[:, 1], 
            jnp.ones(noslip_coords.shape[0])*pin_batch[0],
            jnp.ones(noslip_coords.shape[0])*mu_batch[0]
        )
        v_noslip_ntk = vmap(ntk_fn, (None, None, 0, 0, 0, 0))(
            self.v_net, params, noslip_coords[:, 0], noslip_coords[:, 1], 
            jnp.ones(noslip_coords.shape[0])*pin_batch[0],
            jnp.ones(noslip_coords.shape[0])*mu_batch[0]
        )

        ru_ntk = vmap(ntk_fn, (None, None, 0, 0, 0, 0))(
            self.ru_net, params, batch[:, 0], batch[:, 1], pin_batch, mu_batch
        )
        rv_ntk = vmap(ntk_fn, (None, None, 0, 0, 0, 0))(
            self.rv_net, params, batch[:, 0], batch[:, 1], pin_batch, mu_batch
        )
        rc_ntk = vmap(ntk_fn, (None, None, 0, 0, 0, 0))(
            self.rc_net, params, batch[:, 0], batch[:, 1], pin_batch, mu_batch
        )

        ntk_dict = {
            # "u_in": u_in_ntk,
            "v_in": v_in_ntk,
            # "u_out": u_out_ntk,
            "v_out": v_out_ntk,
            'p_in': p_in_ntk,
            'p_out': p_out_ntk,
            "u_noslip": u_noslip_ntk,
            "v_noslip": v_noslip_ntk,
            "ru": ru_ntk,
            "rv": rv_ntk,
            "rc": rc_ntk,
        }

        return ntk_dict

    @partial(jit, static_argnums=(0,))
    def compute_l2_error(self, params, coords, u_test, v_test):
        u_pred = self.u_pred_fn(params, coords[:, 0], coords[:, 1])
        v_pred = self.v_pred_fn(params, coords[:, 0], coords[:, 1])

        u_error = jnp.linalg.norm(u_pred - u_test) / jnp.linalg.norm(u_test)
        v_error = jnp.linalg.norm(v_pred - v_test) / jnp.linalg.norm(v_test)

        return u_error, v_error
   


class NavierStokesEvaluator(BaseEvaluator):
    def __init__(self, config, model):
        super().__init__(config, model)

    def log_errors(self, params, coords, u_ref, v_ref):
        u_error, v_error = self.model.compute_l2_error(params, coords, u_ref, v_ref)
        self.log_dict["u_error"] = u_error
        self.log_dict["v_error"] = v_error

    # def log_preds(self, params, x_star, y_star):
    #     u_pred = vmap(vmap(model.u_net, (None, None, 0)), (None, 0, None))(params, x_star, y_star)
    #     v_pred = vmap(vmap(model.v_net, (None, None, 0)), (None, 0, None))(params, x_star, y_star)
    #     U_pred = jnp.sqrt(u_pred ** 2 + v_pred ** 2)
    #
    #     fig = plt.figure()
    #     plt.pcolor(U_pred.T, cmap='jet')
    #     log_dict['U_pred'] = fig
    #     fig.close()

    def __call__(self, state, batch, coords, u_ref, v_ref):
        self.log_dict = super().__call__(state, batch)

        if self.config.logging.log_errors:
            self.log_errors(state.params, coords, u_ref, v_ref)

        if self.config.logging.log_preds:
            self.log_preds(state.params, coords)

        return self.log_dict
