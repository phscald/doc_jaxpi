from functools import partial
import time
import os

from absl import logging

from flax.training import checkpoints

import jax
import jax.numpy as jnp
from jax import random, jit, vmap, pmap
from jax.tree_util import tree_map

import scipy.io
import ml_collections

import wandb

import models


from jaxpi.utils import restore_checkpoint

from utils import get_dataset#, parabolic_inflow

import matplotlib.pyplot as plt
import matplotlib.tri as tri


def evaluate(config: ml_collections.ConfigDict, workdir: str):
    # Load dataset
    pin =10
    (
        coords,
        inflow_coords,
        outflow_coords,
        wall_coords,
        #time,
        u0, v0, p0, s0,
        mu0, mu1, rho0, rho1
    ) = get_dataset(pin=pin)
    noslip_coords = wall_coords

    print(f'coords shape:{coords.shape}')
    print(f'inflow coords shape:{inflow_coords.shape}')

    fluid_params = (mu0, mu1, rho0, rho1)
    U_max = .06#.25/3#visc .1
    # pmax = 15

    L_max = .021
    pmax = mu0*U_max/L_max*36
    pin =pin/pmax
    # Re = rho0*U_max*L_max/mu0
    D = 10**(-9)

    T = 1.0  # final time

    # Nondimensionalization
    if config.nondim == True:
        # Nondimensionalization parameters
        U_star = U_max # 0.2  # characteristic velocity
        L_star = L_max #0.1  # characteristic length
        # Re = U_star * L_star / nu

        # Nondimensionalize coordinates and inflow velocity
        inflow_coords = inflow_coords / L_star
        outflow_coords = outflow_coords / L_star
        noslip_coords = noslip_coords / L_star
        coords = coords / L_star

        T_star = L_star/U_star
        p_inflow = (pin / pmax) * jnp.ones((inflow_coords.shape[0]))
        u0, v0, p0 = u0/U_max, v0/U_max, p0/pmax

        # # Nondimensionalization parameters
        # U_star = 1.0  # characteristic veprint(f'coords shape:{coords.shape}')and inflow velocity
        # T = T / T_star
        # inflow_coords = inflow_coords / L_star
        # outflow_coords = outflow_coords / L_star
        # wall_coords = wall_coords / L_star
        # cylinder_coords = cylinder_coords / L_star
        # coords = coords / L_star

        # # Nondimensionalize flow field
        # # u_inflow = u_inflow / U_star
        # u_ref = u_ref / U_star
        # v_ref = v_ref / U_star
        # p_ref = p_ref / U_star ** 2

    # else:
    #     Re = nu

    # # Inflow boundary conditions
    # U_max = 1.5  # maximum velocity
    # inflow_fn = lambda y: parabolic_inflow(y * L_star, U_max)

    # Temporal domain of each time window
    t0 = 0.0
    t1 = 1.0

    temporal_dom = jnp.array([t0, t1 * (1 + 0.05)]) # Must be same as the one used in training

    # Initialize model
    # model = models.NavierStokes2D(config, inflow_fn, temporal_dom, coords, Re)
    model = models.NavierStokes2DwSat(config, pin, temporal_dom, coords, U_max, L_max, fluid_params, D)

    # Restore checkpoint
    ckpt_path = os.path.join(".", "ckpt", config.wandb.name)
    ckpt_path = os.path.abspath(ckpt_path)
    model.state = restore_checkpoint(model.state, ckpt_path)
    params = model.state.params

    # Predict
    u_pred_fn = jit(vmap(vmap(model.u_net, (None, None, 0, 0)), (None, 0, None, None)))
    v_pred_fn = jit(vmap(vmap(model.v_net, (None, None, 0, 0)), (None, 0, None, None)))
    p_pred_fn = jit(vmap(vmap(model.p_net, (None, None, 0, 0)), (None, 0, None, None)))
    s_pred_fn = jit(vmap(vmap(model.s_net, (None, None, 0, 0)), (None, 0, None, None)))

    # t_coords = jnp.linspace(0, t1, 20)[:-1]
    # t_coords = jnp.linspace(0, t1, 4)[:-1]
    t_coords = jnp.array([0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.])

    u_pred_list = []
    v_pred_list = []
    p_pred_list = []
    s_pred_list = []
    # U_pred_list = []

    for idx in range(config.training.num_time_windows):
        # Restore the checkpoint
        ckpt_path = os.path.join('.', 'ckpt', config.wandb.name, 'time_window_{}'.format(idx + 1))
        model.state = restore_checkpoint(model.state, ckpt_path)
        params = model.state.params

        u_pred = u_pred_fn(params, t_coords, coords[:, 0], coords[:, 1])
        v_pred = v_pred_fn(params, t_coords, coords[:, 0], coords[:, 1])
        s_pred = s_pred_fn(params, t_coords, coords[:, 0], coords[:, 1])
        p_pred = p_pred_fn(params, t_coords, coords[:, 0], coords[:, 1])

        u_pred_list.append(u_pred)
        v_pred_list.append(v_pred)
        s_pred_list.append(s_pred)
        p_pred_list.append(p_pred)

    u_pred = jnp.concatenate(u_pred_list, axis=0)
    v_pred = jnp.concatenate(v_pred_list, axis=0)
    p_pred = jnp.concatenate(p_pred_list, axis=0)
    s_pred = jnp.concatenate(s_pred_list, axis=0)

    # Dimensionalize coordinates and flow field
    # if config.nondim == True:
    #     # Dimensionalize coordinates and flow field
    #     coords = coords * L_star

    #     u_ref = u_ref * U_star
    #     v_ref = v_ref * U_star

    #     u_pred = u_pred * U_star
    #     v_pred = v_pred * U_star

    x = coords[:, 0]
    y = coords[:, 1]
    # triang = tri.Triangulation(x, y)

    # Mask the triangles inside the cylinder
    # center = (0.2, 0.2)
    # radius = 0.05

    # x_tri = x[triang.triangles].mean(axis=1)
    # y_tri = y[triang.triangles].mean(axis=1)
    # dist_from_center = jnp.sqrt((x_tri - center[0]) ** 2 + (y_tri - center[1]) ** 2)
    # triang.set_mask(dist_from_center < radius)

    print(f'coords x shape:{x.shape}')
    print(f'coords y shape:{y.shape}')
    print(f'coords u shape:{u_pred[0].shape}')
    # print(len(u_pred))

    # print(dsadsa)
    from matplotlib.animation import FuncAnimation
    fig, ax = plt.subplots()
    m = len(u_pred)
     # Update function for each frame
    def update(frames):
        ax.cla()  # Clear the current axis
        ax.scatter(x, y, s=1, c=s_pred[frames], cmap='jet', vmin=0, vmax=1)
        # plt.colorbar()
        # plt.xlabel("x")
        # plt.ylabel("y")
        # plt.title("Predicted s(x, y) - t = " + str(t_coords[frames]))

    ani = FuncAnimation(fig, update, frames=m, interval=200)
    # Save the animation as a GIF
    ani.save('./video_s_p5.gif', writer='pillow')



    # Plot
    # Save dir
    # save_dir = os.path.join(workdir, "figures", config.wandb.name)
    # if not os.path.isdir(save_dir):
    #     os.makedirs(save_dir)
    # for i in range(len(u_pred)):
    #     fig1 = plt.figure(figsize=(18, 12))

    #     plt.subplot(4, 1, 1)
    #     plt.scatter(x, y, s=1, c=u_pred[i], cmap="jet")#, levels=100)
    #     plt.colorbar()
    #     plt.xlabel("x")
    #     plt.ylabel("y")
    #     plt.title("Predicted u(x, y)")
    #     plt.tight_layout()

    #     plt.subplot(4, 1, 2)
    #     plt.scatter(x, y, s=1, c=v_pred[i], cmap="jet")#, levels=100)
    #     plt.colorbar()
    #     plt.xlabel("x")
    #     plt.ylabel("y")
    #     plt.title("Predicted v(x, y)")
    #     plt.tight_layout()

    #     plt.subplot(4, 1, 3)
    #     plt.scatter(x, y, s=1, c=p_pred[i], cmap="jet")#, levels=100)
    #     plt.colorbar()
    #     plt.xlabel("x")
    #     plt.ylabel("y")
    #     plt.title("Predicted p(x, y)")
    #     plt.tight_layout()

    #     plt.subplot(4, 1, 4)
    #     plt.scatter(x, y, s=1, c=s_pred[i], cmap="jet")#, levels=100)
    #     plt.colorbar()
    #     plt.xlabel("x")
    #     plt.ylabel("y")
    #     plt.title("Predicted s(x, y)")
    #     plt.tight_layout()

    #     save_path = os.path.join(save_dir, 'ns_geomII'+ str(i) +'.png')
    #     fig1.savefig(save_path, bbox_inches="tight", dpi=300)

    #     plt.close()
