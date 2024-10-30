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
    

    
    pin =20
    (
        coords,
        inflow_coords,
        outflow_coords,
        wall_coords,
        #time,
        initial,
        mu0, mu1, rho0, rho1
    ) = get_dataset(pin=pin)
    noslip_coords = wall_coords
    
    u0, v0, p0 = initial[:3]
    u05 = initial[5]
        
    print(jnp.max(coords[:,0]))
    print(jnp.max(coords[:,1]))


    print(f'coords shape:{coords.shape}')
    print(f'inflow coords shape:{inflow_coords.shape}')
    
    fluid_params = (mu0, mu1, rho0, rho1)
    dp = 20
    # pin la em cima
    L_max = 900/1000/100
    U_max = dp*L_max/mu0
    pmax =dp
    Re = rho0*dp*(L_max**2)/(mu0**2)
    print(f'Re={Re}')
    print(f'Re={Re*.112**2}')


    mu = .03
    D =  0*10**(-4)
    t1 = 1 # it is better to change the time in the t_coords array. There it is possible to select the desired percentages of total time solved

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
    
        ind_coords = random.choice(
            random.PRNGKey(1234),
            coords.shape[0],
            shape=(int(coords.shape[0]/6),),
            replace=True,
        )
        coords = coords[ind_coords]

        T_star = L_star/U_star
        p_inflow = (pin / pmax) * jnp.ones((inflow_coords.shape[0]))

    
    t0 = 0.0

    temporal_dom = jnp.array([t0, t1 * (1 + 0.05)]) # Must be same as the one used in training

    # Initialize model
    # model = models.NavierStokes2D(config, inflow_fn, temporal_dom, coords, Re)
    model = models.NavierStokes2DwSat(config, pin/pmax, temporal_dom, coords, U_max, L_max, fluid_params, D)

    # Restore checkpoint
    ckpt_path = os.path.join(".", "ckpt", config.wandb.name)
    ckpt_path = os.path.abspath(ckpt_path)
    model.state = restore_checkpoint(model.state, ckpt_path)
    params = model.state.params

    # Predict
    u_pred_fn = jit(vmap(vmap(model.u_net, (None, None, 0, 0, None)), (None, 0, None, None, None)))
    v_pred_fn = jit(vmap(vmap(model.v_net, (None, None, 0, 0, None)), (None, 0, None, None, None)))
    p_pred_fn = jit(vmap(vmap(model.p_net, (None, None, 0, 0, None)), (None, 0, None, None, None)))
    s_pred_fn = jit(vmap(vmap(model.s_net, (None, None, 0, 0, None)), (None, 0, None, None, None)))

    # t_coords = jnp.linspace(0, t1, 20)[:-1]
    # t_coords = jnp.linspace(0, t1, 4)[:-1]
    t_coords = jnp.array([0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.])*t1#

    u_pred_list = []
    v_pred_list = []
    p_pred_list = []
    s_pred_list = []
    # U_pred_list = []

    for idx in range(config.training.num_time_windows):
        print(f'{idx+1} / {config.training.num_time_windows}' )
        # Restore the checkpoint
        ckpt_path = os.path.join('.', 'ckpt', config.wandb.name, 'time_window_{}'.format(idx + 1))
        model.state = restore_checkpoint(model.state, ckpt_path)
        params = model.state.params
        
        print(f'mu = {mu}')

        u_pred = u_pred_fn(params, t_coords, coords[:, 0], coords[:, 1], mu)
        v_pred = v_pred_fn(params, t_coords, coords[:, 0], coords[:, 1], mu)
        s_pred = s_pred_fn(params, t_coords, coords[:, 0], coords[:, 1], mu)
        p_pred = p_pred_fn(params, t_coords, coords[:, 0], coords[:, 1], mu)

        u_pred_list.append(u_pred)
        v_pred_list.append(v_pred)
        s_pred_list.append(s_pred)
        p_pred_list.append(p_pred)    

    x = coords[:, 0]
    y = coords[:, 1]

    # print(f'coords x shape:{x.shape}')
    # print(f'coords y shape:{y.shape}')
    # print(f'coords u shape:{u_pred[0].shape}')
    # # print(len(u_pred))
    # print(f'u0 min: {jnp.min(u0)}')
    # print(f'u0 max: {jnp.max(u0)}')
    # print(f'v0 min: {jnp.min(v0)}')
    # print(f'v0 max: {jnp.max(v0)}')

    from matplotlib.animation import FuncAnimation
    from functools import partial  # Import partial to pass extra arguments to the update function

    # Create figures and axes once
    figs, axs = plt.subplots()
    figu, axu = plt.subplots()
    figv, axv = plt.subplots()
    figp, axp = plt.subplots()

    m = len(u_pred)  # Assuming u_pred and others are defined

    # Update function for each frame
    def update_s(frames, idx):
        axs.cla()  # Clear the current axis
        axs.scatter(x, y, s=1, c=s_pred_list[idx][frames], cmap='jet', vmin=0, vmax=1)

    def update_u(frames, idx):
        axu.cla()  # Clear the current axis
        axu.scatter(x, y, s=1, c=u_pred_list[idx][frames], cmap='jet', vmin=jnp.min(u05), vmax=jnp.max(u05))

    def update_v(frames, idx):
        axv.cla()  # Clear the current axis
        axv.scatter(x, y, s=1, c=v_pred_list[idx][frames], cmap='jet')#, vmin=jnp.min(v0), vmax=jnp.max(v0))


    def update_p(frames, idx):
        axp.cla()  # Clear the current axis
        axp.scatter(x, y, s=1, c=p_pred_list[idx][frames], cmap='jet', vmin=0, vmax=1)

    # Function to generate GIFs
    def make_gif(idx):
        ani_s = FuncAnimation(figs, partial(update_s, idx=idx-1), frames=m, interval=200)
        ani_s.save(f'./video_s_p5_{idx}.gif', writer='pillow')
        
        ani_u = FuncAnimation(figu, partial(update_u, idx=idx-1), frames=m, interval=200)
        ani_u.save(f'./video_u_p5_{idx}.gif', writer='pillow')
        
        ani_v = FuncAnimation(figv, partial(update_v, idx=idx-1), frames=m, interval=200)
        ani_v.save(f'./video_v_p5_{idx}.gif', writer='pillow')
        
        ani_p = FuncAnimation(figp, partial(update_p, idx=idx-1), frames=m, interval=200)
        ani_p.save(f'./video_p_p5_{idx}.gif', writer='pillow')

    # Generate GIFs for each time window
    for idx in range(1, config.training.num_time_windows + 1):
        make_gif(idx)

    # # Plot
    # # Save dir
    # save_dir = os.path.join(workdir, "figures", config.wandb.name)
    # if not os.path.isdir(save_dir):
    #     os.makedirs(save_dir)
    # for i in range(len(u_pred)):
    #     fig1 = plt.figure(figsize=(18, 12))
        
    #     # U = u_pred[i]
    #     # ind = jnp.where(coords[:, 0]==.2)[0]
    #     # dy = coords[ind[:-1], 1] - coords[ind[1:], 1]
    #     # U = (U[ind[:-1]] + U[ind[1:]])/2
    #     # U_m = jnp.sum(U*dy)/(coords[ind[-1], 1]-coords[ind[-2], 1])
    #     # print(f'U_m: {U_m}')

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
