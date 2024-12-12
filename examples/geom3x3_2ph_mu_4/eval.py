
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
    
    pin = 50
    (
        coords,
        inflow_coords,
        outflow_coords,
        wall_coords,
        initial,
        mu0, mu1, rho0, rho1
    ) = get_dataset(pin=pin)
    noslip_coords = wall_coords
    
    (u0, v0, p0, s0, coords_initial,
            u_fem_s, v_fem_s, p_fem_s, s_fem_s, dt_fem, coords_fem,
            u_fem_q, v_fem_q, p_fem_q, s_fem_q,
            coords_middle, t_middle) = initial
    
    print(jnp.max(coords[:,0]))
    print(jnp.max(coords[:,1]))


    print(f'coords shape:{coords.shape}')
    print(f'inflow coords shape:{inflow_coords.shape}')

    fluid_params = (mu0, mu1, rho0, rho1)
    dp = pin; pmax =dp
    
    L_max = 50/1000/100
    U_max = dp*L_max/mu1
    u_maxs = jnp.max(u_fem_s)/U_max
    v_maxs = jnp.max(v_fem_s)/U_max
    u_maxq = jnp.max(u_fem_q)/U_max
    v_maxq = jnp.max(v_fem_q)/U_max
    
    uv_max = (u_maxq, v_maxq, u_maxs, v_maxs)
    print(f"U_max: {U_max}")
    print(f"u_fem_s/U_max: {jnp.max(u_fem_s)/U_max}")
    print(f"v_fem_s/U_max: {jnp.max(v_fem_s)/U_max}")

    mu = .0056 # mu0 = [.0025, .01]
    t1 = 1 # it is better to change the time in the t_coords array. There it is possible to select the desired percentages of total time solved

    T = 1.0  # final time

    # Nondimensionalization
    if config.nondim == True:
        # Nondimensionalize coordinates and inflow velocity
        inflow_coords = inflow_coords / L_max
        outflow_coords = outflow_coords / L_max
        noslip_coords = noslip_coords / L_max
        coords = coords / L_max    
        
        u0, v0, p0 = u0/U_max, v0/U_max, p0/dp   
    
        ind_coords = random.choice(
            random.PRNGKey(1234),
            coords.shape[0],
            shape=(int(coords.shape[0]/6),),
            replace=True,
        )
        coords = coords[ind_coords]

        p_inflow = (pin / pmax) * jnp.ones((inflow_coords.shape[0]))
    
    t0 = 0.0

    temporal_dom = jnp.array([t0, t1 * (1 + 0.05)]) # Must be same as the one used in training

    # Initialize model
    # model = models.NavierStokes2D(config, inflow_fn, temporal_dom, coords, Re)
    model = models.NavierStokes2DwSat(config, pin/pmax, temporal_dom, coords, U_max, L_max, fluid_params, uv_max) #  no 1

    # Restore checkpoint
    ckpt_path = os.path.join(".", "ckpt", config.wandb.name)
    ckpt_path = os.path.abspath(ckpt_path)
    model.state = restore_checkpoint(model.state, ckpt_path)
    params = model.state.params
    
    # Predict
    u_pred_fn = jit(vmap(vmap(model.u_net, (None, None, 0, 0, None)), (None, 0, None, None, None))) # shape t by xy
    v_pred_fn = jit(vmap(vmap(model.v_net, (None, None, 0, 0, None)), (None, 0, None, None, None)))
    p_pred_fn = jit(vmap(vmap(model.p_net, (None, None, 0, 0, None)), (None, 0, None, None, None)))
    s_pred_fn = jit(vmap(vmap(model.s_net, (None, None, 0, 0, None)), (None, 0, None, None, None)))
    D_pred_fn = jit(vmap(vmap(model.D_net, (None, None, 0, 0, None)), (None, 0, None, None, None)))

    # t_coords = jnp.linspace(0, t1, 20)[:-1]
    # t_coords = jnp.linspace(0, t1, 4)[:-1]
    t_coords = jnp.array([0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.])*t1*.5#
    # t_coords = jnp.array([0,])

    u_pred_list = []
    v_pred_list = []
    p_pred_list = []
    s_pred_list = []

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

    print(f"D: {D_pred_fn(params, t_coords, coords[:, 0], coords[:, 1], mu)[0,0]}")    

    x = coords[:, 0]
    y = coords[:, 1]
    
    # x = coord_intial[:, 0]
    # y = coord_intial[:, 1]

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
        axs.scatter(x, y, s=1, c=s_pred_list[idx][frames], cmap='jet' , vmin=0, vmax=1)

    def update_u(frames, idx):
        axu.cla()  # Clear the current axis
        axu.scatter(x, y, s=1, c=u_pred_list[idx][frames], cmap='jet', vmin=jnp.min(u0), vmax=jnp.max(u0))

    def update_v(frames, idx):
        axv.cla()  # Clear the current axis
        axv.scatter(x, y, s=1, c=v_pred_list[idx][frames], cmap='jet', vmin=jnp.min(v0), vmax=jnp.max(v0))


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

