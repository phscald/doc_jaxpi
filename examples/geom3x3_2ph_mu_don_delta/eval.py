
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
    
    (   initial,
        delta_matrices,
        mu0, mu1, rho0, rho1) = get_dataset()

    fluid_params = (mu0, mu1, rho0, rho1)    
        
    (u0, v0, p0, s0, coords_initial,
            u_fem_s, v_fem_s, p_fem_s, s_fem_s, dt_fem, coords_fem,
            u_fem_q, v_fem_q, p_fem_q, s_fem_q) = initial
    
    _, eigvecs, _, _, _, _, _, _, _  = delta_matrices

    fluid_params = (mu0, mu1, rho0, rho1)
    pin = 50
    dp = pin
    pmax = dp
    L_max = 50/1000/100
    U_max = dp*L_max/mu1

    mu = .0025#.0056 # mu0 = [.0025, .01]
    t1 = 1 # it is better to change the time in the t_coords array. There it is possible to select the desired percentages of total time solved

    T = 1.0  # final time
    tmax =400
    
    coords_fem = coords_fem / L_max
    
    dt_fem = dt_fem / (mu1/dp)
    t_fem = jnp.cumsum(dt_fem)
    idx = jnp.where(t_fem<=tmax)[0]
    
    (u0, v0, p0) = (u0/U_max , v0/U_max , p0/pmax)
    u_fem_s = u_fem_s[idx] / U_max 
    v_fem_s = v_fem_s[idx] / U_max 
    p_fem_s = p_fem_s[idx] / pmax
    s_fem_s = s_fem_s[idx]
    u_fem_q = u_fem_q[idx] / U_max 
    v_fem_q = v_fem_q[idx] / U_max 
    p_fem_q = p_fem_q[idx] / pmax
    s_fem_q = s_fem_q[idx]
    
    t_fem = t_fem[idx]

    t0 = 0.0

    temporal_dom = jnp.array([t0, t1 * (1 + 0.05)]) # Must be same as the one used in training

    # Initialize model
    # model = models.NavierStokes2D(config, inflow_fn, temporal_dom, coords, Re)
    model = models.NavierStokes2DwSat(config, pin/pmax, temporal_dom, U_max, L_max, fluid_params) #  no 1

    # Restore checkpoint
    ckpt_path = os.path.join(".", "ckpt", config.wandb.name)
    ckpt_path = os.path.abspath(ckpt_path)
    model.state = restore_checkpoint(model.state, ckpt_path)
    params = model.state.params
    
    print("oi")
    X = eigvecs[:,:]
    
    # Predict
    u_pred_fn = jit(vmap(vmap(model.u_net, (None, None, 0, None)), (None, 0, None, None))) # shape t by xy
    v_pred_fn = jit(vmap(vmap(model.v_net, (None, None, 0, None)), (None, 0, None, None)))
    p_pred_fn = jit(vmap(vmap(model.p_net, (None, None, 0, None)), (None, 0, None, None)))
    s_pred_fn = jit(vmap(vmap(model.s_net, (None, None, 0, None)), (None, 0, None, None)))
    D_pred_fn = jit(vmap(vmap(model.D_net, (None, None, 0, None)), (None, 0, None, None)))
    
    num_t = 10  # Desired number of points
    idx_t = jnp.linspace(0, t_fem.shape[0] - 1, num_t, dtype=int)
    t_coords = t_fem[idx_t]/tmax    

    # idx_t = 0
    # t_coords = jnp.array([t_fem[idx_t]/tmax])

    # print(jnp.mean(v0)+4*jnp.std(v0))
    # print(jnp.mean(v0)-4*jnp.std(v0))
    # print(jda)

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

        u_pred = u_pred_fn(params, t_coords, X, mu)
        v_pred = v_pred_fn(params, t_coords, X, mu)
        s_pred = s_pred_fn(params, t_coords, X, mu)
        p_pred = p_pred_fn(params, t_coords, X, mu)      

        u_pred_list.append(u_pred)
        v_pred_list.append(v_pred)
        s_pred_list.append(s_pred)
        p_pred_list.append(p_pred)    

    print(f"D: {D_pred_fn(params, t_coords, X, mu)[0,0]}")    

    x = eigvecs[:, 0]
    y = eigvecs[:, 1]
    
    # print(jnp.mean(u0))
    # print(jnp.std(u0))
    # print(jnp.mean(v0))
    # print(jnp.std(v0))
    
    # print(jnp.mean(u_pred_list[0]))
    # print(jnp.std(u_pred_list[0]))
    # print(jnp.mean(v_pred_list[0]))
    # print(jnp.std(v_pred_list[0]))    
    


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
        axu.scatter(x, y, s=1, c=u_pred_list[idx][frames], cmap='jet' )#, vmin=jnp.mean(u0)-2*jnp.std(u0), vmax=jnp.mean(u0)+2*jnp.std(u0))

    def update_v(frames, idx):
        axv.cla()  # Clear the current axis
        axv.scatter(x, y, s=1, c=v_pred_list[idx][frames], cmap='jet' )#, vmin=jnp.mean(v0)-2*jnp.std(v0), vmax=jnp.mean(v0)+2*jnp.std(v0))

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

