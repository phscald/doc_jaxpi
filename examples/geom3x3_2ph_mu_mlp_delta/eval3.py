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
import pickle

def evaluate(config: ml_collections.ConfigDict, workdir: str):
    # Load dataset
    
    (   initial,
        delta_matrices,
        mu0, mu1, rho0, rho1) = get_dataset(config)

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
    
            #    0      1        2       3        4    5      6     7     8
    mu_list = [.0025, .014375, .02625, .038125, .05, .0625, .075, .0875, .1]
    mu = mu_list[0]
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

    u_pred_list = []
    v_pred_list = []
    p_pred_list = []
    s_pred_list = []

    for indx in range(config.training.num_time_windows):
        print(f'{indx+1} / {config.training.num_time_windows}' )
        # Restore the checkpoint
        ckpt_path = os.path.join('.', 'ckpt', config.wandb.name, 'time_window_{}'.format(indx + 1))
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
    
    
    filepath = "./data/chip3x3_mu0_" + str(mu) + "_mu1_0.05.pkl"
    with open(filepath, 'rb') as filepath:
        arquivos = pickle.load(filepath)
    coords_fem = arquivos['coord']
    u_fem = arquivos['u_data'][idx_t] / U_max 
    v_fem = arquivos['v_data'][idx_t] / U_max 
    p_fem = arquivos['p_data'][idx_t] / pmax
    s_fem = arquivos['c_data'][idx_t]
    del arquivos
    
    from matplotlib.animation import FuncAnimation
    from functools import partial  # Import partial to pass extra arguments to the update function

    # Create figures and axes once
    figs, axs = plt.subplots(2, 1, figsize=(6, 10))
    figu, axu = plt.subplots(2, 1, figsize=(6, 10))
    figv, axv = plt.subplots(2, 1, figsize=(6, 10))
    figp, axp = plt.subplots(2, 1, figsize=(6, 10))

    m = len(u_pred)  # Assuming u_pred and others are defined

    # Update function for each frame
    def update_s(frames, indx):
        axs[0].cla()  # Clear the current axis
        axs[0].scatter(x, y, s=1, c=s_pred_list[indx][frames], cmap='jet', vmin=0, vmax=1)
        
        axs[1].cla()  # Clear the current axis
        axs[1].scatter(x, y, s=1, c=s_fem[frames], cmap='jet', vmin=0, vmax=1)
        
        plt.tight_layout() 

    def update_u(frames, indx):
        axu[0].cla()  # Clear the current axis
        axu[0].scatter(x, y, s=1, c=u_pred_list[indx][frames], cmap='jet', vmin=jnp.min(u0), vmax=jnp.max(u0))
        
        axu[1].cla()  # Clear the current axis
        axu[1].scatter(x, y, s=1, c=u_fem[frames], cmap='jet', vmin=jnp.min(u0), vmax=jnp.max(u0))
                
        plt.tight_layout() 

    def update_v(frames, indx):
        axv[0].cla()  # Clear the current axis
        axv[0].scatter(x, y, s=1, c=v_pred_list[indx][frames], cmap='jet', vmin=jnp.min(v0), vmax=jnp.max(v0))
        
        axv[1].cla()  # Clear the current axis
        axv[1].scatter(x, y, s=1, c=v_fem[frames], cmap='jet', vmin=jnp.min(v0), vmax=jnp.max(v0))
           
        plt.tight_layout() 

    def update_p(frames, indx):
        axp[0].cla()  # Clear the current axis
        axp[0].scatter(x, y, s=1, c=p_pred_list[indx][frames], cmap='jet', vmin=0, vmax=1)
                
        axp[1].cla()  # Clear the current axis
        axp[1].scatter(x, y, s=1, c=p_fem[frames], cmap='jet', vmin=0, vmax=1)
                
        plt.tight_layout() 

    # Function to generate GIFs
    def make_gif(indx):
        ani_s = FuncAnimation(figs, partial(update_s, indx=indx-1), frames=m, interval=200)
        ani_s.save(f'./video_s_p5_{indx}.gif', writer='pillow')
        
        ani_u = FuncAnimation(figu, partial(update_u, indx=indx-1), frames=m, interval=200)
        ani_u.save(f'./video_u_p5_{indx}.gif', writer='pillow')
        
        ani_v = FuncAnimation(figv, partial(update_v, indx=indx-1), frames=m, interval=200)
        ani_v.save(f'./video_v_p5_{indx}.gif', writer='pillow')
        
        ani_p = FuncAnimation(figp, partial(update_p, indx=indx-1), frames=m, interval=200)
        ani_p.save(f'./video_p_p5_{indx}.gif', writer='pillow')

    # Generate GIFs for each time window
    for idx in range(1, config.training.num_time_windows + 1):
        make_gif(idx)
