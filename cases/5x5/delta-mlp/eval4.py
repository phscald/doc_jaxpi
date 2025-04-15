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

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import get_dataset

import matplotlib.pyplot as plt
import matplotlib.tri as tri
import pickle

def evaluate(config: ml_collections.ConfigDict, workdir: str):
    # Load dataset
    
    L_max = 50/1000/100
    (   initial,
        delta_matrices,
        mu1, rho0, rho1) = get_dataset(L_max, config)

    fluid_params = (0, mu1, rho0, rho1)    
        
    (u0, v0, p0, s0, coords_initial,
     u_fem, v_fem, p_fem, s_fem, t_fem, 
     coords_fem, mu_list) = initial
    
    _, eigvecs, _, _, _, _, _, _, _  = delta_matrices

    pin = 100
    dp = pin
    pmax = dp
    U_max = dp*L_max/mu1

    # mu_list = [.0025, .0033333333333333335, .005, .01, .05, .0625, .075, .0875, .1]
    # mu_list = [.0033333333333333335, .01, .0625, .0875]
    mu = .0025
    ind_mu = jnp.where(mu_list==mu)[0]
    
    t1 = 1 # it is better to change the time in the t_coords array. There it is possible to select the desired percentages of total time solved

    T = 1.0  # final time
    tmax = 24000
    
    # idx = jnp.where(t_fem<=tmax)[0]
    time = 4825
    idx = jnp.where(t_fem<time)[0]
    idx = jnp.array([idx[-1], idx[-1]+1])

    u_fem = jnp.squeeze( u_fem[ind_mu, idx] )
    v_fem = jnp.squeeze( v_fem[ind_mu, idx] )
    p_fem = jnp.squeeze( p_fem[ind_mu, idx] )
    s_fem = jnp.squeeze( s_fem[ind_mu, idx] )
    
    t_fem = t_fem[idx]

    t0 = 0.0

    temporal_dom = jnp.array([t0, t1 * (1 + 0.05)]) # Must be same as the one used in training
    
    # u_mean = u_fem.mean()
    u_mean = 0
    u_std = u0.std()*3
    # v_mean = v_fem.mean()
    v_mean = 0
    v_std = v0.std()*3
    print(f"ustd: {u_std}")
    print(f"vstd: {v_std}")
    u_stats = (u_mean, u_std, v_mean, v_std)

    # Initialize model
    # model = models.NavierStokes2D(config, inflow_fn, temporal_dom, coords, Re)
    model = models.NavierStokes2DwSat(config, pin/pmax, temporal_dom, U_max, L_max, (fluid_params, u_stats)) #  no 1

    # Restore checkpoint
    ckpt_path = os.path.join(".", "ckpt", config.wandb.name)
    ckpt_path = os.path.abspath(ckpt_path)
    model.state = restore_checkpoint(model.state, ckpt_path)
    params = model.state.params

    X = eigvecs[:,:]
    ind = random.choice(random.PRNGKey(1234), eigvecs.shape[0], shape=(int(eigvecs.shape[0]),) )
    X = X[ind]
    u_fem = u_fem[:,ind]
    v_fem = v_fem[:,ind]
    p_fem = p_fem[:,ind]
    s_fem = s_fem[:,ind]
    
    # Predict
    u_pred_fn = jit(vmap(vmap(model.u_net, (None, None, 0, None)), (None, 0, None, None))) # shape t by xy
    v_pred_fn = jit(vmap(vmap(model.v_net, (None, None, 0, None)), (None, 0, None, None))) 
    p_pred_fn = jit(vmap(vmap(model.p_net, (None, None, 0, None)), (None, 0, None, None))) 
    s_pred_fn = jit(vmap(vmap(model.s_net, (None, None, 0, None)), (None, 0, None, None))) 
    
    t_coords = t_fem/tmax 
    # u_fem = u_fem[idx_t]
    # v_fem = v_fem[idx_t]
    # p_fem = p_fem[idx_t]
    # s_fem = s_fem[idx_t]

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
        
    x = eigvecs[ind, 0]
    y = eigvecs[ind, 1]
    
    from matplotlib.animation import FuncAnimation
    from functools import partial  # Import partial to pass extra arguments to the update function

    m = len(u_pred)  # Assuming u_pred and others are defined
    
    def denormalize_mustd(u, umean, ustd):
        return u*ustd+umean
    def normalize_mustd(u, umean, ustd):
        return (u-umean)/ustd
    # u_pred = denormalize_mustd(u_pred, u_mean, u_std)
    # v_pred = denormalize_mustd(v_pred, v_mean, v_std)
    u_fem = normalize_mustd(u_fem, u_mean, u_std)
    v_fem = normalize_mustd(v_fem, v_mean, v_std)
    u0 = normalize_mustd(u0, u_mean, u_std)
    v0 = normalize_mustd(v0, v_mean, v_std)

    plt.rcParams['font.family'] = 'DeJavu Serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 12  
    
    indx = 0

    # --- Scalar Field (s) ---
    plt.figure()
    plt.scatter(x, y, s=1, c=s_pred[indx], cmap='jet', vmin=0, vmax=1)
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(f'field_delta_s_{mu}_{time}.jpg', format='jpg')

    plt.figure()
    plt.scatter(x, y, s=1, c=s_fem[indx], cmap='jet', vmin=0, vmax=1)
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(f'field_fem_s_{mu}_{time}.jpg', format='jpg')

    # --- U Field ---
    plt.figure()
    plt.scatter(x, y, s=1, c=u_pred[indx], cmap='jet', vmin=-3, vmax=3)
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])  
    plt.tight_layout()
    plt.savefig(f'field_delta_u_{mu}_{time}.jpg', format='jpg')

    plt.figure()
    plt.scatter(x, y, s=1, c=u_fem[indx], cmap='jet', vmin=-3, vmax=3)
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(f'field_fem_u_{mu}_{time}.jpg', format='jpg')

    # --- V Field ---
    plt.figure()
    plt.scatter(x, y, s=1, c=v_pred[indx], cmap='jet', vmin=-3, vmax=3)
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(f'field_delta_v_{mu}_{time}.jpg', format='jpg') 

    plt.figure()
    plt.scatter(x, y, s=1, c=v_fem[indx], cmap='jet', vmin=-3, vmax=3)
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(f'field_fem_v_{mu}_{time}.jpg', format='jpg')

    # --- Pressure Field (p) ---
    plt.figure()
    plt.scatter(x, y, s=1, c=p_pred[indx], cmap='jet', vmin=0, vmax=1)
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(f'field_delta_p_{mu}_{time}.jpg', format='jpg')

    plt.figure()
    plt.scatter(x, y, s=1, c=p_fem[indx], cmap='jet', vmin=0, vmax=1)
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(f'field_fem_p_{mu}_{time}.jpg', format='jpg')

