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

import matplotlib.pyplot as plt
import matplotlib.tri as tri

import wandb

import models

from jaxpi.utils import restore_checkpoint
import pickle
from utils import get_dataset#, parabolic_inflow


def evaluate(config: ml_collections.ConfigDict, workdir: str):
    # Load dataset
    (
        coords,
        inflow_coords,
        outflow_coords,
        wall_coords,
        mu, rho
    ) = get_dataset()
    
    dp = 10
    L_max = 650/1000/1000
    U_max = dp*L_max/mu
    pmax =dp
    Re = rho*dp*(L_max**2)/(mu**2)  

    # Nondimensionalization
    if config.nondim == True:
        # Nondimensionalization parameters
        U_star = U_max # 0.2  # characteristic velocity
        L_star = L_max #0.1  # characteristic length
        Re = rho * U_star * L_star / mu

        # Nondimensionalize coordinates and inflow velocity
        inflow_coords = inflow_coords / L_star
        outflow_coords = outflow_coords / L_star
        wall_coords = wall_coords / L_star
        coords = coords / L_star
        
        # coords = coords_fem

        # Nondimensionalize flow field
        p_inflow = 10 / pmax
    else:
        U_star = 1.0
        L_star = 1.0
        # Re = 1 / nu

    # Initialize model
    model = models.NavierStokes2D(
        config,
        # u_inflow,
        p_inflow,
        inflow_coords,
        outflow_coords,
        wall_coords,
        mu, U_max, pmax,
        Re,
    )

    # Restore checkpoint
    ckpt_path = os.path.join(".", "ckpt", config.wandb.name)
    model.state = restore_checkpoint(model.state, ckpt_path)
    params = model.state.params
    
    # Predict
    u_pred = model.u_pred_fn(params, coords[:, 0], coords[:, 1])
    v_pred = model.v_pred_fn(params, coords[:, 0], coords[:, 1])
    p_pred = model.p_pred_fn(params, coords[:, 0], coords[:, 1])
    # u_pred1 = model.u_pred_fn(params, coords_fem[:, 0], coords_fem[:, 1])
    # v_pred1 = model.v_pred_fn(params, coords_fem[:, 0], coords_fem[:, 1])
    # p_pred1 = model.p_pred_fn(params, coords_fem[:, 0], coords_fem[:, 1])
    
    # Plot
    # Save dir
    save_dir = os.path.join(workdir, "figures", config.wandb.name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    if config.nondim == True:
        # Dimensionalize coordinates and flow field
        coords = coords

        u_pred = u_pred
        v_pred = v_pred
        p_pred = p_pred

    x = coords[:, 0]
    y = coords[:, 1]
    # x1 = coords_fem[:, 0]
    # y1 = coords_fem[:, 1]
    
    print(f'x_max = {jnp.max(x)}')
    print(f'y_max = {jnp.max(y)}')
    print(f'u_pred_min = {jnp.min(u_pred)}')
    print(f'u_pred_max = {jnp.max(u_pred)}')
    print(f'p_pred_min = {jnp.min(p_pred)}')
    print(f'p_pred_max = {jnp.max(p_pred)}')
    
    filepath = './pred_initial_fields.pkl'
    with open(filepath,"wb") as filepath:
        pickle.dump({"u_pred": u_pred,
                     "v_pred": v_pred,
                     "p_pred": p_pred,
                     "coords": coords,}, filepath)
    
    
    
    fig1 = plt.figure()#(figsize=(18, 12))
    # plt.subplot(3, 1, 1)
    # plt.scatter(x1, y1, s=1, c=u_ref, cmap="jet")#, levels=100)
    # plt.colorbar()
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.title("Exact")
    # plt.tight_layout()

    plt.subplot(3, 1, 2)
    plt.scatter(x, y, s=1, c=u_pred, cmap="jet")#, levels=100)
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Predicted u(x, y)")
    plt.tight_layout()

    # plt.subplot(3, 1, 3)
    # plt.scatter(x1, y1, s=1, c=jnp.abs(u_ref - u_pred1), cmap="jet")#, levels=100)
    # plt.colorbar()
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.title("Absolute error")
    # plt.tight_layout()

    save_path = os.path.join(save_dir, "ns_steady_u.jpg")
    fig1.savefig(save_path, bbox_inches="tight", dpi=300)


    fig2 = plt.figure()#(figsize=(18, 12))
    # plt.subplot(3, 1, 1)
    # plt.scatter(x1, y1, s=1, c=v_ref, cmap="jet")#, levels=100)
    # plt.colorbar()
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.title("Exact")
    # plt.tight_layout()

    plt.subplot(3, 1, 2)
    plt.scatter(x, y, s=1, c=v_pred, cmap="jet")#, levels=100)
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Predicted v(x, y)")
    plt.tight_layout()

    # plt.subplot(3, 1, 3)
    # plt.scatter(x1, y1, s=1, c=jnp.abs(v_ref - v_pred1), cmap="jet")#, levels=100)
    # plt.colorbar()
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.title("Absolute error")
    # plt.tight_layout()

    save_path = os.path.join(save_dir, "ns_steady_v.jpg")
    fig2.savefig(save_path, bbox_inches="tight", dpi=300)

    fig3 = plt.figure()#(figsize=(18, 12))
    # plt.subplot(3, 1, 1)
    # plt.scatter(x1, y1, s=1, c=p_ref, cmap="jet")#, levels=100)
    # plt.colorbar()
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.title("Exact")
    # plt.tight_layout()

    plt.subplot(3, 1, 2)
    plt.scatter(x, y, s=1, c=p_pred, cmap="jet")#, levels=100)
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Predicted p(x, y)")
    plt.tight_layout()

    # plt.subplot(3, 1, 3)
    # plt.scatter(x1, y1, s=1, c=jnp.abs(p_ref - p_pred1), cmap="jet")#, levels=100)
    # plt.colorbar()
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.title("Absolute error")
    # plt.tight_layout()

    save_path = os.path.join('./' , save_dir, "ns_steady_p.jpg")
    fig3.savefig(save_path, bbox_inches="tight", dpi=300)
    


    ## compute U
    
    U = u_pred
    ind = jnp.where(coords[:, 0]==.2)[0]
    x_ = coords[ind, 0]
    y_ = coords[ind, 1]
    dy = coords[ind[:-1], 1] - coords[ind[1:], 1]
    U = (U[ind[:-1]] + U[ind[1:]])/2
    U_m = jnp.sum(U*dy)/coords[ind[-1], 1]
    print(f'U_m: {U_m}')
    
    
    U = u_pred
    ind = jnp.where(coords[:, 0]==.5)[0]
    x_ = coords[ind, 0]
    y_ = coords[ind, 1]
    dy = coords[ind[:-1], 1] - coords[ind[1:], 1]
    U = (U[ind[:-1]] + U[ind[1:]])/2
    U_m = jnp.sum(U*dy)/coords[ind[-1], 1]
    print(f'U_m: {U_m}')

    U = u_pred
    ind = jnp.where(coords[:, 0]==.8)[0]
    x_ = coords[ind, 0]
    y_ = coords[ind, 1]
    dy = coords[ind[:-1], 1] - coords[ind[1:], 1]
    U = (U[ind[:-1]] + U[ind[1:]])/2
    U_m = jnp.sum(U*dy)/coords[ind[-1], 1]
    print(f'U_m: {U_m}')