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
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

import wandb

import models

from jaxpi.utils import restore_checkpoint

from utils import get_dataset, parabolic_inflow

def get_coords(xcyl, ycyl):
    dx = .00025
    x1_max = .021
    x2_max = .014

    x1 = np.arange(0,.021+dx/2, dx)
    x2 = np.arange(0,.014+dx/2, dx)

    X = np.zeros((x1.shape[0]*x2.shape[0],2))

    for i in range(x1.shape[0]):
        X[i*x2.shape[0]:(i+1)*x2.shape[0],:] = np.transpose(np.concatenate(([x1[i]*np.ones(x2.shape[0])],[x2]),axis=0))

    # esquerda p entrada
    id = np.where(np.squeeze(np.matmul(X, np.array([[1],[0]]))) == 0)[0]
    inflow_coords = X[id]
    # direita p saida
    id = np.where(np.squeeze(np.matmul(X, np.array([[1],[0]]))) == x1_max)[0]
    outflow_coords = X[id]
    # bottom
    id = np.where(np.squeeze(np.matmul(X, np.array([[0],[1]]))) == 0)[0]
    bot_wall_coords = X[id]
    # top
    id = np.where(np.squeeze(np.matmul(X, np.array([[0],[1]]))) == x2_max)[0]
    top_wall_coords = X[id]
    # wall = top+bottom
    wall_coords = np.concatenate((bot_wall_coords, top_wall_coords), axis=0)

    # cylinder
    radius = np.sqrt(np.power((X[:,0] - (xcyl)) , 2) + np.power((X[:,1] - (ycyl)) , 2))
    inds = np.where(radius <= .0015)

    X = np.delete(X, inds, axis = 0)

    x1_rad = np.array(np.arange(0, 2*np.pi, np.pi/35))
    x2_rad = np.transpose(np.array([ycyl + np.sin(x1_rad) * .0015]))
    x1_rad = np.transpose(np.array([xcyl + np.cos(x1_rad) * .0015]))
    x1_rad = np.concatenate((x1_rad, x2_rad), axis = 1)
    cylinder_coords = x1_rad

    X = np.concatenate((X, x1_rad), axis = 0)

    inds = np.where(X[:,1] > x2_max)
    X = np.delete(X, inds, axis = 0)

    return jax.device_put(X)

def evaluate(config: ml_collections.ConfigDict, workdir: str):
    # Load dataset
    (
        u_fem, v_fem, p_fem, coords_fem,
        coords,
        inflow_coords,
        outflow_coords,
        wall_coords,
        mu, pin,
        cylinder_center,
        cyl_xy, cyl_walls_xy
    ) = get_dataset()

    # U_max = 0.3  # maximum velocity
    # u_inflow, _ = parabolic_inflow(inflow_coords[:, 1], U_max)

    U_max = .10#17#.25/3#visc .1
    # pmax = 15

    L_max = .021
    pmax = mu[0]*U_max/L_max

    # Nondimensionalization
    if config.nondim == True:
        # Nondimensionalization parameters
        U_star = U_max # 0.2  # characteristic velocity
        L_star = L_max #0.1  # characteristic length
        # Re = U_star * L_star / nu

        # Nondimensionalize coordinates and inflow velocity
        inflow_coords = inflow_coords / L_star
        outflow_coords = outflow_coords / L_star
        wall_coords = wall_coords / L_star

        # Nondimensionalize flow field
        # u_inflow = u_inflow / U_star
        u_ref = u_fem #/ U_star
        v_ref = v_fem #/ U_star
        p_ref = p_fem #/ pmax
        p_inflow = 10 / pmax

    # Initialize model
    model = models.NavierStokes2D(
        config,
        wall_coords,
    )

    # Restore checkpoint
    ckpt_path = os.path.join(".", "ckpt", config.wandb.name)
    model.state = restore_checkpoint(model.state, ckpt_path)
    params = model.state.params
    # 0.005, 0.004
    xcyl = .005; ycyl = .007#.014/2
    coords = get_coords(xcyl, ycyl)
    coords = coords / L_star
    xcyl = xcyl / L_star; ycyl = ycyl / L_star
    xcyl = xcyl * jnp.ones(coords.shape[0])
    ycyl = ycyl * jnp.ones(coords.shape[0])

    # Predict
    u_pred = model.u_pred_fn(params, coords[:, 0], coords[:, 1], xcyl, ycyl)
    v_pred = model.v_pred_fn(params, coords[:, 0], coords[:, 1], xcyl, ycyl)
    p_pred = model.p_pred_fn(params, coords[:, 0], coords[:, 1], xcyl, ycyl)

    # Plot
    # Save dir
    save_dir = os.path.join(workdir, "figures", config.wandb.name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    if config.nondim == True:
        # Dimensionalize coordinates and flow field
        coords = coords * L_star

        # print(f'U_star = {U_star}')
        print(jnp.max(u_ref))
        print(jnp.max(u_pred))
        # print(jnp.min(v_ref))
        # print(jnp.min(v_pred))
        # u_ref = u_ref #* U_star
        # v_ref = v_ref #* U_star

        u_pred = u_pred *U_star
        v_pred = v_pred *U_star

        p_pred = p_pred*pmax
        p_ref = p_ref#*pmax

    # Triangulation
    x = coords[:, 0]
    y = coords[:, 1]

    fig1 = plt.figure()#(figsize=(18, 12))
    plt.subplot(3, 1, 1)
    plt.scatter(x, y, s=1, c=u_pred, cmap="jet")#, levels=100)
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("u(x,y)")
    plt.tight_layout()

    plt.subplot(3, 1, 2)
    plt.scatter(x, y, s=1, c=v_pred, cmap="jet")#, levels=100)
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("v(x,y)")
    plt.tight_layout()

    plt.subplot(3, 1, 3)
    plt.scatter(x, y, s=1, c=p_pred, cmap="jet")#, levels=100)
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("p(x,y)")
    plt.tight_layout()

    save_path = os.path.join(save_dir, "aa.pdf")
    fig1.savefig(save_path, bbox_inches="tight", dpi=300)


