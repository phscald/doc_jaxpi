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

from utils import get_dataset, parabolic_inflow


def evaluate(config: ml_collections.ConfigDict, workdir: str):
    # Get dataset
    (
        coords,
        inflow_coords,
        outflow_coords,
        wall_coords,
        # cylinder_center,
        mu,
        p_inflow,
    ) = get_dataset()
    noslip_coords = wall_coords

    U_max = .10#17#.25/3#visc .1

    L_max = .021
    mu_max = 10
    pmax = mu_max*U_max/L_max

    # Nondimensionalization
    if config.nondim == True:
        # Nondimensionalization parameters
        U_star = U_max # 0.2  # characteristic velocity
        L_star = L_max #0.1  # characteristic length
        # Re = U_star * L_star / nu

        # Nondimensionalize coordinates and inflow velocity
        inflow_coords = inflow_coords / L_star
        outflow_coords = outflow_coords / L_star
        # noslip_coords = noslip_coords / L_star
        coords = coords / L_star

    else:
        U_star = 1.0
        L_star = 1.0
        # Re = 1 / nu

    # Initialize model
    model = models.NavierStokes2D(
        config,
        wall_coords, 
    )

    # # Restore checkpoint
    # ckpt_path = os.path.join(".", "ckpt", config.wandb.name)
    # model.state = restore_checkpoint(model.state, ckpt_path)

    # Restore checkpoint
    ckpt_path = os.path.join(".", "ckpt", "default")
    model.state = restore_checkpoint(model.state, ckpt_path)

    params = model.state.params

    mu = jnp.ones(coords[:, 0].shape)*.1
    pin = jnp.ones(coords[:, 0].shape)*10/pmax

    # Predict
    u_pred = model.u_pred_fn(params, coords[:, 0], coords[:, 1], mu, pin)
    v_pred = model.v_pred_fn(params, coords[:, 0], coords[:, 1], mu, pin)
    p_pred = model.p_pred_fn(params, coords[:, 0], coords[:, 1], mu, pin)

    x = coords[:, 0]
    y = coords[:, 1]


    # Plot
    # Save dir
    save_dir = os.path.join(workdir, "figures", config.wandb.name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    if config.nondim == True:
        # Dimensionalize coordinates and flow field
        coords = coords * L_star
        # u_ref = u_ref #* U_star
        # v_ref = v_ref #* U_star

        u_pred = u_pred *U_star
        v_pred = v_pred *U_star

        p_pred = p_pred*pmax
        # p_ref = p_ref#*pmax

    # Triangulation
    x = coords[:, 0]
    y = coords[:, 1]


    fig1 = plt.figure()#(figsize=(18, 12))
    plt.subplot(3, 1, 1)
    plt.scatter(x, y, s=1, c=u_pred, cmap="jet")#, levels=100)
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("u")
    plt.tight_layout()

    plt.subplot(3, 1, 2)
    plt.scatter(x, y, s=1, c=v_pred, cmap="jet")#, levels=100)
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("v")
    plt.tight_layout()

    plt.subplot(3, 1, 3)
    plt.scatter(x, y, s=1, c=p_pred, cmap="jet")#, levels=100)
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("p")
    plt.tight_layout()

    save_path = os.path.join(save_dir, "predicted_fields.pdf")
    fig1.savefig(save_path, bbox_inches="tight", dpi=300)
    # fig1.close()

