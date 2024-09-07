import time
import os

from absl import logging

import jax
import jax.numpy as jnp
from jax import vmap, jacrev
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_map

import ml_collections

import wandb

import matplotlib.pyplot as plt

from jaxpi.samplers import SpaceSampler
from jaxpi.logging import Logger
from jaxpi.utils import save_checkpoint, restore_checkpoint

import models
from utils import get_dataset#, parabolic_inflow

import pickle


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str):
    # Initialize W&B
    wandb_config = config.wandb
    wandb.init(project=wandb_config.project, name=wandb_config.name)

    # Initialize logger
    logger = Logger()

    # Get dataset
    (
        # u_fem, v_fem, p_fem, coords_fem,
        coords,
        inflow_coords,
        outflow_coords,
        wall_coords,
        # cylinder_coords,
        mu, rho
    ) = get_dataset()

    dp = 10
    L_max = 650/1000/1000
    U_max = dp*L_max/mu
    pmax =dp
    Re = rho*dp*(L_max**2)/(mu**2)
    print(f'Re={Re}')

    # pin = 10
    # L_max = 650/1000
    # U_max = 1/8/mu*pin/L_max*(100/1000)**2
    # pmax = mu*U_max/L_max 
    
    # print(pmax)
    # print(dsa)

    # # Inflow boundary conditions
    # U_max = 0.3  # maximum velocity
    # u_inflow, _ = parabolic_inflow(inflow_coords[:, 1], U_max)

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
        # cylinder_coords = cylinder_coords / L_star
        coords = coords / L_star
        # coords_fem = coords_fem / L_star

        # Nondimensionalize flow field
        # u_inflow = u_inflow / U_star
        # u_ref = u_fem / U_star
        # v_ref = v_fem / U_star
        # p_ref = p_fem / pmax
        p_inflow = dp / pmax

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
        # cylinder_coords,
        mu, U_max, pmax,
        Re
    )

    # # Restore checkpoint
    # ckpt_path = os.path.join(".", "ckpt", "default")
    # model.state = restore_checkpoint(model.state, ckpt_path)
    # print(f'{model.state}')
    # print(dsa)


    # Initialize evaluator
    evaluator = models.NavierStokesEvaluator(config, model)

    # Initialize  residual sampler
    res_sampler = iter(SpaceSampler(coords, config.training.batch_size_per_device))

    # jit warm up
    print("Waiting for JIT...")
    for step in range(config.training.max_steps):
        start_time = time.time()

        batch = next(res_sampler)
        model.state = model.step(model.state, batch)

        # Update weights if necessary
        if config.weighting.scheme in ["grad_norm", "ntk"]:
            if step % config.weighting.update_every_steps == 0:
                model.state = model.update_weights(model.state, batch)

        # Log training metrics, only use host 0 to record results
        if jax.process_index() == 0:
            if step % config.logging.log_every_steps == 0: 
                print(f'step: {step} / {config.training.max_steps}')
                # # Get the first replica of the state and batch
                # state = jax.device_get(tree_map(lambda x: x[0], model.state))
                # batch = jax.device_get(tree_map(lambda x: x[0], batch))
                # log_dict = evaluator(state, batch, coords_fem, u_ref, v_ref)
                # wandb.log(log_dict, step)

                # end_time = time.time()
                # # Report training metrics
                # logger.log_iter(step, start_time, end_time, log_dict)

        # Save checkpoint
        if config.saving.save_every_steps is not None:
            if (step + 1) % config.saving.save_every_steps == 0 or (
                step + 1
            ) == config.training.max_steps:
                path = os.path.join(workdir, "ckpt", config.wandb.name)
                path = os.path.abspath(path)
                save_checkpoint(model.state, path, keep=config.saving.num_keep_ckpts)

    return model
