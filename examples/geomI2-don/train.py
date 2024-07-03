import functools
from functools import partial

import time
import os

from absl import logging

import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap, jacrev
from jax import random, vmap, pmap, local_device_count
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_map

import ml_collections

import wandb

import matplotlib.pyplot as plt

from jaxpi.samplers import SpaceSampler, BaseSampler
from jaxpi.logging import Logger
from jaxpi.utils import save_checkpoint

import models
from utils import get_dataset


class mpSpaceSampler(BaseSampler):
    def __init__(
        self, mu_dom, pin_dom, inflow_coords, outflow_coords, spatial_coords, batch_size, rng_key=random.PRNGKey(1234)
    ):
        super().__init__(batch_size, rng_key)

        self.mu_dom = mu_dom
        self.pin_dom = pin_dom
        self.inflow_coords = inflow_coords
        self.outflow_coords = outflow_coords
        self.spatial_coords = spatial_coords


    @partial(pmap, static_broadcasted_argnums=(0,))
    def data_generation(self, key):
        "Generates data containing batch_size samples"
        key1, key2, key3, key4, key5 = random.split(key, 5)

        mu_batch = jax.random.uniform(
            key1,
            shape=(self.batch_size, ),
            minval=self.mu_dom[0],
            maxval=self.mu_dom[1],
        )

        pin_batch = jax.random.uniform(
            key2,
            shape=(self.batch_size, ),
            minval=self.pin_dom[0],
            maxval=self.pin_dom[1],
        )

        inflow_idx = random.choice(
            key3, self.inflow_coords.shape[0], shape=(self.batch_size,)
        )
        inflow_batch = self.inflow_coords[inflow_idx, :]

        outflow_idx = random.choice(
            key4, self.outflow_coords.shape[0], shape=(self.batch_size,)
        )
        outflow_batch = self.outflow_coords[outflow_idx, :]

        spatial_idx = random.choice(
            key5, self.spatial_coords.shape[0], shape=(self.batch_size,)
        )
        spatial_batch = self.spatial_coords[spatial_idx, :]
        # batch = jnp.concatenate([mu_batch, pin_batch, spatial_batch], axis=1)
        batch = ((mu_batch), (pin_batch), inflow_batch, outflow_batch, spatial_batch)

        return batch


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str):
    # Initialize W&B
    wandb_config = config.wandb
    wandb.init(project=wandb_config.project, name=wandb_config.name)

    # Initialize logger
    logger = Logger()

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

    U_max = .15#.25/3#visc .1
    # pmax = 15

    L_max = .021
    mu_max = .10
    pmax = mu_max*U_max/L_max

    # # Inflow boundary conditions
    # U_max = 0.3  # maximum velocity
    # u_inflow, _ = parabolic_inflow(inflow_coords[:, 1], U_max)

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
        coords = coords / L_star


        # Nondimensionalize flow field
        # u_inflow = u_inflow / U_star

        p_inflow = jnp.array(p_inflow) / pmax

    else:
        U_star = 1.0
        L_star = 1.0
        # Re = 1 / nu

    # Initialize model
    model = models.NavierStokes2D(
        config,
        10/pmax,
        wall_coords, 
    )

    res_sampler = iter(
                mpSpaceSampler(
                    mu,
                    p_inflow,
                    inflow_coords,
                    outflow_coords,
                    coords,
                    config.training.batch_size_per_device,
                    rng_key=random.PRNGKey(1234)
                )
            )
    
    # batch = next(res_sampler)
    # (mu_batch, pin_batch, inflow_batch, outflow_batch, batch) = batch
    # print(pin_batch.max(axis=0))
    # print(dsad)
    

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
            if step % config.logging.log_every_steps == 0: print(f'step: {step}')
        #         # Get the first replica of the state and batch
        #         state = jax.device_get(tree_map(lambda x: x[0], model.state))
        #         batch = jax.device_get(tree_map(lambda x: x[0], batch))
        #         log_dict = evaluator(state, batch, coords_fem, u_ref, v_ref)
        #         wandb.log(log_dict, step)

        #         end_time = time.time()
        #         # Report training metrics
        #         logger.log_iter(step, start_time, end_time, log_dict)

        # Save checkpoint
        if config.saving.save_every_steps is not None:
            if (step + 1) % config.saving.save_every_steps == 0 or (
                step + 1
            ) == config.training.max_steps:
                path = os.path.join(workdir, "ckpt", config.wandb.name)
                path = os.path.abspath(path)
                save_checkpoint(model.state, path, keep=config.saving.num_keep_ckpts)

    return model
