import functools
from functools import partial
import time
import os

from absl import logging

import jax
import jax.numpy as jnp
from jax import random
from jax import pmap, vmap, jacrev
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_map

import ml_collections

import wandb

import matplotlib.pyplot as plt

from jaxpi.samplers import SpaceSampler, BaseSampler
from jaxpi.logging import Logger
from jaxpi.utils import save_checkpoint, restore_checkpoint

import models
from utils import get_dataset, parabolic_inflow

import pickle


class ResSampler(BaseSampler):
    def __init__(
        self,
        mu_dom,
        pin_dom,
        coords,
        inflow_coords,
        outflow_coords,
        # wall_coords,
        cylinder_center,
        rng_key=random.PRNGKey(1234),
    ):
        super().__init__(rng_key)

        self.mu_dom = mu_dom
        self.pin_dom = pin_dom
        self.coords = coords
        self.inflow_coords = inflow_coords
        self.outflow_coords = outflow_coords
        # self.wall_coords = wall_coords
        self.cyl_center = cylinder_center
        self.batch_size = 10

    def get_wall(self, cyl_center):

        cyl_walls_x = jnp.zeros((cyl_center.shape[0], jnp.arange(0, 2, 1/35).shape[0]))
        cyl_walls_y = jnp.zeros((cyl_center.shape[0], jnp.arange(0, 2, 1/35).shape[0]))
        cyl_x     = jnp.zeros((cyl_center.shape[0], jnp.arange(0, 2, 1/35).shape[0]))
        cyl_y     = jnp.zeros((cyl_center.shape[0], jnp.arange(0, 2, 1/35).shape[0]))
        for i in range(cyl_center.shape[0]):
            L_max = .021
            R = .0015 / L_max
            x1_rad =(jnp.arange(0, 2*jnp.pi, jnp.pi/35))
            x2_rad = cyl_center[i,1] + jnp.sin(x1_rad) * R
            x1_rad = cyl_center[i,0] + jnp.cos(x1_rad) * R
            cyl_walls_x.at[i,:].set(x1_rad)
            cyl_walls_y.at[i,:].set(x2_rad)
            cyl_x.at[i,:].set(jnp.ones(cyl_x.shape[1])*cyl_center[i,0])
            cyl_y.at[i,:].set(jnp.ones(cyl_x.shape[1])*cyl_center[i,1])
        cyl_walls_x = jnp.reshape(cyl_walls_x, (-1,1))
        cyl_walls_y = jnp.reshape(cyl_walls_y, (-1,1))
        cyl_walls_xy = jnp.concatenate((cyl_walls_x, cyl_walls_y), axis =1)
        cyl_x = jnp.reshape(cyl_x, (-1,1))
        cyl_y = jnp.reshape(cyl_y, (-1,1))
        cyl_xy = jnp.concatenate((cyl_x, cyl_y), axis =1)
        return cyl_xy, cyl_walls_xy

    @partial(pmap, static_broadcasted_argnums=(0,))
    def data_generation(self, key):
        "Generates data containing batch_size samples"
        key1, key2, key3, key4, key5, key6 = random.split(key, 6)

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
            key5, self.coords.shape[0], shape=(self.batch_size,)
        )
        spatial_batch = self.coords[spatial_idx, :]

        cyl_idx = random.choice(
            key6, self.cyl_center.shape[0], shape=(self.batch_size,)
        )
        cyl_batch = self.cyl_center[cyl_idx, :]

        L_max = .021
        R = .0015 / L_max
        distances = jnp.sqrt(jnp.square(spatial_batch[:,0] - cyl_batch[:,0]) + jnp.square(spatial_batch[:,1] - cyl_batch[:,1]))
        mask = distances < R

        spatial_batch = spatial_batch.at[:,0].set(jnp.where(mask, 0.0, spatial_batch[:, 0]))
        spatial_batch = spatial_batch.at[:,1].set(jnp.where(mask, 0.0, spatial_batch[:, 1]))

        cyl_xy, cyl_walls = self.get_wall(cyl_batch[0:15])
        # xcyl_batch = self.get_wall(cyl_batch[0:15])

        batch = ((mu_batch), (pin_batch), inflow_batch, outflow_batch, spatial_batch, cyl_batch, cyl_walls, cyl_xy)

        return batch


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str):
    # Initialize W&B
    wandb_config = config.wandb
    wandb.init(project=wandb_config.project, name=wandb_config.name)

    # Initialize logger
    logger = Logger()

    # Get dataset
    (
        u_fem, v_fem, p_fem, coords_fem,
        coords,
        inflow_coords,
        outflow_coords,
        wall_coords,
        mu, pin,
        cylinder_center
    ) = get_dataset()

    U_max = .10#17#.25/3#visc .1

    L_max = .021
    pmax = mu[0]*U_max/L_max

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
        cylinder_center = cylinder_center / L_star
        coords = coords / L_star
        # coords_fem = coords_fem / L_star

        # Nondimensionalize flow field
        # u_inflow = u_inflow / U_star
        # u_ref = u_fem / U_star
        # v_ref = v_fem / U_star
        # p_ref = p_fem / pmax
        # p_inflow = 10 / pmax
        pin = jnp.array(pin) / pmax

    # else:
    #     U_star = 1.0
    #     L_star = 1.0
        # Re = 1 / nu

    # Initialize model
    model = models.NavierStokes2D(
        config,
        wall_coords,
    )

    # # Restore checkpoint
    # ckpt_path = os.path.join(".", "ckpt", "default")
    # model.state = restore_checkpoint(model.state, ckpt_path)
    # print(f'{model.state}')
    # print(dsa)


    # Initialize evaluator
    evaluator = models.NavierStokesEvaluator(config, model)

    # Initialize  residual sampler
    res_sampler = iter(ResSampler(mu, pin, coords, inflow_coords, outflow_coords, cylinder_center, rng_key=random.PRNGKey(0)))


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
            if step % config.logging.log_every_steps == 0: print(f'step: {step} / {config.training.max_steps}')
                # Get the first replica of the state and batch
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
