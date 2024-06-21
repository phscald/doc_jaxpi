import time
import os

from absl import logging

import jax
import jax.numpy as jnp
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
from utils import get_dataset, parabolic_inflow

# class ResSampler(BaseSampler):
#     def __init__(
#         self,
#         coords,
#         inflow_coords,
#         outflow_coords,
#         wall_coords,
#         cylinder_center,
#         rng_key=random.PRNGKey(1234),
#     ):
#         super().__init__(rng_key)

#         self.coords = coords
#         self.inflow_coords = inflow_coords
#         self.outflow_coords = outflow_coords
#         self.wall_coords = wall_coords
#         self.cylinder_center = cylinder_center
#         self.batch_size = 10
        

def get_center(coords, cylinder_center, idx):
    X = coords
    cyl_center = cylinder_center[idx]
    # cylinder
    radius = jnp.sqrt(jnp.power((X[:,0] - (cyl_center[0,0])) , 2) + jnp.power((X[:,1] - (cyl_center[0,1])) , 2))
    inds = jnp.where(radius <= .0015)[0]

    X = jnp.delete(X, inds, axis = 0)

    x1_rad = jnp.array(jnp.arange(0, 2*jnp.pi, jnp.pi/35))
    x2_rad = jnp.transpose(jnp.array([cyl_center[0,1] + jnp.sin(x1_rad) * .0015]))
    x1_rad = jnp.transpose(jnp.array([cyl_center[0,0] + jnp.cos(x1_rad) * .0015]))
    x1_rad = jnp.concatenate((x1_rad, x2_rad), axis = 1)
    # cylinder_coords = x1_rad
    L_star = .021

    X = jnp.concatenate((X, x1_rad), axis = 0)/ L_star

    return X#, cylinder_coords

    # @partial(pmap, static_broadcasted_argnums=(0,))
    # def data_generation(self, key):
    #     "Generates data containing batch_size samples"
    #     subkeys = random.split(key, 4)

    #     idx = random.choice(
    #         subkeys[1],
    #         jnp.arange(start=0, stop=self.cylinder_center.shape[0]),
    #         shape=(self.batch_size,),
    #         replace=True,
    #         )

    #     X = []; wall = []
    #     for i in range(self.batch_size):
    #         X_, wall_ = self.get_center(idx)
    #         X.append(X_)
    #         wall.append(wall_)

    #     X = jnp.vstack(X)
    #     wall = jnp.vstack(wall)
    #     X  = random.permutation(
    #         subkeys[3], X) 
    #     wall  = random.permutation(
    #         subkeys[3], wall)

    #     batch = (X,
    #         self.inflow_coords,
    #         self.outflow_coords,
    #         self.wall_coords,)

    #     return batch




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
        cylinder_center,
        mu,
    ) = get_dataset()

    cylinder_center = jnp.ones(cylinder_center.shape)*[.0105, .007]

    U_max = .10#17#.25/3#visc .1

    L_max = .021
    pmax = mu*U_max/L_max
    num_centers = 1

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
        # wall_coords = wall_coords / L_star
        # cylinder_coords = cylinder_coords / L_star
        # coords = coords / L_star
        # coords_fem = coords_fem / L_star

        # Nondimensionalize flow field
        # u_inflow = u_inflow / U_star
        # u_ref = u_fem / U_star
        # v_ref = v_fem / U_star
        # p_ref = p_fem / pmax
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
        # cylinder_coords,
        mu, U_max, pmax
        # Re,
    )
    
   # Initialize evaluator
    evaluator = models.NavierStokesEvaluator(config, model)

    # Initialize  residual sampler
    # res_sampler = iter(SpaceSampler(coords, config.training.batch_size_per_device))

    # jit warm up
    print("Waiting for JIT...")
    for step in range(config.training.max_steps):
        start_time = time.time()
        
        subkeys = random.split(random.PRNGKey(0), 3)
        idx = random.choice(
            subkeys[0],
            jnp.arange(start=0, stop=cylinder_center.shape[0]),
            shape=(num_centers,),
            replace=True,
            )

        # X = []; cyl_center_X = []
        # for i in range(num_centers):
        X = get_center(coords, cylinder_center, idx)
            # X.append(X_)
            # wall.append(wall_)
            
        cyl_center_X = jnp.ones((X.shape[0],2))*cylinder_center[idx]
            # cyl_center_wall.append(jnp.ones((X_.shape[0],2))*cylinder_center[idx[i]])

        # X = jnp.vstack(X)
        # wall = jnp.vstack(wall)
        # cyl_center_X = jnp.vstack(cyl_center_X)
        # cyl_center_wall = jnp.vstack(cyl_center_wall)
        # idx_x = jax.random.permutation(subkeys[1], jnp.arange(X.shape[0]))
        # idx_w = jax.random.permutation(subkeys[2], jnp.arange(wall.shape[0]))

        # X = X[idx_x]
        # cyl_center_X = cyl_center_X[idx_x]
        # wall = wall[idx_w]
        # cyl_center_wall = cyl_center_wall[idx_w]

        # batch = next(res_sampler)
        # batch = (X, cyl_center_X)
        # print(len(batch))
        # print(X.shape)
        # print(cyl_center_X.shape)
        batch = (jnp.concatenate((X, cyl_center_X), axis=1))
        res_sampler = iter(SpaceSampler(batch, config.training.batch_size_per_device))
        batch = next(res_sampler)

        # print(X.shape)
        # print(cyl_center_X)
        # print(model.state)
        model.state = model.step(model.state, batch)

        # Update weights if necessary
        if config.weighting.scheme in ["grad_norm", "ntk"]:
            if step % config.weighting.update_every_steps == 0:
                model.state = model.update_weights(model.state, batch)

        # Log training metrics, only use host 0 to record results
        if jax.process_index() == 0:
            if step % config.logging.log_every_steps == 0:
                print(f'step: {step}')
            #     # Get the first replica of the state and batch
            #     state = jax.device_get(tree_map(lambda x: x[0], model.state))
            #     batch = jax.device_get(tree_map(lambda x: x[0], batch))
            #     log_dict = evaluator(state, batch, coords_fem, u_ref, v_ref)
            #     wandb.log(log_dict, step)

            #     end_time = time.time()
            #     # Report training metrics
            #     logger.log_iter(step, start_time, end_time, log_dict)

        # Save checkpoint
        if config.saving.save_every_steps is not None:
            if (step + 1) % config.saving.save_every_steps == 0 or (
                step + 1
            ) == config.training.max_steps:
                path = os.path.join(workdir, "ckpt", config.wandb.name)
                path = os.path.abspath(path)
                save_checkpoint(model.state, path, keep=config.saving.num_keep_ckpts)

    return model
