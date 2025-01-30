import functools
from functools import partial
import time
import os

from absl import logging

import jax

import jax.numpy as jnp
from jax import random, vmap, pmap, local_device_count, jit
from jax.tree_util import tree_map

import matplotlib.pyplot as plt

import numpy as np
import scipy.io
import ml_collections

import wandb

import models

from jaxpi.samplers import BaseSampler, SpaceSampler, TimeSpaceSampler
from jaxpi.logging import Logger
from jaxpi.utils import save_checkpoint

from jaxpi.utils import restore_checkpoint
from flax.jax_utils import replicate


from utils import get_dataset#, get_fine_mesh, parabolic_inflow

    

class resSampler(BaseSampler):
    def __init__(self, delta_matrices, mu, initial, batch_size, rng_key=random.PRNGKey(1234)):
        super().__init__(batch_size, rng_key)
        
        self.delta_matrices = delta_matrices
        self.mu = mu
        self.initial = initial
    #     self.step = 0
        
    # def update_step(self, step):
    #     self.step = step
        
    @partial(pmap, static_broadcasted_argnums=(0,))
    def data_generation(self, key):
        "Generates data containing batch_size samples"

        
        (  u0,   v0,   p0,   s0,   coords_initial,
              u_fem_s,   v_fem_s,   p_fem_s,   s_fem_s,   t_fem,  coords_fem,
              u_fem_q,   v_fem_q,   p_fem_q,   s_fem_q) = self.initial
        
        (idx_bcs, eigvecs, map_elements_vertexes, B_matrices, A_matrices, M_matrices, N_matrices) = self.delta_matrices
        
        
        #1st step: sample of elements to be considered by the loss terms
        
        key1, key = random.split(key, 2)
        idx_elem = random.choice(key1, map_elements_vertexes.shape[0], shape=(self.batch_size,) )
        
        idx_fem = map_elements_vertexes[idx_elem]
        idx_fem = jnp.reshape(idx_fem, (-1,))
        eigvecs_elem = jnp.reshape(eigvecs[idx_fem][jnp.newaxis, :, :], (self.batch_size, 3, 52))
        eigvecs_elem = eigvecs_elem[:,:,:2]
        eigvecs = eigvecs[:,:2]
        
        (idx_inlet, idx_outlet, idx_noslip) = idx_bcs
        key1, key = random.split(key, 2)
        idx_idxnos = random.choice(key1, idx_noslip.shape[0], shape=(256,) )

        key1, key = random.split(key, 2)
        idx_idxnos = random.choice(key1, idx_noslip.shape[0], shape=(256,) )
        mu_inlet = random.uniform(key1, shape=(idx_inlet.shape[0],), minval = self.mu[0], maxval = self.mu[1]) 
        t_inlet = random.choice(key1, t_fem.shape[0], shape=(idx_inlet.shape[0],) )
        mu_noslip = random.uniform(key1, shape=(idx_idxnos.shape[0],), minval = self.mu[0], maxval = self.mu[1]) 
        t_noslip = random.choice(key1, t_fem.shape[0], shape=(idx_idxnos.shape[0],) )
        X_bc = (eigvecs[idx_inlet], eigvecs[idx_outlet], eigvecs[idx_noslip[idx_idxnos]], mu_inlet, t_inlet, mu_noslip, t_noslip)        

        X = eigvecs_elem 
        
        matrices = (eigvecs_elem,  
                    N_matrices[idx_elem],
                    B_matrices[idx_elem], 
                    A_matrices[idx_elem],
                    M_matrices[idx_elem])
        

        #2nd step: sample of time points

        key1, key = random.split(key, 2)
        idx_t = random.choice(key1, t_fem.shape[0], shape=(self.batch_size,) )
        
        idx_fem = map_elements_vertexes[idx_elem]
        # idx_fem = jnp.reshape(idx_fem, (-1,))
               
        t = t_fem[idx_t]
        
        # idx_t = jnp.repeat(idx_t, 3)
        u0_b, v0_b, p0_b, s0_b = [], [], [], []
        u_fem_s_b, v_fem_s_b, p_fem_s_b, s_fem_s_b = [], [], [], []
        u_fem_q_b, v_fem_q_b, p_fem_q_b, s_fem_q_b = [], [], [], []
        X_fem = []
        
        for i in range(3):
            X_fem.append(eigvecs[idx_fem[:,i]])
            
            u0_b.append(u0[idx_fem[:,i]])
            v0_b.append(v0[idx_fem[:,i]])
            p0_b.append(p0[idx_fem[:,i]])
            s0_b.append(s0[idx_fem[:,i]])
            
            u_fem_s_b.append(u_fem_s[idx_t, idx_fem[:,i]])
            v_fem_s_b.append(v_fem_s[idx_t, idx_fem[:,i]])
            p_fem_s_b.append(p_fem_s[idx_t, idx_fem[:,i]])
            s_fem_s_b.append(s_fem_s[idx_t, idx_fem[:,i]])
            
            u_fem_q_b.append(u_fem_q[idx_t, idx_fem[:,i]])
            v_fem_q_b.append(v_fem_q[idx_t, idx_fem[:,i]])
            p_fem_q_b.append(p_fem_q[idx_t, idx_fem[:,i]])
            s_fem_q_b.append(s_fem_q[idx_t, idx_fem[:,i]])
                     
        X_fem = jnp.concatenate(X_fem, axis=0)
        t_fem = jnp.concatenate((t,t,t), axis=0)
        u0_b, v0_b, p0_b, s0_b = jnp.concatenate(u0_b, axis=0), jnp.concatenate(v0_b, axis=0), jnp.concatenate(p0_b, axis=0), jnp.concatenate(s0_b, axis=0)
        u_fem_s_b, v_fem_s_b, p_fem_s_b, s_fem_s_b = jnp.concatenate(u_fem_s_b, axis=0), jnp.concatenate(v_fem_s_b, axis=0), jnp.concatenate(p_fem_s_b, axis=0), jnp.concatenate(s_fem_s_b, axis=0)
        u_fem_q_b, v_fem_q_b, p_fem_q_b, s_fem_q_b = jnp.concatenate(u_fem_q_b, axis=0), jnp.concatenate(v_fem_q_b, axis=0), jnp.concatenate(p_fem_q_b, axis=0), jnp.concatenate(s_fem_q_b, axis=0)
        key1, key = random.split(key, 2)
        mu_batch = random.uniform(key1, shape=(self.batch_size,), minval = self.mu[0], maxval = self.mu[1])  
        mu_fem = jnp.concatenate((mu_batch,mu_batch,mu_batch), axis=0)
        
                
        fields = (X_fem, t_fem, mu_fem, u_fem_q_b, v_fem_q_b, p_fem_q_b, s_fem_q_b, u_fem_s_b, v_fem_s_b, p_fem_s_b, s_fem_s_b)
        fields_ic = (u0_b, v0_b, p0_b, s0_b)  
        
        batch = (t, X, X_bc, mu_batch, matrices, fields, fields_ic)

        return batch

        

def train_one_window(config, workdir, model, samplers, idx):
    
    # upd_stp = 100
    # Initialize evaluator
    evaluator = models.NavierStokesEvaluator(config, model)

    # Initialize logger
    logger = Logger()

    step_offset = idx * config.training.max_steps

    # jit warm up
    # batch = {}
    # samplers["res"].update_RAD(model, 0, config.training.max_steps, k=1, c=1)
    # samplers["res"]._iterator = iter(samplers["res"])

    # for key, sampler in samplers.items():
    #     if key == "res": #and step % 50 ==0:
    #         batch[key] = next(sampler._iterator)
    #     else:
    #         batch[key] = next(sampler)
    print("Waiting for JIT...")
    step = 0
    while step < config.training.max_steps:
        start_time = time.time()

        # if step % 1000 ==0:
        #     samplers["res"].update_RAD(model, step, config.training.max_steps, k=1, c=1)
        #     samplers["res"]._iterator = iter(samplers["res"])
        # samplers["res"].update_step(step)
        
        
        # batch = update_batch(step, samplers, batch)
        # Sample mini-batch        
        batch = {}

        for key, sampler in samplers.items():
            batch[key] = next(sampler)
            # if key == "res": #and step % 50 ==0:
            #     batch[key] = next(sampler._iterator)
            # else:
            #     batch[key] = next(sampler)
            
            
        # model.update_delta_matrices(batch["res"][3])
        
        for _ in range(100):   
            model.state = model.step(model.state, batch)
            step +=1

        # Update weights if necessary
        if config.weighting.scheme in ["grad_norm", "ntk"]:
            if step % config.weighting.update_every_steps == 0:
                model.state = model.update_weights(model.state, batch)

        # Log training metrics, only use host 0 to record results
        if jax.process_index() == 0:
            if step % config.logging.log_every_steps == 0:
                print(f"step: {step}")
                # # Get the first replica of the state and batch
                # state = jax.device_get(tree_map(lambda x: x[0], model.state))
                # batch = jax.device_get(tree_map(lambda x: x[0], batch))
                # log_dict = evaluator(state, batch)
                # wandb.log(log_dict, step + step_offset)

                # end_time = time.time()
                # # Report training metrics
                # logger.log_iter(step, start_time, end_time, log_dict)

        # Save checkpoint
        if config.saving.save_every_steps is not None:
            if (step ) % config.saving.save_every_steps == 0 or (
                step 
            ) == config.training.max_steps:
                path = os.path.join(
                    workdir, "ckpt", config.wandb.name, "time_window_{}".format(idx + 1)
                )
                path = os.path.abspath(path)
                save_checkpoint(model.state, path, keep=config.saving.num_keep_ckpts)

    return model

def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str):
    # Initialize W&B
    wandb_config = config.wandb
    wandb.init(project=wandb_config.project, name=wandb_config.name)
    
    (   initial,
        delta_matrices,
        mu0, mu1, rho0, rho1) = get_dataset()

    fluid_params = (mu0, mu1, rho0, rho1)    
        
    (u0, v0, p0, s0, coords_initial,
            u_fem_s, v_fem_s, p_fem_s, s_fem_s, dt_fem, coords_fem,
            u_fem_q, v_fem_q, p_fem_q, s_fem_q) = initial
    
    pin = 50
    dp = pin
    L_max = 50/1000/100
    U_max = dp*L_max/mu1
    u_maxs = jnp.max(u_fem_s)/U_max
    v_maxs = jnp.max(v_fem_s)/U_max
    u_maxq = jnp.max(u_fem_q)/U_max
    v_maxq = jnp.max(v_fem_q)/U_max

    print(f"U_max: {U_max}")
    print(f"u_fem_s/U_max: {jnp.max(u_fem_s)/U_max}")
    print(f"v_fem_s/U_max: {jnp.max(v_fem_s)/U_max}")
    
    pmax = dp
    Re = rho0*dp*(L_max**2)/(mu1**2)
    print(f'Re={Re}')
    print(f'max_Steps: {config.training.max_steps}')

    t1 = 400


    coords_fem = coords_fem / L_max
    
    dt_fem = dt_fem / (mu1/dp)
    t_fem = jnp.cumsum(dt_fem)
    idx = jnp.where(t_fem<=t1)[0]
    
    (u0, v0, p0) = (u0/U_max , v0/U_max , p0/pmax)
    u_fem_s = u_fem_s[idx] / U_max 
    v_fem_s = v_fem_s[idx] / U_max 
    p_fem_s = p_fem_s[idx] / pmax
    s_fem_s = s_fem_s[idx]
    u_fem_q = u_fem_q[idx] / U_max 
    v_fem_q = v_fem_q[idx] / U_max 
    p_fem_q = p_fem_q[idx] / pmax
    s_fem_q = s_fem_q[idx]
    
    initial = (u0, v0, p0, s0, coords_initial,
            u_fem_s, v_fem_s, p_fem_s, s_fem_s, t_fem, coords_fem,
            u_fem_q, v_fem_q, p_fem_q, s_fem_q)
    
    idx_bcs, eigvecs, vertices, map_elements_vertexes, centroid, B_matrices, A_matrices, M_matrices, N_matrices  = delta_matrices
    # vertices = vertices / L_max
    # centroid = centroid / L_max
    delta_matrices = (idx_bcs, eigvecs, map_elements_vertexes, B_matrices, A_matrices, M_matrices, N_matrices )

    # Temporal domain of each time window
    t0 = 0.0

    temporal_dom = jnp.array([t0, t1 * (1 + 0.05)])

    for idx in range(config.training.num_time_windows):
        logging.info("Training time window {}".format(idx + 1))
        
        # Initialize model 
        model = models.NavierStokes2DwSat(config, pin/pmax, temporal_dom, U_max, L_max, fluid_params) #  no 1
        
        # Initialize Sampler
        keys = random.PRNGKey(0)
        
        res_sampler = resSampler(
                delta_matrices,
                mu0,
                initial,
                config.training.res_batch_size,
                # rng_key=keys,
            )
                    
        samplers = {
            "res": res_sampler,
        }

        
        # if config.training.fine_tune:
        #     ckpt_path = os.path.join(".", "ckpt", config.wandb.name, "time_window_1")
        #     ckpt_path = os.path.abspath(ckpt_path)
        #     state = restore_checkpoint(model.state, ckpt_path)
        #     model.state =  replicate(state)
        
        # Train model for the current time window
        model = train_one_window(config, workdir, model, samplers, idx)

        # # Update the initial condition for the next time window
        # if config.training.num_time_windows > 1:
        #     state = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], model.state))
        #     params = state.params
        #     # u0 = vmap(model.u_net, (None, None, 0, 0))(
        #     #     params, t1, coords[:, 0], coords[:, 1]
        #     # )
        #     # v0 = vmap(model.v_net, (None, None, 0, 0))(
        #     #     params, t1, coords[:, 0], coords[:, 1]
        #     # )
        #     # p0 = vmap(model.p_net, (None, None, 0, 0))(
        #     #     params, t1, coords[:, 0], coords[:, 1]
        #     # )
        #     s0 = vmap(model.s_net, (None, None, 0, 0))(
        #         params, t1, coords[:, 0], coords[:, 1]
        #     )

            # del model, state, params
