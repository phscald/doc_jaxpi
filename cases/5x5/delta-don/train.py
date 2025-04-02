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
from flax import linen as nn

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import get_dataset
    
def normalize_mustd(u, umean, ustd):
    return (u-umean)/ustd     

class resSampler(BaseSampler):
    def __init__(self, delta_matrices, initial, u_stats, indx_extremes, batch_size, max_steps=None, rng_key=random.PRNGKey(1234)):
        super().__init__(batch_size, rng_key)       
        
        self.delta_matrices = delta_matrices
        self.initial = initial
        self.u_stats = u_stats
        self.indx_extremes = indx_extremes
        self.step = 0
        self.max_steps = max_steps
        
    def update_initial(self, initial):
        self.initial = initial 
        
    def update_step(self, step):
        self.step = step     
        
    @partial(pmap, static_broadcasted_argnums=(0,))
    def data_generation(self, key):
        "Generates data containing batch_size samples"
        
        (  u0, v0, p0, s0, coords_initial,
              u_fem, v_fem, p_fem, s_fem, t_fem, coords_fem, mu_list) = self.initial
        
        u_mean, u_std, v_mean, v_std = self.u_stats
        
        u_fem = normalize_mustd(u_fem, u_mean, u_std)
        v_fem = normalize_mustd(v_fem, v_mean, v_std)
        u0    = normalize_mustd(u0, u_mean, u_std)
        v0    = normalize_mustd(v0, v_mean, v_std)
        
        (idx_bcs, eigvecs, map_elements_vertexes, B_matrices, A_matrices, M_matrices, N_matrices) = self.delta_matrices
                
        #1st step: sample of elements to be considered by the loss terms
        
        key1, key = random.split(key, 2)
        idx_elem = random.choice(key1, map_elements_vertexes.shape[0], shape=(self.batch_size,) )
        idx_elem = jnp.concatenate((idx_elem, self.indx_extremes))
        
        idx_fem = map_elements_vertexes[idx_elem]
        idx_fem = jnp.reshape(idx_fem, (-1,))
        eigvecs_elem = jnp.reshape(eigvecs[idx_fem][jnp.newaxis, :, :], (-1, 3, 22))
        eigvecs_elem = eigvecs_elem[:,:,:]
        eigvecs = eigvecs[:,:]
        
        (idx_inlet, idx_outlet, idx_noslip) = idx_bcs     

        X = eigvecs_elem 
        
        matrices = (eigvecs_elem,  
                    N_matrices[idx_elem],
                    B_matrices[idx_elem], 
                    A_matrices[idx_elem],
                    M_matrices[idx_elem])

        #2nd step: sample of time points

        key1, key2, key = random.split(key, 3)
        idx_t = random.choice(key1, t_fem.shape[0], shape=(self.batch_size+self.indx_extremes.shape[0],) )
        idx_mu = random.choice(key2, mu_list.shape[0], shape=(self.batch_size+self.indx_extremes.shape[0],) )
        
        
        idx_fem = map_elements_vertexes[idx_elem]
        # idx_fem = jnp.reshape(idx_fem, (-1,))
               
        t = t_fem[idx_t]
        
        # idx_t = jnp.repeat(idx_t, 3)
        u0_b, v0_b, p0_b, s0_b = [], [], [], []
        u_fem_b, v_fem_b, p_fem_b, s_fem_b = [], [], [], []
        X_fem = []
        
        for i in range(3):
            X_fem.append(eigvecs[idx_fem[:,i]])
            
            u0_b.append(u0[idx_fem[:,i]])
            v0_b.append(v0[idx_fem[:,i]])
            p0_b.append(p0[idx_fem[:,i]])
            s0_b.append(s0[idx_fem[:,i]])
            
            u_fem_b.append(u_fem[idx_mu, idx_t, idx_fem[:,i]])
            v_fem_b.append(v_fem[idx_mu, idx_t, idx_fem[:,i]])
            p_fem_b.append(p_fem[idx_mu, idx_t, idx_fem[:,i]])
            s_fem_b.append(s_fem[idx_mu, idx_t, idx_fem[:,i]])
            
                     
        X_fem = jnp.concatenate(X_fem, axis=0)
        t_fem = jnp.concatenate((t,t,t), axis=0)
        u0_b, v0_b, p0_b, s0_b = jnp.concatenate(u0_b, axis=0), jnp.concatenate(v0_b, axis=0), jnp.concatenate(p0_b, axis=0), jnp.concatenate(s0_b, axis=0)
        u_fem_b, v_fem_b, p_fem_b, s_fem_b = jnp.concatenate(u_fem_b, axis=0), jnp.concatenate(v_fem_b, axis=0), jnp.concatenate(p_fem_b, axis=0), jnp.concatenate(s_fem_b, axis=0)

        mu_batch = mu_list[idx_mu]
        mu_fem = jnp.concatenate((mu_batch,mu_batch,mu_batch), axis=0)
        mu_batch2 =  jnp.array([.0033333333333333335, .01, .0625, .0875])
        mu_batch2 = mu_batch2[random.choice(key1, mu_batch2.shape[0], shape=(1024,) )]
        mu_batch = jnp.concatenate((mu_batch[:mu_batch.shape[0]-1024],mu_batch2), axis=0)
                
        fields = (X_fem, t_fem, mu_fem, u_fem_b, v_fem_b, p_fem_b, s_fem_b)
        fields_ic = (u0_b, v0_b, p0_b, s0_b)  

        batch = (t, X, mu_batch, matrices, fields, fields_ic)

        return batch

def train_one_window(config, workdir, model, samplers, idx, initial):
    
    # upd_stp = 100
    # Initialize evaluator
    evaluator = models.NavierStokesEvaluator(config, model)

    # Initialize logger
    logger = Logger()

    step_offset = idx * config.training.max_steps

    # jit warm up
    print("Waiting for JIT...")
    step = 0
    start_time = time.time() 
    batch = {}
    while step < config.training.max_steps:
        
        if step%2500==0:
            ind = np.random.choice(initial[9].shape[0], int(initial[9].shape[0]/6))
            initial_ = list(initial)
            for i in range(5, 9):
                initial_[i] = jax.device_put(initial[i][:, ind])
            initial_[9] = jax.device_put(initial[9][ind])
            samplers["res"].update_initial(initial_)
                
        for key, sampler in samplers.items():
            batch[key] = next(sampler)
            
        # Update weights if necessary
        if config.weighting.scheme in ["grad_norm", "ntk"]:
            if step % config.weighting.update_every_steps == 0:
                model.state = model.update_weights(model.state, batch)
        
        for _ in range(100):   
            model.state = model.step(model.state, batch)
            # model.update_epoch()
            step +=1
            
        samplers["res"].update_step(step)

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
    
    L_max = 50/1000/100
    (   initial,
        delta_matrices,
        mu1, rho0, rho1) = get_dataset(L_max)

    fluid_params = (0, mu1, rho0, rho1)    
        
    pin = 100
    dp = pin
    
    U_max = dp*L_max/mu1
    print(f"U_max: {U_max}")

    pmax = dp
    Re = rho0*dp*(L_max**2)/(mu1**2)
    print(f'Re={Re}')
    print(f'max_Steps: {config.training.max_steps}')


    (u0, v0, p0, s0, coords_initial,
     u_fem, v_fem, p_fem, s_fem, t_fem, 
     coords_fem, mu_list) = initial
    
    print('u')
    print(f'mean: {u_fem.mean()}')
    print(f'std: {u_fem.std()}')
    print('v')
    print(f'mean: {v_fem.mean()}')
    print(f'std: {v_fem.std()}')
    
    # u_mean = u_fem.mean()
    u_mean = 0
    u_std = u0.std()*3
    # v_mean = v_fem.mean()
    v_mean = 0
    v_std = v0.std()*3
    u_stats = (u_mean, u_std, v_mean, v_std)
        
      
    print(f't_fem max fem {t_fem.max()}')
    t1 =  24000
    
    idx = jnp.where(t_fem<=t1)[0]
    
    u_fem = u_fem[:, idx]
    v_fem = v_fem[:, idx]
    p_fem = p_fem[:, idx]
    s_fem = s_fem[:, idx]
    
    t_fem = t_fem[idx]
    
    idx_bcs, eigvecs, vertices, map_elements_vertexes, _, B_matrices, A_matrices, M_matrices, N_matrices  = delta_matrices
 
    indx_extremes = []
    for i in range(vertices.shape[0]):
        ind = jnp.where(vertices[i,:,0]==coords_fem[:,0].min())[0]
        if ind.shape[0] >= 1:
            indx_extremes.append(i)
        ind = jnp.where(vertices[i,:,0]==coords_fem[:,0].max())[0]
        if ind.shape[0] >= 1:
            indx_extremes.append(i)
    indx_extremes = jnp.array(indx_extremes)
    
    delta_matrices = (idx_bcs, eigvecs, map_elements_vertexes, B_matrices, A_matrices, M_matrices, N_matrices )
    
    # Temporal domain of each time window
    t0 = 0.0

    temporal_dom = jnp.array([t0, t1 * (1 + 0.05)])

    for idx in range(config.training.num_time_windows):
        logging.info("Training time window {}".format(idx + 1))
        
        # Initialize model 
        model = models.NavierStokes2DwSat(config, pin/pmax, temporal_dom, U_max, L_max, (fluid_params, u_stats)) #  no 1
        
        # Initialize Sampler
        keys = random.PRNGKey(0)
        
        res_sampler = resSampler(
            delta_matrices,
            initial,
            u_stats,
            indx_extremes,
            config.training.res_batch_size,
            config.training.max_steps,
            rng_key=keys,
        ) 
                    
        samplers = {
            "res": res_sampler,
        }
        # batch = {}
        # for key, sampler in samplers.items():
        #     batch[key] = next(sampler)    
        
        if config.training.fine_tune:
            ckpt_path = os.path.join(".", "ckpt", config.wandb.name, "time_window_1")
            ckpt_path = os.path.abspath(ckpt_path)
            state = restore_checkpoint(model.state, ckpt_path)
            model.state =  replicate(state)
        
        # Train model for the current time window
        model = train_one_window(config, workdir, model, samplers, idx, initial)

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
