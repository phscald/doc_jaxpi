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

from utils import get_dataset#, parabolic_inflow

import matplotlib.pyplot as plt
import matplotlib.tri as tri
import pickle

def evaluate(config: ml_collections.ConfigDict, workdir: str):
    # Load dataset
    

    
       
    
    from matplotlib.animation import FuncAnimation
    from functools import partial  # Import partial to pass extra arguments to the update function

    # Create figures and axes once
    figs, axs = plt.subplots(2, 1, figsize=(6, 10))
    figu, axu = plt.subplots(2, 1, figsize=(6, 10))
    figv, axv = plt.subplots(2, 1, figsize=(6, 10))
    figp, axp = plt.subplots(2, 1, figsize=(6, 10))

    m = len(u_pred)  # Assuming u_pred and others are defined
    
    s_pred = jnp.clip(s_pred, 0, 1)
    matrix1 = [s_pred, u_pred, v_pred, p_pred]
    matrix2 = [s_fem, u_fem, v_fem, p_fem]
    label   = ["s", "u", "v", "p"]
    for i in range(4):
        print(f'==={label[i]}===')
        mse = jnp.mean((matrix1[i] - matrix2[i]) ** 2)   
        mse_rel = jnp.mean((matrix1[i] - matrix2[i]) ** 2)  /  jnp.mean((matrix2[i]) ** 2)
        l1_relative_error = jnp.sum(jnp.abs(matrix1[i] - matrix2[i])) / jnp.sum(jnp.abs(matrix1[i]))
        print(f"MSE: {mse}")
        print(f"MSErel: {mse_rel*100:.2f}%")
        print(f"L1-relative error: {l1_relative_error*100:.2f}")

    # Update function for each frame
    def update_s(frames, indx):
        axs[0].cla()  # Clear the current axis
        axs[0].scatter(x, y, s=1, c=s_pred_list[indx][frames], cmap='jet', vmin=0, vmax=1)
        
        axs[1].cla()  # Clear the current axis
        axs[1].scatter(x, y, s=1, c=s_fem[frames], cmap='jet', vmin=0, vmax=1)
        
        plt.tight_layout() 

    def update_u(frames, indx):
        axu[0].cla()  # Clear the current axis
        axu[0].scatter(x, y, s=1, c=u_pred_list[indx][frames], cmap='jet', vmin=jnp.min(u0), vmax=jnp.max(u0))
        
        axu[1].cla()  # Clear the current axis
        axu[1].scatter(x, y, s=1, c=u_fem[frames], cmap='jet', vmin=jnp.min(u0), vmax=jnp.max(u0))
                
        plt.tight_layout() 

    def update_v(frames, indx):
        axv[0].cla()  # Clear the current axis
        axv[0].scatter(x, y, s=1, c=v_pred_list[indx][frames], cmap='jet', vmin=jnp.min(v0), vmax=jnp.max(v0))
        
        axv[1].cla()  # Clear the current axis
        axv[1].scatter(x, y, s=1, c=v_fem[frames], cmap='jet', vmin=jnp.min(v0), vmax=jnp.max(v0))
           
        plt.tight_layout() 

    def update_p(frames, indx):
        axp[0].cla()  # Clear the current axis
        axp[0].scatter(x, y, s=1, c=p_pred_list[indx][frames], cmap='jet', vmin=0, vmax=1)
                
        axp[1].cla()  # Clear the current axis
        axp[1].scatter(x, y, s=1, c=p_fem[frames], cmap='jet', vmin=0, vmax=1)
                
        plt.tight_layout() 

    # Function to generate GIFs
    def make_gif(indx):
        ani_s = FuncAnimation(figs, partial(update_s, indx=indx-1), frames=m, interval=200)
        ani_s.save(f'./video_s_p5_{indx}.gif', writer='pillow')
        
        ani_u = FuncAnimation(figu, partial(update_u, indx=indx-1), frames=m, interval=200)
        ani_u.save(f'./video_u_p5_{indx}.gif', writer='pillow')
        
        ani_v = FuncAnimation(figv, partial(update_v, indx=indx-1), frames=m, interval=200)
        ani_v.save(f'./video_v_p5_{indx}.gif', writer='pillow')
        
        ani_p = FuncAnimation(figp, partial(update_p, indx=indx-1), frames=m, interval=200)
        ani_p.save(f'./video_p_p5_{indx}.gif', writer='pillow')

    # # Generate GIFs for each time window
    # for idx in range(1, config.training.num_time_windows + 1):
    #     make_gif(idx)
