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
import seaborn as sns
import pickle

def compute_saturation_curve(s_pred):
    sat_curve = jnp.mean(s_pred, axis=1)
    return sat_curve
    
def compute_outlet_flow_ratios(u_pred, s_pred, vertices, map_elements_vertexes, xmax, xmin):
    out_flow_ratio_w = [] # 1 é água
    out_flow_ratio_o = [] # 0 é óleo
    for i in range(vertices.shape[0]):
        ind = jnp.where(vertices[i,:,0]==xmax)[0]
        if ind.shape[0] == 2:
            ind1 = map_elements_vertexes[i,ind[0]]
            ind2 = map_elements_vertexes[i,ind[1]]
            dy = jnp.abs(vertices[i,ind[0],1] - vertices[i,ind[1],1])
            out_flow_ratio_w.append( (s_pred[:,ind1] * u_pred[:,ind1] + s_pred[:,ind2] * u_pred[:,ind2]) /2 * dy )
            out_flow_ratio_o.append( ( (1-s_pred[:,ind1]) * u_pred[:,ind1] + (1-s_pred[:,ind2]) * u_pred[:,ind2] ) /2 * dy )
    out_flow_ratio_w = jnp.stack(out_flow_ratio_w)
    out_flow_ratio_o = jnp.stack(out_flow_ratio_o)
    out_flow_ratio_w = jnp.sum(out_flow_ratio_w, axis=0)
    out_flow_ratio_o = jnp.sum(out_flow_ratio_o, axis=0)
    in_flow_ratio = compute_inlet_flow_ratio(u_pred, vertices, map_elements_vertexes, xmin)
    krw = out_flow_ratio_w / in_flow_ratio
    kro = out_flow_ratio_o / in_flow_ratio
    return out_flow_ratio_w, out_flow_ratio_o, krw, kro


def compute_inlet_flow_ratio(u_pred, vertices, map_elements_vertexes, xmin):
    in_flow_ratio = []
    for i in range(vertices.shape[0]):
        ind = jnp.where(vertices[i,:,0]==xmin)[0]
        if ind.shape[0] == 2:
            ind1 = map_elements_vertexes[i,ind[0]]
            ind2 = map_elements_vertexes[i,ind[1]]
            dy = jnp.abs(vertices[i,ind[0],1] - vertices[i,ind[1],1])
            in_flow_ratio.append( (u_pred[:,ind1] + u_pred[:,ind2]) /2 * dy )
    in_flow_ratio = jnp.stack(in_flow_ratio)
    in_flow_ratio = jnp.sum(in_flow_ratio, axis=0)
    return in_flow_ratio

def get_curves(u_pred, s_pred, vertices, map_elements_vertexes, xmax, xmin):
    out_flow_ratio_w, out_flow_ratio_o, krw, kro = compute_outlet_flow_ratios(u_pred, s_pred, vertices, map_elements_vertexes, xmax, xmin)
    in_flow_ratio = compute_inlet_flow_ratio(u_pred, vertices, map_elements_vertexes, xmin)
    sat_curve = compute_saturation_curve(s_pred)
    return out_flow_ratio_w, out_flow_ratio_o, krw, kro, in_flow_ratio, sat_curve

def predict(model, t_coords, X, mu, label, config):
    # Predict
        # Restore checkpoint
    # ckpt_path = os.path.join(".", "ckpt", config.wandb.name)
    # ckpt_path = os.path.abspath(ckpt_path)
    # model.state = restore_checkpoint(model.state, ckpt_path)
    # params = model.state.params
    
    u_pred_fn = jit(vmap(vmap(model.u_net, (None, None, 0, None)), (None, 0, None, None))) # shape t by xy
    s_pred_fn = jit(vmap(vmap(model.s_net, (None, None, 0, None)), (None, 0, None, None)))
    for indx in range(config.training.num_time_windows):
        print(f'{indx+1} / {config.training.num_time_windows}' )
        # Restore the checkpoint
        ckpt_path = os.path.join('.', 'ckpt', label, 'time_window_{}'.format(indx + 1))
        model.state = restore_checkpoint(model.state, ckpt_path)
        params = model.state.params

        print(f'mu = {mu}')

        u_pred = u_pred_fn(params, t_coords, X, mu)
        s_pred = s_pred_fn(params, t_coords, X, mu)  
        
    return u_pred, s_pred
            
def evaluate(config: ml_collections.ConfigDict, workdir: str):
    # Load dataset
    
    config.mode = "eval" 
    
    L_max = 50/1000/100
    (   initial,
        delta_matrices,
        mu1, rho0, rho1) = get_dataset(L_max, config)

    fluid_params = (0, mu1, rho0, rho1)    
        
    (u0, v0, p0, s0, coords_initial,
     u_fem, v_fem, p_fem, s_fem, t_fem, 
     coords_fem, mu_list) = initial

    _, eigvecs, vertices, map_elements_vertexes, _, _, _, _, _  = delta_matrices

    pin = 100
    dp = pin
    pmax = dp
    U_max = dp*L_max/mu1

    mu = .1 #mu_list = [.0025, .014375, .02625, .038125, .05, .0625, .075, .0875, .1]
    ind_mu = jnp.where(mu_list==mu)[0]
    
    t1 = 1 # it is better to change the time in the t_coords array. There it is possible to select the desired percentages of total time solved

    T = 1.0  # final time
    tmax = 6000
    
    idx = jnp.where(t_fem<=tmax)[0]
    
    u_fem = jnp.squeeze( u_fem[ind_mu, idx] )
    v_fem = jnp.squeeze( v_fem[ind_mu, idx] )
    p_fem = jnp.squeeze( p_fem[ind_mu, idx] )
    s_fem = jnp.squeeze( s_fem[ind_mu, idx] )
    
    t_fem = t_fem[idx]

    t0 = 0.0

    temporal_dom = jnp.array([t0, t1 * (1 + 0.05)]) # Must be same as the one used in training

    # Initialize model
    # model = models.NavierStokes2D(config, inflow_fn, temporal_dom, coords, Re)
    model = models.NavierStokes2DwSat(config, pin/pmax, temporal_dom, U_max, L_max, fluid_params) #  no 1

    # Restore checkpoint
    ckpt_path = os.path.join(".", "ckpt", config.wandb.name)
    ckpt_path = os.path.abspath(ckpt_path)
    model.state = restore_checkpoint(model.state, ckpt_path)
    params = model.state.params
    
    X = eigvecs[:,:]
    
    # Predict
    u_pred_fn = jit(vmap(vmap(model.u_net, (None, None, 0, None)), (None, 0, None, None))) # shape t by xy
    v_pred_fn = jit(vmap(vmap(model.v_net, (None, None, 0, None)), (None, 0, None, None)))
    p_pred_fn = jit(vmap(vmap(model.p_net, (None, None, 0, None)), (None, 0, None, None)))
    s_pred_fn = jit(vmap(vmap(model.s_net, (None, None, 0, None)), (None, 0, None, None)))
    
    num_t = 10  # Desired number of points
    idx_t = jnp.linspace(0, t_fem.shape[0] - 1, num_t, dtype=int)
    t_coords = t_fem[idx_t]/tmax 
    u_fem = u_fem[idx_t]
    v_fem = v_fem[idx_t]
    p_fem = p_fem[idx_t]
    s_fem = s_fem[idx_t]

    u_pred_list = []
    v_pred_list = []
    p_pred_list = []
    s_pred_list = []

    for indx in range(config.training.num_time_windows):
        print(f'{indx+1} / {config.training.num_time_windows}' )
        # Restore the checkpoint
        ckpt_path = os.path.join('.', 'ckpt', config.wandb.name, 'time_window_{}'.format(indx + 1))
        model.state = restore_checkpoint(model.state, ckpt_path)
        params = model.state.params

        print(f'mu = {mu}')

        u_pred = u_pred_fn(params, t_coords, X, mu)
        v_pred = v_pred_fn(params, t_coords, X, mu)
        s_pred = s_pred_fn(params, t_coords, X, mu)
        p_pred = p_pred_fn(params, t_coords, X, mu)      

        u_pred_list.append(u_pred)
        v_pred_list.append(v_pred)
        s_pred_list.append(s_pred)
        p_pred_list.append(p_pred)     


    x = eigvecs[:, 0]
    y = eigvecs[:, 1]
    
    u_mean = 0
    u_std = u_fem.std()*3
    
    def denormalize_mustd(u, umean, ustd):
        return u*ustd+umean
    def normalize_mustd(u, umean, ustd):
        return (u-umean)/ustd
    u_pred = denormalize_mustd(u_pred, u_mean, u_std)
    
    filepath = './pred.pkl'
    with open(filepath,"wb") as filepath:
        pickle.dump({"u_pred": u_pred,
                     "s_pred": s_pred,
                     "vertices": vertices,
                     "map_elements_vertexes": map_elements_vertexes,
                     "xmax": x.max(),
                     "xmin": x.min(),
                     "t_coords": t_coords,
                     "tmax": tmax,
                     }, filepath)
        
    filepath = './pred_fem.pkl'
    with open(filepath,"wb") as filepath:
        pickle.dump({"u_pred": u_fem,
                     "s_pred": s_fem,
                     "vertices": vertices,
                     "map_elements_vertexes": map_elements_vertexes,
                     "xmax": x.max(),
                     "xmin": x.min(),
                     "t_coords": t_coords,
                     "tmax": tmax,
                     }, filepath)