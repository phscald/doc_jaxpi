import jax.numpy as jnp
import jax
import numpy as np
import pickle
import scipy.io


def get_fem_data():
    filepath = './steady_3x3.pkl'
    with open(filepath, 'rb') as filepath:
        arquivos = pickle.load(filepath)

    coord = arquivos['coord']

    u_fem = arquivos['u']
    v_fem = arquivos['v']
    p_fem = arquivos['p']

    return jax.device_put(u_fem), jax.device_put(v_fem), jax.device_put(p_fem), jax.device_put(coord)

def get_coords():
    filepath = '3x3.pkl'
    with open(filepath, 'rb') as filepath:
        arquivos = pickle.load(filepath)

    contour_points = arquivos['contour_points']
    X = arquivos['outside_points']
       
    inflow_coords = np.arange(0, np.max(contour_points[:,1])+.1)[:,np.newaxis]
    inflow_coords = np.concatenate((np.zeros(inflow_coords.shape), inflow_coords), axis=1)
    
    outflow_coords = np.arange(0, np.max(contour_points[:,1])+.1)[:,np.newaxis]
    outflow_coords = np.concatenate((np.ones(outflow_coords.shape)*np.max(contour_points[:,0]), outflow_coords), axis=1)
    
    bot_coords = np.arange(0, np.max(contour_points[:,0])+.1)[:,np.newaxis]
    bot_coords = np.concatenate((bot_coords, np.zeros(bot_coords.shape)), axis=1)
    
    top_coords = np.arange(0, np.max(contour_points[:,0])+.1)[:,np.newaxis]
    top_coords = np.concatenate((top_coords, np.ones(top_coords.shape)*np.max(contour_points[:,1])), axis=1)
    
    wall_coords = np.concatenate((bot_coords, top_coords), axis=0)
    
    h = np.max(X[:,1])
    
    X[:,1] = -1*X[:,1] + h
    inflow_coords[:,1] = -1*inflow_coords[:,1] + h
    outflow_coords[:,1] = -1*outflow_coords[:,1] + h
    wall_coords[:,1] = -1*wall_coords[:,1] + h
    contour_points[:,1] = -1*contour_points[:,1] + h

    return jax.device_put(X), \
            jax.device_put(inflow_coords), \
            jax.device_put(outflow_coords), \
            jax.device_put(wall_coords), \
            jax.device_put(contour_points)


def get_dataset():
    u_fem, v_fem, p_fem, coords_fem = get_fem_data()
    coords, inflow_coords, outflow_coords, wall_coords, cylinder_coords = get_coords()
    scale = 1/1000/100
    coords, inflow_coords, outflow_coords, wall_coords, cylinder_coords = (
        coords*scale,
        inflow_coords*scale, 
        outflow_coords*scale, 
        wall_coords*scale, 
        cylinder_coords*scale
    )
    mu = .05
    rho = 1000

    return (
        u_fem, v_fem, p_fem, coords_fem,
        coords,
        inflow_coords,
        outflow_coords,
        wall_coords,
        cylinder_coords,
        mu, rho
    ) 
