import jax.numpy as jnp
import jax
from jax import random
import numpy as np
import pickle
import scipy.io


def parabolic_inflow(y, U_max):
    u = 4 * U_max * y * (0.41 - y) / (0.41**2)
    v = jnp.zeros_like(y)
    return u, v

def get_fem_data():
    filepath = '../../../pinn/compare_files/uvp_coord_hole.pkl'
    with open(filepath, 'rb') as filepath:
        arquivos = pickle.load(filepath)

    uvp = arquivos['data_uvp']
    coord = arquivos['coord']

    tam = int((uvp.shape[0])/3)
    u_fem = uvp[0:tam]
    v_fem = uvp[tam:2*tam]
    p_fem = uvp[2*tam:3*tam]

    return jax.device_put(u_fem), jax.device_put(v_fem), jax.device_put(p_fem), jax.device_put(coord)

def get_coords():
    dx = .0001
    x1_max = .021
    x2_max = .014

    x1 = np.arange(0,.021+dx/2, dx)
    x2 = np.arange(0,.014+dx/2, dx)

    X = np.zeros((x1.shape[0]*x2.shape[0],2))

    for i in range(x1.shape[0]):
        X[i*x2.shape[0]:(i+1)*x2.shape[0],:] = np.transpose(np.concatenate(([x1[i]*np.ones(x2.shape[0])],[x2]),axis=0))

    # esquerda p entrada
    id = np.where(np.squeeze(np.matmul(X, np.array([[1],[0]]))) == 0)[0]
    inflow_coords = X[id]
    # direita p saida
    id = np.where(np.squeeze(np.matmul(X, np.array([[1],[0]]))) == x1_max)[0]
    outflow_coords = X[id]
    # bottom
    id = np.where(np.squeeze(np.matmul(X, np.array([[0],[1]]))) == 0)[0]
    bot_wall_coords = X[id]
    # top
    id = np.where(np.squeeze(np.matmul(X, np.array([[0],[1]]))) == x2_max)[0]
    top_wall_coords = X[id]
    # wall = top+bottom
    wall_coords = np.concatenate((bot_wall_coords, top_wall_coords), axis=0)

    inds = np.where(X[:,1] > x2_max)
    X = np.delete(X, inds, axis = 0)

    return jax.device_put(X), \
            jax.device_put(inflow_coords), \
            jax.device_put(outflow_coords), \
            jax.device_put(wall_coords)


def get_dataset():
    u_fem, v_fem, p_fem, coords_fem = get_fem_data()
    coords, inflow_coords, outflow_coords, wall_coords = get_coords()
    mu = [.1, .10001]
    pin = [10, 10.001]

    key = random.PRNGKey(0)
    cylinder_center = random.uniform(key, shape=(100,2)) 
    minval = jnp.array([0.0045, 0.0165]) 
    maxval = jnp.array([0.0095, 0.0195]) 
    cylinder_center = minval + cylinder_center * (maxval - minval)

    return (
        u_fem, v_fem, p_fem, coords_fem,
        coords,
        inflow_coords,
        outflow_coords,
        wall_coords,
        mu, pin,
        cylinder_center
    )
