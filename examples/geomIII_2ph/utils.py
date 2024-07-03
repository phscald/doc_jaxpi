import jax.numpy as jnp
import jax
import numpy as np
import pickle
import scipy.io


def get_fem_data():
    filepath = './uvp_coord3.pkl'
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

    filepath = './data_porousmedia.pkl'
    with open(filepath, 'rb') as filepath:
        arquivos = pickle.load(filepath)

    X = arquivos["X"]
    idx = arquivos["idx"]

    # esquerda p entrada
    id = np.where(idx == 2)[0]
    inflow_coords = X[id]
    # direita p saida
    id = np.where(idx == 3)[0]
    outflow_coords = X[id]
    # bottom
    id = np.where(idx == 4)[0]
    bot_wall_coords = X[id]
    # top
    id = np.where(idx == 5)[0]
    top_wall_coords = X[id]
    # wall = top+bottom
    wall_coords = np.concatenate((bot_wall_coords, top_wall_coords), axis=0)

    # grains
    id = np.where(idx == 1)[0]
    cylinder_coords = X[id]

    return jax.device_put(X), \
            jax.device_put(inflow_coords), \
            jax.device_put(outflow_coords), \
            jax.device_put(wall_coords), \
            jax.device_put(cylinder_coords)


def get_dataset():
    u0, v0, p0, coords_fem = get_fem_data()
    s0 = jnp.zeros(p0.shape)
    idx = jnp.where(coords_fem==0)[0]
    s0 = s0.at[idx].set(1.0)
    coords, inflow_coords, outflow_coords, wall_coords, cylinder_coords = get_coords()
    mu0 = .1; mu1 = .15; rho0 = 1000; rho1 = 1000

    return (
        coords,
        inflow_coords,
        outflow_coords,
        wall_coords,
        cylinder_coords,
        u0, v0, p0, s0, coords_fem,
        mu0, mu1, rho0, rho1
    )