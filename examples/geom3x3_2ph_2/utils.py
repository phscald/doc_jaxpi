import jax.numpy as jnp
import jax
import numpy as np
import pickle
import scipy.io

def get_coords2():
    dx = .0001
    x1_max = 900/1000/1000
    x2_max = 900/1000/1000

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
    print(f'wall coords: {wall_coords.shape}')

    inds = np.where(X[:,1] > x2_max)
    X = np.delete(X, inds, axis = 0)

    return jax.device_put(X), \
            jax.device_put(inflow_coords), \
            jax.device_put(outflow_coords), \
            jax.device_put(wall_coords)



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

    return jax.device_put(X/1000/1000), \
            jax.device_put(inflow_coords/1000/1000), \
            jax.device_put(outflow_coords/1000/1000), \
            jax.device_put(wall_coords/1000/1000), \
            jax.device_put(contour_points/1000/1000)


def initial_fields(coords):
    
    filepath = '../geom3x3/pred_initial_fields.pkl'
    with open(filepath, 'rb') as filepath:
        arquivos = pickle.load(filepath)
    coords_initial = arquivos['coords']
    u0 = arquivos['u_pred']
    v0 = arquivos['v_pred']
    p0 = arquivos['p_pred']
    del arquivos

    # u0 = jnp.zeros(coords.shape[0])
    # v0 = jnp.zeros(coords.shape[0])
    # p0 = jnp.zeros(coords.shape[0])
    s0 = jnp.zeros(coords.shape[0])
    # s0[jnp.where(coords[0]==0)] = 1
    s0.at[jnp.where(coords[:,0]<=.05*jnp.max(coords[:,0]))].set(1.0)
    return u0, v0, p0, s0, coords_initial


def get_dataset():

    coords, inflow_coords, outflow_coords, wall_coords = get_coords2() 
    # coords, inflow_coords, outflow_coords, wall_coords, cylinder_coords = get_coords() # esse
    # cylinder_coords1 = cylinder_coords.copy()
    # cylinder_coords1 = cylinder_coords1.at[:, 1].set(cylinder_coords[:, 1] * 0)

    # cylinder_coords2 = cylinder_coords.copy()
    # cylinder_coords2 = cylinder_coords2.at[:, 1].set(cylinder_coords[:, 1] * jnp.max(wall_coords[:, 1]))
    # cylinder_coords = jnp.concatenate((cylinder_coords1, cylinder_coords2), axis=0)
    # wall_coords = jnp.concatenate((wall_coords, cylinder_coords), axis=0) # esse
    mu0 = .1 # oil
    mu1 = .1 # water
    rho0 = 1000; rho1 = 1000
    # u0, v0, p0, s0, coords_initial = initial_fields(coords)
    coords_initial = coords
    u0 = jnp.zeros(coords_initial.shape[0])
    v0 = jnp.zeros(coords_initial.shape[0]); p0 = jnp.zeros(coords_initial.shape[0]); s0= jnp.zeros(coords_initial.shape[0])
    s0.at[jnp.where(coords[:,0]<=.05*jnp.max(coords[:,0]))].set(1.0)
    return (
        coords,
        inflow_coords,
        outflow_coords,
        wall_coords,
        u0, v0, p0, s0, coords_initial, # coords_initial is already nondimensional
        mu0, mu1, rho0, rho1
    )
