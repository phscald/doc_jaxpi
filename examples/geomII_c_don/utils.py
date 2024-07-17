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
    
    
    # cylinder
    # radius = np.sqrt(np.power((X[:,0] - (x1_max/2)) , 2) + np.power((X[:,1] - (x2_max/2)) , 2))
    # inds = np.where(radius <= .0015)

    # X = np.delete(X, inds, axis = 0)
    

    return jax.device_put(X), \
            jax.device_put(inflow_coords), \
            jax.device_put(outflow_coords), \
            jax.device_put(wall_coords), 
            
def get_wall(cyl_center, R =.0015):
    # cyl_center = np.array(cyl_center)
    x1_rad = jnp.array(jnp.arange(0, 2*jnp.pi, jnp.pi/30))
    cyl_walls_xy = jnp.zeros((0,2))#jnp.zeros((cyl_center.shape[0]*x1_rad.shape[0], 2))
    cyl_xy = jnp.zeros((0,2))#jnp.zeros((cyl_center.shape[0]*x1_rad.shape[0], 2))

    for i in range(cyl_center.shape[0]):
        x1_rad = jnp.array(jnp.arange(0, 2*jnp.pi, jnp.pi/30))
        x2_rad = jnp.transpose(jnp.array([cyl_center[i,1] + jnp.sin(x1_rad) * R]))
        x1_rad = jnp.transpose(jnp.array([cyl_center[i,0] + jnp.cos(x1_rad) * R]))
        x1_rad = jnp.concatenate((x1_rad, x2_rad), axis = 1)
        cyl_walls_xy = jnp.concatenate((cyl_walls_xy, x1_rad), axis=0)
        cyl_xy = jnp.concatenate((cyl_xy, jnp.ones((x1_rad.shape[0],2))*cyl_center[i]), axis=0)
        # cyl_walls_xy.at[i*x1_rad.shape[0]:(i+1)*x1_rad.shape[0],:].set(x1_rad)
        # cyl_walls_xy[i*x1_rad.shape[0]:(i+1)*x1_rad.shape[0]] = x1_rad
        # cyl_xy.at[i*x1_rad.shape[0]:(i+1)*x1_rad.shape[0],:].set(cyl_center[i])
        # cyl_xy[i*x1_rad.shape[0]:(i+1)*x1_rad.shape[0]] = cyl_center[i]
    
    return cyl_xy, cyl_walls_xy

def get_dataset():
    u_fem, v_fem, p_fem, coords_fem = get_fem_data()
    coords, inflow_coords, outflow_coords, wall_coords = get_coords()
    mu = [.1, .10001]
    pin = [10, 10.001]

    # cylinder_center = random.uniform(random.PRNGKey(0), shape=(100,2)) 
    cylinder_center = random.uniform(random.PRNGKey(1), shape=(500,2)) 
    # minval = jnp.array([0.005, 0.002]) 
    # maxval = jnp.array([0.01500, 0.019]) 
    minval = jnp.array([0.0105, 0.006]) 
    maxval = jnp.array([0.010501, 0.008]) 
    cylinder_center = minval + cylinder_center * (maxval - minval)
    
    cyl_xy, cyl_walls_xy = get_wall(cylinder_center)
    

    return (
        u_fem, v_fem, p_fem, coords_fem,
        coords,
        inflow_coords,
        outflow_coords,
        wall_coords,
        mu, pin,
        cylinder_center,
        # jnp.concatenate((cylinder_center, cyl_walls_xy), axis = 0),
        cyl_xy, cyl_walls_xy
    )
