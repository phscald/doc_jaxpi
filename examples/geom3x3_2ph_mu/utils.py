import jax.numpy as jnp
import jax
import numpy as np
import pickle
import scipy.io

          
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
    
    wall_coords = np.concatenate((bot_coords, top_coords, contour_points), axis=0)
    
    h = np.max(X[:,1])
    
    X[:,1] = -1*X[:,1] + h
    inflow_coords[:,1] = -1*inflow_coords[:,1] + h
    outflow_coords[:,1] = -1*outflow_coords[:,1] + h
    wall_coords[:,1] = -1*wall_coords[:,1] + h
    # contour_points[:,1] = -1*contour_points[:,1] + h

    return jax.device_put(X), \
            jax.device_put(inflow_coords), \
            jax.device_put(outflow_coords), \
            jax.device_put(wall_coords) #, \
            # jax.device_put(contour_points) 
            
def initial_fields(coords):
    
    filepath = '../geom3x3/pred_initial_fields.pkl'
    with open(filepath, 'rb') as filepath:
        arquivos = pickle.load(filepath)
    coords_initial = arquivos['coords']
    u0 = arquivos['u_pred']
    v0 = arquivos['v_pred']
    p0 = arquivos['p_pred']
    del arquivos
    
    filepath = '../geom3x3/pred_initial_fields_05.pkl'
    with open(filepath, 'rb') as filepath:
        arquivos = pickle.load(filepath)
    coords_initial = arquivos['coords']
    u0_5 = arquivos['u_pred']
    v0_5 = arquivos['v_pred']
    del arquivos
    
    filepath = './chip3x3_inv1.pkl'
    with open(filepath, 'rb') as filepath:
        arquivos = pickle.load(filepath)
    coords_fem = arquivos['coord']
    u_fem_1 = arquivos['u_data']
    v_fem_1 = arquivos['v_data']
    p_fem_1 = arquivos['p_data']
    s_fem_1 = arquivos['c_data']
    dt_fem_1 = arquivos['dt_data']
    del arquivos
    
    filepath = './chip3x3_inv2.pkl'
    with open(filepath, 'rb') as filepath:
        arquivos = pickle.load(filepath)
    coords_fem = arquivos['coord']
    u_fem_2 = arquivos['u_data']
    v_fem_2 = arquivos['v_data']
    p_fem_2 = arquivos['p_data']
    s_fem_2 = arquivos['c_data']
    dt_fem_2 = arquivos['dt_data']
    del arquivos
    
    filepath = './chip3x3_inv5.pkl'
    with open(filepath, 'rb') as filepath:
        arquivos = pickle.load(filepath)
    coords_fem = arquivos['coord']
    u_fem_5 = arquivos['u_data']
    v_fem_5 = arquivos['v_data']
    p_fem_5 = arquivos['p_data']
    s_fem_5 = arquivos['c_data']
    dt_fem_5 = arquivos['dt_data']
    del arquivos
    

    # u0 = jnp.zeros(coords.shape[0])
    # v0 = jnp.zeros(coords.shape[0])
    # p0 = jnp.zeros(coords.shape[0])
    s0 = jnp.zeros(coords.shape[0])
    # s0[jnp.where(coords[0]==0)] = 1
    s0.at[jnp.where(coords[:,0]<=.05*jnp.max(coords[:,0]))].set(1.0)
    coords_initial =coords
    # u0=s0; v0=s0; p0=s0
    return (u0, v0, p0, s0, coords_initial, u0_5, v0_5,
            jax.device_put(u_fem_1), jax.device_put(v_fem_1), jax.device_put(p_fem_1), jax.device_put(s_fem_1), jax.device_put(dt_fem_1), jax.device_put(coords_fem),
            jax.device_put(u_fem_5), jax.device_put(v_fem_5), jax.device_put(p_fem_5), jax.device_put(s_fem_5),
            jax.device_put(u_fem_2), jax.device_put(v_fem_2), jax.device_put(p_fem_2), jax.device_put(s_fem_2),)


def get_dataset(pin):

    coords, inflow_coords, outflow_coords, wall_coords = get_coords()
    scale = 1/1000/100
    (coords, inflow_coords, outflow_coords, wall_coords) = (
                                                                coords*scale, 
                                                                inflow_coords*scale, 
                                                                outflow_coords*scale, 
                                                                wall_coords*scale)
    mu0 = .01#.02
    mu1 = [.01, .0503]
    rho0 = 1000; rho1 = 1000
    # u0, v0, p0, s0, _, u0_5, v0_5 = initial_fields(coords)
    initial = initial_fields(coords)
    return (
        coords,
        inflow_coords,
        outflow_coords,
        wall_coords,
        initial,
        mu0, mu1, rho0, rho1
    )
