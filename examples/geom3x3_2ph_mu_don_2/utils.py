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

    filepath = './chip3x3_steady_mu_50cp.pkl'
    with open(filepath, 'rb') as filepath:
        arquivos = pickle.load(filepath)
    # coords_initial = arquivos['coord']
    u0 = np.squeeze(arquivos['u_data'])
    v0 = np.squeeze(arquivos['v_data'])
    p0 = np.squeeze(arquivos['p_data'])
    del arquivos
        
    filepath = './chip3x3_inv0.pkl'
    with open(filepath, 'rb') as filepath:
        arquivos = pickle.load(filepath)
    coords_fem = arquivos['coord']
    u_fem_s = arquivos['u_data']
    v_fem_s = arquivos['v_data']
    p_fem_s = arquivos['p_data']
    s_fem_s = arquivos['c_data']
    dt_fem_s = arquivos['dt_data']
    del arquivos
        
    filepath = './chip3x3_inv20.pkl'
    with open(filepath, 'rb') as filepath:
        arquivos = pickle.load(filepath)
    coords_fem = arquivos['coord']
    u_fem_q = arquivos['u_data']
    v_fem_q = arquivos['v_data']
    p_fem_q = arquivos['p_data']
    s_fem_q = arquivos['c_data']
    dt_fem_q = arquivos['dt_data']
    del arquivos
    
    coords_fem = jax.device_put(coords_fem)
    s0 = jnp.zeros(coords_fem.shape[0])  # Initialize with zeros
    condition = jnp.where(coords_fem[:, 0] <= 0.0111 * jnp.max(coords_fem[:, 0]))
    s0 = s0.at[condition].set(1.0)  # Assign the result back to s0
    coords_initial = coords_fem
    return (jax.device_put(u0), jax.device_put(v0), jax.device_put(p0), jax.device_put(s0), jax.device_put(coords_initial),
            jax.device_put(u_fem_s), jax.device_put(v_fem_s), jax.device_put(p_fem_s), jax.device_put(s_fem_s), jax.device_put(dt_fem_s), coords_fem,
            jax.device_put(u_fem_q), jax.device_put(v_fem_q), jax.device_put(p_fem_q), jax.device_put(s_fem_q),)


def get_dataset(pin):

    coords, inflow_coords, outflow_coords, wall_coords = get_coords()
    scale = 1/1000/100
    (coords, inflow_coords, outflow_coords, wall_coords) = (
                                                                coords*scale, 
                                                                inflow_coords*scale, 
                                                                outflow_coords*scale, 
                                                                wall_coords*scale)
    # mu0 = [.0025, .05]#.02
    
    # mu0 = [.05, .1]
    mu0 = [.04, .1]
    mu1 = .05
    rho0 = 1000; rho1 = 1000
    initial = initial_fields(coords)
    return (
        coords,
        inflow_coords,
        outflow_coords,
        wall_coords,
        initial,
        mu0, mu1, rho0, rho1
    )
