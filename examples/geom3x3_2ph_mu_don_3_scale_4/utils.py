import jax.numpy as jnp
import jax
import numpy as np
import pickle
import scipy.io
import matplotlib.pyplot as plt

          
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
    
    x_out = np.where((wall_coords[:,0]>=0) & (wall_coords[:,0]<=150) & ((wall_coords[:,1]==0) | (wall_coords[:,1]==900)))[0]
    wall_coords = np.delete(wall_coords, x_out, axis=0)
    x_out = np.where((wall_coords[:,0]>=250) & (wall_coords[:,0]<=400) & ((wall_coords[:,1]==0) | (wall_coords[:,1]==900)))[0]
    wall_coords = np.delete(wall_coords, x_out, axis=0)
    x_out = np.where((wall_coords[:,0]>=500) & (wall_coords[:,0]<=650) & ((wall_coords[:,1]==0) | (wall_coords[:,1]==900)))[0]
    wall_coords = np.delete(wall_coords, x_out, axis=0)
    x_out = np.where((wall_coords[:,0]>=750) & (wall_coords[:,0]<=900) & ((wall_coords[:,1]==0) | (wall_coords[:,1]==900)))[0]
    wall_coords = np.delete(wall_coords, x_out, axis=0)
    
    # plt.scatter(wall_coords[:,0], wall_coords[:,1])
    # plt.savefig("asa.jpeg")
    
    h = np.max(X[:,1])
    
    X[:,1] = -1*X[:,1] + h
    inflow_coords[:,1] = -1*inflow_coords[:,1] + h
    outflow_coords[:,1] = -1*outflow_coords[:,1] + h
    wall_coords[:,1] = -1*wall_coords[:,1] + h
    # contour_points[:,1] = -1*contour_points[:,1] + h
    
    # plt.scatter(wall_coords[:,0], wall_coords[:,1])
    # plt.savefig("asa.jpeg")

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
    
    print(f"medio: {np.max(u0)}")
        
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
    
    print(f"slow: {np.max(u_fem_s[-1])}")

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
    
    print(f"quick: {np.max(u_fem_q[-1])}")
    
    
    X_middle, t_middle = get_points_middle_shock(s_fem_s, s_fem_q, coords_fem, dt_fem_s)
    
    coords_fem = jax.device_put(coords_fem)
    s0 = jnp.zeros(coords_fem.shape[0])  # Initialize with zeros
    condition = jnp.where(coords_fem[:, 0] <= 0.0111 * jnp.max(coords_fem[:, 0]))
    s0 = s0.at[condition].set(1.0)  # Assign the result back to s0
    coords_initial = coords_fem
    return (jax.device_put(u0), jax.device_put(v0), jax.device_put(p0), jax.device_put(s0), jax.device_put(coords_initial),
            jax.device_put(u_fem_s), jax.device_put(v_fem_s), jax.device_put(p_fem_s), jax.device_put(s_fem_s), jax.device_put(dt_fem_s), coords_fem,
            jax.device_put(u_fem_q), jax.device_put(v_fem_q), jax.device_put(p_fem_q), jax.device_put(s_fem_q),
            jax.device_put(X_middle), jax.device_put(t_middle))
    
def get_points_middle_shock(s_fem_s, s_fem_q, coords_fem, dt_fem_s):
    upper_lim_s = .3
    lower_lim_s = .2
    upper_lim_q = .8
    lower_lim_q = .7
    
    y_mains = np.array([[150, 250], [400, 500], [650, 750]])/100/1000
    
    X = []
    t = []

    cont = 1
    for i in range(s_fem_s.shape[0]):
        inds_s = np.where((s_fem_s[i]>=lower_lim_s) & (s_fem_s[i]<=upper_lim_s))[0]
        inds_q = np.where((s_fem_q[i]>=lower_lim_q) & (s_fem_q[i]<=upper_lim_q))[0]    
        
        cont2 = 0 
        for j in range(y_mains.shape[0]): 
            
            inds_main = np.where((coords_fem[:,1]>=y_mains[j,0]) & (coords_fem[:,1]<=y_mains[j,1]))[0]
            inds_back = np.intersect1d(inds_s, inds_main)
            inds_front = np.intersect1d(inds_q, inds_main)
            
            if inds_back.size == 0 or inds_front.size == 0:
                continue
            
            x_max = np.max(coords_fem[inds_front,0])
            x_min = np.max(coords_fem[inds_back,0])
            
            X.append([x_min, x_max])
            t.append([cont * dt_fem_s[0], cont2])
            cont2 += 1
        
        cont += 1
    
    X = np.array(X)
    t = np.array(t)
                                       
    return X, t
    

def get_dataset(pin):

    coords, inflow_coords, outflow_coords, wall_coords = get_coords()
    scale = 1/1000/100
    (coords, inflow_coords, outflow_coords, wall_coords) = (
                                                                coords*scale, 
                                                                inflow_coords*scale, 
                                                                outflow_coords*scale, 
                                                                wall_coords*scale)
    # mu0 = [.0025, .05]#.02
    
    mu0 = [.0025, .006]
    # mu0 = [.0025, .0051]
    # mu0 = [.05, .101]
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
