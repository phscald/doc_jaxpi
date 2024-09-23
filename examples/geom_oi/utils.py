import jax.numpy as jnp
import jax
import numpy as np
import pickle
import scipy.io


# def get_coords():
#     dx = .0001
#     x1_max = .021
#     x2_max = .014

#     x1 = np.arange(0,.021+dx/2, dx)
#     x2 = np.arange(0,.014+dx/2, dx)

#     X = np.zeros((x1.shape[0]*x2.shape[0],2))

#     for i in range(x1.shape[0]):
#         X[i*x2.shape[0]:(i+1)*x2.shape[0],:] = np.transpose(np.concatenate(([x1[i]*np.ones(x2.shape[0])],[x2]),axis=0))

#     # esquerda p entrada
#     id = np.where(np.squeeze(np.matmul(X, np.array([[1],[0]]))) == 0)[0]
#     inflow_coords = X[id]
#     # direita p saida
#     id = np.where(np.squeeze(np.matmul(X, np.array([[1],[0]]))) == x1_max)[0]
#     outflow_coords = X[id]
#     # bottom
#     id = np.where(np.squeeze(np.matmul(X, np.array([[0],[1]]))) == 0)[0]
#     bot_wall_coords = X[id]
#     # top
#     id = np.where(np.squeeze(np.matmul(X, np.array([[0],[1]]))) == x2_max)[0]
#     top_wall_coords = X[id]
#     # wall = top+bottom
#     wall_coords = np.concatenate((bot_wall_coords, top_wall_coords), axis=0)

#     # cylinder
#     radius = np.sqrt(np.power((X[:,0] - (x1_max/2)) , 2) + np.power((X[:,1] - (x2_max/2)) , 2))
#     inds = np.where(radius <= .0015)

#     X = np.delete(X, inds, axis = 0)

#     x1_rad = np.array(np.arange(0, 2*np.pi, np.pi/35))
#     x2_rad = np.transpose(np.array([x2_max/2 + np.sin(x1_rad) * .0015]))
#     x1_rad = np.transpose(np.array([x1_max/2 + np.cos(x1_rad) * .0015]))
#     x1_rad = np.concatenate((x1_rad, x2_rad), axis = 1)

#     X = np.concatenate((X, x1_rad), axis = 0)
#     wall_coords = np.concatenate((wall_coords, x1_rad), axis = 0)


#     inds = np.where(X[:,1] > x2_max)
#     X = np.delete(X, inds, axis = 0)

#     return jax.device_put(X), \
#             jax.device_put(inflow_coords), \
#             jax.device_put(outflow_coords), \
#             jax.device_put(wall_coords)


# def initial_fields(coords, mu, pin):

#     u0 = jnp.zeros(coords.shape[0])
#     v0 = jnp.zeros(coords.shape[0])
#     p0 = jnp.zeros(coords.shape[0])
#     s0 = jnp.zeros(coords.shape[0])
#     # s0[jnp.where(coords[0]==0)] = 1
#     s0.at[jnp.where(coords[:,0]<=.05*jnp.max(coords[:,0]))].set(1.0)
#     return u0, v0, p0, s0


def get_coords():
    filepath = 'oi.pkl'
    with open(filepath, 'rb') as filepath:
        arquivos = pickle.load(filepath)

    contour_points = arquivos['contour_points']
    X = arquivos['outside_points']
    
    ###########
    
    indx = np.where(contour_points[:,1]>=150)[0]
    contour_points = contour_points[indx]
    indx = np.where(contour_points[:,1]<=250)[0]
    contour_points = contour_points[indx]

    addx = np.arange(151,250)[:,np.newaxis]
    addy = 150*np.ones(addx.shape)
    contour_points = np.concatenate((contour_points, np.concatenate((addx, addy), axis=1)), axis=0)
    addy = 250*np.ones(addx.shape)
    contour_points = np.concatenate((contour_points, np.concatenate((addx, addy), axis=1)), axis=0)

    addx = np.arange(401,500)[:,np.newaxis]
    addy = 150*np.ones(addx.shape)
    contour_points = np.concatenate((contour_points, np.concatenate((addx, addy), axis=1)), axis=0)
    addy = 250*np.ones(addx.shape)
    contour_points = np.concatenate((contour_points, np.concatenate((addx, addy), axis=1)), axis=0)
    contour_points[:,1] = contour_points[:,1]-150
    
    indx = np.where(X[:,1]>150)[0]
    X = X[indx]
    indx = np.where(X[:,1]<250)[0]
    X = X[indx]
    X[:,1] = X[:,1]-150
        
    ##################  
    idx = np.where(X[:,0] == np.min(X[:,0]))[0]
    inflow_coords = X[idx,:]
    
    idx = np.where(X[:,0] == np.max(X[:,0]))[0]
    outflow_coords = X[idx,:]
    
    idx = np.where(X[:,1] == np.min(X[:,1]))[0]
    bot_coords = X[idx,:]
    
    idx = np.where(X[:,1] == np.max(X[:,1]))[0]
    top_coords = X[idx,:]
    
    ##
    # idx = np.where(contour_points[:,1]<50)[0]
    # contour_points[idx,1] = 0
    # idx = np.where(contour_points[:,1]>50)[0]
    # contour_points[idx,1] = 100
    ##
        
    wall_coords = np.concatenate((bot_coords, top_coords, contour_points), axis=0)

    return jax.device_put(X), \
            jax.device_put(inflow_coords), \
            jax.device_put(outflow_coords), \
            jax.device_put(wall_coords) #, \
            # jax.device_put(contour_points/1000)
           
            
def initial_fields(coords):
    
    filepath = '../geom_oi0/pred_initial_fields.pkl'
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
    coords_initial =coords
    # u0=s0; v0=s0; p0=s0
    return u0, v0, p0, s0, coords_initial


def get_dataset(pin):

    coords, inflow_coords, outflow_coords, wall_coords = get_coords()
    scale = 1/1000/100
    (coords, inflow_coords, outflow_coords, wall_coords) = (
                                                                coords*scale, 
                                                                inflow_coords*scale, 
                                                                outflow_coords*scale, 
                                                                wall_coords*scale)
    mu0 = .01#.02
    mu1 = .01
    rho0 = 1000; rho1 = 1000
    u0, v0, p0, s0, _ = initial_fields(coords)
    return (
        coords,
        inflow_coords,
        outflow_coords,
        wall_coords,
        u0, v0, p0, s0,
        mu0, mu1, rho0, rho1
    )
