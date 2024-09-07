import jax.numpy as jnp
import jax
import numpy as np
import pickle
import scipy.io


# def get_fem_data():
#     filepath = './steady_3x3.pkl'
#     with open(filepath, 'rb') as filepath:
#         arquivos = pickle.load(filepath)

#     coord = arquivos['coord']

#     u_fem = arquivos['u']
#     v_fem = arquivos['v']
#     p_fem = arquivos['p']

#     return jax.device_put(u_fem), jax.device_put(v_fem), jax.device_put(p_fem), jax.device_put(coord)

def get_coords():
    filepath = '1x2.pkl'
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


def get_dataset():
    # u_fem, v_fem, p_fem, coords_fem = get_fem_data()
    coords, inflow_coords, outflow_coords, wall_coords = get_coords()
    scale = 1/1000/1000
    (coords, inflow_coords, outflow_coords, wall_coords) = (
                                                                coords*scale, 
                                                                inflow_coords*scale, 
                                                                outflow_coords*scale, 
                                                                wall_coords*scale)
    mu = .001
    rho = 1000

    return (
        # u_fem, v_fem, p_fem, coords_fem,
        coords,
        inflow_coords,
        outflow_coords,
        wall_coords,
        mu, rho
    ) 
