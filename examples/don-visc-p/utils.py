import jax.numpy as jnp
import jax
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
    dx = .00025
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

    # # cylinder
    # radius = np.sqrt(np.power((X[:,0] - (cylinder_center[0])) , 2) + np.power((X[:,1] - (cylinder_center[1])) , 2))
    # inds = np.where(radius <= .0015)

    # X = np.delete(X, inds, axis = 0)

    # x1_rad = np.array(np.arange(0, 2*np.pi, np.pi/35))
    # x2_rad = np.transpose(np.array([cylinder_center[1] + np.sin(x1_rad) * .0015]))
    # x1_rad = np.transpose(np.array([cylinder_center[0] + np.cos(x1_rad) * .0015]))
    # x1_rad = np.concatenate((x1_rad, x2_rad), axis = 1)
    # # cylinder_coords = x1_rad
    # wall_coords = np.concatenate((wall_coords, x1_rad), axis=0)

    # X = np.concatenate((X, x1_rad), axis = 0)

    inds = np.where(X[:,1] > x2_max)
    X = np.delete(X, inds, axis = 0)

    return (
            jax.device_put(X), 
            jax.device_put(inflow_coords), 
            jax.device_put(outflow_coords), 
            jax.device_put(wall_coords),
    )


#, \
            # jax.device_put(cylinder_coords)


def get_dataset():


    # u_fem, v_fem, p_fem, coords_fem = get_fem_data()

    coords= []; inflow_coords= []; outflow_coords= []; wall_coords= []

    # key = jax.random.PRNGKey(0)
    # cylinder_center = jax.random.uniform(key, shape=(100,2))
    # minval = jnp.array([0.0165, 0.0045])
    # maxval = jnp.array([0.0195, 0.0095])
    # cylinder_center = minval + cylinder_center * (maxval - minval)

    coords, inflow_coords, outflow_coords, wall_coords = get_coords()

    # for _ in range(cylinder_center.shape[0]):
    #     coords_, inflow_coords_, outflow_coords_, wall_coords_= get_coords()
    #     coords.append(coords_)
    #     inflow_coords.append(inflow_coords_)
    #     outflow_coords.append(outflow_coords_)
    #     wall_coords.append(wall_coords_)
    mu = [.1, .1001]
    pin = [10, 10.001]

    # data = jnp.load("data/ns_steady.npy", allow_pickle=True).item()
    # u_ref = jnp.array(data["u"])
    # v_ref = jnp.array(data["v"])
    # p_ref = jnp.array(data["p"])
    # coords = jnp.array(data["coords"])
    # inflow_coords = jnp.array(data["inflow_coords"])
    # outflow_coords = jnp.array(data["outflow_coords"])
    # wall_coords = jnp.array(data["wall_coords"])
    # cylinder_coords = jnp.array(data["cylinder_coords"])
    # nu = jnp.array(data["nu"])

    return (
        # u_fem, v_fem, p_fem, coords_fem,
        # u_ref,
        # v_ref,
        # p_ref,
        coords,
        inflow_coords,
        outflow_coords,
        wall_coords,
        # cylinder_center,
        # cylinder_coords,
        mu,
        pin
    )
