import jax.numpy as jnp
import jax
import numpy as np
import pickle
import scipy.io


# def parabolic_inflow(y, U_max):
#     u = 4 * U_max * y * (0.41 - y) / (0.41**2)
#     v = jnp.zeros_like(y)
#     return u, v


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

# def get_time(space):
#     time = jnp.arange(0, 1.001, 0.01)
#     time_repeated = jnp.tile(time, (space.shape[0], 1)).flatten()
#     space_repeated = jnp.repeat(space, time.shape[0], axis=0)
#     coord = jnp.column_stack((time_repeated, space_repeated))
#     return coord


def initial_fields(coords, mu, pin):
    # self.coords.max(axis=0)
    u0 = 1/(2*mu)*(pin/coords[:,0].max())*(coords[:,1]**2-coords[:,1].max()*coords[:,1])
    v0 = jnp.zeros(coords.shape[0])
    p0 = pin - (coords[:,0]/.021)*pin
    s0 = jnp.zeros(coords.shape[0])
    return u0, v0, p0, s0

def get_dataset(pin):

    coords, inflow_coords, outflow_coords, wall_coords = get_coords()
    mu0 = .1
    mu1 = .15
    rho0 = 1000; rho1 = 1000
    u0, v0, p0, s0 = initial_fields(coords, mu0, pin)
    time = jnp.arange(0, 1.001, 0.01)
    return (
        coords,
        inflow_coords,
        outflow_coords,
        wall_coords,
        #time,
        u0, v0, p0, s0,
        mu0, mu1, rho0, rho1
    )

# def get_dataset():
#     # data = jnp.load("data/ns_unsteady.npy", allow_pickle=True).item()
#     # u_ref = jnp.array(data["u"])
#     # v_ref = jnp.array(data["v"])
#     # p_ref = jnp.array(data["p"])
#     t = jnp.array(data["t"])
#     coords = jnp.array(data["coords"])
#     inflow_coords = jnp.array(data["inflow_coords"])
#     outflow_coords = jnp.array(data["outflow_coords"])
#     wall_coords = jnp.array(data["wall_coords"])
#     cylinder_coords = jnp.array(data["cylinder_coords"])
#     nu = jnp.array(data["nu"])

#     return (
#         u_ref,
#         v_ref,
#         p_ref,
#         coords,
#         inflow_coords,
#         outflow_coords,
#         wall_coords,
#         cylinder_coords,
#         nu,
#     )


# def get_fine_mesh():
#     data = jnp.load("data/fine_mesh.npy", allow_pickle=True).item()
#     fine_coords = jnp.array(data["coords"])

#     data = jnp.load("data/fine_mesh_near_cylinder.npy", allow_pickle=True).item()
#     fine_coords_near_cyl = jnp.array(data["coords"])

#     return fine_coords, fine_coords_near_cyl
