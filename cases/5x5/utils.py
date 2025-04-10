import jax.numpy as jnp
import jax
import numpy as np
import pickle
import scipy.io
import matplotlib.pyplot as plt
from jax.numpy.linalg import norm as distance
from jax.numpy.linalg import inv as invert

def nondimensionalize(data, mu1, dp, L_max):
    U_max = dp*L_max/mu1
    coords = data[1]
    coords = coords/L_max
    t = data[2]
    t = t / (mu1/dp)
    u = data[0][0] / U_max
    v = data[0][1] / U_max
    p = data[0][2] / dp
    s = data[0][3]
        
    umax = np.max(u)
    vmax = np.max(v)
    tmax = np.max(t)

    return ([u, v, p, s], coords, t, [umax, vmax, tmax])

def load_dataset(filepath):
    with open(filepath, 'rb') as filepath:
        arquivos = pickle.load(filepath)
    coords_fem = arquivos['coord']
    u_fem = arquivos['u_data'][1:]
    v_fem = arquivos['v_data'][1:]
    p_fem = arquivos['p_data'][1:]
    s_fem = arquivos['c_data'][1:]
    s_fem[s_fem > 0.01] = 1 # s_fem[s_fem > 0.49] = 1
    s_fem[s_fem < 0.01] = 0 # s_fem[s_fem < 0.49] = 0
    dt_fem = arquivos['dt_data'][1:]
    t_fem = np.cumsum(dt_fem)
    del arquivos
    
    return ((u_fem, v_fem, p_fem, s_fem), coords_fem, t_fem)


def get_fem(filepath_list, mu_list, mu1, dp, L_max):
    data_all = []
    flag_umax = 0
    
    for file in filepath_list:
        data = load_dataset(file)
        data = nondimensionalize(data, mu1, dp, L_max)
        if flag_umax == 0: 
            umax = data[3]
            flag_umax = 1
        data_all.append(data[0])
    coords = data[1]
    t = data[2]
        
    return data_all, coords, t, mu_list, umax
            
def initial_fields(L_max, config):

    filepath = '../data/chip3x3_steady_mu_50cp.pkl'
    with open(filepath, 'rb') as filepath:
        arquivos = pickle.load(filepath)
    u0 = np.squeeze(arquivos['u_data'])
    v0 = np.squeeze(arquivos['v_data'])
    p0 = np.squeeze(arquivos['p_data'])
    del arquivos
    
    if config is None:
        file_list = [
            "../data/chip3x3_new_mu0_0.0025_mu1_0.05.pkl",
            "../data/chip3x3_new_mu0_0.0033333333333333335_mu1_0.05.pkl",
            "../data/chip3x3_new_mu0_0.05_mu1_0.05.pkl",
            "../data/chip3x3_new_mu0_0.075_mu1_0.05.pkl",
            "../data/chip3x3_new_mu0_0.1_mu1_0.05.pkl",
        ]
        mu_list = [.0025, .005, .05, .075, .1]
        # mu_list = [ .1]
    elif config.mode == "eval":
        file_list = [
            "../data/chip3x3_new_mu0_0.0025_mu1_0.05.pkl",
            "../data/chip3x3_new_mu0_0.0033333333333333335_mu1_0.05.pkl",
            "../data/chip3x3_new_mu0_0.005_mu1_0.05.pkl",
            "../data/chip3x3_new_mu0_0.01_mu1_0.05.pkl",
            "../data/chip3x3_new_mu0_0.05_mu1_0.05.pkl",
            "../data/chip3x3_new_mu0_0.0625_mu1_0.05.pkl",
            "../data/chip3x3_new_mu0_0.075_mu1_0.05.pkl",
            "../data/chip3x3_new_mu0_0.0875_mu1_0.05.pkl",
            "../data/chip3x3_new_mu0_0.1_mu1_0.05.pkl",
        ]
        mu_list = [.0025, .0033333333333333335, .005, .01, .05, .0625, .075, .0875, .1]
        # mu_list = [ .1]
    
    (data_fem, coords_fem, t_fem, _, _) = get_fem(file_list, mu_list, mu1=.05, dp=100, L_max=L_max)
    pmax = 100
    dp = pmax
    mu1 = .05
    U_max = dp*L_max/mu1
    u0, v0, p0 = u0 / U_max, v0 / U_max, p0 / pmax 

    i=0
    u_fem = np.concatenate(
        [data_fem[j][i][np.newaxis, :, :] for j in range(len(data_fem))], 
        axis=0
    )

    i=1
    v_fem = np.concatenate(
        [data_fem[j][i][np.newaxis, :, :] for j in range(len(data_fem))], 
        axis=0
    )

    i=2
    p_fem = np.concatenate(
        [data_fem[j][i][np.newaxis, :, :] for j in range(len(data_fem))], 
        axis=0
    )

    i=3
    s_fem = np.concatenate(
        [data_fem[j][i][np.newaxis, :, :] for j in range(len(data_fem))], 
        axis=0
    )
    
    coords_fem = jax.device_put(coords_fem)
    s0 = jnp.zeros(coords_fem.shape[0])  # Initialize with zeros
    condition = jnp.where(coords_fem[:, 0] <= 0.0111 * jnp.max(coords_fem[:, 0]))
    s0 = s0.at[condition].set(1.0)  # Assign the result back to s0
    coords_initial = coords_fem
    return (jax.device_put(u0), jax.device_put(v0), jax.device_put(p0), jax.device_put(s0), jax.device_put(coords_initial),
            u_fem, v_fem, p_fem, s_fem, t_fem, 
            # jax.device_put(u_fem), jax.device_put(v_fem), jax.device_put(p_fem), jax.device_put(s_fem), jax.device_put(t_fem), 
            coords_fem, jnp.array(mu_list))
    
    
def get_delta_matrices():
    
    filepath = '../data/matrices.pkl'
    with open(filepath, 'rb') as filepath:
        arquivos = pickle.load(filepath)
    eigvecs = np.squeeze(arquivos['eigvecs'])
    del arquivos
    eigvecs /= np.linalg.norm(eigvecs, axis=1, keepdims=True)
       
    filepath = '../data/matrices2.pkl'
    with open(filepath, 'rb') as filepath:
        arquivos = pickle.load(filepath)
    subdomain_id = np.squeeze(arquivos['subdomain_id'])
    vertices = np.squeeze(arquivos['vertices'])
    mesh_vertices = arquivos['mesh_vertices']
    centroid = np.squeeze(arquivos['centroid'])
    A_matrices = np.squeeze(arquivos['A_matrices'])
    B_matrices = np.squeeze(arquivos['B_matrices'])
    M_matrices = np.squeeze(arquivos['M_matrices'])
    N_matrices = arquivos['N_matrices']
    del arquivos
    
    subdomain_id = jax.device_put(subdomain_id)
    eigvecs = jax.device_put(eigvecs)
    vertices = jax.device_put(np.stack(vertices))
    mesh_vertices = jax.device_put(mesh_vertices[:,:-1])
    centroid =  jax.device_put(np.stack(centroid))
    B_matrices = jax.device_put(np.stack(B_matrices))
    A_matrices = jax.device_put(np.stack(A_matrices))
    M_matrices = jax.device_put(np.stack(M_matrices))
    N_matrices = jax.device_put(np.stack(N_matrices))
       
    return (
        subdomain_id,
        eigvecs, 
        vertices,
        mesh_vertices,
        centroid, 
        B_matrices, 
        A_matrices, 
        M_matrices, 
        N_matrices
    )
    
def get_noslip(subdomain_id, vertices, mesh_vertices):
    gap_btw_sides = .001
    sq_side = .0015
    side_gap = sq_side + gap_btw_sides
    radius = sq_side/2
    num_sqrs_x = 4
    num_sqrs_y = 4
    centers = jnp.arange(4) * side_gap + radius
    centers = jnp.concatenate([
                               jnp.concatenate([centers[:,jnp.newaxis] , jnp.ones((4,1)) *centers[0] ], axis=1),
                               jnp.concatenate([centers[:,jnp.newaxis] , jnp.ones((4,1)) *centers[1] ], axis=1),
                               jnp.concatenate([centers[:,jnp.newaxis] , jnp.ones((4,1)) *centers[2] ], axis=1),
                               jnp.concatenate([centers[:,jnp.newaxis] , jnp.ones((4,1)) *centers[3] ], axis=1),
    ], axis =0)

    idx_innerwalls = jnp.where(subdomain_id==1)[0]

    idx_mesh_col = []
    for i in idx_innerwalls:
        verts = vertices[i]
        dist = distance(centers-verts[0]*jnp.ones(centers.shape), axis=1) 
        dist = jnp.abs(dist)      
        min_dist_idx = jnp.where(dist==jnp.min(dist))[0]
        center = jnp.squeeze(centers[min_dist_idx])
        dist = distance(verts-center*jnp.ones(verts.shape), axis=1) 
        dist = jnp.abs(dist)     
        min_dist_idx = jnp.where(dist==jnp.min(dist))[0][0]
        idx_mesh = jnp.where((mesh_vertices[:,0]==verts[min_dist_idx,0])&(mesh_vertices[:,1]==verts[min_dist_idx,1]))[0][0]
        idx_mesh_col.append(idx_mesh)
    
    idx_mesh_col = jnp.stack(idx_mesh_col) 
    return idx_mesh_col

       
def relationship_element_vertex(L_max, delta_matrices, config):
    
    subdomain_id, eigvecs, vertices, mesh_vertices, centroid, B_matrices, A_matrices, M_matrices, N_matrices = delta_matrices

    eigvecs = jnp.concatenate([mesh_vertices/L_max, eigvecs], axis=1)
    idx_inlet = jnp.where(mesh_vertices[:,0]==jnp.min(mesh_vertices[:,0]))[0]
    idx_outlet = jnp.where(mesh_vertices[:,0]==jnp.max(mesh_vertices[:,0]))[0]
    idx_bottom = jnp.where(mesh_vertices[:,1]==jnp.min(mesh_vertices[:,1]))[0]
    idx_top = jnp.where(mesh_vertices[:,1]==jnp.max(mesh_vertices[:,1]))[0]
    idx_inwalls = get_noslip(subdomain_id, vertices, mesh_vertices)
    
    idx_noslip = jnp.concatenate([idx_bottom, idx_top, idx_bottom, idx_top, idx_inwalls])
    idx_bcs = (idx_inlet, idx_outlet, idx_noslip)
       
    if False:
        map_elements_vertexes = []
        for i in range(vertices.shape[0]):
            idxs = []
            for j in range(vertices.shape[1]):
                idx = jnp.where((mesh_vertices[:,0]==vertices[i,j,0]) & (mesh_vertices[:,1]==vertices[i,j,1]))[0]
                idxs.append(idx)
            idxs = jnp.squeeze(jnp.stack(idxs))
            map_elements_vertexes.append(idxs)
        map_elements_vertexes = jnp.stack(map_elements_vertexes)
            
        filepath = '../data/map_elements_vertexes.pkl'
        with open(filepath,"wb") as filepath:
            pickle.dump({
                        "map_elements_vertexes": map_elements_vertexes,
                    }, filepath)
    else:
        filepath = '../data/map_elements_vertexes.pkl'
        with open(filepath, 'rb') as filepath:
            arquivos = pickle.load(filepath)
        map_elements_vertexes = arquivos['map_elements_vertexes']
        del arquivos

    
    if config is None:
        print("Inverting Ms")
        for i in range(M_matrices.shape[0]):
            M_matrices = M_matrices.at[i].set(invert(M_matrices[i]))
        print("Inverting done")
    elif config.mode == "eval":
        print(" ")
        
    delta_matrices = (idx_bcs, eigvecs, vertices/L_max, map_elements_vertexes, centroid, B_matrices, A_matrices, M_matrices, N_matrices)
    return delta_matrices

def get_dataset(L_max, config=None):

    mu1 = .05
    rho0 = 1000; rho1 = 1000
    initial = initial_fields(L_max, config)
    delta_matrices = get_delta_matrices()
    delta_matrices = relationship_element_vertex(L_max, delta_matrices, config)
        
    return (
        initial,
        delta_matrices,
        mu1, rho0, rho1
    )

    # print(jnp.where(subdomain_id==0)[0].shape) # fluid
    # print(jnp.where(subdomain_id==1)[0].shape) # InnerWalls
    # print(jnp.where(subdomain_id==2)[0].shape) # Inlet
    # print(jnp.where(subdomain_id==3)[0].shape) # Outlet
    # print(jnp.where(subdomain_id==4)[0].shape) # TopWall
    # print(jnp.where(subdomain_id==5)[0].shape) # BottomWall