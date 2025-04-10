import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

def compute_saturation_curve(s_pred):
    sat_curve = jnp.mean(s_pred, axis=1)
    return sat_curve
    
def compute_outlet_flow_ratios(u_pred, s_pred, u0, vertices, map_elements_vertexes, xmax, xmin, mu):
    out_flow_ratio_w = [] # 1 é água
    out_flow_ratio_o = [] # 0 é óleo
    for i in range(vertices.shape[0]):
        ind = jnp.where(vertices[i,:,0]==xmax)[0]
        if ind.shape[0] == 2:
            ind1 = map_elements_vertexes[i,ind[0]]
            ind2 = map_elements_vertexes[i,ind[1]]
            dy = jnp.abs(vertices[i,ind[0],1] - vertices[i,ind[1],1])
            out_flow_ratio_w.append( (s_pred[:,ind1] * u_pred[:,ind1] + s_pred[:,ind2] * u_pred[:,ind2]) /2 * dy )
            out_flow_ratio_o.append( ( (1-s_pred[:,ind1]) * u_pred[:,ind1] + (1-s_pred[:,ind2]) * u_pred[:,ind2] ) /2 * dy )
    out_flow_ratio_w = jnp.stack(out_flow_ratio_w)
    out_flow_ratio_o = jnp.stack(out_flow_ratio_o)
    out_flow_ratio_w = jnp.sum(out_flow_ratio_w, axis=0)
    out_flow_ratio_o = jnp.sum(out_flow_ratio_o, axis=0)
    in_flow_ratio0 = compute_inlet_flow_ratio(u0, vertices, map_elements_vertexes, xmax)
    mu_w = mu
    mu_o = .05
    krw = out_flow_ratio_w / in_flow_ratio0 * (mu_w/mu_o)
    kro = out_flow_ratio_o / in_flow_ratio0
    return out_flow_ratio_w, out_flow_ratio_o, krw, kro


def compute_inlet_flow_ratio(u_pred, vertices, map_elements_vertexes, xmin):
    in_flow_ratio = []
    for i in range(vertices.shape[0]):
        ind = jnp.where(vertices[i,:,0]==xmin)[0]
        if ind.shape[0] == 2:
            ind1 = map_elements_vertexes[i,ind[0]]
            ind2 = map_elements_vertexes[i,ind[1]]
            dy = jnp.abs(vertices[i,ind[0],1] - vertices[i,ind[1],1])
            in_flow_ratio.append( (u_pred[:,ind1] + u_pred[:,ind2]) /2 * dy )
    in_flow_ratio = jnp.stack(in_flow_ratio)
    in_flow_ratio = jnp.sum(in_flow_ratio, axis=0)
    return in_flow_ratio

def get_curves(u_pred, s_pred, u0, vertices, map_elements_vertexes, xmax, xmin, mu):
    print("chega aqui")
    out_flow_ratio_w, out_flow_ratio_o, krw, kro = compute_outlet_flow_ratios(u_pred, s_pred, u0, vertices, map_elements_vertexes, xmax, xmin, mu)
    print("chega aqui")
    in_flow_ratio = compute_inlet_flow_ratio(u_pred, vertices, map_elements_vertexes, xmin)
    print("chega aqui")
    sat_curve = compute_saturation_curve(s_pred)
    print("chega aqui")
    return np.array(out_flow_ratio_w), \
        np.array(out_flow_ratio_o), \
        np.array(krw), \
        np.array(kro), \
        np.array(in_flow_ratio),\
        np.array(sat_curve)

def load_data(filepath):
    with open(filepath, 'rb') as filepath:
        arquivos = pickle.load(filepath)
    u_pred = arquivos['u_pred']
    s_pred = arquivos['s_pred']
    vertices = arquivos['vertices']
    map_elements_vertexes = arquivos['map_elements_vertexes']
    xmax = arquivos['xmax']
    xmin = arquivos['xmin']
    t_coords = arquivos['t_coords']
    tmax = arquivos['tmax']
    del arquivos
    s_pred = jnp.clip(s_pred, 0, 1)
    s_pred = jnp.round(s_pred)
    return [u_pred, s_pred], vertices, map_elements_vertexes, xmax, xmin, t_coords, tmax

def load_data_fem(filepath):
    with open(filepath, 'rb') as filepath:
        arquivos = pickle.load(filepath)
    u_pred = arquivos['u_pred']
    u0 = arquivos['u0'][jnp.newaxis,:]
    s_pred = arquivos['s_pred']
    vertices = arquivos['vertices']
    map_elements_vertexes = arquivos['map_elements_vertexes']
    xmax = arquivos['xmax']
    xmin = arquivos['xmin']
    t_coords = arquivos['t_coords']
    tmax = arquivos['tmax']
    del arquivos
    s_pred = jnp.clip(s_pred, 0, 1)
    s_pred = jnp.round(s_pred)
    return [u_pred, s_pred], u0, vertices, map_elements_vertexes, xmax, xmin, t_coords, tmax

# mu_list = [.0025, .0033333333333333335, .005, .01, .05, .0625, .075, .0875, .1]
# mu_list = [.0033333333333333335, .01, .0625, .0875]
mu = .0875
print(f"mu: {mu}")

pred_data, vertices, map_elements_vertexes, xmax, xmin, _, _ = load_data('./pred'+ str(mu) +'.pkl')
# pred_pinn, _, _, _, _, _, _                                  = load_data('pred_pinn.pkl')
pred_fem, u0 , _, _, _, _, t_coords, tmax                    = load_data_fem('./pred_fem'+ str(mu) +'.pkl')
u_fem, s_fem = pred_fem

curves_nn = get_curves(pred_data[0], pred_data[1], u0, vertices, map_elements_vertexes, xmax, xmin, mu)
# curves_pinn = get_curves(pred_pinn[0], pred_pinn[1], vertices, map_elements_vertexes, xmax, xmin)
curves_fem  = get_curves(u_fem       , s_fem       , u0, vertices, map_elements_vertexes, xmax, xmin, mu)
#                        0                 1    2    3              4          5

filepath = './curves_' + str(mu) + '.pkl'
with open(filepath,"wb") as filepath:
    pickle.dump({"curves_nn": curves_nn,
                 "curves_fem": curves_fem}, filepath)


#curves = out_flow_ratio_w, out_flow_ratio_o, krw, kro, in_flow_ratio, sat_curve
# colors = ['#D81B60', '#1E88E5', '#FFC107', '#004D40']

# plt.figure()
# sns.lineplot(x= t_coords*tmax, y= curves_data[5], color=colors[0], linestyle=':', label="saturation", )
# sns.lineplot(x= t_coords*tmax, y= curves_fem[5], color=colors[0], linestyle='-', label="saturation", )
# plt.legend(loc="best")
# plt.xlabel("time")
# plt.savefig('saturation.jpg', format='jpg')

# plt.figure()
# sns.lineplot(x= t_coords*tmax, y= curves_data[0], color=colors[1], linestyle=':', label="Vw_out(t)", )
# sns.lineplot(x= t_coords*tmax, y= curves_data[1], color=colors[2], linestyle=':', label="Vo_out(t)", )
# sns.lineplot(x= t_coords*tmax, y= curves_data[4], color=colors[3], linestyle=':', label="Vw_in(t)", )

# # sns.lineplot(x= t_coords*tmax, y= curves_pinn[5], color=colors[0], linestyle='--', label="saturation", )
# # sns.lineplot(x= t_coords*tmax, y= curves_pinn[0], color=colors[1], linestyle='--', label="Vw_out(t)", )
# # sns.lineplot(x= t_coords*tmax, y= curves_pinn[1], color=colors[2], linestyle='--', label="Vo_out(t)", )
# # sns.lineplot(x= t_coords*tmax, y= curves_pinn[4], color=colors[3], linestyle='--', label="Vw_in(t)", )


# sns.lineplot(x= t_coords*tmax, y= curves_fem[0], color=colors[1], linestyle='-', label="Vw_out(t)", )
# sns.lineplot(x= t_coords*tmax, y= curves_fem[1], color=colors[2], linestyle='-', label="Vo_out(t)", )
# sns.lineplot(x= t_coords*tmax, y= curves_fem[4], color=colors[3], linestyle='-', label="Vw_in(t)", )

# plt.legend(loc="best")
# plt.xlabel("time")
# plt.savefig('curves.jpg', format='jpg')

# def reescale(y):
#     return (y- np.min(y)) / (y[0] - np.min(y))

# plt.rcParams['font.family'] = 'DeJavu Serif'
# plt.rcParams['font.serif'] = ['Times New Roman']
# plt.rcParams['font.size'] = 12   

# plt.figure()
# sns.lineplot(x= curves_data[5], y=           curves_data[2], color=colors[0], linestyle=':', label="krw")
# sns.lineplot(x= curves_data[5], y= reescale(curves_data[3]), color=colors[1], linestyle=':', label="kro")

# sns.lineplot(x= t_coords*tmax, y= curves_pinn[2], color=colors[0], linestyle='--', label="krw")
# sns.lineplot(x= t_coords*tmax, y= curves_pinn[3], color=colors[1], linestyle='--', label="kro")

# sns.lineplot(x= curves_fem[5], y=           curves_fem[2], color=colors[0], linestyle='-', label="krw")
# sns.lineplot(x= curves_fem[5], y= reescale(curves_fem[3]), color=colors[1], linestyle='-', label="kro")

# plt.legend(loc="best")
# plt.xlabel("Sw")
# plt.savefig('k_curves0025.jpg', format='jpg')