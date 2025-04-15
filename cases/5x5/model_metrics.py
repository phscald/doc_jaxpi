import jax
import jax.numpy as jnp
import pickle
from tabulate import tabulate
from matplotlib import pyplot as plt
import seaborn as sns

def load_data(folder, mu):
    # print()
    # print(f'=== {mu} ===')
    filepath = folder+'/matrices_' + str(mu) + '.pkl'
    with open(filepath, 'rb') as filepath:
        arquivos = pickle.load(filepath)
    matrix1 = arquivos['matrix1']
    matrix2 = arquivos['matrix2']
    del arquivos
    
    label   = ["s", "u", "v", "p"]
    metrics = {}
    for i in range(4):
        # print(f'==={label[i]}===')
        mse = jnp.mean((matrix1[i] - matrix2[i]) ** 2)   
        l2_relative_error = jnp.sum((matrix1[i] - matrix2[i])**2) / jnp.sum((matrix2[i])**2)
        # R2 Score
        ss_res = jnp.sum((matrix2[i] - matrix1[i]) ** 2)
        ss_tot = jnp.sum((matrix2[i] - jnp.mean(matrix2[i])) ** 2)
        r2 = 1 - ss_res / ss_tot
        metrics[label[i]] = [mse, l2_relative_error, r2]
   
    return matrix1, matrix2, metrics
    
def get_metrics(folder):  

    mu_list = [.0033333333333333335, .01, .0625, .0875]
    # mu_list = [.0033333333333333335, .0625, .0875]
    matrix1, matrix2, metrics = load_data(folder, mu_list[0])
    metrics_dict = {}
    metrics_dict[str(mu_list[0])] = metrics

    for i in range(1, len(mu_list)):
        m1, m2, metrics = load_data(folder, mu_list[i])
        metrics_dict[str(mu_list[i])] = metrics
        
        for j in range(4):
            matrix1[j] = jnp.concatenate([matrix1[j], m1[j]], axis=0)
            matrix2[j] = jnp.concatenate([matrix2[j], m2[j]], axis=0)

    label   = ["s", "u", "v", "p"]
    metrics = {}
    # print()
    # print("=== GLOBAL ===")
    for i in range(4):
        # print(f'==={label[i]}===')
        mse = jnp.mean((matrix1[i] - matrix2[i]) ** 2)   
        l2_relative_error = jnp.sum((matrix1[i] - matrix2[i])**2) / jnp.sum((matrix2[i])**2)
        # R2 Score
        ss_res = jnp.sum((matrix2[i] - matrix1[i]) ** 2)
        ss_tot = jnp.sum((matrix2[i] - jnp.mean(matrix2[i])) ** 2)
        r2 = 1 - ss_res / ss_tot
        metrics[label[i]] = [mse, l2_relative_error, r2]
    metrics_dict["global"] = metrics
    
    return metrics_dict

def get_metrics_all_fields(folder):  

    mu_list = [.0033333333333333335, .01, .0625, .0875]
    metrics_dict = {}
    for i in range(len(mu_list)):

        matrix1, matrix2, _ = load_data(folder, mu_list[i])
        matrix1 = jnp.concatenate([matrix1[0], matrix1[1], matrix1[2], matrix1[3]], axis=0)
        matrix2 = jnp.concatenate([matrix2[0], matrix2[1], matrix2[2], matrix2[3]], axis=0)
        
        mse = jnp.mean((matrix1 - matrix2) ** 2)   
        l2_relative_error = jnp.sum((matrix1 - matrix2)**2) / jnp.sum((matrix2)**2)
        # R2 Score
        ss_res = jnp.sum((matrix2 - matrix1) ** 2)
        ss_tot = jnp.sum((matrix2 - jnp.mean(matrix2)) ** 2)
        r2 = 1 - ss_res / ss_tot
        metrics_dict[str(mu_list[i])] = [mse, l2_relative_error, r2]
        
    return metrics_dict

        
mu_list = [.0033333333333333335, .01, .0625, .0875]        
folders = ["mlp", "delta-mlp", "ffmlp"]
metrics_dict = {}
for folder in folders:
       metrics_dict[folder] = get_metrics(folder)

label   = ["s", "u", "v", "p"]
table = {"fields": ["u", "u", "u", "v", "v", "v", "p", "p", "p", "s", "s", "s"],
         "metrics": ["MSE", "L2", "R2", "MSE", "L2", "R2", "MSE", "L2", "R2", "MSE", "L2", "R2"],
         folders[2]: [metrics_dict["ffmlp"]["global"]["u"][0], metrics_dict["ffmlp"]["global"]["u"][1], metrics_dict["ffmlp"]["global"]["u"][2],
                      metrics_dict["ffmlp"]["global"]["v"][0], metrics_dict["ffmlp"]["global"]["v"][1], metrics_dict["ffmlp"]["global"]["v"][2],
                      metrics_dict["ffmlp"]["global"]["p"][0], metrics_dict["ffmlp"]["global"]["p"][1], metrics_dict["ffmlp"]["global"]["p"][2],
                      metrics_dict["ffmlp"]["global"]["s"][0], metrics_dict["ffmlp"]["global"]["s"][1], metrics_dict["ffmlp"]["global"]["s"][2]],
         folders[0]: [metrics_dict["mlp"]["global"]["u"][0], metrics_dict["mlp"]["global"]["u"][1], metrics_dict["mlp"]["global"]["u"][2],
                      metrics_dict["mlp"]["global"]["v"][0], metrics_dict["mlp"]["global"]["v"][1], metrics_dict["mlp"]["global"]["v"][2],
                      metrics_dict["mlp"]["global"]["p"][0], metrics_dict["mlp"]["global"]["p"][1], metrics_dict["mlp"]["global"]["p"][2],
                      metrics_dict["mlp"]["global"]["s"][0], metrics_dict["mlp"]["global"]["s"][1], metrics_dict["mlp"]["global"]["s"][2]],
         folders[1]: [metrics_dict["delta-mlp"]["global"]["u"][0], metrics_dict["delta-mlp"]["global"]["u"][1], metrics_dict["delta-mlp"]["global"]["u"][2],
                      metrics_dict["delta-mlp"]["global"]["v"][0], metrics_dict["delta-mlp"]["global"]["v"][1], metrics_dict["delta-mlp"]["global"]["v"][2],
                      metrics_dict["delta-mlp"]["global"]["p"][0], metrics_dict["delta-mlp"]["global"]["p"][1], metrics_dict["delta-mlp"]["global"]["p"][2],
                      metrics_dict["delta-mlp"]["global"]["s"][0], metrics_dict["delta-mlp"]["global"]["s"][1], metrics_dict["delta-mlp"]["global"]["s"][2]]
         }

print(tabulate(table, headers="keys", tablefmt="latex"))

metrics_dict_all_fields = {}
for folder in folders:
       metrics_dict[folder] = get_metrics_all_fields(folder)
       
colors = ['#D81B60', '#1E88E5', '#FFC107', '#004D40']  
tags = ["MLP", "Delta-MLP", "FFMLP"] 

plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12    
       
plt.figure()
for i in range(len(folders)):
    vec_metrics =  [metrics_dict[folders[i]][str(mu_list[j])][0] for j in range(4)]
    sns.lineplot(x= jnp.array(mu_list), y= jnp.array(vec_metrics), color=colors[i], linestyle=':', marker='D', label=tags[i])
plt.legend(loc="best")
plt.ylabel(r"$MSE$")
plt.xlabel(r"$\mu$")
plt.savefig('./images/MSE_all_fields.jpg', format='jpg')

plt.figure()
for i in range(len(folders)):
    vec_metrics =  [metrics_dict[folders[i]][str(mu_list[j])][1] for j in range(4)]
    sns.lineplot(x= jnp.array(mu_list), y= jnp.array(vec_metrics), color=colors[i], linestyle=':', marker='D', label=tags[i])
plt.legend(loc="best")
plt.ylabel(r"$Relative\,\,L2\,\,Error$")
plt.xlabel(r"$\mu$")
plt.savefig('./images/L2_all_fields.jpg', format='jpg')

plt.figure()
for i in range(len(folders)):
    vec_metrics =  [metrics_dict[folders[i]][str(mu_list[j])][2] for j in range(4)]
    sns.lineplot(x= jnp.array(mu_list), y= jnp.array(vec_metrics), color=colors[i], linestyle=':', marker='D', label=tags[i])
plt.legend(loc="best")
plt.ylabel(r"$R^2$")
plt.xlabel(r"$\mu$")
plt.ylim(-1, 1)
plt.savefig('./images/R2_all_fields.jpg', format='jpg')
