from tabulate import tabulate
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pickle

def load_data(folder, mu_list):

    curves_nn_kro = []
    curves_nn_krw = []
    curves_fem_kro = []
    curves_fem_krw = []
    for mu in mu_list:

        filepath = folder+'/curves_' + str(mu) + '.pkl'
        with open(filepath, 'rb') as filepath:
            arquivos = pickle.load(filepath)
        curves_nn_kro.append(arquivos['curves_nn'][3])
        curves_nn_krw.append(arquivos['curves_nn'][2])
        curves_fem_kro.append(arquivos['curves_fem'][3])
        curves_fem_krw.append(arquivos['curves_fem'][2])
        del arquivos
    
    curves_nn_krw = np.stack(curves_nn_krw)
    curves_nn_kro = np.stack(curves_nn_kro)
    curves_fem_krw = np.stack(curves_fem_krw)
    curves_fem_kro = np.stack(curves_fem_kro)
   
    return curves_nn_krw, curves_nn_kro, curves_fem_krw, curves_fem_kro


def get_metrics(A_pred, B_truth):

    mse = np.mean((A_pred - B_truth) ** 2)   
    l2_relative_error = np.sum((A_pred - B_truth)**2) / np.sum((B_truth)**2)
    # R2 Score
    ss_res = np.sum((B_truth - A_pred) ** 2)
    ss_tot = np.sum((B_truth - np.mean(B_truth)) ** 2)
    r2 = 1 - ss_res / ss_tot
        
    return mse, l2_relative_error, r2

folders = ['ffmlp', 'mlp', 'delta-mlp'] 
mu_list = [.0033333333333333335, .0625, .0875]

krw_metrics_folder = []
kro_metrics_folder = []
for i in range(len(folders)):
    print(f'=== folder: {folders[i]} ===')
    curves_nn_krw, curves_nn_kro, curves_fem_krw, curves_fem_kro = load_data(folders[i], mu_list)
    metrics_krw = get_metrics(curves_nn_krw, curves_fem_krw)
    metrics_kro = get_metrics(curves_nn_kro, curves_fem_kro)
    krw_metrics_folder.append(metrics_krw)
    kro_metrics_folder.append(metrics_kro)
    print(f"metrics krw: mse: {metrics_krw[0]}, L2: {metrics_krw[1]}, R2: {metrics_krw[2]}" )
    print(f"metrics kro: mse: {metrics_kro[0]}, L2: {metrics_kro[1]}, R2: {metrics_kro[2]}" )  
    
def reescale(y):
    y[y<1] = (y[y<1]- np.min(y)) / (y[0] - np.min(y))
    return y
    
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12  
    
colors = ['#D81B60', '#1E88E5', '#FFC107', '#004D40']
labels = ['--', ':', '-.']
mu_list = [.0033333333333333335, .0875]
for j in range(len(mu_list)):
    plt.figure()
    for i in range(len(folders)):
        print(f'=== folder: {folders[i]} ===')
        filepath = folders[i]+'/curves_' + str(mu_list[j]) + '.pkl'
        with open(filepath, 'rb') as filepath:
            arquivos = pickle.load(filepath)
        curves_nn = arquivos['curves_nn']
        curves_fem = arquivos['curves_fem']
            
        if i == 0:
            sns.lineplot(x= curves_fem[5], y=           curves_fem[2], color=colors[0], linestyle='-', label="krw ")
            sns.lineplot(x= curves_fem[5], y= reescale(curves_fem[3]), color=colors[1], linestyle='-', label="kro ")
        sns.lineplot(x= curves_fem[5], y=           curves_nn[2], color=colors[0], linestyle=labels[i], label="krw "+folders[i])
        sns.lineplot(x= curves_fem[5], y= reescale(curves_nn[3]), color=colors[1], linestyle=labels[i], label="kro "+folders[i])
        
    # plt.legend(loc="best")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.xlabel("Sw")
    plt.savefig('k_curves'+str(mu_list[j])+'.jpg', format='jpg', bbox_inches='tight')
    

table = {"fields": ["k_{rw}", "k_{rw}", "k_{rw}", "k_{ro}", "k_{ro}", "k_{ro}"],
         "metrics": ["MSE", "L2", "R2", "MSE", "L2", "R2"],
         folders[0]: [krw_metrics_folder[0][0], krw_metrics_folder[0][1], krw_metrics_folder[0][2], kro_metrics_folder[0][0],kro_metrics_folder[0][1], kro_metrics_folder[0][2]],
         folders[1]: [krw_metrics_folder[1][0], krw_metrics_folder[1][1], krw_metrics_folder[1][2], kro_metrics_folder[1][0],kro_metrics_folder[1][1], kro_metrics_folder[1][2]],
         folders[2]: [krw_metrics_folder[2][0], krw_metrics_folder[2][1], krw_metrics_folder[2][2], kro_metrics_folder[2][0],kro_metrics_folder[2][1], kro_metrics_folder[2][2]],
         }

print(tabulate(table, headers="keys", tablefmt="latex"))
        
        