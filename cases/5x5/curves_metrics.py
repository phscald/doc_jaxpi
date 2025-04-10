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
        print(mu)
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

folders = ['delta-mlp'] 
mu_list = [.0033333333333333335, .01, .0625, .0875]

curves_nn_krw, curves_nn_kro, curves_fem_krw, curves_fem_kro = load_data(folders[0], mu_list)
metrics_krw = get_metrics(curves_nn_krw, curves_fem_krw)
metrics_kro = get_metrics(curves_nn_kro, curves_fem_kro)

print(f"metrics krw: mse: {metrics_krw[0]}, L2: {metrics_krw[1]}, R2: {metrics_krw[2]}" )
print(f"metrics kro: mse: {metrics_kro[0]}, L2: {metrics_kro[1]}, R2: {metrics_kro[2]}" )  