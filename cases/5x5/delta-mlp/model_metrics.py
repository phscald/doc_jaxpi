import jax
import jax.numpy as jnp
import pickle

def load_data(mu):
    print()
    print(f'=== {mu} ===')
    filepath = './matrices_' + str(mu) + '.pkl'
    with open(filepath, 'rb') as filepath:
        arquivos = pickle.load(filepath)
    matrix1 = arquivos['matrix1']
    matrix2 = arquivos['matrix2']
    del arquivos
    
    label   = ["s", "u", "v", "p"]
    for i in range(4):
        print(f'==={label[i]}===')
        mse = jnp.mean((matrix1[i] - matrix2[i]) ** 2)   
        l2_relative_error = jnp.sum((matrix1[i] - matrix2[i])**2) / jnp.sum((matrix2[i])**2)
        # R2 Score
        ss_res = jnp.sum((matrix2[i] - matrix1[i]) ** 2)
        ss_tot = jnp.sum((matrix2[i] - jnp.mean(matrix2[i])) ** 2)
        r2 = 1 - ss_res / ss_tot
        print(f"R² score: {r2:.4f}")
        print(f"MSE: {mse}")
        print(f"L2-relative error: {l2_relative_error*100:.2f}%")
   
    return matrix1, matrix2
    
    

mu_list = [.0033333333333333335, .01, .0625, .0875]
matrix1, matrix2 = load_data(mu_list[0])

for i in range(1, len(mu_list)):
    m1, m2 = load_data(mu_list[i])
    
    for j in range(4):
        matrix1[j] = jnp.concatenate([matrix1[j], m1[j]], axis=0)
        matrix2[j] = jnp.concatenate([matrix2[j], m2[j]], axis=0)

label   = ["s", "u", "v", "p"]
print()
print("=== GLOBAL ===")
for i in range(4):
    print(f'==={label[i]}===')
    mse = jnp.mean((matrix1[i] - matrix2[i]) ** 2)   
    l2_relative_error = jnp.sum((matrix1[i] - matrix2[i])**2) / jnp.sum((matrix2[i])**2)
    # R2 Score
    ss_res = jnp.sum((matrix2[i] - matrix1[i]) ** 2)
    ss_tot = jnp.sum((matrix2[i] - jnp.mean(matrix2[i])) ** 2)
    r2 = 1 - ss_res / ss_tot
    print(f"R² score: {r2:.4f}")
    print(f"MSE: {mse}")
    print(f"L2-relative error: {l2_relative_error*100:.2f}%")