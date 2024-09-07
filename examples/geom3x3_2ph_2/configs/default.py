import ml_collections
import jax.numpy as jnp

def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    config.mode = "train"
    # reduzir tamanho do passo? (testando agora)
    # estava chegando a +0  edepois voltava lá pra cima
    # tentar um grid 2x2 ao inves de 3x3? (proxima tentativa)

    # Weights & Biases
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "geomII-unsteady"
    wandb.name = "default"
    wandb.tag = None

    # Nondimensionalization
    config.nondim = True

    # Arch
    config.arch = arch = ml_collections.ConfigDict()
    arch.arch_name = "ModifiedMlp"
    arch.num_layers = 7
    arch.hidden_dim = 100 #720
    arch.out_dim = 4
    arch.activation = "tanh"  # gelu works better than tanh for this problem
    arch.periodicity = ml_collections.ConfigDict(
        {"period": (1.0,1.0),
         "axis": (1,2), 
         "trainable": (True,True)}
        ) # False
    # arch.periodicity = ml_collections.ConfigDict(
    #     {"period": (1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0),
    #      "axis": (1,2,3,4,5,6,7,8,9,10), 
    #      "trainable": (True,True,True,True,True,True,True,True,True,True)}
    #     ) # False
    arch.fourier_emb = ml_collections.ConfigDict({"embed_scale": 1.0, "embed_dim": arch.hidden_dim})
    arch.reparam = ml_collections.ConfigDict(
        {"type": "weight_fact", "mean": 1.0, "stddev": 0.1}
    )

    # Optim
    config.optim = optim = ml_collections.ConfigDict()
    optim.optimizer = "Adam"
    optim.beta1 = 0.9
    optim.beta2 = 0.999
    optim.eps = 1e-8
    optim.learning_rate = 1e-3
    optim.decay_rate = 0.88
    optim.decay_steps = 5000
    optim.grad_accum_steps = 0

    # Training
    config.training = training = ml_collections.ConfigDict()
    training.max_steps = 75000
    training.num_time_windows = 1# 10

    div=4
    training.inflow_batch_size = int(2048/div)
    training.outflow_batch_size = int(2048/div)
    training.noslip_batch_size = int(2048/div)
    training.ic_batch_size = int(2048/div)
    training.res_batch_size =  int(4096/div)

    # Weighting
    config.weighting = weighting = ml_collections.ConfigDict()
    weighting.scheme = "grad_norm"
    weighting.init_weights = {
        # "u_ic": 1.0,
        # "v_ic": 1.0,
        # "p_ic": 1.0,
        "s_ic": 1.0,
        "p_in": 1.0,
        "p_out": 1.0,
        "s_in": 1.0,
        "v_in": 1.0,
        "v_out": 1.0,
        "u_noslip": 1.0,
        "v_noslip": 1.0,
        "ru": 1.0,
        "rv": 1.0,
        "rc": 1.0,
        "rs": 1.0,
    }

    weighting.momentum = 0.9 
    weighting.update_every_steps = 5000  # 100 for grad norm and 1000 for ntk

    weighting.use_causal = True
    weighting.causal_tol = 1.0
    weighting.num_chunks = 32 # getting a larger number might improve

    # Logging
    config.logging = logging = ml_collections.ConfigDict()
    logging.log_every_steps = 100
    logging.log_errors = True
    logging.log_losses = True
    logging.log_weights = True
    logging.log_grads = False
    logging.log_ntk = False
    logging.log_preds = False

    # Saving
    config.saving = saving = ml_collections.ConfigDict()
    saving.save_every_steps = 75000 #training.max_steps
    saving.num_keep_ckpts = 5

    # Input shape for initializing Flax models
    config.input_dim = 3

    # Integer for PRNG random seed.
    config.seed = 42

    return config
