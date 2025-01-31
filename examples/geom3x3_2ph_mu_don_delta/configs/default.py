import ml_collections
import jax.numpy as jnp

def get_config():
    
    upd_stp = 1
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    config.mode = "train"
    # config.mode = "eval"
    # ver o coeficiente do causal depois
    # Weights & Biases
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "geom1x2-unsteady"
    wandb.name = "default"
    wandb.tag = None

    # Nondimensionalization
    config.nondim = True

    # Arch
    config.arch = arch = ml_collections.ConfigDict()
    arch.arch_name = "DeepONet3wD"
    # arch.num_layers = 8
    arch.num_trunk_layers = 4 # mu 2
    arch.num_branch_layers = 7 # t 6
    # arch.num_branch_layers2 = 7 # xy v(xy)
    arch.hidden_dim = 100
    arch.out_dim = 4
    arch.activation = "tanh"  # gelu works better than tanh for this problem
    arch.periodicity = False
    arch.fourier_emb = False # ml_collections.ConfigDict({"embed_scale": 5.0, "embed_dim": arch.hidden_dim})
    arch.reparam = ml_collections.ConfigDict(
        {"type": "weight_fact", "mean": 1, "stddev": 0.1}
    )

    # Optim
    config.optim = optim = ml_collections.ConfigDict()
    optim.optimizer = "Adam"
    optim.beta1 = 0.9
    optim.beta2 = 0.999
    optim.eps = 1e-8
    optim.learning_rate = 1e-4#1e-4
    optim.decay_rate = 1 #0.98
    optim.decay_steps = 5000
    optim.grad_accum_steps = 0

    # Training
    config.training = training = ml_collections.ConfigDict()
    training.max_steps = 200000#int(2*10**(5)/upd_stp*2)
    training.fine_tune = True
    training.num_time_windows = 1

    div = 2
    training.inflow_batch_size = 32  #int(2048/div)
    training.outflow_batch_size = 32 #int(2048/div)
    training.noslip_batch_size = 128 #512int(2048/div)
    training.ic_batch_size = 512  #512 #int(2048/div)
    training.res_batch_size = 2048 + 1024#512 #+512#int(2*2048/div)

    # Weighting
    config.weighting = weighting = ml_collections.ConfigDict()
    weighting.scheme = "grad_norm"
    
    weighting.init_weights = {
        "u_data": 1.0,
        "v_data": 1.0,
        "p_data": 1.0,
        "s_data": 1.0,
        # "u_ic": 1.0,
        # "v_ic": 1.0,
        # "p_ic": 1.0,
        # "s_ic": 1.0,
        "noslip": 1.0,
        "sin": 1.0,
        "dp": 1.0,
        # "ru": 1.0,
        # "rv": 1.0,
        # "rc": 1.0,
        # "rs": 1.0,
    }

    weighting.momentum = 0.9
    weighting.update_every_steps = 2500#int(5000/upd_stp) # 100 for grad norm and 1000 for ntk

    weighting.use_causal = False  ###################### CAUSALITY
    weighting.causal_tol = .1
    weighting.num_chunks = 16

    # Logging
    config.logging = logging = ml_collections.ConfigDict()
    logging.log_every_steps = 100
    logging.log_errors  = True
    logging.log_lossesn = True
    logging.log_weights = True
    logging.log_grads   = False
    logging.log_ntk     = False
    logging.log_preds   = False

    # Saving
    config.saving = saving = ml_collections.ConfigDict()
    saving.save_every_steps = 50000#int(50000/upd_stp)
    saving.num_keep_ckpts = 10

    # Input shape for initializing Flax models
    # config.input_dim = 4
    # config.input_branch = 3
    # # config.input_branch = 2
    # config.input_trunk = 1
    
    config.input_dim = 4
    config.input_branch = 1   # t
    config.input_branch2 = 52 # xy v(xy)
    config.input_trunk = 1    # mu 

    
    # Integer for PRNG random seed.
    config.seed = 42
    

    return config
