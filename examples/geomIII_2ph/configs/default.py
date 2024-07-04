import ml_collections

import jax.numpy as jnp


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    config.mode = "eval"

    # Weights & Biases
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "geomIII-unsteady"
    wandb.name = "default"
    wandb.tag = None

    # Nondimensionalization
    config.nondim = True

    # Arch
    config.arch = arch = ml_collections.ConfigDict()
    arch.arch_name = "Mlp"
    div =2
    arch.num_layers =15# 7 300
    arch.hidden_dim = 180
    arch.out_dim = 4
    arch.activation = "gelu"  # gelu works better than tanh for this problem
    arch.periodicity = False # ml_collections.ConfigDict(
        # {"period": (1.0,1.0), "axis": (1,2), "trainable": (True,)}
        # )
    arch.fourier_emb = ml_collections.ConfigDict(
        {"embed_scale": 10.0, "embed_dim": arch.hidden_dim} # 128
    )
    arch.reparam = ml_collections.ConfigDict(
        {"type": "weight_fact", "mean": 0.5, "stddev": 0.1}
    )

    # Optim
    config.optim = optim = ml_collections.ConfigDict()
    optim.optimizer = "Adam"
    optim.beta1 = 0.9
    optim.beta2 = 0.999
    optim.eps = 1e-8
    optim.learning_rate = .5e-3
    optim.decay_rate = 0.9
    optim.decay_steps = 10000
    optim.grad_accum_steps = 0

    # Training
    config.training = training = ml_collections.ConfigDict()
    training.max_steps = 100000
    training.num_time_windows = 1# 10

    training.inflow_batch_size = int(2048/div)
    training.outflow_batch_size = int(2048/div)
    training.noslip_batch_size = int(2048/div)
    training.ic_batch_size = int(2048/div)
    training.res_batch_size = int(4096//div)

    # Weighting
    config.weighting = weighting = ml_collections.ConfigDict()
    weighting.scheme = "grad_norm"
    weighting.init_weights = {
        "u_ic": 1.0,
        "v_ic": 1.0,
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
    weighting.update_every_steps = 7000  # 100 for grad norm and 1000 for ntk

    weighting.use_causal = True
    weighting.causal_tol = 1.0
    weighting.num_chunks = 16

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
    saving.save_every_steps = training.max_steps
    saving.num_keep_ckpts = 10

    # Input shape for initializing Flax models
    config.input_dim = 3

    # Integer for PRNG random seed.
    config.seed = 42

    return config
