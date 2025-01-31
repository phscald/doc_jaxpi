from functools import partial
from typing import Any, Callable, Sequence, Tuple, Optional, Union, Dict

from flax import linen as nn
from flax.core.frozen_dict import freeze

import jax
from jax import random, jit, vmap
import jax.numpy as jnp
from jax.nn.initializers import glorot_normal, normal, zeros, constant

activation_fn = {
    "relu": nn.relu,
    "gelu": nn.gelu,
    "swish": nn.swish,
    "sigmoid": nn.sigmoid,
    "tanh": jnp.tanh,
    "sin": jnp.sin,
}


def _get_activation(str):
    if str in activation_fn:
        return activation_fn[str]

    else:
        raise NotImplementedError(f"Activation {str} not supported yet!")


def _weight_fact(init_fn, mean, stddev):
    def init(key, shape):
        key1, key2 = random.split(key)
        w = init_fn(key1, shape)
        g = mean + normal(stddev)(key2, (shape[-1],))
        g = jnp.exp(g)
        v = w / (g)
        return g, v

    return init


class PeriodEmbs(nn.Module):
    period: Tuple[float]  # Periods for different axes
    axis: Tuple[int]  # Axes where the period embeddings are to be applied
    trainable: Tuple[
        bool
    ]  # Specifies whether the period for each axis is trainable or not

    def setup(self):
        # Initialize period parameters as trainable or constant and store them in a flax frozen dict
        period_params = {}
        for idx, is_trainable in enumerate(self.trainable):
            if is_trainable:
                period_params[f"period_{idx}"] = self.param(
                    f"period_{idx}", constant(self.period[idx]), ()
                )
            else:
                period_params[f"period_{idx}"] = self.period[idx]

        self.period_params = freeze(period_params)

    @nn.compact
    def __call__(self, x):
        """
        Apply the period embeddings to the specified axes.
        """
        y = []

        for i, xi in enumerate(x):
            if i in self.axis:
                idx = self.axis.index(i)
                period = self.period_params[f"period_{idx}"]
                y.extend([jnp.cos(period * xi), jnp.sin(period * xi)])
            else:
                y.append(xi)

        return jnp.hstack(y)


class FourierEmbs(nn.Module):
    embed_scale: float
    embed_dim: int

    @nn.compact
    def __call__(self, x):
        kernel = self.param(
            "kernel", normal(self.embed_scale), (x.shape[-1], self.embed_dim // 2)
        )
        y = jnp.concatenate(
            [jnp.cos(jnp.dot(x, kernel)), jnp.sin(jnp.dot(x, kernel))], axis=-1
        )
        return y


class Dense(nn.Module):
    features: int
    kernel_init: Callable = glorot_normal()
    bias_init: Callable = zeros
    reparam: Union[None, Dict] = None

    @nn.compact
    def __call__(self, x):
        if self.reparam is None:
            kernel = self.param(
                "kernel", self.kernel_init, (x.shape[-1], self.features)
            )

        elif self.reparam["type"] == "weight_fact":
            g, v = self.param(
                "kernel",
                _weight_fact(
                    self.kernel_init,
                    mean=self.reparam["mean"],
                    stddev=self.reparam["stddev"],
                ),
                (x.shape[-1], self.features),
            )
            kernel = g * v

        bias = self.param("bias", self.bias_init, (self.features,))

        y = jnp.dot(x, kernel) + bias

        return y


class Conv1D(nn.Module):
    features: int
    kernel_size: int = 1
    kernel_init: Callable = glorot_normal()
    bias_init: Callable = zeros
    reparam: Union[None, Dict] = None

    @nn.compact
    def __call__(self, x):
        """
        x: Input tensor of shape (batch_size, seq_length, in_channels)
        """
        # Reshape for efficient batch processing (optional if batch_size=1)
        # x = x.reshape(-1, x.shape[1], x.shape[2])
        x = x[jnp.newaxis, :, jnp.newaxis]
        kernel_shape = (1,1,1) #(self.features, x.shape[0], 1)
        if self.reparam is None:
            kernel = self.param("kernel", self.kernel_init, kernel_shape)
        elif self.reparam["type"] == "weight_fact":
            g, v = self.param(
                "kernel",
                _weight_fact(
                    self.kernel_init,
                    mean=self.reparam["mean"],
                    stddev=self.reparam["stddev"],
                ),
                (1,1,1),
            )
            kernel = g * v 
            
        bias = self.param("bias", self.bias_init, (self.features,))
        
        # # Apply 1D convolution
        y = jax.lax.conv_general_dilated(
            x,                      # Input tensor
            kernel,                 # Convolution kernel
            window_strides=(1,),    # Stride of 1
            padding="SAME",         # Zero-padding to keep output the same size as input
            dimension_numbers=("NWC", "WIO", "NWC"),  # Input format, kernel format, output format
        )

        # x = x[jnp.newaxis, :, jnp.newaxis]
        # y = 
        y = jnp.squeeze(y)
        y += bias  # Add bias
        return y

# TODO: Make it more general, e.g. imposing periodicity for the given axis

class ResNet(nn.Module):
    arch_name: Optional[str] = "ResNet"
    num_layers: int = 4
    hidden_dim: int = 128
    out_dim: int = 1
    activation: str = "tanh"
    periodicity: Union[None, Dict] = None
    fourier_emb: Union[None, Dict] = None
    reparam: Union[None, Dict] = None

    def setup(self):
        self.activation_fn = _get_activation(self.activation)
    @nn.compact
    def __call__(self, x):
        if self.periodicity:
            x = PeriodEmbs(**self.periodicity)(x)

        if self.fourier_emb:
            x = FourierEmbs(**self.fourier_emb)(x)
            
        x_previous = x
        for _ in range(self.num_layers):

            x = Dense(features=self.hidden_dim, reparam=self.reparam)(x_previous)
            x = self.activation_fn(x)
            x_previous = jnp.concatenate(
                     [x, x_previous], axis=-1 )

        x = Dense(features=self.out_dim, reparam=self.reparam)(x)
        return x

class Mlp(nn.Module):
    
    arch_name: Optional[str] = "Mlp"
    num_layers: int = 4
    hidden_dim: int = 256
    out_dim: int = 1
    activation: str = "tanh"
    periodicity: Union[None, Dict] = None
    fourier_emb: Union[None, Dict] = None
    reparam: Union[None, Dict] = None

    def setup(self):
        self.activation_fn = _get_activation(self.activation)

    @nn.compact
    def __call__(self, x):
        if self.periodicity:
            x = PeriodEmbs(**self.periodicity)(x)

        if self.fourier_emb:
            x = FourierEmbs(**self.fourier_emb)(x)

        for _ in range(self.num_layers):
            x = Dense(features=self.hidden_dim, reparam=self.reparam)(x)
            x = self.activation_fn(x)

        x = Dense(features=self.out_dim, reparam=self.reparam)(x)
        return x
    
class SharedMlp(nn.Module):
        
    arch_name: Optional[str] = "SharedMlp"
    num_layers: int = 4
    hidden_dim: int = 256
    out_dim: int = 1
    activation: str = "tanh"
    periodicity: Union[None, Dict] = None
    fourier_emb: Union[None, Dict] = None
    reparam: Union[None, Dict] = None

    def setup(self):
        self.activation_fn = _get_activation(self.activation)

    @nn.compact
    def __call__(self, x):
        
        if self.periodicity:
            x = PeriodEmbs(**self.periodicity)(x)

        if self.fourier_emb:
            x = FourierEmbs(**self.fourier_emb)(x)
            
        for _ in range(self.num_layers):
            x = Conv1D(features=self.hidden_dim, reparam=self.reparam)(x)
            x = self.activation_fn(x)
            
        x = Conv1D(features=self.out_dim, reparam=self.reparam)(x)
        
        return x
    

class ModifiedMlp(nn.Module):
    arch_name: Optional[str] = "ModifiedMlp"
    num_layers: int = 4
    hidden_dim: int = 256
    out_dim: int = 1
    activation: str = "tanh"
    periodicity: Union[None, Dict] = None
    fourier_emb: Union[None, Dict] = None
    reparam: Union[None, Dict] = None

    def setup(self):
        self.activation_fn = _get_activation(self.activation)

    @nn.compact
    def __call__(self, x):
        if self.periodicity:
            x = PeriodEmbs(**self.periodicity)(x)

        if self.fourier_emb:
            x = FourierEmbs(**self.fourier_emb)(x)

        u = Dense(features=self.hidden_dim, reparam=self.reparam)(x)
        v = Dense(features=self.hidden_dim, reparam=self.reparam)(x)

        u = self.activation_fn(u)
        v = self.activation_fn(v)

        for _ in range(self.num_layers):
            x = Dense(features=self.hidden_dim, reparam=self.reparam)(x)
            x = self.activation_fn(x)
            x = x * u + (1 - x) * v

        x = Dense(features=self.out_dim, reparam=self.reparam)(x)
        return x
    
class ModifiedMlp(nn.Module):
    arch_name: Optional[str] = "ModSharedMlp"
    num_layers: int = 4
    hidden_dim: int = 256
    out_dim: int = 1
    activation: str = "tanh"
    periodicity: Union[None, Dict] = None
    fourier_emb: Union[None, Dict] = None
    reparam: Union[None, Dict] = None

    def setup(self):
        self.activation_fn = _get_activation(self.activation)

    @nn.compact
    def __call__(self, x):
        if self.periodicity:
            x = PeriodEmbs(**self.periodicity)(x)

        if self.fourier_emb:
            x = FourierEmbs(**self.fourier_emb)(x)

        u = Dense(features=self.hidden_dim, reparam=self.reparam)(x)
        v = Dense(features=self.hidden_dim, reparam=self.reparam)(x)

        u = self.activation_fn(u)
        v = self.activation_fn(v)

        for _ in range(self.num_layers):
            x = Dense(features=self.hidden_dim, reparam=self.reparam)(x)
            x = self.activation_fn(x)
            x = x * u + (1 - x) * v

        x = Dense(features=self.out_dim, reparam=self.reparam)(x)
        return x

class ModifiedResNet(nn.Module):
    arch_name: Optional[str] = "ModifiedResNet"
    num_layers: int = 4
    hidden_dim: int = 256
    out_dim: int = 1
    activation: str = "tanh"
    periodicity: Union[None, Dict] = None
    fourier_emb: Union[None, Dict] = None
    reparam: Union[None, Dict] = None

    def setup(self):
        self.activation_fn = _get_activation(self.activation)

    @nn.compact
    def __call__(self, x):
        if self.periodicity:
            x = PeriodEmbs(**self.periodicity)(x)

        if self.fourier_emb:
            x = FourierEmbs(**self.fourier_emb)(x)

        u = Dense(features=self.hidden_dim, reparam=self.reparam)(x)
        v = Dense(features=self.hidden_dim, reparam=self.reparam)(x)

        u = self.activation_fn(u)
        v = self.activation_fn(v)

        x_previous = x
        for _ in range(self.num_layers):
            x = Dense(features=self.hidden_dim, reparam=self.reparam)(x_previous)
            x = self.activation_fn(x)
            x = x * u + (1 - x) * v
            x_previous = jnp.concatenate(
                     [x, x_previous], axis=-1 )

        x = Dense(features=self.out_dim, reparam=self.reparam)(x)
        return x


class MlpBlock(nn.Module):
    num_layers: int
    hidden_dim: int
    out_dim: int
    activation: str
    reparam: Union[None, Dict]
    final_activation: bool

    def setup(self):
        self.activation_fn = _get_activation(self.activation)

    @nn.compact
    def __call__(self, x):
        for _ in range(self.num_layers):
            x = Dense(features=self.hidden_dim, reparam=self.reparam)(x)
            x = self.activation_fn(x)

        x = Dense(features=self.out_dim, reparam=self.reparam)(x)
        if self.final_activation:
            x = self.activation_fn(x)

        return x


class DeepONet(nn.Module):
    arch_name: Optional[str] = "DeepONet"
    num_branch_layers: int = 4
    num_trunk_layers: int = 4 # (u, x) : u é o branch, x é o trunk
    hidden_dim: int = 256
    out_dim: int = 1
    activation: str = "tanh"
    periodicity: Union[None, Dict] = None
    fourier_emb: Union[None, Dict] = None
    reparam: Union[None, Dict] = None

    def setup(self):
        self.activation_fn = _get_activation(self.activation)

    @nn.compact
    def __call__(self, u, x):
        u = ModifiedMlp(#MlpBlock(
            num_layers=self.num_branch_layers,
            hidden_dim=self.hidden_dim,
            out_dim=self.hidden_dim,
            activation=self.activation,
            #final_activation=False,
            reparam=self.reparam,
            periodicity=self.periodicity,
            fourier_emb=self.fourier_emb,
        )(u)

        x = ModifiedMlp(#Mlp(
            num_layers=self.num_trunk_layers,
            hidden_dim=self.hidden_dim,
            out_dim=self.hidden_dim,
            activation=self.activation,
            periodicity=self.periodicity,
            fourier_emb=self.fourier_emb,
            reparam=self.reparam,
        )(x)

        y = u * x #nn.sigmoid(x)
        y = self.activation_fn(y)
        y = Dense(features=self.out_dim, reparam=self.reparam)(y)
        return y
    
class DeepONetwD(nn.Module):
    arch_name: Optional[str] = "DeepONetwD"
    num_branch_layers: int = 4
    num_trunk_layers: int = 4 # (u, x) : u é o branch, x é o trunk
    hidden_dim: int = 256
    out_dim: int = 1
    activation: str = "tanh"
    periodicity: Union[None, Dict] = None
    fourier_emb: Union[None, Dict] = None
    reparam: Union[None, Dict] = None

    def setup(self):
        self.activation_fn = _get_activation(self.activation)

    @nn.compact
    def __call__(self, u, x, pde_param):
        # (u, x) : u é o branch, x é o trunk
        #  u: x y t mu - branch
        #  x: mu       - trunk
        pde_param = Dense(features=1)(pde_param)         
        
        u = ModifiedMlp(#MlpBlock(
            num_layers=self.num_branch_layers,
            hidden_dim=self.hidden_dim,
            out_dim=self.hidden_dim,
            activation=self.activation,
            #final_activation=False,
            reparam=self.reparam,
            periodicity=self.periodicity,
            fourier_emb=self.fourier_emb,
        )(u)

        x = ModifiedMlp(#Mlp(
            num_layers=self.num_trunk_layers,
            hidden_dim=self.hidden_dim,
            out_dim=self.hidden_dim,
            activation=self.activation,
            periodicity=self.periodicity,
            fourier_emb=self.fourier_emb,
            reparam=self.reparam,
        )(x)
        # x = nn.softplus(x)
               
        y = u * x 
        
        y1 = self.activation_fn(y)
        y1 = Dense(features=self.out_dim-1, reparam=self.reparam)(y1)
        
        y2 = nn.sigmoid(y)
        uv_scale = Dense(features=2, reparam=self.reparam)(y2)
        y2 = Dense(features=1, reparam=self.reparam)(y2)
        
        y = jnp.concatenate( [y1, y2, pde_param, uv_scale], axis=-1 )
        return y

class DeepONetwDssep(nn.Module):
    arch_name: Optional[str] = "DeepONetwDssep"
    num_branch_layers: int = 4
    num_trunk_layers: int = 4 # (u, x) : u é o branch, x é o trunk
    hidden_dim: int = 256
    out_dim: int = 1
    activation: str = "tanh"
    periodicity: Union[None, Dict] = None
    fourier_emb: Union[None, Dict] = None
    reparam: Union[None, Dict] = None

    def setup(self):
        self.activation_fn = _get_activation(self.activation)

    @nn.compact
    def __call__(self, u, x, pde_param):
        # (u, x) : u é o branch, x é o trunk
        #  u: x y t - branch
        #  x: mu       - trunk
        pde_param = Dense(features=1)(pde_param)
        
        s = ModifiedMlp(#MlpBlock(
            num_layers=self.num_branch_layers,
            hidden_dim=self.hidden_dim,
            out_dim=1,
            activation="sigmoid",
            #final_activation=False,
            reparam=self.reparam,
            periodicity=self.periodicity,
            fourier_emb=self.fourier_emb,
        )(u)
                
        u = ModifiedMlp(#MlpBlock(
            num_layers=self.num_branch_layers,
            hidden_dim=self.hidden_dim,
            out_dim=self.hidden_dim,
            activation=self.activation,
            #final_activation=False,
            reparam=self.reparam,
            periodicity=self.periodicity,
            fourier_emb=self.fourier_emb,
        )(u)

        x = ModifiedMlp(#Mlp(
            num_layers=self.num_trunk_layers,
            hidden_dim=self.hidden_dim,
            out_dim=self.hidden_dim,
            activation="sigmoid",
            periodicity=self.periodicity,
            fourier_emb=self.fourier_emb,
            reparam=self.reparam,
        )(x)
        # x = nn.sigmoid( x )
               
        s = nn.sigmoid( s * x )
        s = Dense(features=1, reparam=self.reparam)(s)
        
        y = u * x 
        y1 = self.activation_fn(y)
        y1 = Dense(features=self.out_dim-1, reparam=self.reparam)(y1)  
        uv_scale = Dense(features=2, reparam=self.reparam)( nn.sigmoid(y) )
             
        y = jnp.concatenate( [y1, s, pde_param, uv_scale], axis=-1 )
        return y

class DeepONet3wD(nn.Module):
    arch_name: Optional[str] = "DeepONet3wD"
    num_branch_layers: int = 4
    # num_branch_layers2: int = 4
    num_trunk_layers: int = 4 # (u, x) : u é o branch, x é o trunk
    hidden_dim: int = 256
    out_dim: int = 1
    activation: str = "tanh"
    periodicity: Union[None, Dict] = None
    fourier_emb: Union[None, Dict] = None
    reparam: Union[None, Dict] = None

    def setup(self):
        self.activation_fn = _get_activation(self.activation)

    @nn.compact
    def __call__(self, u1, u2, x, pde_param):
        # (u, x) : u é o branch, x é o trunk
        #  u1: t              - branch1
        #  u2: x, y, v(x,y)   - branch2
        #  x: mu              - trunk
        pde_param = Dense(features=1)(pde_param)   
        
        u1 = jnp.concatenate([u1, u2])
      
        
        u1 = ModifiedMlp(#MlpBlock(
            num_layers=self.num_branch_layers,
            hidden_dim=self.hidden_dim,
            out_dim=self.hidden_dim,
            activation=self.activation,
            #final_activation=False,
            reparam=self.reparam,
            periodicity=self.periodicity,
            fourier_emb=self.fourier_emb,
        )(u1)
        
        # u2 = ModifiedMlp(#Mlp(
        #     num_layers=self.num_branch_layers2,
        #     hidden_dim=self.hidden_dim,
        #     out_dim=self.hidden_dim,
        #     activation=self.activation,
        #     periodicity=self.periodicity,
        #     fourier_emb=self.fourier_emb,
        #     reparam=self.reparam,
        # )(u2)

        x = ModifiedMlp(#Mlp(
            num_layers=self.num_trunk_layers,
            hidden_dim=self.hidden_dim,
            out_dim=self.hidden_dim,
            activation=self.activation,
            periodicity=self.periodicity,
            fourier_emb=self.fourier_emb,
            reparam=self.reparam,
        )(x)
   
        # u2 = jnp.squeeze(u2)
        # u1 = u1 * u2
        u1 = self.activation_fn(u1)
        y = u1 * x
        
        y_p = self.activation_fn(y)
        y_u = self.activation_fn(Dense(features=int(self.hidden_dim/self.out_dim), reparam=self.reparam)(y_p))
        y_v = self.activation_fn(Dense(features=int(self.hidden_dim/self.out_dim), reparam=self.reparam)(y_p))
        y_p = self.activation_fn(Dense(features=int(self.hidden_dim/self.out_dim), reparam=self.reparam)(y_p))
        y_u = Dense(features=1, reparam=self.reparam)(y_u)
        y_v = Dense(features=1, reparam=self.reparam)(y_v)
        y_p = Dense(features=1, reparam=self.reparam)(y_p)
        
        y_s = nn.sigmoid(y)
        y_s = nn.sigmoid(Dense(features=int(self.hidden_dim/self.out_dim), reparam=self.reparam)(y_s))
        y_s = Dense(features=1, reparam=self.reparam)(y_s)
        
        y = jnp.concatenate( [y_u, y_v, y_p, y_s, pde_param], axis=-1 )
        return y
    
 
class DeepOResNet(nn.Module):
    arch_name: Optional[str] = "DeepOResNet"
    num_branch_layers: int = 4
    num_trunk_layers: int = 4 # (u, x) : u é o branch, x é o trunk
    hidden_dim: int = 256
    out_dim: int = 1
    activation: str = "tanh"
    periodicity: Union[None, Dict] = None
    fourier_emb: Union[None, Dict] = None
    reparam: Union[None, Dict] = None

    def setup(self):
        self.activation_fn = _get_activation(self.activation)

    @nn.compact
    def __call__(self, u, x):
        u = ModifiedResNet(#MlpBlock(
            num_layers=self.num_branch_layers,
            hidden_dim=self.hidden_dim,
            out_dim=  self.num_branch_layers*self.hidden_dim,
            activation=self.activation,
            #final_activation=False,
            reparam=self.reparam,
            
            periodicity=self.periodicity,
            fourier_emb=self.fourier_emb,
        )(u)

        x = ModifiedResNet(#Mlp(
            num_layers=self.num_trunk_layers,
            hidden_dim=self.hidden_dim,
            out_dim=  self.num_branch_layers*self.hidden_dim,
            activation=self.activation,
            periodicity=self.periodicity,
            fourier_emb=self.fourier_emb,
            reparam=self.reparam,
        )(x)

        y  = u * x
        y = self.activation_fn(y)

        y = Dense(features=self.out_dim, reparam=self.reparam)(y)
        return y

