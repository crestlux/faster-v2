from faster.networks.resnet import ImageStateEncoder, get_resnet18
from typing import Optional, Type
import flax.linen as nn
import jax.numpy as jnp

from faster.networks.mlp import default_init

class StateValue(nn.Module):
    base_cls: nn.Module
    obs_encoder_cls: Optional[Type[nn.Module]] = None

    @nn.compact
    def __call__(self, observations, *args, **kwargs) -> jnp.ndarray:
        if self.obs_encoder_cls is not None:
            obs_encoded = self.obs_encoder_cls()(observations, training=kwargs.get("training", False))
        elif isinstance(observations, dict) and "image" in observations:
            obs_encoded = ImageStateEncoder(encoder_cls=get_resnet18)(observations, training=kwargs.get("training", False))
        elif isinstance(observations, dict):
            obs_encoded = observations["state"]
        else:
            obs_encoded = observations
            
        outputs = self.base_cls()(obs_encoded, *args, **kwargs)
        value = nn.Dense(1, kernel_init=default_init())(outputs)
        return jnp.squeeze(value, -1)


class StateActionValue(nn.Module):
    base_cls: nn.Module
    obs_encoder_cls: Optional[Type[nn.Module]] = None

    @nn.compact
    def __call__(self, observations, actions: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        if self.obs_encoder_cls is not None:
            obs_encoded = self.obs_encoder_cls()(observations, training=kwargs.get("training", False))
        elif isinstance(observations, dict) and "image" in observations:
            obs_encoded = ImageStateEncoder(encoder_cls=get_resnet18)(observations, training=kwargs.get("training", False))
        elif isinstance(observations, dict):
            obs_encoded = observations["state"]
        else:
            obs_encoded = observations
            
        if actions.shape[0] != obs_encoded.shape[0]:
            if actions.shape[0] % obs_encoded.shape[0] == 0:
                repeat_factor = actions.shape[0] // obs_encoded.shape[0]
                obs_encoded = jnp.repeat(obs_encoded, repeat_factor, axis=0)

        inputs = jnp.concatenate([obs_encoded, actions], axis=-1)
        outputs = self.base_cls()(inputs, *args, **kwargs)

        value = nn.Dense(1, kernel_init=default_init())(outputs)

        return jnp.squeeze(value, -1)

class StateActionEncoder(nn.Module):
    base_cls: nn.Module
    obs_encoder_cls: Optional[Type[nn.Module]] = None

    @nn.compact
    def __call__(self, observations, actions: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        if self.obs_encoder_cls is not None:
            obs_encoded = self.obs_encoder_cls()(observations, training=kwargs.get("training", False))
        elif isinstance(observations, dict) and "image" in observations:
            obs_encoded = ImageStateEncoder(encoder_cls=get_resnet18)(observations, training=kwargs.get("training", False))
        elif isinstance(observations, dict):
            obs_encoded = observations["state"]
        else:
            obs_encoded = observations
            
        if actions.shape[0] != obs_encoded.shape[0]:
            if actions.shape[0] % obs_encoded.shape[0] == 0:
                repeat_factor = actions.shape[0] // obs_encoded.shape[0]
                obs_encoded = jnp.repeat(obs_encoded, repeat_factor, axis=0)

        inputs = jnp.concatenate([obs_encoded, actions], axis=-1)
        return self.base_cls()(inputs, *args, **kwargs)
