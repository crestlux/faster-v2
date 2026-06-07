from faster.networks.resnet import ImageStateEncoder, get_resnet18
from faster.networks.ensemble import Ensemble
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
    """Q-network head that operates on pre-encoded flat observations.

    Observations must be a flat array (e.g. output of SharedEncoderEnsembleCritic's
    single shared encoder).  Dict inputs with an "image" key are not supported here —
    use SharedEncoderEnsembleCritic to handle the image encoder outside the ensemble.
    """
    base_cls: nn.Module

    @nn.compact
    def __call__(self, observations, actions: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        if isinstance(observations, dict):
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
    """Encoder-head that operates on pre-encoded flat observations (returns features, not scalar).

    Like StateActionValue but returns the hidden representation rather than a Q scalar.
    Used for the edit actor, which always receives flat actor-encoded features.
    """
    base_cls: nn.Module

    @nn.compact
    def __call__(self, observations, actions: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        if isinstance(observations, dict):
            obs_encoded = observations["state"]
        else:
            obs_encoded = observations

        if actions.shape[0] != obs_encoded.shape[0]:
            if actions.shape[0] % obs_encoded.shape[0] == 0:
                repeat_factor = actions.shape[0] // obs_encoded.shape[0]
                obs_encoded = jnp.repeat(obs_encoded, repeat_factor, axis=0)

        inputs = jnp.concatenate([obs_encoded, actions], axis=-1)
        return self.base_cls()(inputs, *args, **kwargs)


class SharedEncoderEnsembleCritic(nn.Module):
    """Single shared image encoder feeding an ensemble of Q heads (RedQ-style).

    One encoder (TD-trained, named ``ImageStateEncoder_0``) sits outside the vmap so
    it is used consistently at training, Bellman-target, and inference time.
    The Q heads (under ``Ensemble_0``) receive the encoder's flat output.

    ``subsample_ensemble`` only touches ``Ensemble_0`` and leaves the shared encoder
    intact, so RedQ's random min-of-k target works without duplicating the encoder.

    Contrast with the old ``Ensemble(StateActionValue(...))`` pattern, which gave
    every one of the num_qs heads its own ResNet (10× compute) that was then bypassed
    at inference, causing a train/inference representation mismatch.
    """
    encoder_cls: type
    net_cls: type
    num_qs: int

    @nn.compact
    def __call__(self, observations, actions, *args):
        feat = self.encoder_cls()(observations)
        return Ensemble(self.net_cls, num=self.num_qs)(feat, actions, *args)
