from typing import Optional, Type

import flax.linen as nn
import jax.numpy as jnp

from faster.networks.ensemble import Ensemble
from faster.networks.mlp import default_init
from faster.networks.resnet import ImageStateEncoder, get_resnet18


class StateValue(nn.Module):
    base_cls: nn.Module

    @nn.compact
    def __call__(self, observations: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        outputs = self.base_cls()(observations, *args, **kwargs)
        value = nn.Dense(1, kernel_init=default_init())(outputs)
        return jnp.squeeze(value, -1)


class StateActionValue(nn.Module):
    """Q-network head that operates on pre-encoded flat observations.

    Observations must be a flat array (e.g. the output of SharedEncoderEnsembleCritic's
    single shared encoder, or actor-encoded features). The image encoder lives OUTSIDE
    this head (in SharedEncoderEnsembleCritic) so the num_qs ensemble does not duplicate
    a ResNet per head. The repeat_factor branch lets a single (B, d) observation score
    multiple (B*K, a) candidate actions without materialising the broadcast upstream.
    """

    base_cls: nn.Module

    @nn.compact
    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
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
    """Like StateActionValue but returns the hidden representation rather than a Q scalar.

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

    One encoder (TD-trained, named ``ImageStateEncoder_0``) sits outside the vmap so it
    is used consistently at training, Bellman-target, and inference time. The Q heads
    (under ``Ensemble_0``) receive the encoder's flat output.

    ``subsample_ensemble`` only touches ``Ensemble_0`` and leaves the shared encoder
    intact, so RedQ's random min-of-k target works without duplicating the encoder.

    For low-dim (non-image) observations the ImageStateEncoder passes the flat array
    through unchanged, so this wrapper is a no-op there.
    """

    encoder_cls: type
    net_cls: type
    num_qs: int

    @nn.compact
    def __call__(self, observations, actions, *args):
        feat = self.encoder_cls()(observations)
        return Ensemble(self.net_cls, num=self.num_qs)(feat, actions, *args)
