from functools import partial
from typing import Optional, Sequence

import flax
import flax.linen as nn
import gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct
from flax.training.train_state import TrainState
from ml_collections import config_dict

from faster.agents.agent import Agent
from faster.agents.temperature import Temperature
from faster.data.dataset import DatasetDict
from faster.distributions import TanhNormal
from faster.networks.resnet import ImageStateEncoder, get_resnet18, augment_obs_batch
from faster.networks import (
    DDPM,
    MLP,
    DiffusionMLP,
    DiffusionMLPResNet,
    Ensemble,
    FourierFeatures,
    MLPResNetV2,
    SharedEncoderEnsembleCritic,
    StateActionValue,
    StateActionEncoder,
    cosine_beta_schedule,
    ddim_sampler,
    subsample_ensemble,
    vp_beta_schedule,
)


def decay_mask_fn(params):
    flat_params = flax.traverse_util.flatten_dict(params)
    flat_mask = {path: path[-1] != "bias" for path in flat_params}
    return flax.core.FrozenDict(flax.traverse_util.unflatten_dict(flat_mask))


def _load_pretrained_resnet18_conv_weights(params, in_channels=3):
    """Replace conv kernels in any img_encoder within params with ImageNet pretrained weights.

    GroupNorm params are left at their random init values (they adapt during fine-tuning).
    For in_channels != 3: first conv weights are adapted by averaging over the 3 input
    channels and tiling, preserving the pretrained filter magnitudes.
    """
    try:
        import torch
        from torchvision.models import resnet18, ResNet18_Weights
        pt = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).state_dict()
    except Exception as e:
        print(f"[pretrained_encoder] skipping: {e}")
        return params

    def pt_conv(t):
        """PyTorch (out,in,H,W) → JAX/Flax (H,W,in,out)."""
        return jnp.array(t.detach().cpu().float().numpy().transpose(2, 3, 1, 0))

    def adapt_first_conv(t, in_ch):
        """Average 3-ch pretrained weights across channels, tile to in_ch."""
        w = t.detach().cpu().float().numpy()  # (64, 3, 7, 7)
        w_avg = w.mean(axis=1, keepdims=True)  # (64, 1, 7, 7)
        w_tiled = np.tile(w_avg, (1, in_ch, 1, 1))  # (64, in_ch, 7, 7)
        return jnp.array(w_tiled.transpose(2, 3, 1, 0))  # (7, 7, in_ch, 64)

    # Maps PyTorch key → suffix path within "img_encoder/..."
    MAPPING = {
        "conv1.weight":                 "img_encoder/conv_init/kernel",
        "layer1.0.conv1.weight":        "img_encoder/ResNetBlock_0/Conv_0/kernel",
        "layer1.0.conv2.weight":        "img_encoder/ResNetBlock_0/Conv_1/kernel",
        "layer1.1.conv1.weight":        "img_encoder/ResNetBlock_1/Conv_0/kernel",
        "layer1.1.conv2.weight":        "img_encoder/ResNetBlock_1/Conv_1/kernel",
        "layer2.0.conv1.weight":        "img_encoder/ResNetBlock_2/Conv_0/kernel",
        "layer2.0.downsample.0.weight": "img_encoder/ResNetBlock_2/conv_proj/kernel",
        "layer2.0.conv2.weight":        "img_encoder/ResNetBlock_2/Conv_1/kernel",
        "layer2.1.conv1.weight":        "img_encoder/ResNetBlock_3/Conv_0/kernel",
        "layer2.1.conv2.weight":        "img_encoder/ResNetBlock_3/Conv_1/kernel",
        "layer3.0.conv1.weight":        "img_encoder/ResNetBlock_4/Conv_0/kernel",
        "layer3.0.downsample.0.weight": "img_encoder/ResNetBlock_4/conv_proj/kernel",
        "layer3.0.conv2.weight":        "img_encoder/ResNetBlock_4/Conv_1/kernel",
        "layer3.1.conv1.weight":        "img_encoder/ResNetBlock_5/Conv_0/kernel",
        "layer3.1.conv2.weight":        "img_encoder/ResNetBlock_5/Conv_1/kernel",
        "layer4.0.conv1.weight":        "img_encoder/ResNetBlock_6/Conv_0/kernel",
        "layer4.0.downsample.0.weight": "img_encoder/ResNetBlock_6/conv_proj/kernel",
        "layer4.0.conv2.weight":        "img_encoder/ResNetBlock_6/Conv_1/kernel",
        "layer4.1.conv1.weight":        "img_encoder/ResNetBlock_7/Conv_0/kernel",
        "layer4.1.conv2.weight":        "img_encoder/ResNetBlock_7/Conv_1/kernel",
    }

    params_dict = flax.core.unfreeze(params) if isinstance(params, flax.core.FrozenDict) else dict(params)
    flat = flax.traverse_util.flatten_dict(params_dict, sep="/")

    loaded = 0
    for pt_key, suffix in MAPPING.items():
        if pt_key not in pt:
            continue
        for flat_key in list(flat.keys()):
            if flat_key.endswith(suffix):
                if pt_key == "conv1.weight":
                    flat[flat_key] = adapt_first_conv(pt[pt_key], in_channels)
                else:
                    flat[flat_key] = pt_conv(pt[pt_key])
                loaded += 1

    print(f"[pretrained_encoder] loaded {loaded}/{len(MAPPING)} conv layers from ImageNet ResNet18")
    return flax.traverse_util.unflatten_dict(flat, sep="/")


def _make_split_encoder_tx(params, base_tx, encoder_lr=1e-5):
    """Wrap base_tx so ImageStateEncoder_0 params use a fixed smaller encoder_lr."""
    try:
        params_dict = flax.core.unfreeze(params) if isinstance(params, flax.core.FrozenDict) else dict(params)
        flat = flax.traverse_util.flatten_dict(params_dict, sep="/")
        labels = {k: ("encoder" if "ImageStateEncoder_0" in k else "model") for k in flat}
        label_tree = flax.traverse_util.unflatten_dict(labels, sep="/")
        return optax.multi_transform(
            transforms={"encoder": optax.adamw(encoder_lr), "model": base_tx},
            param_labels=label_tree,
        )
    except Exception as e:
        print(f"[split_encoder_tx] fallback to base_tx: {e}")
        return base_tx


@partial(jax.jit, static_argnames=("critic_fn",))
def compute_q(critic_fn, critic_params, observations, actions):
    q_values = critic_fn({"params": critic_params}, observations, actions)
    return q_values.min(axis=0)


@partial(jax.jit, static_argnames=("apply_fn",))
def _sample_actions(rng, apply_fn, params, *args) -> np.ndarray:
    key, rng = jax.random.split(rng)
    dist = apply_fn({"params": params}, *args)
    return dist.sample(seed=key), rng


@jax.jit
def _sample_actions_jit(agent, observations):
    rng = agent.rng
    observations = jax.tree_map(lambda x: jnp.squeeze(x), observations)
    observations = jax.tree_map(lambda x: jnp.expand_dims(x, axis=0) if x.ndim == 1 or x.ndim == 3 else x, observations)

    # Always compute actor encoding for actor scan efficiency (avoids T×ResNet18 per step).
    # Use the ONLINE actor (not target) for action generation: the critic/filter are trained
    # to score candidates produced by self.actor's denoising (see update/_sample_candidates),
    # and with actor_tau=0.001 the target actor lags far behind during fine-tuning. Acting with
    # the target here makes eval/rollout both stale and inconsistent with training -> ~0% success
    # even when the trained policy works (verified: lift current-actor 60% vs target 0% @1250).
    actor_obs_enc = agent._get_obs_encoding(observations, agent.actor.params)

    actor_params = agent.actor.params
    # obs_for_filter: raw obs needed so actor's denoising scan has the dict for obs_flat;
    #   filter scoring uses actor_obs_enc_rep inside ddim_sampler_hidden_filter (Fix 1).
    # obs_for_critic: the outer critic owns its encoder, so it gets the RAW obs and encodes
    #   internally — exactly as it is trained (share_encoder=False). Only when share_encoder=True
    #   (critic has no own encoder) does it receive the flat actor features.
    obs_for_filter = actor_obs_enc if agent.share_encoder else observations
    obs_for_critic = actor_obs_enc if agent.share_encoder else observations

    if agent.filter_enabled and agent.filter_at_eval:
        actions, _, rng = agent._sample_candidates(rng, obs_for_filter, agent.N, actor_params, agent.filter_temperature_train, actor_obs_enc=actor_obs_enc, k_keep=min(agent.n_base_deploy, agent.N))
    else:
        actions, rng = ddim_sampler(
            agent.actor.apply_fn,
            actor_params,
            agent.T,
            rng,
            agent.action_dim,
            actor_obs_enc,  # flat features → actor skips internal encoder
            agent.alphas,
            agent.alpha_hats,
            agent.betas,
            agent.M,
            eta=agent.ddim_eta,
        )
        actions = actions.reshape(1, agent.N, -1)

    diffusion_actions = actions.squeeze(axis=0)  # (N, action_dim)

    if diffusion_actions.shape[0] > 1 or agent.ne_samples > 0:
        key, rng = jax.random.split(rng)
        target_params = subsample_ensemble(key, agent.target_critic.params, agent.num_min_qs, agent.num_qs)
        all_actions = diffusion_actions  # (N, action_dim)

        if agent.ne_samples > 0:
            key, rng = jax.random.split(rng)
            # Tile base candidates to fill ne_samples edit slots.
            # When filter leaves 1 candidate, tile it ne_samples times and generate
            # independent stochastic edits (TanhNormal sample differs each call).
            n_base = diffusion_actions.shape[0]  # static Python int in JIT
            if n_base >= agent.ne_samples:
                edit_base = diffusion_actions[: agent.ne_samples]
            else:
                tile_factor = (agent.ne_samples + n_base - 1) // n_base
                edit_base = jnp.tile(diffusion_actions, (tile_factor, 1))[: agent.ne_samples]
            d_exec = edit_base[:, : agent.critic_action_dim]  # (ne, exec_dim)
            # Use actor_obs_enc: consistent with update_edit_actor training path.
            # StateActionEncoder handles (1, enc_dim) → repeats to match (ne, enc_dim).
            # EXPO/EXPO-FT: edits are STOCHASTIC samples (one per base candidate); the top-Q argmax
            # below selects among the 8 base + 8 edit candidates (a* = argmax_a Q). The critic must
            # be calibrated over edited actions (it is, online: Eq.5 backs up from a*).
            r_exec, rng = _sample_actions(key, agent.edit_actor.apply_fn, agent.edit_actor.params, actor_obs_enc, d_exec)
            r_exec = agent._apply_residual_action_mask(r_exec * agent.r_action_scale) + d_exec
            r_exec = jnp.clip(r_exec, -1.0, 1.0)
            r_full = jnp.concatenate([r_exec, edit_base[:, agent.critic_action_dim :]], axis=-1)
            all_actions = jnp.concatenate([all_actions, r_full], axis=0)  # (N+ne, action_dim)

        # EXPO-FT: Q evaluates executed portion only: Q(s, a_{t:t+C}); deterministic top-Q selection.
        exec_all = all_actions[:, : agent.critic_action_dim]  # (N+ne, critic_action_dim)
        qs = compute_q(agent.target_critic.apply_fn, target_params, obs_for_critic, exec_all)
        idx = jnp.argmax(qs)
        action = all_actions[idx]  # return full action_dim for ActionChunkWrapper
    else:
        action = diffusion_actions[0]

    rng, _ = jax.random.split(rng)
    return action, rng


@jax.jit
def _eval_actions_jit(agent, observations):
    """JIT-compiled eval path using filter_temperature_eval (deterministic) vs train temp."""
    rng = agent.rng
    observations = jax.tree_map(lambda x: jnp.squeeze(x), observations)
    observations = jax.tree_map(lambda x: jnp.expand_dims(x, axis=0) if x.ndim == 1 or x.ndim == 3 else x, observations)

    # Use the ONLINE actor (not target) for action generation: the critic/filter are trained
    # to score candidates produced by self.actor's denoising (see update/_sample_candidates),
    # and with actor_tau=0.001 the target actor lags far behind during fine-tuning. Acting with
    # the target here makes eval/rollout both stale and inconsistent with training -> ~0% success
    # even when the trained policy works (verified: lift current-actor 60% vs target 0% @1250).
    import os as _os
    # HYPOTHESIS TEST: official uses target_actor (EMA, stable) for eval/deploy candidates;
    # our fork switched to live self.actor (image-encoder-lag motivated, but harmful for low-dim
    # where there's no encoder -> noisy oscillating eval). EVAL_USE_TARGET_ACTOR=1 reverts to official.
    _eval_params = agent.target_actor.params if _os.environ.get("EVAL_USE_TARGET_ACTOR") == "1" else agent.actor.params
    actor_obs_enc = agent._get_obs_encoding(observations, _eval_params)
    actor_params = _eval_params
    obs_for_filter = actor_obs_enc if agent.share_encoder else observations
    # Critic owns its encoder → raw obs when share_encoder=False; flat features only when True.
    obs_for_critic = actor_obs_enc if agent.share_encoder else observations

    if agent.filter_enabled and agent.filter_at_eval:
        actions, _, rng = agent._sample_candidates(rng, obs_for_filter, agent.N, actor_params, agent.filter_temperature_eval, actor_obs_enc=actor_obs_enc, k_keep=min(agent.n_base_deploy, agent.N))
    else:
        actions, rng = ddim_sampler(
            agent.actor.apply_fn, actor_params, agent.T, rng, agent.action_dim,
            actor_obs_enc, agent.alphas, agent.alpha_hats, agent.betas, agent.M, eta=agent.ddim_eta,
        )
        actions = actions.reshape(1, agent.N, -1)

    diffusion_actions = actions.squeeze(axis=0)  # (N, action_dim)

    if diffusion_actions.shape[0] > 1 or agent.ne_samples > 0:
        key, rng = jax.random.split(rng)
        target_params = subsample_ensemble(key, agent.target_critic.params, agent.num_min_qs, agent.num_qs)
        all_actions = diffusion_actions  # (N, action_dim)

        if agent.ne_samples > 0:
            key, rng = jax.random.split(rng)
            n_base = diffusion_actions.shape[0]  # static
            if n_base >= agent.ne_samples:
                edit_base = diffusion_actions[: agent.ne_samples]
            else:
                tile_factor = (agent.ne_samples + n_base - 1) // n_base
                edit_base = jnp.tile(diffusion_actions, (tile_factor, 1))[: agent.ne_samples]
            d_exec = edit_base[:, : agent.critic_action_dim]
            # EXPO/EXPO-FT: stochastic edit samples (one per base candidate); top-Q argmax selects.
            r_exec, rng = _sample_actions(key, agent.edit_actor.apply_fn, agent.edit_actor.params, actor_obs_enc, d_exec)
            r_exec = agent._apply_residual_action_mask(r_exec * agent.r_action_scale) + d_exec
            r_exec = jnp.clip(r_exec, -1.0, 1.0)
            r_full = jnp.concatenate([r_exec, edit_base[:, agent.critic_action_dim :]], axis=-1)
            all_actions = jnp.concatenate([all_actions, r_full], axis=0)

        exec_all = all_actions[:, : agent.critic_action_dim]
        qs = compute_q(agent.target_critic.apply_fn, target_params, obs_for_critic, exec_all)
        idx = jnp.argmax(qs)
        action = all_actions[idx]
    else:
        action = diffusion_actions[0]

    rng, _ = jax.random.split(rng)
    return action, rng


_ALLOWED_SAMPLING_MODES = ("zscore", "plain")


def _validate_sampling_mode(mode: str) -> None:
    if mode not in _ALLOWED_SAMPLING_MODES:
        raise ValueError(f"Invalid sampling_mode={mode}. Allowed: {_ALLOWED_SAMPLING_MODES}")


def _z_score_normalize(values: jnp.ndarray, axis: int, eps: float = 1e-6) -> jnp.ndarray:
    mean = values.mean(axis=axis, keepdims=True)
    std = values.std(axis=axis, keepdims=True)
    return (values - mean) / jnp.maximum(std, eps)


def _gumbel_topk(key, logits, k):
    u = jax.random.uniform(key, logits.shape, minval=1e-6, maxval=1.0 - 1e-6)
    g = -jnp.log(-jnp.log(u))
    _, idx = jax.lax.top_k(logits + g, k)
    return idx


def sample_k_indices(key, scores: jnp.ndarray, k: int, *, temperature: jnp.ndarray, mode: str = "zscore") -> jnp.ndarray:
    _validate_sampling_mode(mode)
    k = int(k)
    if k < 1:
        raise ValueError(f"k must be >= 1; got k={k}")

    scores = jnp.asarray(scores)
    if scores.ndim < 1:
        raise ValueError(f"scores must have ndim >= 1; got shape={scores.shape}")

    n = scores.shape[-1]
    if k > n:
        raise ValueError(f"k must be <= scores.shape[-1]; got k={k}, n={n}")

    prefix = scores.shape[:-1]
    batch = int(np.prod(prefix)) if prefix else 1
    s2 = scores.reshape(batch, n)

    proc = s2 if mode == "plain" else _z_score_normalize(s2, axis=1)

    temp = jnp.asarray(temperature, dtype=proc.dtype)
    temp = jnp.broadcast_to(temp, (batch,))
    do_sample = temp > 0
    temp_safe = jnp.where(do_sample, temp, jnp.asarray(1.0, dtype=proc.dtype))
    logits = proc / temp_safe[:, None]

    idx_sample = _gumbel_topk(key, logits, k)
    idx_det = jax.lax.top_k(s2, k)[1]
    idx = jnp.where(do_sample[:, None], idx_sample, idx_det)
    return idx.reshape(prefix + (k,))


def _gather_axis1(x, idx):
    if x is None:
        return None
    idx = jnp.asarray(idx)
    def _gather_fn(arr):
        batch = jnp.arange(arr.shape[0]).reshape((arr.shape[0],) + (1,) * (idx.ndim - 1))
        batch = jnp.broadcast_to(batch, idx.shape)
        return arr[batch, idx]
    return jax.tree_map(_gather_fn, x)


default_init = nn.initializers.xavier_uniform


class DenoisingStateActionValue(nn.Module):
    base_cls: nn.Module
    cond_encoder_cls: nn.Module
    time_preprocess_cls: nn.Module
    obs_encoder_cls: Optional[type] = None  # if set, used instead of default ImageStateEncoder

    @nn.compact
    def __call__(self, observations, actions: jnp.ndarray, time: jnp.ndarray, training: bool = False):
        t_ff = self.time_preprocess_cls()(time)
        cond = self.cond_encoder_cls()(t_ff, training=training)
        if self.obs_encoder_cls is not None:
            obs_encoded = self.obs_encoder_cls()(observations, training=training)
        elif isinstance(observations, dict) and "image" in observations:
            obs_encoded = ImageStateEncoder(encoder_cls=get_resnet18)(observations, training=training)
        elif isinstance(observations, dict):
            obs_encoded = observations["state"]
        else:
            obs_encoded = observations
        inputs = jnp.concatenate([obs_encoded, actions, cond], axis=-1)
        outputs = self.base_cls()(inputs, training=training)
        value = nn.Dense(1, kernel_init=default_init())(outputs)
        return jnp.squeeze(value, -1)


@partial(jax.jit, static_argnames=("actor_apply_fn", "hidden_apply_fn", "act_dim", "filter_act_dim", "T", "N", "training", "filter_temperature_mode", "k_keep"))
def ddim_sampler_hidden_filter(
    actor_apply_fn,
    actor_params,
    hidden_apply_fn,
    hidden_params,
    T,
    N,
    rng,
    act_dim,
    observations,
    alphas,
    alpha_hats,
    betas,
    repeat_last_step,
    filter_temperature: float = 0.0,
    filter_temperature_mode: str = "plain",
    ddim_eta: float = 0.0,
    training: bool = False,
    init_noise=None,
    filter_act_dim: int = None,
    actor_obs_enc=None,  # pre-computed actor encoding (batch, enc_dim); avoids T×ResNet18 in scan
    k_keep: int = 1,     # # noise candidates the filter keeps to fully denoise (FASTER=1; EXPO-FT deploy=N_base)
):
    batch_size = jax.tree_util.tree_leaves(observations)[0].shape[0]
    obs = jax.tree_map(lambda x: jnp.broadcast_to(x[:, None, ...], (batch_size, N, *x.shape[1:])), observations)
    rng, init_key, noise_key = jax.random.split(rng, 3)
    x = jax.random.normal(init_key, (batch_size, N, act_dim)) if init_noise is None else init_noise
    temp = jnp.asarray(filter_temperature, dtype=jnp.float32)

    # Broadcast actor_obs_enc to (batch, N, enc_dim); gathered after filtering to match obs/x shape.
    if actor_obs_enc is not None:
        actor_obs_enc_rep = jnp.broadcast_to(actor_obs_enc[:, None, :], (batch_size, N, actor_obs_enc.shape[-1]))
    else:
        actor_obs_enc_rep = None

    x_eval = x
    stored = (x_eval,)
    if x.shape[1] > 1:
        if hidden_apply_fn is None or hidden_params is None:
            raise ValueError("hidden_apply_fn and hidden_params are required when filtering more than one candidate.")
        x_flat = x_eval.reshape(-1, act_dim)
        obs_flat = jax.tree_map(lambda x: x.reshape(-1, *x.shape[2:]), obs)
        time = jnp.full((x_flat.shape[0], 1), T, dtype=jnp.int32)
        # Trim to exec portion only when filter_act_dim is specified.
        # Non-exec dimensions of the initial noise are irrelevant to Q(s, a_exec).
        x_flat_for_filter = x_flat[:, :filter_act_dim] if filter_act_dim is not None else x_flat
        # Use actor-encoded features for filter scoring: consistent with filter training
        # (update_critic_from_candidates trains filter on next_obs_enc = actor-encoded flat).
        # Filter critic's obs_encoder_cls is bypassed (flat vector → ImageStateEncoder → pass-through).
        filter_obs = (
            actor_obs_enc_rep.reshape(-1, actor_obs_enc_rep.shape[-1])
            if actor_obs_enc is not None else obs_flat
        )
        q_values_sel = hidden_apply_fn({"params": hidden_params}, filter_obs, x_flat_for_filter, time)
        q_sel = q_values_sel.min(axis=0).reshape(batch_size, -1)
        rng, key = jax.random.split(rng)

        def select_idx(key_, scores_, k_):
            return sample_k_indices(key_, scores_, k_, temperature=temp, mode=filter_temperature_mode)

        # FASTER keeps the single best-scored noise (k_keep=1) to fully denoise; EXPO-FT deploy keeps
        # N_base candidates so the top-Q selection ranks multiple base chunks (paper: 8 base + 8 edit).
        k_keep = min(int(k_keep), N)
        idx = select_idx(key, q_sel, k_keep)
        x = _gather_axis1(x, idx)
        obs = _gather_axis1(obs, idx)
        stored = tuple(_gather_axis1(s, idx) for s in stored)
        if actor_obs_enc is not None:
            actor_obs_enc_rep = _gather_axis1(actor_obs_enc_rep, idx)

    # Define step and denoise_segment AFTER filtering so they close over the gathered
    # actor_obs_enc_rep. When actor_obs_enc is provided, actor skips T×ResNet18 per step.
    def step(x_, obs_, t_):
        x_flat = x_.reshape(-1, act_dim)
        obs_flat = jax.tree_map(lambda x: x.reshape(-1, *x.shape[2:]), obs_)
        t_ = jnp.asarray(t_, dtype=jnp.int32)
        time = jnp.full((x_flat.shape[0], 1), t_)
        if actor_obs_enc is not None:
            enc_flat = actor_obs_enc_rep.reshape(-1, actor_obs_enc_rep.shape[-1])
        else:
            enc_flat = None
        eps_pred = actor_apply_fn({"params": actor_params}, obs_flat, x_flat, time, training=training, obs_encoding=enc_flat)

        def ddim_step(_):
            alpha_hat_t = alpha_hats[t_]
            sqrt_alpha_hat_t = jnp.sqrt(alpha_hat_t)
            sqrt_one_minus_alpha_hat_t = jnp.sqrt(1.0 - alpha_hat_t)
            x0_pred = (x_flat - sqrt_one_minus_alpha_hat_t * eps_pred) / sqrt_alpha_hat_t
            alpha_hat_prev = jnp.where(t_ > 0, alpha_hats[jnp.maximum(0, t_ - 1)], jnp.asarray(1.0, dtype=alpha_hat_t.dtype))
            eta = jnp.asarray(ddim_eta, dtype=x_flat.dtype)

            def deterministic(_):
                x_next = jnp.sqrt(alpha_hat_prev) * x0_pred + jnp.sqrt(1.0 - alpha_hat_prev) * eps_pred
                return x_next, x0_pred

            def stochastic(_):
                sigma = eta * jnp.sqrt((1.0 - alpha_hat_prev) / (1.0 - alpha_hat_t) * (1.0 - alpha_hat_t / alpha_hat_prev))
                z = jax.random.normal(jax.random.fold_in(noise_key, t_), x_flat.shape)
                noise = sigma * z
                eps_scale = jnp.sqrt(jnp.maximum(0.0, 1.0 - alpha_hat_prev - sigma**2))
                x_next = jnp.sqrt(alpha_hat_prev) * x0_pred + eps_scale * eps_pred + noise
                return x_next, x0_pred

            return jax.lax.cond(eta == 0.0, deterministic, stochastic, operand=None)

        x_next, x_eval = ddim_step(None)
        x_next = jnp.clip(x_next, -1, 1)
        x_eval = jnp.clip(x_eval, -1, 1)
        return x_next.reshape(x_.shape), x_eval.reshape(x_.shape)

    def denoise_segment(x_, obs_, t_start, t_stop_exclusive):
        times = jnp.arange(t_start, t_stop_exclusive, -1)

        def body(x__, t_):
            x__, _ = step(x__, obs_, t_)
            return x__, ()

        x_, _ = jax.lax.scan(body, x_, times)
        return x_

    if T > 0:
        x = denoise_segment(x, obs, T - 1, -1)

    def repeat_body(_, x_):
        x_, _ = step(x_, obs, 0)
        return x_

    x = jax.lax.fori_loop(0, repeat_last_step, repeat_body, x)

    x = jnp.clip(x, -1, 1)
    return x, stored, rng


def _copy_actor_enc_to_ensemble_params(ensemble_params, actor_enc_params):
    """Copy actor's pretrained ImageStateEncoder_0 weights into all ensemble members.

    Ensemble conv/dense params have a leading batch dim (num_qs, ...) from nn.vmap.
    This broadcasts the actor's single-member params across all ensemble members,
    giving the critic/filter_critic a pretrained visual backbone from day 1.
    """
    # Recursively unfreeze to plain dicts so flatten_dict works with all Flax versions.
    params_dict = flax.core.unfreeze(ensemble_params)
    actor_enc_dict = flax.core.unfreeze(actor_enc_params)

    flat_ens = flax.traverse_util.flatten_dict(params_dict, sep="/")
    flat_actor = flax.traverse_util.flatten_dict({"ImageStateEncoder_0": actor_enc_dict}, sep="/")

    loaded = 0
    for actor_suffix, actor_val in flat_actor.items():
        actor_arr = jnp.asarray(actor_val)
        for ens_key in list(flat_ens.keys()):
            if ens_key.endswith(actor_suffix):
                ens_shape = flat_ens[ens_key].shape
                if ens_shape == actor_arr.shape:
                    flat_ens[ens_key] = actor_arr
                    loaded += 1
                elif len(ens_shape) == len(actor_arr.shape) + 1 and ens_shape[1:] == actor_arr.shape:
                    # vmap adds a leading ensemble dimension — broadcast to all members
                    flat_ens[ens_key] = jnp.broadcast_to(actor_arr[None], ens_shape)
                    loaded += 1

    print(f"[copy_actor_enc] copied {loaded}/{len(flat_actor)} params to ensemble critics")
    # Return plain dict (same as _load_pretrained_resnet18_conv_weights) so that
    # _make_split_encoder_tx and optax.multi_transform can traverse it without issues.
    return flax.traverse_util.unflatten_dict(flat_ens, sep="/")


class FasterEXPOLearner(Agent):
    critic: TrainState
    target_critic: TrainState
    actor: TrainState
    target_actor: TrainState
    edit_actor: TrainState
    temp: TrainState
    betas: jnp.ndarray
    alphas: jnp.ndarray
    alpha_hats: jnp.ndarray

    filter_critic: Optional[TrainState]
    target_filter_critic: Optional[TrainState]
    filter_enabled: bool = struct.field(pytree_node=False)
    filter_at_eval: bool = struct.field(pytree_node=False)
    filter_temperature_train: float = struct.field(pytree_node=False)
    filter_temperature_eval: float = struct.field(pytree_node=False)
    filter_temperature_mode: str = struct.field(pytree_node=False)
    chunk_discount_mode: str = struct.field(pytree_node=False)  # "per_chunk" (paper γ¹) | "per_step" (γ^exec_horizon)

    action_dim: int = struct.field(pytree_node=False)       # full chunk dim = chunk_size * orig
    orig_action_dim: int = struct.field(pytree_node=False)  # per-step dim (e.g. 7 for robomimic)
    chunk_size: int = struct.field(pytree_node=False)        # prediction horizon H
    exec_horizon: int = struct.field(pytree_node=False)      # execution horizon C (≤ chunk_size)
    critic_action_dim: int = struct.field(pytree_node=False) # C * orig_dim — Q sees executed portion only
    T: int = struct.field(pytree_node=False)
    N: int = struct.field(pytree_node=False)
    train_N: int = struct.field(pytree_node=False)
    ne_samples: int = struct.field(pytree_node=False)
    ne_samples_train: int = struct.field(pytree_node=False)
    r_action_scale: float = struct.field(pytree_node=False)
    residual_action_mask: Optional[np.ndarray] = struct.field(pytree_node=False)
    M: int = struct.field(pytree_node=False)
    ddim_eta: float = struct.field(pytree_node=False)
    actor_tau: float
    tau: float
    discount: float
    target_entropy: float

    num_qs: int = struct.field(pytree_node=False)
    num_min_qs: Optional[int] = struct.field(pytree_node=False)
    filter_num_min_qs: Optional[int] = struct.field(pytree_node=False)
    share_encoder: bool = struct.field(pytree_node=False)
    state_proj_dim: int = struct.field(pytree_node=False)
    vision_pool: str = struct.field(pytree_node=False)
    num_kp: int = struct.field(pytree_node=False)
    augment: bool = struct.field(pytree_node=False)
    n_base_deploy: int = struct.field(pytree_node=False)

    def _get_obs_encoding(self, observations, actor_params):
        """Encode observations using the actor's image encoder with configured state_proj_dim."""
        if not (isinstance(observations, (dict, flax.core.FrozenDict)) and "image" in observations):
            return observations
        if "ImageStateEncoder_0" not in actor_params:
            return observations
        return ImageStateEncoder(encoder_cls=get_resnet18, state_proj_dim=self.state_proj_dim, pool=self.vision_pool, num_kp=self.num_kp).apply(
            {"params": actor_params["ImageStateEncoder_0"]},
            observations,
            training=False,
        )

    def _get_obs_encoding_chunked(self, observations, actor_params, n_chunks):
        """Encode a large batch in n_chunks sequential chunks via lax.map.
        Peak GPU memory = 1 chunk instead of the full batch.
        Falls back to _get_obs_encoding for non-image / low-dim obs.
        n_chunks=1: skip lax.map entirely, single direct pass.
        """
        if not (isinstance(observations, (dict, flax.core.FrozenDict)) and "image" in observations):
            return observations
        if "ImageStateEncoder_0" not in actor_params:
            return observations
        if n_chunks == 1:
            return self._get_obs_encoding(observations, actor_params)
        encoder_params = actor_params["ImageStateEncoder_0"]
        total = jax.tree_util.tree_leaves(observations)[0].shape[0]
        chunk_size = total // n_chunks
        obs_chunks = jax.tree_map(lambda x: x.reshape(n_chunks, chunk_size, *x.shape[1:]), observations)
        enc_chunks = jax.lax.map(
            lambda obs_chunk: ImageStateEncoder(encoder_cls=get_resnet18, state_proj_dim=self.state_proj_dim, pool=self.vision_pool, num_kp=self.num_kp).apply(
                {"params": encoder_params}, obs_chunk, training=False
            ),
            obs_chunks,
        )
        return enc_chunks.reshape(total, -1)

    @classmethod
    def create(
        cls,
        seed: int,
        observation_space,
        action_space,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        temp_lr: float = 3e-4,
        actor_encoder_lr: float = 1e-5,   # image encoder LR for the actor (split from the MLP heads).
        critic_encoder_lr: float = 1e-4,  # image encoder LR for the critic (warm-started from actor enc).
        hidden_dims: Sequence[int] = (256, 256),
        critic_hidden_dims: Optional[Sequence[int]] = None,
        outer_critic_hidden_dims: Optional[Sequence[int]] = None,
        filter_critic_hidden_dims: Optional[Sequence[int]] = None,
        discount: float = 0.99,
        tau: float = 0.005,
        num_qs: int = 2,
        num_min_qs: Optional[int] = None,
        filter_num_min_qs: Optional[int] = None,
        critic_dropout_rate: Optional[float] = None,
        critic_weight_decay: Optional[float] = None,
        critic_layer_norm: bool = False,
        target_entropy: Optional[float] = None,
        adjust_target_entropy: Optional[bool] = False,
        init_temperature: float = 1.0,
        use_pnorm: bool = False,
        use_critic_resnet: bool = False,
        time_dim: int = 128,
        actor_drop: Optional[float] = None,
        d_actor_drop: Optional[float] = None,
        T: int = 10,
        N: int = 32,
        train_N: int = 32,
        M: int = 0,
        ne_samples: int = 0,
        ne_samples_train: int = 0,
        r_action_scale: float = 1.0,
        residual_action_mask: Optional[Sequence[float]] = None,
        actor_layer_norm: bool = True,
        decay_steps: Optional[int] = int(3e6),
        actor_tau: float = 0.001,
        actor_num_blocks: int = 3,
        ddim_eta: float = 0.0,
        beta_schedule: str = "vp",
        exec_horizon: int = 1,
        chunk_size: int = 1,
        filter_enabled: bool = False,
        filter_at_eval: bool = False,
        filter_temperature_train: Optional[float] = None,
        filter_temperature_eval: Optional[float] = None,
        filter_temperature_mode: str = "plain",
        chunk_discount_mode: str = "per_chunk",
        share_encoder: bool = False,
        state_proj_dim: int = 0,
        vision_pool: str = "gap",
        num_kp: int = 32,
        augment: bool = False,
        n_base_deploy: int = 8,
    ):
        action_dim = action_space.shape[-1]
        chunk_size = int(chunk_size)
        exec_horizon = int(exec_horizon)
        # EXPO-FT style: critic and edit_actor operate on the *executed* portion of the chunk.
        # orig_action_dim: single-step action dim (e.g. 7 for robomimic)
        # critic_action_dim: C * orig_dim — what actually gets executed and evaluated by Q.
        orig_action_dim = action_dim // chunk_size          # 7 for chunk=8, 7 for chunk=1
        critic_action_dim = exec_horizon * orig_action_dim  # 28 for chunk=8/exec=4, 7 for chunk=1

        if residual_action_mask is not None:
            residual_action_mask = np.asarray(residual_action_mask, dtype=np.float32)
            # mask applies to the exec portion (critic_action_dim), not the full chunk
            if residual_action_mask.shape != (critic_action_dim,):
                raise ValueError(
                    f"Expected residual_action_mask shape ({critic_action_dim},) "
                    f"[exec_horizon={exec_horizon} * orig_action_dim={orig_action_dim}], "
                    f"got {residual_action_mask.shape}"
                )

        ddim_eta = float(ddim_eta)
        assert ddim_eta >= 0.0

        target_num_qs = num_min_qs if num_min_qs is not None else num_qs
        target_filter_num_qs = filter_num_min_qs if filter_num_min_qs is not None else num_qs

        assert target_num_qs >= 2, target_num_qs
        _validate_sampling_mode(filter_temperature_mode)
        assert chunk_discount_mode in ("per_chunk", "per_step"), chunk_discount_mode

        if isinstance(action_space, gym.Space):
            observations = observation_space.sample()
            actions = action_space.sample()
        else:
            observations = observation_space
            actions = action_space
        hidden_dims = tuple(hidden_dims)
        critic_hidden_dims = hidden_dims if critic_hidden_dims is None else tuple(critic_hidden_dims)
        outer_critic_hidden_dims = critic_hidden_dims if outer_critic_hidden_dims is None else tuple(outer_critic_hidden_dims)
        filter_critic_hidden_dims = critic_hidden_dims if filter_critic_hidden_dims is None else tuple(filter_critic_hidden_dims)
        if target_entropy is None:
            if adjust_target_entropy:
                # entropy target based on exec portion dimension
                target_entropy = -critic_action_dim / 2 + critic_action_dim * jnp.log(r_action_scale)
            else:
                target_entropy = -critic_action_dim / 2

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)

        preprocess_time_cls = partial(FourierFeatures, output_size=time_dim, learnable=True)
        cond_model_cls = partial(DiffusionMLP, hidden_dims=(time_dim * 2, time_dim * 2), activations=nn.swish, activate_final=False)

        if decay_steps is not None:
            actor_lr = optax.cosine_decay_schedule(actor_lr, decay_steps)
        base_model_cls = partial(
            DiffusionMLPResNet,
            use_layer_norm=actor_layer_norm,
            num_blocks=actor_num_blocks,
            dropout_rate=d_actor_drop,
            out_dim=action_dim,
            activations=nn.swish,
        )
        # Encoder factory: consistent state_proj_dim across actor and all critic/filter networks.
        from functools import partial as _partial
        enc_factory = _partial(ImageStateEncoder, encoder_cls=get_resnet18, state_proj_dim=state_proj_dim, pool=vision_pool, num_kp=num_kp)

        actor_def = DDPM(time_preprocess_cls=preprocess_time_cls, cond_encoder_cls=cond_model_cls,
                         reverse_encoder_cls=base_model_cls, obs_encoder_cls=enc_factory)

        time = jnp.zeros((1, 1))
        observations = jax.tree_map(lambda x: jnp.expand_dims(jnp.asarray(x), axis=0), observations)
        actions = jax.tree_map(lambda x: jnp.expand_dims(jnp.asarray(x), axis=0), actions)
        actor_params = actor_def.init(actor_key, observations, actions, time)["params"]
        _is_image = isinstance(observations, dict) and "image" in observations and "ImageStateEncoder_0" in actor_params
        if _is_image:
            _in_ch = int(jnp.asarray(observations["image"]).shape[-1])
            actor_params = _load_pretrained_resnet18_conv_weights(actor_params, _in_ch)
            actor_tx = _make_split_encoder_tx(actor_params, optax.adamw(learning_rate=actor_lr), encoder_lr=actor_encoder_lr)
        else:
            actor_tx = optax.adamw(learning_rate=actor_lr)
        actor = TrainState.create(apply_fn=actor_def.apply, params=actor_params, tx=actor_tx)
        target_actor = TrainState.create(
            apply_fn=actor_def.apply, params=actor_params, tx=optax.GradientTransformation(lambda _: None, lambda _: None)
        )

        # share_encoder=True: critic/filter use actor encoder (stop_gradient at train time).
        # share_encoder=False: each network builds its own encoder via enc_factory.
        if share_encoder and isinstance(observations, dict) and "image" in observations and "ImageStateEncoder_0" in actor_params:
            example_obs_enc = enc_factory().apply(
                {"params": actor_params["ImageStateEncoder_0"]}, observations, training=False
            )
        else:
            example_obs_enc = observations

        if beta_schedule == "cosine":
            betas = jnp.array(cosine_beta_schedule(T))
        elif beta_schedule == "linear":
            betas = jnp.linspace(1e-4, 2e-2, T)
        elif beta_schedule == "vp":
            betas = jnp.array(vp_beta_schedule(T))
        else:
            raise ValueError(f"Invalid beta schedule: {beta_schedule}")

        alphas = 1 - betas
        alpha_hat = jnp.array([jnp.prod(alphas[: i + 1]) for i in range(T)])

        edit_actor_base_cls = partial(MLP, hidden_dims=hidden_dims, dropout_rate=actor_drop, activate_final=True, use_pnorm=use_pnorm)
        edit_actor_base_cls = partial(StateActionEncoder, base_cls=edit_actor_base_cls)
        # EXPO-FT: edit_actor outputs critic_action_dim (exec portion), not full action_dim.
        # Input conditioning: executed chunk (critic_action_dim), not full prediction.
        edit_actor_def = TanhNormal(edit_actor_base_cls, critic_action_dim)
        edit_actor_init_actions = jnp.ones((1, critic_action_dim))
        # The edit_actor is ALWAYS fed actor-encoded flat features (state_proj_dim-respecting),
        # never a raw obs dict — at both training (update_edit_actor) and candidate selection.
        # So initialise it on that same flat encoding. Initialising on the raw dict instead would
        # encode via a default ImageStateEncoder with state_proj_dim=0, mismatching the inference
        # encoding whenever state_proj_dim>0 (Dense shape error). This also drops the vestigial
        # ResNet18 the edit_actor would otherwise carry (it never uses it) — same as the filter critic.
        if _is_image and "ImageStateEncoder_0" in actor_params:
            edit_actor_example_obs = ImageStateEncoder(
                encoder_cls=get_resnet18, state_proj_dim=state_proj_dim, pool=vision_pool, num_kp=num_kp
            ).apply({"params": actor_params["ImageStateEncoder_0"]}, observations, training=False)
        else:
            edit_actor_example_obs = example_obs_enc
        edit_actor_params = edit_actor_def.init(actor_key, edit_actor_example_obs, edit_actor_init_actions)["params"]
        if _is_image:
            edit_actor_params = _load_pretrained_resnet18_conv_weights(edit_actor_params, _in_ch)
            edit_actor_tx = _make_split_encoder_tx(edit_actor_params, optax.adam(learning_rate=actor_lr), encoder_lr=actor_encoder_lr)
        else:
            edit_actor_tx = optax.adam(learning_rate=actor_lr)
        edit_actor = TrainState.create(apply_fn=edit_actor_def.apply, params=edit_actor_params, tx=edit_actor_tx)

        def make_critic_base_cls(critic_hidden_dims_):
            if use_critic_resnet:
                return partial(MLPResNetV2, num_blocks=1)
            return partial(
                MLP,
                hidden_dims=critic_hidden_dims_,
                activate_final=True,
                dropout_rate=critic_dropout_rate,
                use_layer_norm=critic_layer_norm,
                use_pnorm=use_pnorm,
            )

        def make_outer_critic_cls(critic_hidden_dims_):
            # Heads receive pre-encoded flat features from SharedEncoderEnsembleCritic's single encoder.
            return partial(StateActionValue, base_cls=make_critic_base_cls(critic_hidden_dims_))

        def make_filter_critic_cls(critic_hidden_dims_):
            return partial(
                DenoisingStateActionValue,
                base_cls=make_critic_base_cls(critic_hidden_dims_),
                cond_encoder_cls=cond_model_cls,
                time_preprocess_cls=preprocess_time_cls,
                obs_encoder_cls=enc_factory,
            )

        outer_critic_cls = make_outer_critic_cls(outer_critic_hidden_dims)
        filter_critic_cls = make_filter_critic_cls(filter_critic_hidden_dims)
        # Single shared encoder + ensemble of Q heads (see SharedEncoderEnsembleCritic).
        critic_def = SharedEncoderEnsembleCritic(encoder_cls=enc_factory, net_cls=outer_critic_cls, num_qs=num_qs)
        # EXPO-FT: critic Q(s, a_{t:t+C}) — initialise with exec portion only.
        critic_init_actions = actions[:, :critic_action_dim]
        # share_encoder=False: example_obs_enc is the RAW obs dict → the shared encoder is built
        # and (below) gets the TD gradient. share_encoder=True: it's flat actor features → the
        # encoder passes through and no ImageStateEncoder_0 params are created.
        critic_params = critic_def.init(critic_key, example_obs_enc, critic_init_actions)["params"]
        # Warm-start the critic's single ResNet18 from the actor's pretrained (ImageNet) encoder.
        if _is_image and not share_encoder and "ImageStateEncoder_0" in actor_params:
            critic_params = _copy_actor_enc_to_ensemble_params(critic_params, actor_params["ImageStateEncoder_0"])
        if critic_weight_decay is not None:
            base_tx = optax.adamw(learning_rate=critic_lr, weight_decay=critic_weight_decay, mask=decay_mask_fn)
        else:
            base_tx = optax.adam(learning_rate=critic_lr)
        # Split-LR optimizer: the shared encoder (ImageStateEncoder_0) learns at 1e-4 (slower than
        # the MLP heads at critic_lr), preserving pretrained features while adapting to the TD signal.
        tx = _make_split_encoder_tx(critic_params, base_tx, encoder_lr=critic_encoder_lr) if _is_image and not share_encoder else base_tx
        critic = TrainState.create(apply_fn=critic_def.apply, params=critic_params, tx=tx)
        target_critic_def = SharedEncoderEnsembleCritic(encoder_cls=enc_factory, net_cls=outer_critic_cls, num_qs=target_num_qs)
        target_critic = TrainState.create(
            apply_fn=target_critic_def.apply, params=critic_params, tx=optax.GradientTransformation(lambda _: None, lambda _: None)
        )

        temp_def = Temperature(init_temperature)
        temp_params = temp_def.init(temp_key)["params"]
        temp = TrainState.create(apply_fn=temp_def.apply, params=temp_params, tx=optax.adam(learning_rate=temp_lr))

        filter_critic = None
        target_filter_critic = None
        htemp_train = 0.0
        htemp_eval = 0.0

        if filter_enabled:
            assert train_N >= 1
            if filter_at_eval:
                assert N >= 1
            htemp_train = 1.0 if filter_temperature_train is None else float(filter_temperature_train)
            htemp_eval = 1.0 if filter_temperature_eval is None else float(filter_temperature_eval)

            rng, hidden_key = jax.random.split(rng)
            hidden_def = Ensemble(filter_critic_cls, num=num_qs)
            # Filter critic conditions on the FULL noise vector (action_dim), not just the exec
            # portion. The DDIM-denoised action's executed portion a_{t:t+C} depends on ALL
            # action_dim noise dimensions (the diffusion model couples them), so its regression
            # target Q^a(s, a_exec) is only fully determined by the complete initial noise.
            # Trimming to the exec dims (critic_action_dim) hid information the target depends on,
            # injecting irreducible noise into the filter's candidate ranking.
            filter_init_actions = actions
            # Filter always receives actor_obs_enc (flat encoded vector) at both inference
            # (Fix 1: actor_obs_enc_rep) and training (filter_obs_enc from update calls).
            # Initialising with pre-encoded flat features removes the ResNet18 from filter params:
            #   - raw-dict init → 44.7M ResNet18 params (98.6% of filter) never updated (dead weight)
            #   - flat-enc init →  618K MLP-only params, all actually trained
            # The DenoisingStateActionValue obs_encoder_cls path returns the flat vector unchanged
            # (not a dict → else branch), so no img_encoder sub-module is created in params.
            if _is_image and "ImageStateEncoder_0" in actor_params:
                filter_example_obs = ImageStateEncoder(
                    encoder_cls=get_resnet18, state_proj_dim=state_proj_dim, pool=vision_pool, num_kp=num_kp
                ).apply({"params": actor_params["ImageStateEncoder_0"]}, observations, training=False)
            else:
                filter_example_obs = example_obs_enc
            hidden_params = hidden_def.init(hidden_key, filter_example_obs, filter_init_actions, time)["params"]
            filter_critic = TrainState.create(apply_fn=hidden_def.apply, params=hidden_params, tx=optax.adam(learning_rate=critic_lr))
            target_hidden_def = Ensemble(filter_critic_cls, num=target_filter_num_qs)
            target_filter_critic = TrainState.create(
                apply_fn=target_hidden_def.apply, params=hidden_params, tx=optax.GradientTransformation(lambda _: None, lambda _: None)
            )

        return cls(
            rng=rng,
            actor=actor,
            target_actor=target_actor,
            edit_actor=edit_actor,
            betas=betas,
            alphas=alphas,
            alpha_hats=alpha_hat,
            action_dim=action_dim,
            orig_action_dim=orig_action_dim,
            chunk_size=chunk_size,
            exec_horizon=exec_horizon,
            critic_action_dim=critic_action_dim,
            T=T,
            N=N,
            train_N=train_N,
            ne_samples=ne_samples,
            ne_samples_train=ne_samples_train,
            r_action_scale=r_action_scale,
            residual_action_mask=residual_action_mask,
            M=M,
            actor_tau=actor_tau,
            ddim_eta=float(ddim_eta),
            critic=critic,
            target_critic=target_critic,
            temp=temp,
            target_entropy=target_entropy,
            tau=tau,
            discount=discount,
            num_qs=num_qs,
            num_min_qs=num_min_qs,
            filter_num_min_qs=filter_num_min_qs,
            filter_critic=filter_critic,
            target_filter_critic=target_filter_critic,
            filter_enabled=filter_enabled,
            filter_at_eval=filter_at_eval,
            filter_temperature_train=htemp_train,
            filter_temperature_eval=htemp_eval,
            filter_temperature_mode=filter_temperature_mode,
            chunk_discount_mode=chunk_discount_mode,
            share_encoder=bool(share_encoder),
            state_proj_dim=int(state_proj_dim),
            vision_pool=str(vision_pool),
            num_kp=int(num_kp),
            augment=bool(augment),
            n_base_deploy=int(n_base_deploy),
        )

    def _sample_candidates(self, rng, observations, N, actor_params, filter_temperature, actor_obs_enc=None, k_keep=1):
        """Sample N action candidates; the filter keeps the best k_keep to fully denoise.

        observations: raw obs dict when share_encoder=False; flat encoded when True.
        actor_obs_enc: pre-computed actor encoding (batch, enc_dim) for the denoising scan.
            Avoids T×ResNet18 per step when share_encoder=False. Pass None when not needed.
        k_keep: # base candidates kept after filtering (FASTER training=1; EXPO-FT deploy=N_base).
        """
        observations = jax.tree_map(lambda x: jax.device_put(jnp.asarray(x)), observations)
        batch_size = jax.tree_util.tree_leaves(observations)[0].shape[0]

        if not self.filter_enabled:
            # Use actor_obs_enc as flat obs for the scan so actor skips internal ResNet18.
            obs_for_scan = actor_obs_enc if actor_obs_enc is not None else observations
            obs_rep = jax.tree_map(
                lambda x: jnp.broadcast_to(x[:, None, ...], (batch_size, N, *x.shape[1:])).reshape(-1, *x.shape[1:]),
                obs_for_scan,
            )
            actions_flat, rng = ddim_sampler(
                self.actor.apply_fn,
                actor_params,
                self.T,
                rng,
                self.action_dim,
                obs_rep,
                self.alphas,
                self.alpha_hats,
                self.betas,
                self.M,
                eta=self.ddim_eta,
            )
            actions = actions_flat.reshape(batch_size, N, -1)
            return actions, (), rng

        hidden_apply_fn = None
        hidden_params = None
        if N > 1:
            assert self.target_filter_critic is not None
            key, rng = jax.random.split(rng)
            hidden_apply_fn = self.target_filter_critic.apply_fn
            hidden_params = subsample_ensemble(key, self.target_filter_critic.params, self.filter_num_min_qs, self.num_qs)

        actions, stored, rng = ddim_sampler_hidden_filter(
            self.actor.apply_fn,
            actor_params,
            hidden_apply_fn,
            hidden_params,
            self.T,
            N,
            rng,
            self.action_dim,
            observations,
            self.alphas,
            self.alpha_hats,
            self.betas,
            self.M,
            filter_temperature=filter_temperature,
            filter_temperature_mode=self.filter_temperature_mode,
            ddim_eta=self.ddim_eta,
            init_noise=None,
            filter_act_dim=None,  # condition filter on the full noise vector (action_dim)
            actor_obs_enc=actor_obs_enc,
            k_keep=k_keep,
        )
        return actions, stored, rng

    def _outer_backup_q_scores(self, target_params, observations, actions):
        # EXPO-FT / RedQ: take the MINIMUM over the subsampled target heads (num_min_qs, e.g. 2)
        # and use it for BOTH candidate selection and the Bellman backup — matching the paper's
        # "draw 2 networks at random ... take their minimum to reduce overestimation".
        # (Previously used a decorrelated select(head0)/eval(head1) split — a double-Q variant that
        #  also controls selection bias — but it deviated from the paper's literal min-of-2.)
        q_values_sel = self.target_critic.apply_fn({"params": target_params}, observations, actions)
        q = q_values_sel.min(axis=0)
        return q, q

    def _apply_residual_action_mask(self, residual_actions):
        if self.residual_action_mask is None:
            return residual_actions
        mask = jnp.asarray(self.residual_action_mask, dtype=residual_actions.dtype)
        return residual_actions * mask

    def _select_best_actions(self, rng, edit_observations, critic_observations, actions, target_params):
        # actions: (batch, N, action_dim) — full prediction chunks.
        # edit_observations: actor-encoded features for the edit actor (its input space).
        # critic_observations: obs for the outer critic Q — RAW dict (share_encoder=False) so the
        #   critic's own encoder runs, or flat actor features (share_encoder=True).
        batch_size = jax.tree_util.tree_leaves(edit_observations)[0].shape[0]
        num_candidates = actions.shape[1]
        actions_all = actions  # (batch, N, action_dim)

        if self.ne_samples_train > 0:
            key, rng = jax.random.split(rng)
            r_observations = edit_observations
            ne = self.ne_samples_train
            nc = num_candidates  # static Python int (JAX shape)
            # When filter reduces candidates to < ne_samples_train (e.g. k_keep=1), tile the
            # filtered candidate(s) so the stochastic edit actor generates ne diverse variants.
            if nc >= ne:
                edit_base = actions[:, :ne, :]   # (batch, ne, action_dim)
            else:
                tile_factor = (ne + nc - 1) // nc
                edit_base = jnp.tile(actions, (1, tile_factor, 1))[:, :ne, :]  # (batch, ne, action_dim)
            d_exec = edit_base[:, :, : self.critic_action_dim].reshape(-1, self.critic_action_dim)
            r_exec, rng = _sample_actions(
                key, self.edit_actor.apply_fn, self.edit_actor.params, r_observations, d_exec
            )
            r_exec = self._apply_residual_action_mask(r_exec * self.r_action_scale) + d_exec
            r_exec = jnp.clip(r_exec, -1.0, 1.0)
            # Reconstruct full action: edited exec + original non-exec portion
            r_full = jnp.concatenate([
                r_exec.reshape(batch_size, ne, self.critic_action_dim),
                edit_base[:, :, self.critic_action_dim :],
            ], axis=-1)  # (batch, ne, action_dim)
            actions_all = jnp.concatenate([actions, r_full], axis=1)  # (batch, N+ne, action_dim)

        # EXPO-FT: Q(s, a_{t:t+C}) — evaluate exec portion only.
        # critic_observations is the un-repeated (batch,...) obs; the critic encodes it once and
        # StateActionValue repeats the FEATURES to match exec_flat's (batch*total_candidates) rows.
        exec_flat = actions_all[:, :, : self.critic_action_dim].reshape(-1, self.critic_action_dim)
        q_sel, q_eval = self._outer_backup_q_scores(target_params, critic_observations, exec_flat)

        total_candidates = num_candidates + self.ne_samples_train
        q_sel = q_sel.reshape(batch_size, total_candidates)
        q_eval = q_eval.reshape(batch_size, total_candidates)

        best_indices = jnp.argmax(q_sel, axis=1)
        batch_indices = jnp.arange(batch_size)
        best_actions = actions_all[batch_indices, best_indices]  # full action_dim
        best_q = q_eval[batch_indices, best_indices]
        return best_actions, best_q, best_indices, rng

    def eval_actions(self, observations):
        action, rng = _eval_actions_jit(self, observations)
        return np.array(action.squeeze()), self.replace(rng=rng)

    def sample_actions(self, observations):
        action, rng = _sample_actions_jit(self, observations)
        return np.array(action.squeeze()), self.replace(rng=rng)

    @partial(jax.jit, static_argnames=("has_critic_obs_enc",))
    def update_edit_actor(self, batch: DatasetDict, critic_obs_enc=None, *, has_critic_obs_enc: bool = False):
        # Edit actor input: actor-encoded flat features (its own image encoder is bypassed —
        # flat vector → StateActionEncoder → pass-through). Consistent with _select_best_actions
        # and inference, which all feed the edit actor actor_obs_enc.
        if has_critic_obs_enc:
            obs_enc = jax.lax.stop_gradient(critic_obs_enc)
        else:
            obs_enc = jax.lax.stop_gradient(
                self._get_obs_encoding(batch["observations"], self.target_actor.params)
            )
        # Critic Q input: the outer critic uses its OWN encoder on RAW obs (share_encoder=False),
        # matching how it is trained / queried at inference. Only share_encoder=True feeds it the
        # flat actor features. The critic params are fixed here (gradient flows to the edit actor
        # only), so its encoder just runs forward.
        critic_obs = obs_enc if self.share_encoder else batch["observations"]

        key, rng = jax.random.split(self.rng)
        key2, rng = jax.random.split(rng)
        dropout_key, rng = jax.random.split(rng)

        def edit_actor_loss_fn(actor_params):
            # EXPO-FT: edit_actor operates on exec portion of the chunk
            exec_chunk = batch["actions"][:, : self.critic_action_dim]  # (B, critic_action_dim)
            dist = self.edit_actor.apply_fn(
                {"params": actor_params}, obs_enc, exec_chunk,
                training=True, rngs={"dropout": dropout_key}
            )
            actions = dist.sample(seed=key)  # (B, critic_action_dim)
            # Entropy regularization is applied to the *unscaled* residual policy (the TanhNormal
            # over [-1, 1]). r_action_scale is a fixed, deterministic transform of the sampled
            # action, so it must NOT enter the entropy/temperature bookkeeping.
            #
            # The previous change-of-variables term (-n_active*log(r_action_scale) ≈ +53 for
            # 28 dims @ scale 0.15) shifted reported entropy down by ~53, making both the plain
            # SAC target (-critic_action_dim/2 = -14) and the "adjusted" target (-67) unreachable.
            # The temperature loss α·(entropy - target) was then always positive, so α decayed
            # monotonically to 0 → no entropy regularization and a frozen edit policy whenever Q≈0.
            # Using the unscaled TanhNormal entropy with target = -critic_action_dim/2 is achievable,
            # so α settles at a positive value and the entropy term stays active.
            log_probs = dist.log_prob(actions)
            actions = self._apply_residual_action_mask(actions * self.r_action_scale)
            actions = actions + exec_chunk
            actions = jnp.clip(actions, -1.0, 1.0)
            # critic Q(s, edited_exec_chunk) — critic encodes critic_obs with its own encoder
            qs = self.critic.apply_fn({"params": self.critic.params}, critic_obs, actions, True, rngs={"dropout": key2})
            q = qs.mean(axis=0)
            loss = (log_probs * self.temp.apply_fn({"params": self.temp.params}) - q).mean()
            return loss, {"edit_q": q.mean(), "edit_actor_loss": loss, "entropy": -log_probs.mean()}

        grads, actor_info = jax.grad(edit_actor_loss_fn, has_aux=True)(self.edit_actor.params)
        edit_actor = self.edit_actor.apply_gradients(grads=grads)
        return self.replace(edit_actor=edit_actor, rng=rng), actor_info

    @jax.jit
    def update_actor(self, batch: DatasetDict):
        rng = self.rng
        key, rng = jax.random.split(rng)
        time = jax.random.randint(key, (batch["actions"].shape[0],), 0, self.T)
        key, rng = jax.random.split(rng)
        noise_sample = jax.random.normal(key, (batch["actions"].shape[0], self.action_dim))

        alpha_hats = self.alpha_hats[time]
        time = jnp.expand_dims(time, axis=1)
        alpha_1 = jnp.expand_dims(jnp.sqrt(alpha_hats), axis=1)
        alpha_2 = jnp.expand_dims(jnp.sqrt(1 - alpha_hats), axis=1)
        noisy_actions = alpha_1 * batch["actions"] + alpha_2 * noise_sample

        key, rng = jax.random.split(rng)

        def actor_loss_fn(score_model_params):
            eps_pred = self.actor.apply_fn(
                {"params": score_model_params}, batch["observations"], noisy_actions, time, rngs={"dropout": key}, training=True
            )
            loss = (((eps_pred - noise_sample) ** 2).sum(axis=-1)).mean()
            return loss, {"actor_loss": loss}

        grads, info = jax.grad(actor_loss_fn, has_aux=True)(self.actor.params)
        actor = self.actor.apply_gradients(grads=grads)
        target_params = optax.incremental_update(actor.params, self.target_actor.params, self.actor_tau)
        target_actor = self.target_actor.replace(params=target_params)
        return self.replace(actor=actor, target_actor=target_actor, rng=rng), info

    @jax.jit
    def update_temperature(self, entropy: float):
        def temperature_loss_fn(temp_params):
            temperature = self.temp.apply_fn({"params": temp_params})
            loss = temperature * (entropy - self.target_entropy).mean()
            return loss, {"temperature": temperature, "temperature_loss": loss}

        grads, temp_info = jax.grad(temperature_loss_fn, has_aux=True)(self.temp.params)
        temp = self.temp.apply_gradients(grads=grads)
        return self.replace(temp=temp), temp_info

    @jax.jit
    def update_critic_from_candidates(self, batch: DatasetDict, next_actions_candidates: jnp.ndarray, stored, rng, actor_enc_next=None):
        # batch["observations"] / ["next_observations"] are RAW obs (share_encoder=False) so the
        # outer critic encodes them with its OWN (TD-trained) encoder — consistently across the
        # main loss, the Bellman target, and the filter target. actor_enc_next is the actor-encoded
        # next_obs, used only where actor features are required (the edit actor's input and the
        # filter critic's obs conditioning). For share_encoder=True both are the same flat features.
        critic_next_obs = batch["next_observations"]
        edit_filter_next = actor_enc_next if actor_enc_next is not None else critic_next_obs

        key, rng = jax.random.split(rng)
        target_params = subsample_ensemble(key, self.target_critic.params, self.num_min_qs, self.num_qs)

        next_actions, next_q, _, rng = self._select_best_actions(
            rng, edit_filter_next, critic_next_obs, next_actions_candidates, target_params
        )
        # Chunk Bellman backup. `batch["rewards"]` is the (undiscounted) sum of rewards over the
        # executed window — matching the online ActionChunkWrapper (offline↔online consistency;
        # intra-chunk discounting is negligible for robomimic's sparse success reward).
        #
        # Two modes (chunk_discount_mode), differing only in the bootstrap discount:
        #   "per_chunk" (DEFAULT, PAPER-FAITHFUL): Q(s_t,a_{t:t+C}) = r_t + γ · Q(s_{t+C}, ã*)
        #       This is the EXPO-FT paper's LITERAL update (Sec 4.2 Eq.5) — a BARE γ (verified
        #       verbatim: no γ^C anywhere in the chunk Bellman; r_t is the sparse binary success
        #       reward). Formally it treats the chunk as one macro-step (a chunk-MDP). NOTE the
        #       paper is internally inconsistent: its Sec-3 objective E[Σ γ^t r_t] is per-env-step
        #       (which would call for γ^C), but the actual update uses bare γ — we follow the update.
        #   "per_step":  Q = r_t + γ^exec_horizon · Q(s_{t+C}, ã*)
        #       The exact n-step backup for the per-env-step objective (options/semi-MDP), γ=0.99 per
        #       env-step fixed across chunk sizes. Use only for a controlled chunk_size *ablation*.
        #       At exec_horizon=1 both modes coincide (γ^1 = γ).
        if self.chunk_discount_mode == "per_chunk":
            chunk_discount = self.discount
        else:  # "per_step"
            chunk_discount = self.discount ** self.exec_horizon
        target_q = batch["rewards"] + chunk_discount * batch["masks"] * next_q

        key, rng = jax.random.split(rng)

        def critic_loss_fn(critic_params):
            # EXPO-FT: Q(s, a_{t:t+C}) — critic trained on exec portion only.
            # batch["observations"] is RAW → the critic's shared encoder runs here and receives
            # the TD gradient (the whole point of share_encoder=False).
            exec_actions = batch["actions"][:, : self.critic_action_dim]
            qs = self.critic.apply_fn({"params": critic_params}, batch["observations"], exec_actions, True, rngs={"dropout": key})
            loss = ((qs - target_q) ** 2).mean()
            return loss, {"critic_loss": loss, "q": qs.mean()}

        grads, info = jax.grad(critic_loss_fn, has_aux=True)(self.critic.params)
        critic = self.critic.apply_gradients(grads=grads)
        target_critic_params = optax.incremental_update(critic.params, self.target_critic.params, self.tau)
        target_critic = self.target_critic.replace(params=target_critic_params)

        filter_critic = self.filter_critic
        target_filter_critic = self.target_filter_critic
        if self.filter_enabled and stored and filter_critic is not None and target_filter_critic is not None:
            k_final = next_actions_candidates.shape[1]
            # Filter critic INPUT obs = actor-encoded next_obs (flat), consistent with inference
            # (ddim_sampler_hidden_filter feeds it actor_obs_enc_rep). Broadcast to k_final.
            obs_rep = jax.tree_map(
                lambda x: jnp.broadcast_to(x[:, None, ...], (x.shape[0], k_final, *x.shape[1:])).reshape(-1, *x.shape[1:]),
                edit_filter_next
            )
            # Filter TARGET = outer critic Q^a(s', a0_exec). The outer critic uses its OWN encoder,
            # so feed it RAW next_obs (un-repeated; StateActionValue repeats features to match a0).
            a0_flat = next_actions_candidates.reshape(-1, next_actions_candidates.shape[-1])
            a0_exec = a0_flat[:, : self.critic_action_dim]
            q0 = compute_q(self.target_critic.apply_fn, target_params, critic_next_obs, a0_exec).reshape(-1, k_final)
            q0 = jax.lax.stop_gradient(q0)
            rng, drop_key = jax.random.split(rng)
            num_stages = len(stored)
            stored_actions = jnp.stack(stored, axis=0)
            # Filter critic conditions on the FULL noise vector (action_dim) — no exec trim.
            # Its target q0 is still Q^a on the exec portion (a0_exec above), but the *input*
            # is the complete initial noise that determines the denoised action (see create()).
            a_stacked = stored_actions.reshape(-1, stored_actions.shape[-1])
            q0_flat = q0.reshape(-1)
            obs_stacked = jax.tree_map(
                lambda x: jnp.broadcast_to(x[None, ...], (num_stages, *x.shape)).reshape(-1, *x.shape[1:]),
                obs_rep
            )
            t_stages = jnp.asarray((self.T,), dtype=jnp.int32)
            q_targets = jnp.broadcast_to(q0_flat[None, :], (num_stages, q0_flat.shape[0]))
            time_stacked = jnp.broadcast_to(t_stages[:, None, None], (num_stages, q0_flat.shape[0], 1)).reshape(-1, 1)

            def hidden_loss_fn(hidden_params):
                qs = filter_critic.apply_fn(
                    {"params": hidden_params}, obs_stacked, a_stacked, time_stacked, True, rngs={"dropout": drop_key}
                )
                actual_nq = qs.shape[0]
                qs_r = qs.reshape(actual_nq, num_stages, -1)
                diffs = qs_r - q_targets[None, :, :]
                stage_losses = (diffs**2).mean(axis=(0, 2))
                loss = stage_losses.mean()
                q_min = qs_r.min(axis=0)
                pred_mean = q_min.mean()
                q0_mean = q0.mean()
                return loss, {"filter_critic_loss": loss, "hidden_q": pred_mean, "hidden_target_q": q0_mean}

            grads, hidden_info = jax.grad(hidden_loss_fn, has_aux=True)(filter_critic.params)
            filter_critic = filter_critic.apply_gradients(grads=grads)
            target_hidden_params = optax.incremental_update(filter_critic.params, target_filter_critic.params, self.tau)
            target_filter_critic = target_filter_critic.replace(params=target_hidden_params)
            info.update(hidden_info)

        agent = self.replace(
            critic=critic, target_critic=target_critic, filter_critic=filter_critic, target_filter_critic=target_filter_critic, rng=rng
        )
        return agent, info

    @jax.jit
    def update_critic(self, batch: DatasetDict, raw_next_obs=None):
        rng = self.rng
        actor_params = self.actor.params
        if self.share_encoder:
            obs_enc = jax.lax.stop_gradient(self._get_obs_encoding(batch["observations"], self.target_actor.params))
            next_obs_enc = jax.lax.stop_gradient(self._get_obs_encoding(batch["next_observations"], self.target_actor.params))
            encoded_batch = {**batch, "observations": obs_enc, "next_observations": next_obs_enc}
            next_actions_candidates, stored, rng = self._sample_candidates(
                rng, next_obs_enc, self.train_N, actor_params, self.filter_temperature_train
            )
            return self.update_critic_from_candidates(encoded_batch, next_actions_candidates, stored, rng, actor_enc_next=next_obs_enc)
        else:
            # Critic gets RAW obs (own encoder); actor uses its enc shortcut for candidate sampling.
            actor_next_obs_enc = jax.lax.stop_gradient(self._get_obs_encoding(batch["next_observations"], self.target_actor.params))
            next_actions_candidates, stored, rng = self._sample_candidates(
                rng, batch["next_observations"], self.train_N, actor_params, self.filter_temperature_train,
                actor_obs_enc=actor_next_obs_enc,
            )
            return self.update_critic_from_candidates(batch, next_actions_candidates, stored, rng, actor_enc_next=actor_next_obs_enc)

    @partial(jax.jit, static_argnames=("utd_ratio", "pretrain_q", "pretrain_r"))
    def update_offline(self, batch: DatasetDict, utd_ratio: int, pretrain_q: bool, pretrain_r: bool):
        assert utd_ratio > 0
        assert jax.tree_util.tree_leaves(batch["observations"])[0].shape[0] % utd_ratio == 0
        mini_batch_size = jax.tree_util.tree_leaves(batch["observations"])[0].shape[0] // utd_ratio

        # Augment (conditioned on self.augment flag; no-op for low-dim obs)
        rng = self.rng
        if self.augment:
            rng, obs_key, next_key = jax.random.split(rng, 3)
            batch = {**batch,
                     "observations": augment_obs_batch(obs_key, batch["observations"]),
                     "next_observations": augment_obs_batch(next_key, batch["next_observations"])}

        n_enc_chunks = max(1, utd_ratio // 4)

        # Actor encoder (stop-grad): ONE pass over the full batch (chunked for memory). Used for
        # candidate sampling, the edit actor's input, and the filter critic's obs conditioning.
        obs_enc = jax.lax.stop_gradient(self._get_obs_encoding_chunked(batch["observations"], self.target_actor.params, n_enc_chunks))
        next_obs_enc = jax.lax.stop_gradient(self._get_obs_encoding_chunked(batch["next_observations"], self.target_actor.params, n_enc_chunks))
        # Outer-critic obs: RAW (share_encoder=False) so the critic's own TD-trained encoder runs
        # inside each per-mini-update loss; flat actor features only when share_encoder=True.
        critic_batch = batch if not self.share_encoder else {**batch, "observations": obs_enc, "next_observations": next_obs_enc}

        if self.share_encoder:
            next_obs_for_cands = next_obs_enc
            actor_obs_enc_for_cands = None
        else:
            # Candidate sampling: filter critic needs raw obs; actor uses enc shortcut
            next_obs_for_cands = batch["next_observations"]
            actor_obs_enc_for_cands = next_obs_enc

        def get_mini_batch(i):
            start = i * mini_batch_size
            return jax.tree_util.tree_map(lambda x: jax.lax.dynamic_slice_in_dim(x, start, mini_batch_size, axis=0), critic_batch)

        def get_actor_enc_next(i):
            start = i * mini_batch_size
            return jax.lax.dynamic_slice_in_dim(next_obs_enc, start, mini_batch_size, axis=0)

        raw_actor_batch = jax.tree_util.tree_map(
            lambda x: jax.lax.dynamic_slice_in_dim(x, (utd_ratio - 1) * mini_batch_size, mini_batch_size, axis=0), batch
        )

        new_agent = self
        critic_info = {}
        if pretrain_q:
            # Pre-sample candidates once for full next_obs — actor params fixed during critic UTD loop
            # Use rng that is already advanced past augmentation split (no reset to self.rng)
            next_cands_full, stored_full, rng = self._sample_candidates(
                rng, next_obs_for_cands, self.train_N, self.actor.params, self.filter_temperature_train,
                actor_obs_enc=actor_obs_enc_for_cands,
            )
            new_agent = self.replace(rng=rng)

            def get_mini_candidates(i):
                start = i * mini_batch_size
                cands = jax.lax.dynamic_slice_in_dim(next_cands_full, start, mini_batch_size, axis=0)
                stored = tuple(jax.lax.dynamic_slice_in_dim(s, start, mini_batch_size, axis=0) for s in stored_full)
                return cands, stored

            def body(i, carry):
                agent, _ = carry
                cands, stored = get_mini_candidates(i)
                agent, info = agent.update_critic_from_candidates(
                    get_mini_batch(i), cands, stored, agent.rng, actor_enc_next=get_actor_enc_next(i)
                )
                return agent, info

            cands0, stored0 = get_mini_candidates(0)
            new_agent, critic_info = new_agent.update_critic_from_candidates(
                get_mini_batch(0), cands0, stored0, new_agent.rng, actor_enc_next=get_actor_enc_next(0)
            )
            new_agent, critic_info = jax.lax.fori_loop(1, utd_ratio, body, (new_agent, critic_info))

        new_agent, actor_info = new_agent.update_actor(raw_actor_batch)
        if pretrain_r and (self.ne_samples + self.ne_samples_train > 0):
            actor_obs_enc_slice = jax.lax.dynamic_slice_in_dim(
                obs_enc, (utd_ratio - 1) * mini_batch_size, mini_batch_size, axis=0
            )
            new_agent, edit_info = new_agent.update_edit_actor(
                raw_actor_batch, actor_obs_enc_slice, has_critic_obs_enc=True
            )
            new_agent, temp_info = new_agent.update_temperature(edit_info["entropy"])
            actor_info.update(edit_info)
            actor_info.update(temp_info)
        return new_agent, {**actor_info, **critic_info}

    @partial(jax.jit, static_argnames="utd_ratio")
    def update(self, batch: DatasetDict, utd_ratio: int):
        assert utd_ratio > 0
        assert jax.tree_util.tree_leaves(batch["observations"])[0].shape[0] % utd_ratio == 0
        mini_batch_size = jax.tree_util.tree_leaves(batch["observations"])[0].shape[0] // utd_ratio

        rng = self.rng
        if self.augment:
            rng, obs_key, next_key = jax.random.split(rng, 3)
            batch = {**batch,
                     "observations": augment_obs_batch(obs_key, batch["observations"]),
                     "next_observations": augment_obs_batch(next_key, batch["next_observations"])}

        # Encode full batch in sequential chunks (lax.map) to bound peak GPU memory.
        # n_enc_chunks=max(1, utd_ratio//4) → ~80% GPU utilization vs per-step encoding.
        # Always pre-encode with actor encoder for the UTD fori_loop (1 encoder pass total).
        n_enc_chunks = max(1, utd_ratio // 4)

        obs_enc = jax.lax.stop_gradient(self._get_obs_encoding_chunked(batch["observations"], self.target_actor.params, n_enc_chunks))
        next_obs_enc = jax.lax.stop_gradient(self._get_obs_encoding_chunked(batch["next_observations"], self.target_actor.params, n_enc_chunks))
        # Outer-critic obs: RAW (share_encoder=False) so the critic's own TD-trained encoder runs
        # inside each per-mini-update loss; flat actor features only when share_encoder=True.
        critic_batch = batch if not self.share_encoder else {**batch, "observations": obs_enc, "next_observations": next_obs_enc}

        if self.share_encoder:
            next_obs_for_cands = next_obs_enc
            actor_obs_enc_for_cands = None
        else:
            # Filter critic needs raw obs; actor shortcut uses next_obs_enc
            next_obs_for_cands = batch["next_observations"]
            actor_obs_enc_for_cands = next_obs_enc

        def get_mini_batch(i):
            start = i * mini_batch_size
            return jax.tree_util.tree_map(lambda x: jax.lax.dynamic_slice_in_dim(x, start, mini_batch_size, axis=0), critic_batch)

        def get_actor_enc_next(i):
            start = i * mini_batch_size
            return jax.lax.dynamic_slice_in_dim(next_obs_enc, start, mini_batch_size, axis=0)

        raw_actor_batch = jax.tree_util.tree_map(
            lambda x: jax.lax.dynamic_slice_in_dim(x, (utd_ratio - 1) * mini_batch_size, mini_batch_size, axis=0), batch
        )

        # Pre-sample candidates once for full next_obs — rng continues from augmentation split
        next_cands_full, stored_full, rng = self._sample_candidates(
            rng, next_obs_for_cands, self.train_N, self.actor.params, self.filter_temperature_train,
            actor_obs_enc=actor_obs_enc_for_cands,
        )
        new_agent = self.replace(rng=rng)

        def get_mini_candidates(i):
            start = i * mini_batch_size
            cands = jax.lax.dynamic_slice_in_dim(next_cands_full, start, mini_batch_size, axis=0)
            stored = tuple(jax.lax.dynamic_slice_in_dim(s, start, mini_batch_size, axis=0) for s in stored_full)
            return cands, stored

        def body(i, carry):
            agent, _ = carry
            cands, stored = get_mini_candidates(i)
            agent, info = agent.update_critic_from_candidates(
                get_mini_batch(i), cands, stored, agent.rng, actor_enc_next=get_actor_enc_next(i)
            )
            return agent, info

        cands0, stored0 = get_mini_candidates(0)
        new_agent, critic_info = new_agent.update_critic_from_candidates(
            get_mini_batch(0), cands0, stored0, new_agent.rng, actor_enc_next=get_actor_enc_next(0)
        )
        new_agent, critic_info = jax.lax.fori_loop(1, utd_ratio, body, (new_agent, critic_info))

        new_agent, actor_info = new_agent.update_actor(raw_actor_batch)
        if self.ne_samples + self.ne_samples_train > 0:
            # Always pass pre-computed actor_obs_enc slice (avoids extra ResNet18 call).
            # update_edit_actor now uses actor-encoded features regardless of share_encoder.
            actor_obs_enc_slice = jax.lax.dynamic_slice_in_dim(
                obs_enc, (utd_ratio - 1) * mini_batch_size, mini_batch_size, axis=0
            )
            new_agent, edit_info = new_agent.update_edit_actor(
                raw_actor_batch, actor_obs_enc_slice, has_critic_obs_enc=True
            )
            new_agent, temp_info = new_agent.update_temperature(edit_info["entropy"])
            actor_info.update(edit_info)
            actor_info.update(temp_info)
        return new_agent, {**actor_info, **critic_info}


def get_config():
    from configs import base_config

    config = base_config.get_config()

    config.model_cls = "FasterEXPOLearner"

    config.critic_lr = 3e-4
    config.temp_lr = 3e-4

    config.init_temperature = 1.0
    # adjust_target_entropy=False: target_entropy = -critic_action_dim/2 (standard SAC heuristic).
    # update_edit_actor now reports the *unscaled* TanhNormal entropy (the r_action_scale
    # change-of-variables term was removed), so this target is achievable and α settles at a
    # positive value instead of collapsing to 0. Setting this True re-enables the old
    # scale-shifted target (-67), which is unreachable and drives α → 0 — do not use.
    config.adjust_target_entropy = False
    config.discount = 0.99
    config.tau = 0.005
    config.hidden_dims = (256, 256, 256)
    config.critic_hidden_dims = config_dict.placeholder(tuple)
    config.outer_critic_hidden_dims = config_dict.placeholder(tuple)
    config.filter_critic_hidden_dims = config_dict.placeholder(tuple)
    config.critic_dropout_rate = None
    config.critic_weight_decay = None
    config.use_pnorm = False
    config.use_critic_resnet = False

    config.num_qs = 10
    config.num_min_qs = 2
    config.filter_num_min_qs = 2
    config.critic_layer_norm = True

    config.T = 10
    config.time_dim = 128
    config.N = 8
    config.train_N = 8
    config.M = 0
    config.ddim_eta = 0.0
    config.beta_schedule = "vp"

    config.filter_enabled = True
    config.filter_at_eval = True
    config.filter_temperature_train = 1.0
    config.filter_temperature_eval = 0.0
    config.filter_temperature_mode = "zscore"
    # Chunk Bellman discount. DEFAULT "per_chunk" matches the EXPO-FT paper's literal update
    # (Sec 4.2 Eq.5): Q(s_t,a_{t:t+C}) = r_t + γ·Q(s_{t+C}, ...) — a BARE γ (no exponent), with r_t
    # the sparse binary success reward (undiscounted). Verified verbatim: γ has no exponent anywhere
    # in the chunk Bellman; only the Sec-3 objective uses γ^t. (The paper is internally a bit
    # inconsistent — its Sec-3 objective E[Σ γ^t r_t] is per-env-step, which would imply γ^C — but
    # the actual chunk update uses bare γ, so per_chunk is the paper-faithful choice.)
    # "per_step" = γ^exec_horizon bootstrap: the exact n-step backup for the per-env-step objective
    # (options/semi-MDP), fixed across chunk sizes — use only for a controlled chunk_size ablation.
    # Identical at exec_horizon=1. See CHANGES_FROM_OFFICIAL.md §3.5.
    config.chunk_discount_mode = "per_chunk"

    config.ne_samples = 1
    config.ne_samples_train = 1
    config.r_action_scale = 0.15
    config.residual_action_mask = config_dict.placeholder(tuple)  # None = no mask
    # EXPO-FT: chunk_size injected from --chunk_size flag; kept here as documentation.
    # exec_horizon likewise injected from --exec_horizon; both control critic_action_dim.

    config.actor_drop = 0.0
    config.d_actor_drop = 0.0
    config.actor_lr = 3e-4
    # Image-encoder learning rates, split from the MLP heads (optax.multi_transform).
    # Default 1e-5 keeps the warm-started ImageNet encoder nearly frozen — adequate for easy
    # tasks (lift/can) but too slow for tasks needing precise visual localization (square,
    # tool_hang). Raise actor_encoder_lr (e.g. 1e-4) to let the encoder adapt to robosuite
    # pixels; verified to drop BC actor_loss from ~0.25 (frozen) to ~0.16 on square.
    config.actor_encoder_lr = 1e-5
    config.critic_encoder_lr = 1e-4
    config.actor_layer_norm = True
    # Polyak rate for target_actor. After the online-actor eval fix, target_actor is used ONLY as a
    # stable encoder snapshot for training-time candidate/edit obs encoding (eval/rollout use the
    # live self.actor). With actor_encoder_lr raised to 1e-4 the encoder moves faster, so 0.001
    # (EMA window ~1000 steps) lagged too far behind the live encoder. 0.005 (window ~200, = critic
    # tau) tracks it closely → smaller train/eval encoder gap, negligible added bootstrap variance
    # (the actor is BC-only and target_critic already stabilizes the value target).
    config.actor_tau = 0.005
    config.actor_num_blocks = 4
    config.decay_steps = int(3e6)
    config.share_encoder = False
    config.state_proj_dim = 0
    # Vision pooling head for the image ResNet encoder. "gap" reproduces the 5% baseline;
    # "spatial_softmax" replaces global avg pooling with soft-argmax keypoints (Arm A).
    config.vision_pool = "gap"
    config.num_kp = 32
    config.augment = False
    # EXPO-FT deploy candidate pool: at eval/rollout the filter keeps N_base base chunks (paper: 8)
    # for the top-Q selection, instead of FASTER's single best (k_keep=1). With ne_samples stochastic
    # edits this realises the paper's "N base + N edit → deterministic top-Q" (Eq.2: a* = argmax_a Q).
    # If N (noise pool) > n_base_deploy the filter prunes N→n_base_deploy (FASTER efficiency); if equal
    # it keeps all. Under-representing the base (the old k_keep=1) made a flat/uncalibrated critic pick
    # noisy edits → deploy collapse; 8 base candidates keep the deploy robust. Training keeps k_keep=1.
    config.n_base_deploy = 8

    return config
