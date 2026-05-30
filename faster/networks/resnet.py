import functools
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Callable, Sequence, Tuple, Optional

ModuleDef = Any

class ResNetBlock(nn.Module):
    filters: int
    conv: ModuleDef
    norm: ModuleDef
    act: Callable
    strides: Tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(self, x):
        residual = x
        y = self.conv(self.filters, (3, 3), self.strides)(x)
        y = self.norm(name='norm_1')(y)
        y = self.act(y)
        y = self.conv(self.filters, (3, 3))(y)
        y = self.norm(scale_init=nn.initializers.zeros, name='norm_2')(y)

        if residual.shape != y.shape:
            residual = self.conv(self.filters, (1, 1), self.strides, name='conv_proj')(residual)
            residual = self.norm(name='norm_proj')(residual)

        return self.act(residual + y)

class ResNet(nn.Module):
    stage_sizes: Sequence[int]
    block_cls: ModuleDef
    num_filters: int = 64
    dtype: Any = jnp.float32
    act: Callable = nn.relu

    @nn.compact
    def __call__(self, x, training: bool = True):
        conv = functools.partial(nn.Conv, use_bias=False, dtype=self.dtype)
        norm = functools.partial(nn.GroupNorm, num_groups=32, epsilon=1e-5, dtype=self.dtype)

        x = conv(self.num_filters, (7, 7), (2, 2), padding=[(3, 3), (3, 3)], name='conv_init')(x)
        x = norm(name='gn_init')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')

        for i, block_size in enumerate(self.stage_sizes):
            for j in range(block_size):
                strides = (2, 2) if i > 0 and j == 0 else (1, 1)
                x = self.block_cls(self.num_filters * (2 ** i), strides=strides, conv=conv, norm=norm, act=self.act)(x)

        x = jnp.mean(x, axis=(-3, -2))
        return x

ResNet18 = functools.partial(ResNet, stage_sizes=[2, 2, 2, 2], block_cls=ResNetBlock)


# ---------------------------------------------------------------------------
# Image augmentation — paper-style (OpenPI): random crop+resize + color jitter.
# Applied per sample, per view independently. No-op for non-image observations.
# ---------------------------------------------------------------------------

def _augment_single_view(key: jnp.ndarray, img: jnp.ndarray, crop_ratio: float = 0.95,
                          brightness: float = 0.1, contrast: float = 0.1,
                          saturation: float = 0.1) -> jnp.ndarray:
    """Augment a single (H, W, 3) float32 [0,1] RGB view.

    Called per-sample per-view via vmap — C is always 3 here.
    """
    H, W = img.shape[0], img.shape[1]
    crop_H = int(H * crop_ratio)
    crop_W = int(W * crop_ratio)

    k_crop, k_b, k_c, k_s = jax.random.split(key, 4)
    kh, kw = jax.random.split(k_crop)

    # Random crop + bilinear resize back to original resolution
    h0 = jax.random.randint(kh, (), 0, max(1, H - crop_H + 1))
    w0 = jax.random.randint(kw, (), 0, max(1, W - crop_W + 1))
    patch = jax.lax.dynamic_slice(img, (h0, w0, 0), (crop_H, crop_W, 3))
    img = jax.image.resize(patch, (H, W, 3), method='bilinear')

    # Brightness: additive uniform offset
    b = jax.random.uniform(k_b, (), minval=-brightness, maxval=brightness)
    img = jnp.clip(img + b, 0.0, 1.0)

    # Contrast: scale deviations from channel mean
    c_fac = jax.random.uniform(k_c, (), minval=1.0 - contrast, maxval=1.0 + contrast)
    mean = img.mean(axis=(0, 1), keepdims=True)
    img = jnp.clip((img - mean) * c_fac + mean, 0.0, 1.0)

    # Saturation: blend toward grayscale (BT.601 luma coefficients)
    s_fac = jax.random.uniform(k_s, (), minval=1.0 - saturation, maxval=1.0 + saturation)
    gray = 0.299 * img[..., 0:1] + 0.587 * img[..., 1:2] + 0.114 * img[..., 2:3]
    img = jnp.clip(img * s_fac + gray * (1.0 - s_fac), 0.0, 1.0)

    return img


def augment_obs_batch(rng: jnp.ndarray, obs, crop_ratio: float = 0.95,
                      brightness: float = 0.1, contrast: float = 0.1,
                      saturation: float = 0.1):
    """Apply independent augmentation per sample and per camera view.

    obs: dict with "image" key (B, H, W, C) where C must be a multiple of 3, or raw array.
    Images are expected as uint8 [0,255] or float32 [0,255] from the replay buffer.
    Returns obs with augmented "image" as float32 [0,255] (matching ImageStateEncoder input format).
    No-op for low-dim (non-image) observations, or when C % 3 != 0.
    """
    if not isinstance(obs, dict) or "image" not in obs:
        return obs

    image = obs["image"]
    C = image.shape[-1]
    # Only augment stacked-RGB images (robomimic uses 3 or 6 channels)
    if C % 3 != 0:
        return obs

    # Convert to float32 [0,1] for augmentation ops.
    # Images arrive from replay buffer as uint8 or float32 [0,255].
    if image.dtype == jnp.uint8:
        image = image.astype(jnp.float32) / 255.0
    else:
        image = image.astype(jnp.float32) / 255.0  # always normalize; safe for uint8-range floats

    B, H, W, _ = image.shape
    n_views = C // 3

    # Reshape to (B, n_views, H, W, 3) for independent per-view augmentation
    image_views = image.reshape(B, H, W, n_views, 3).transpose(0, 3, 1, 2, 4)

    # Generate B * n_views independent RNG keys
    keys = jax.random.split(rng, B * n_views).reshape(B, n_views, 2)

    # vmap over (batch, view) — each (key, img_HW3) pair augmented independently
    augmented = jax.vmap(
        jax.vmap(
            lambda k, img: _augment_single_view(k, img, crop_ratio, brightness, contrast, saturation)
        )
    )(keys, image_views)
    # augmented: (B, n_views, H, W, 3) in [0,1]

    # Reshape back to (B, H, W, C) and convert to [0,255] float32
    image_out = augmented.transpose(0, 2, 3, 1, 4).reshape(B, H, W, C) * 255.0
    return {**obs, "image": image_out}


class ImageStateEncoder(nn.Module):
    encoder_cls: ModuleDef
    state_proj_dim: int = 0  # >0: project state to this dim before concat (paper: 64)

    @nn.compact
    def __call__(self, observations, training: bool = False):
        if hasattr(observations, "keys") and "image" in observations:
            state = observations["state"]
            image = observations["image"]

            if image.dtype != jnp.float32:
                image = image.astype(jnp.float32) / 255.0
            else:
                image = image / 255.0

            mean = jnp.array([0.485, 0.456, 0.406], dtype=jnp.float32)
            std = jnp.array([0.229, 0.224, 0.225], dtype=jnp.float32)

            C = image.shape[-1]
            if C % 3 == 0:
                repeats = C // 3
                mean = jnp.tile(mean, repeats)
                std = jnp.tile(std, repeats)

            image = (image - mean) / std

            img_embed = self.encoder_cls(name='img_encoder')(image, training=training)

            # Optional state projection to fixed dim (paper: 64-dim proprioception embed)
            if self.state_proj_dim > 0:
                state = nn.Dense(self.state_proj_dim, name='state_proj')(state)
                state = nn.tanh(state)

            fused = jnp.concatenate([state, img_embed], axis=-1)
            return fused
        else:
            return observations

def get_resnet18(**kwargs):
    return ResNet18(**kwargs)
