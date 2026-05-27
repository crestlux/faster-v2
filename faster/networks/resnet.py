import functools
import flax.linen as nn
import jax.numpy as jnp
from typing import Any, Callable, Sequence, Tuple

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
        # Using GroupNorm or LayerNorm to avoid BatchNorm state tracking complexity in RL
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

class ImageStateEncoder(nn.Module):
    encoder_cls: ModuleDef
    
    @nn.compact
    def __call__(self, observations, training: bool = False):
        if isinstance(observations, dict):
            state = observations["state"]
            image = observations["image"]
            
            # image is expected to be [B, H, W, C] and likely uint8 or float [0, 255]
            # normalize to roughly [-1, 1] or [0, 1]
            if image.dtype != jnp.float32:
                image = image.astype(jnp.float32) / 255.0            # Apply ImageNet normalization for pretrained weights compatibility
            mean = jnp.array([0.485, 0.456, 0.406], dtype=jnp.float32)
            std = jnp.array([0.229, 0.224, 0.225], dtype=jnp.float32)
            
            C = image.shape[-1]
            if C % 3 == 0:
                repeats = C // 3
                mean = jnp.tile(mean, repeats)
                std = jnp.tile(std, repeats)
                
            image = (image - mean) / std

                
            img_embed = self.encoder_cls(name='img_encoder')(image, training=training)
            
            # fuse state and image
            fused = jnp.concatenate([state, img_embed], axis=-1)
            return fused
        else:
            return observations

def get_resnet18(**kwargs):
    return ResNet18(**kwargs)

