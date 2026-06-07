"""Diagnostic: verify offline-dataset obs and env-returned obs are on the same scale.
A scale/key mismatch makes a perfectly trained policy score 0% at eval.
Run: CUDA_VISIBLE_DEVICES=0 python _diag_obs.py lift
"""
import sys
import numpy as np
import h5py
from pathlib import Path

env_name = sys.argv[1] if len(sys.argv) > 1 else "lift"
h5_path = f"datasets/robomimic/image/{env_name}/ph/image_v141.hdf5"

from faster.data.robomimic_datasets import (
    PROPRIOCEPTION_KEYS, IMAGE_OBS_KEYS, get_robomimic_env,
)

# --- 1. Inspect raw H5 offline obs ---
print(f"=== Offline H5: {h5_path} ===")
with h5py.File(h5_path, "r") as f:
    demo0 = f["data/demo_0"]
    print("obs keys:", list(demo0["obs"].keys()))
    for k in PROPRIOCEPTION_KEYS:
        v = demo0[f"obs/{k}"][:]
        print(f"  state[{k}]: shape={v.shape} dtype={v.dtype} range=[{v.min():.4f},{v.max():.4f}] mean={v.mean():.4f}")
    for k in IMAGE_OBS_KEYS:
        v = demo0[f"obs/{k}"][:]
        print(f"  image[{k}]: shape={v.shape} dtype={v.dtype} range=[{v.min()},{v.max()}] mean={v.mean():.2f}")
    acts = demo0["actions"][:]
    print(f"  actions: shape={acts.shape} range=[{acts.min():.3f},{acts.max():.3f}]")
    rews = demo0["rewards"][:]
    print(f"  rewards: shape={rews.shape} sum={rews.sum()} unique={np.unique(rews)}")

# --- 2. Inspect env-returned processed obs ---
print(f"\n=== Env processed obs ({env_name}) ===")
example_action = acts[0][np.newaxis]
env = get_robomimic_env(h5_path, example_action, env_name, use_image_obs=True)
obs = env.reset()
print("processed obs keys:", list(obs.keys()))
s = obs["state"]; img = obs["image"]
print(f"  state: shape={s.shape} dtype={s.dtype} range=[{s.min():.4f},{s.max():.4f}] mean={s.mean():.4f}")
print(f"  image: shape={img.shape} dtype={img.dtype} range=[{img.min()},{img.max()}] mean={float(img.mean()):.2f}")

# step a few random actions to confirm stability
for _ in range(3):
    a = np.random.uniform(-1, 1, size=example_action.shape[1]).astype(np.float32)
    obs, r, d, info = env.step(a)
img2 = obs["image"]
print(f"  image after steps: dtype={img2.dtype} range=[{img2.min()},{img2.max()}] mean={float(img2.mean()):.2f}")

print("\n=== SCALE MATCH CHECK ===")
print(f"  H5 image mean ~ {v.mean():.1f} (uint8 expected ~100-220)")
print(f"  env image mean ~ {float(img.mean()):.1f}")
print(f"  state dim H5={sum(h5py.File(h5_path,'r')[f'data/demo_0/obs/{k}'].shape[1] for k in PROPRIOCEPTION_KEYS)} env={s.shape[-1]}")
