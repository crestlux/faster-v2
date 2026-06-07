"""Inspect the edit actor's output distribution (loc / std / entropy) at a checkpoint.
Tells us whether the stochastic edit has degenerated to ~uniform noise (std saturated, entropy
near max) — i.e. whether the deploy collapse is the edit becoming meaningless.

Usage: CUDA_VISIBLE_DEVICES=0 python _inspect_edit.py <run_dir> <ckpt_step>
"""
import json, sys
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp
import flax

sys.path.insert(0, ".")
from faster.agents import FasterEXPOLearner
from faster.data.robomimic_datasets import ENV_TO_HORIZON_MAP, get_robomimic_env
from faster.networks.evaluation_wandb_video import get_video_robomimic_env
from faster.train_robo_env_utils import _resolve_robomimic_dataset_path
from faster.utils import robomimic_datasets_root

run_dir = Path(sys.argv[1]); ckpt_step = int(sys.argv[2])
flags = json.load(open(run_dir / "flags.json")); cfg = flags["config"]
env_name = flags["env_name"]; use_image = flags["use_image_obs"]
chunk_size = int(flags["chunk_size"]); exec_h = int(flags["exec_horizon"]) or chunk_size
seed = int(flags["seed"]); orig = 7; action_dim = chunk_size * orig

root = robomimic_datasets_root(Path("datasets/robomimic"))
dpath = _resolve_robomimic_dataset_path(root, env_name, "ph", use_image)
env = get_video_robomimic_env(str(dpath), np.zeros((1, orig), np.float32), env_name, render_offscreen=False, use_image_obs=use_image)
ex_obs = jax.tree_map(lambda x: jnp.asarray(x), env.reset())

kwargs = {k: v for k, v in cfg.items() if k != "model_cls"}
kwargs.update(chunk_size=chunk_size, exec_horizon=exec_h,
              state_proj_dim=int(flags["state_proj_dim"]), share_encoder=bool(flags["share_encoder"]),
              augment=bool(flags["augment_obs"]))
for tk in ("hidden_dims","critic_hidden_dims","outer_critic_hidden_dims","filter_critic_hidden_dims","residual_action_mask"):
    if isinstance(kwargs.get(tk), list): kwargs[tk] = tuple(kwargs[tk])
agent = FasterEXPOLearner.create(seed, ex_obs, np.zeros((action_dim,), np.float32), **kwargs)
for c in [run_dir/"checkpoints"/f"checkpoint_offline_{ckpt_step}.msgpack", run_dir/"checkpoints"/f"checkpoint_{ckpt_step}.msgpack"]:
    if c.exists():
        agent = flax.serialization.from_bytes(agent, open(c,"rb").read()); print(f"[loaded] {c}"); break

# Collect a batch of observations via resets (robust; std is what we care about)
obs_list = [env.reset() for _ in range(8)]
def stack(obs):
    if isinstance(obs[0], dict):
        return {k: jnp.asarray(np.stack([np.asarray(b[k]) for b in obs])) for k in obs[0]}
    return jnp.asarray(np.stack([np.asarray(b) for b in obs]))
obs_b = stack(obs_list)

cad = agent.critic_action_dim
enc = agent._get_obs_encoding(obs_b, agent.actor.params)   # (16, enc)
# condition the edit on a plausible base action (zeros = neutral); the loc/std reflect the policy
d_exec = jnp.zeros((enc.shape[0], cad), jnp.float32)
dist = agent.edit_actor.apply_fn({"params": agent.edit_actor.params}, enc, d_exec)
base = dist.distribution  # underlying MultivariateNormalDiag (pre-tanh)
loc = np.asarray(base.loc); scale = np.asarray(base.scale.diag)
x = np.asarray(dist.sample(seed=jax.random.PRNGKey(0)))   # tanh-squashed samples in (-1,1)
mode = np.asarray(dist.mode())
try:
    ent = float(np.asarray(dist.distribution.entropy()).mean())  # pre-tanh Normal entropy
except Exception:
    ent = float("nan")
beta = float(cfg.get("r_action_scale"))
print(f"=== edit actor @ {ckpt_step}  (critic_action_dim={cad}, r_action_scale={beta}) ===")
print(f"  pre-tanh loc : mean={loc.mean():+.3f} |mean|={np.abs(loc).mean():.3f} max|{np.abs(loc).max():.3f}|")
print(f"  pre-tanh std : mean={scale.mean():.3f} min={scale.min():.3f} max={scale.max():.3f}   (log_std_max clip = std<=e^2=7.39)")
print(f"  sampled x    : |mean|={np.abs(x).mean():.3f}  (|x|->1 means saturated/uniform-ish)")
print(f"  mode (tanh loc): |mean|={np.abs(mode).mean():.3f}")
print(f"  => deployed edit magnitude: sample beta*|x|~{beta*np.abs(x).mean():.3f}, mode beta*|mode|~{beta*np.abs(mode).mean():.3f}  (per-dim, action range 2)")
