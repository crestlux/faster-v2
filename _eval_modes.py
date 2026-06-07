"""Load a FasterEXPOLearner checkpoint and evaluate the SAME policy under several
action-selection modes, to isolate whether Q-guided selection (filter + edit) helps
or hurts vs the raw diffusion BC policy.

Modes:
  raw     : N=1, no filter, no edit  -> pure single diffusion sample (BC)
  filterN : N=8 filter pick 1, no edit
  full    : N=8 filter + edit (ne_samples) + outer-Q argmax  (the deployed policy)

Usage:
  CUDA_VISIBLE_DEVICES=0 python _eval_modes.py <run_dir> <checkpoint_step> [num_eps]
e.g.
  python _eval_modes.py exp/lift_ph_image_..._s42 6000 30
"""
import json
import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import flax

sys.path.insert(0, ".")
from faster.agents import FasterEXPOLearner
from faster.data.robomimic_datasets import ENV_TO_HORIZON_MAP, get_robomimic_env
from faster.networks.evaluation_wandb_video import get_video_robomimic_env
from faster.evaluation import ActionChunkWrapper
from faster.train_robo_env_utils import _resolve_robomimic_dataset_path
from faster.utils import robomimic_datasets_root

run_dir = Path(sys.argv[1])
ckpt_step = int(sys.argv[2])
num_eps = int(sys.argv[3]) if len(sys.argv) > 3 else 30

flags = json.load(open(run_dir / "flags.json"))
cfg = flags["config"]
env_name = flags["env_name"]
use_image = flags["use_image_obs"]
chunk_size = int(flags["chunk_size"])
exec_horizon = int(flags["exec_horizon"]) if int(flags["exec_horizon"]) > 0 else chunk_size
seed = int(flags["seed"])
orig_action_dim = 7  # robomimic
action_dim = chunk_size * orig_action_dim

print(f"[cfg] env={env_name} image={use_image} chunk={chunk_size} exec={exec_horizon} "
      f"ne_samples={cfg.get('ne_samples')} N={cfg.get('N')} r_scale={cfg.get('r_action_scale')}")

robomimic_root = robomimic_datasets_root(Path("datasets/robomimic"))
dataset_path = _resolve_robomimic_dataset_path(robomimic_root, env_name, "ph", use_image)

example_action = np.zeros((1, orig_action_dim), dtype=np.float32)
eval_env = get_video_robomimic_env(str(dataset_path), example_action, env_name,
                                   render_offscreen=False, use_image_obs=use_image)
if chunk_size > 1:
    eval_env = ActionChunkWrapper(eval_env, chunk_size, exec_horizon=exec_horizon)
max_traj_len = ENV_TO_HORIZON_MAP[env_name]

# Build example obs from the env (matches training obs structure / dtype)
ex_obs = eval_env.reset()
ex_obs_sq = jax.tree_map(lambda x: jnp.asarray(x), ex_obs)
ex_act = np.zeros((action_dim,), dtype=np.float32)

# Reconstruct create kwargs from config (drop model_cls; inject chunk/exec)
kwargs = {k: v for k, v in cfg.items() if k != "model_cls"}
# ml_collections placeholders may serialize as None; create() handles None.
kwargs["chunk_size"] = chunk_size
kwargs["exec_horizon"] = exec_horizon
# train_robo.py writes flags.json BEFORE injecting these flags into config, so config holds
# stale defaults. The actual agent was built from the TOP-LEVEL flag values -> use those.
kwargs["state_proj_dim"] = int(flags["state_proj_dim"])
kwargs["share_encoder"] = bool(flags["share_encoder"])
kwargs["augment"] = bool(flags["augment_obs"])
# vision_pool/num_kp also live at top-level (config holds stale "gap"); .get for old ckpts.
kwargs["vision_pool"] = flags.get("vision_pool", "gap")
kwargs["num_kp"] = int(flags.get("num_kp", 32))
# tuple fields come back as lists from json -> coerce
for tk in ("hidden_dims", "critic_hidden_dims", "outer_critic_hidden_dims",
           "filter_critic_hidden_dims", "residual_action_mask"):
    if isinstance(kwargs.get(tk), list):
        kwargs[tk] = tuple(kwargs[tk])

agent = FasterEXPOLearner.create(seed, ex_obs_sq, ex_act, **kwargs)

_cands = [run_dir / "checkpoints" / f"checkpoint_offline_{ckpt_step}.msgpack",
          run_dir / "checkpoints" / f"checkpoint_{ckpt_step}.msgpack"]
ckpt = next((c for c in _cands if c.exists()), None)
assert ckpt is not None, f"no checkpoint found among {[str(c) for c in _cands]}"
with open(ckpt, "rb") as f:
    agent = flax.serialization.from_bytes(agent, f.read())
print(f"[loaded] {ckpt}")


def run_eval(ag, label, n):
    succ = 0
    rets = []
    for ep in range(n):
        obs = eval_env.reset()
        done = False
        steps = 0
        ep_ret = 0.0
        while not done and steps < max_traj_len:
            a, ag = ag.eval_actions(obs)
            obs, r, done, info = eval_env.step(np.asarray(a))
            ep_ret += r
            steps += int((info or {}).get("chunk_steps", 1))
        rets.append(ep_ret)
        if ep_ret > 0:
            succ += 1
    print(f"  [{label:8s}] success={succ}/{n} ({100*succ/n:.0f}%)  mean_return={np.mean(rets):.2f}")
    return succ / n


print(f"\n=== Eval ({num_eps} eps each) — deploy candidate scheme ===")
N = 8
# raw single diffusion sample (no filter pick, no edit)
run_eval(agent.replace(N=1, ne_samples=0, filter_at_eval=True, n_base_deploy=1), "raw_BC", num_eps)
# FASTER: filter -> 1 base, no edit
run_eval(agent.replace(N=N, ne_samples=0, filter_at_eval=True, n_base_deploy=1), "filter_1base", num_eps)
# EXPO-FT base pool: 8 base, top-Q, no edit
run_eval(agent.replace(N=N, ne_samples=0, filter_at_eval=True, n_base_deploy=N), f"{N}base_noedit", num_eps)
# EXPO-FT: 8 base + 8 stochastic edit, deterministic top-Q (the deployed policy)
run_eval(agent.replace(N=N, ne_samples=N, filter_at_eval=True, n_base_deploy=N), f"EXPO-FT_{N}base+{N}edit", num_eps)
