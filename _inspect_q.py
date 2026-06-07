"""Measure WHY stochastic edits collapse: compare Q(base) vs Q(base+edit) and inspect the edit
(per-dim magnitude, saturation, gripper). Uses the loaded checkpoint's ONLINE critic (10 heads).

Usage: CUDA_VISIBLE_DEVICES=0 python _inspect_q.py <run_dir> <ckpt_step>
"""
import json, sys
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp
import flax

sys.path.insert(0, ".")
from faster.agents import FasterEXPOLearner
from faster.data.robomimic_datasets import ENV_TO_HORIZON_MAP
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

kwargs = {k: v for k, v in cfg.items() if k != "model_cls"}
kwargs.update(chunk_size=chunk_size, exec_horizon=exec_h, state_proj_dim=int(flags["state_proj_dim"]),
              share_encoder=bool(flags["share_encoder"]), augment=bool(flags["augment_obs"]))
for tk in ("hidden_dims","critic_hidden_dims","outer_critic_hidden_dims","filter_critic_hidden_dims","residual_action_mask"):
    if isinstance(kwargs.get(tk), list): kwargs[tk] = tuple(kwargs[tk])
agent = FasterEXPOLearner.create(seed, jax.tree_map(lambda x: jnp.asarray(x), env.reset()), np.zeros((action_dim,), np.float32), **kwargs)
for c in [run_dir/"checkpoints"/f"checkpoint_offline_{ckpt_step}.msgpack", run_dir/"checkpoints"/f"checkpoint_{ckpt_step}.msgpack"]:
    if c.exists(): agent = flax.serialization.from_bytes(agent, open(c,"rb").read()); print(f"[loaded] {c}"); break

cad = agent.critic_action_dim; beta = float(cfg.get("r_action_scale"))
rng = jax.random.PRNGKey(0)
qb_all, qe_all, edit_abs, sat_base, sat_edit, win_edit = [], [], [], [], [], []
grip_edit = []
for trial in range(6):
    obs = env.reset()
    obs_b = jax.tree_map(lambda x: jnp.asarray(x)[None], obs)
    enc = agent._get_obs_encoding(obs_b, agent.actor.params)
    # 8 base candidates via the deploy path (filter keeps all 8)
    rng, k = jax.random.split(rng)
    cands, _, rng = agent._sample_candidates(rng, obs_b, 8, agent.actor.params, 0.0, actor_obs_enc=enc, k_keep=8)
    base = cands[0]  # (8, action_dim)
    d_exec = base[:, :cad]
    # stochastic edit (deploy form)
    rng, k = jax.random.split(rng)
    dist = agent.edit_actor.apply_fn({"params": agent.edit_actor.params}, jnp.broadcast_to(enc, (8, enc.shape[-1])), d_exec)
    x = dist.sample(seed=k)
    r_exec = jnp.clip(x * beta + d_exec, -1.0, 1.0)
    edited = jnp.concatenate([r_exec, base[:, cad:]], axis=-1)
    # Q via ONLINE critic (10 heads), min over heads
    qb = agent.critic.apply_fn({"params": agent.critic.params}, obs_b, base[:, :cad]).min(0)   # (8,)
    qe = agent.critic.apply_fn({"params": agent.critic.params}, obs_b, edited[:, :cad]).min(0)  # (8,)
    qb_all.append(np.asarray(qb)); qe_all.append(np.asarray(qe))
    edit_abs.append(np.abs(np.asarray(r_exec - d_exec)))     # (8, cad)
    sat_base.append((np.abs(np.asarray(d_exec)) > 0.98).mean())
    sat_edit.append((np.abs(np.asarray(r_exec)) > 0.98).mean())
    grip_edit.append(np.abs(np.asarray(r_exec - d_exec))[:, orig-1::orig].mean())  # gripper dim of each step
    win_edit.append(float(np.asarray(qe).max() > np.asarray(qb).max()))

qb_all = np.concatenate(qb_all); qe_all = np.concatenate(qe_all); edit_abs = np.concatenate(edit_abs)
print(f"=== ckpt {ckpt_step}  (critic_action_dim={cad}, beta={beta}) — ONLINE critic min-of-10 ===")
print(f"  Q(base) : mean={qb_all.mean():.3f}  max-per-state-avg={np.mean([q.max() for q in qb_all.reshape(6,8)]):.3f}")
print(f"  Q(edit) : mean={qe_all.mean():.3f}")
print(f"  Q(edit)-Q(base) per candidate: mean={float((qe_all-qb_all).mean()):+.3f}  (>0 => edit overvalued)")
print(f"  fraction of states where best edit Q > best base Q: {np.mean(win_edit):.2f}")
print(f"  edit |Δ| per dim: mean={edit_abs.mean():.3f} max={edit_abs.max():.3f}")
print(f"  gripper-dim edit |Δ|: mean={np.mean(grip_edit):.3f}")
print(f"  saturation frac (|a|>0.98): base={np.mean(sat_base):.3f} -> edited={np.mean(sat_edit):.3f}")
