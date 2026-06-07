#!/usr/bin/env python
"""End-to-end verification with the FULL current EXPO-FT-aligned config (lift, chunk1 baseline).
Exercises: state_proj_dim=64 (+edit_actor init fix), backup min-of-2, chunk_discount=per_chunk,
actor_encoder_lr=1e-4, augment, batch=64/utd=20, per-task r_action_scale, online-actor eval, RedQ.
Prints losses (sensible/finite?) and the REAL eval-path success (learns / no training harm?)."""
import os, sys, time
import numpy as np, jax, jax.numpy as jnp
sys.argv = [sys.argv[0]]
from faster.agents import FasterEXPOLearner
from faster.data.robomimic_datasets import RoboD4RLDataset
from faster.networks.evaluation_wandb_video import get_video_robomimic_env
from faster.evaluation import evaluate_robo
from faster.agents.faster_expo_learner import get_config
from faster.utils import _load_robomimic_dataset
from faster.data.replay_buffer import _device_put_numeric_leaves

ds_path = "datasets/robomimic/image-only/lift/ph/image_v141.hdf5"
STEPS, BS, UTD = int(os.environ.get("STEPS","4000")), 64, 20

print("[v] loading lift dataset...", flush=True); t=time.time()
ds = RoboD4RLDataset(env=None, num_data=0, custom_dataset=_load_robomimic_dataset(ds_path, use_image_obs=True)); ds.seed(42)
print(f"[v] loaded {len(ds.dataset_dict['actions'])} trans in {time.time()-t:.0f}s", flush=True)

# FULL EXPO-FT-aligned config (mirrors scripts/faster_expo_online_lift_image.sh)
c = get_config()
c.filter_critic_hidden_dims=(512,512,512); c.state_proj_dim=64; c.actor_encoder_lr=1e-4
c.augment=True; c.r_action_scale=0.2; c.ne_samples=1; c.ne_samples_train=1
print(f"[v] cfg: state_proj={c.state_proj_dim} enc_lr={c.actor_encoder_lr} r_scale={c.r_action_scale} "
      f"chunk_discount={c.chunk_discount_mode} actor_tau={c.actor_tau} num_qs={c.num_qs} num_min_qs={c.num_min_qs} augment={c.augment}", flush=True)
kw=dict(c); kw.pop("model_cls"); kw.setdefault("chunk_size",1); kw.setdefault("exec_horizon",1)
obs0={"state":ds.dataset_dict["observations"]["state"][0],"image":ds.dataset_dict["observations"]["image"][0]}
agent=FasterEXPOLearner.create(42, obs0, ds.dataset_dict["actions"][0], **kw)
eval_env=get_video_robomimic_env(ds_path, np.zeros((1,7),np.float32), "lift", render_offscreen=False, use_image_obs=True)

print("[v] training (batch=64, utd=20)...", flush=True); t=time.time()
for i in range(1, STEPS+1):
    batch=_device_put_numeric_leaves(ds.sample(BS*UTD))
    agent, info = agent.update_offline(batch, UTD, True, True)
    if i % 500 == 0:
        al,cl,q,eq,ent = (float(info.get(k,float('nan'))) for k in ["actor_loss","critic_loss","q","edit_q","entropy"])
        print(f"[v] step {i}/{STEPS} actor_loss={al:.3f} critic_loss={cl:.4f} q={q:.3f} edit_q={eq:.3f} entropy={ent:.2f} ({(time.time()-t)/i:.2f}s/it)", flush=True)
        assert all(np.isfinite(x) for x in [al,cl,q]), "NON-FINITE LOSS!"
    if i % 2000 == 0:
        m = evaluate_robo(agent, eval_env, num_episodes=12, max_traj_len=400)  # REAL eval path (eval_actions: online actor + filter + edit + RedQ)
        print(f"[v] === step {i}: REAL-eval(full path) success_rate={m['return']:.3f} len={m['length']:.0f} ===", flush=True)
print("[v] DONE — no crash, losses finite", flush=True)
