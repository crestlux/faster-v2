#!/usr/bin/env bash
# Stage 1b — FAITHFUL official low-dim integrity: EXACT official config.
# vs prior Stage1, the ONLY changes are the 2 config diffs found vs upstream:
#   actor_tau 0.005->0.001 (our fork raised it for the IMAGE encoder, irrelevant/harmful for low-dim)
#   actor_num_blocks 4->3.
# Everything else already matched official (T=10, beta=vp, hidden_dims, num_qs=10, N=8, lr=3e-4, ...).
# If THIS reaches stable ~0.8 -> our core reproduces official (the fork's actor_tau drift was the cause);
# if it STILL plateaus ~0.4 -> deeper core-code regression (diff update logic vs upstream).
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_MODE=offline
source .env && python train_robo.py \
  --config=faster/agents/faster_expo_learner.py \
  --dataset_dir=ph --env_name=square \
  --use_image_obs=False \
  --batch_size=256 --utd_ratio=20 \
  --pretrain_steps=0 --max_steps=200000 --start_training=5000 \
  --chunk_size=1 \
  --config.ne_samples=1 --config.ne_samples_train=1 \
  --config.r_action_scale=0.15 \
  --config.actor_tau=0.001 \
  --config.actor_num_blocks=3 \
  --eval_interval=10000 --eval_episodes=50 --offline_eval_interval=200000 \
  --checkpoint_model=True --checkpoint_keep=2 --save_video=False \
  --run_tag=lowdim_integrity_officialcfg --wandb_run_group=integrity_check
