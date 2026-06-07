#!/usr/bin/env bash
# Stage 1 — CODE INTEGRITY CHECK: reproduce official LOW-DIM square ~0.8 with OUR code.
# Official recipe (code defaults): batch=256, chunk=1, ne=1, r_scale=0.15, pretrain=0, utd=20.
# Official reaches ~0.8 by ~100k env steps. If ours climbs similarly -> core intact (our image
# underperformance is a budget artifact, not a code regression). Cheap (low-dim = no vision).
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
  --eval_interval=10000 --eval_episodes=50 --offline_eval_interval=200000 \
  --checkpoint_model=True --checkpoint_keep=2 --save_video=False \
  --run_tag=lowdim_integrity_official --wandb_run_group=integrity_check
