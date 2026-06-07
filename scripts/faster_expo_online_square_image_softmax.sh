#!/usr/bin/env bash
# Arm A: square image + SpatialSoftmax pooling head (vs GAP baseline = 5%).
# IDENTICAL to faster_expo_online_square_image.sh EXCEPT --vision_pool=spatial_softmax.
#   Single-variable ablation: aug stays at neutral baseline defaults (crop=0.95, rotation off),
#   state_proj_dim=64, batch=64, chunk1, ne=1 — all matching the 5% GAP baseline.
# WANDB offline (no surprise publish / no login hang on a background run); `wandb sync` later.
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.6
export XLA_FLAGS="--xla_gpu_strict_conv_algorithm_picker=false"
export WANDB_MODE=offline
source .env && python train_robo.py \
  --config=faster/agents/faster_expo_learner.py \
  --dataset_dir=ph \
  --env_name=square $@ \
  --use_image_obs=True \
  --batch_size=64 \
  --save_video=True \
  --utd_ratio=20 \
  --checkpoint_model=True \
  --checkpoint_keep=3 \
  --eval_interval=5000 \
  --eval_episodes=20 \
  --offline_eval_interval=10000 \
  --pretrain_steps=60000 \
  --max_steps=80000 \
  --chunk_size=1 \
  --config.ne_samples=1 \
  --config.ne_samples_train=1 \
  --config.actor_num_blocks=4 \
  --config.actor_encoder_lr=1e-4 \
  --config.r_action_scale=0.05 \
  --augment_obs=True \
  --state_proj_dim=64 \
  --vision_pool=spatial_softmax \
  --num_kp=32 \
  "--config.filter_critic_hidden_dims=(512,512,512)" \
  --noshare_encoder \
  --run_tag=chunk1_spatialsoftmax_armA \
  --wandb_run_group=square_vision_pool_ablation
