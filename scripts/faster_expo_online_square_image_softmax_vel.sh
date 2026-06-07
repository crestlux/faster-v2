#!/usr/bin/env bash
# Arm A2: square image + SpatialSoftmax pooling + VELOCITY proprio (state 9->17).
# Single-variable vs Arm A (faster_expo_online_square_image_softmax.sh): ONLY adds PROPRIO_VELOCITY=1.
#   vision_pool=spatial_softmax (same as Arm A), aug neutral defaults, batch=64, chunk1, ne=1.
#   PROPRIO_VELOCITY=1 -> robomimic_datasets appends eef_vel_lin/ang + gripper_qvel; object stays excluded.
# Tests whether richer proprio (VITA-style) helps ON TOP of the winning SpatialSoftmax config.
# Runs on GPU 1 (parallel with Arm A on GPU 0). WANDB offline.
export CUDA_VISIBLE_DEVICES=1
export PROPRIO_VELOCITY=1
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
  --run_tag=chunk1_softmax_vel_armA2 \
  --wandb_run_group=square_vision_pool_ablation
