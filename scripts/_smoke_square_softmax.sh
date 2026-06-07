#!/usr/bin/env bash
# Short end-to-end smoke: square image + vision_pool=spatial_softmax.
# Validates create()/pretrained-conv-load/offline-update-jit/inference-encoder, then exits.
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_strict_conv_algorithm_picker=false"
export WANDB_MODE=disabled
source .env 2>/dev/null
python train_robo.py \
  --config=faster/agents/faster_expo_learner.py \
  --dataset_dir=ph --env_name=square --use_image_obs=True \
  --batch_size=64 --utd_ratio=20 \
  --save_video=False --checkpoint_model=False \
  --eval_interval=100000 --eval_episodes=1 --offline_eval_interval=100000 \
  --pretrain_steps=3 --max_steps=4 --log_interval=1 \
  --chunk_size=1 \
  --config.ne_samples=1 --config.ne_samples_train=1 \
  --config.actor_num_blocks=4 --config.actor_encoder_lr=1e-4 --config.r_action_scale=0.05 \
  --augment_obs=True --state_proj_dim=64 \
  "--config.filter_critic_hidden_dims=(512,512,512)" \
  --noshare_encoder \
  --vision_pool=spatial_softmax --num_kp=32 \
  --run_tag=smoke_softmax --wandb_run_group=smoke
