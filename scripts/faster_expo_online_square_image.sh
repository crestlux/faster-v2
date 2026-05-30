#!/usr/bin/env bash
# Baseline (no action chunk, single action, share_encoder=False)
#   wandb name: square_ph_image_lowdim_<timestamp>__s42
#   역할: action chunk ablation 비교 기준선 (chunk_size=1)
#
# 공정 비교: chunk_size(1 vs 8)·exec_horizon만 _ac.sh와 다르게 두고, 나머지 인자는 전부 동일.
#   동일하게 맞춘 항목: batch_size=256, utd_ratio=8, pretrain_steps=10000, max_steps=150000,
#                      ne_samples=8, ne_samples_train=8, actor_num_blocks=4, filter=512, wandb_group.
#   (chunk_size=1 → exec_horizon은 자동으로 1; ne_samples는 EXPO edit 메커니즘이라 양쪽 동일하게 유지)
# 메모리: batch=256 → peak ~6.6GB. _ac.sh와 동일 GPU 공존: 각 MEM_FRACTION=0.4 → 합 0.8(19.2GB) < 24GB.
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4
export XLA_FLAGS="--xla_gpu_strict_conv_algorithm_picker=false"
source .env && python train_robo.py \
  --config=faster/agents/faster_expo_learner.py \
  --dataset_dir=ph \
  --env_name=square $@ \
  --use_image_obs=True \
  --batch_size=256 \
  --save_video=True \
  --utd_ratio=8 \
  --checkpoint_model=True \
  --checkpoint_keep=3 \
  --eval_interval=5000 \
  --eval_episodes=20 \
  --pretrain_steps=10000 \
  --max_steps=150000 \
  --chunk_size=1 \
  --config.ne_samples=8 \
  --config.ne_samples_train=8 \
  --config.actor_num_blocks=4 \
  "--config.filter_critic_hidden_dims=(512,512,512)" \
  --noshare_encoder \
  --run_tag=chunk1_baseline_b256 \
  --wandb_run_group=ac8e4_ablation
