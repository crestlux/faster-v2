#!/usr/bin/env bash
# Ablation A — OLD BASELINE: ne=1, utd=4, blocks=3, OLD code (filter_dim=56)
#   wandb name: square_ph_image_ac8e4_ne1utd4old_<timestamp>__s42
#   wandb group: ac8e4_ablation
#   현재 돌리는 실험과 동일한 하이퍼파라미터 (재현용)
#   주의: 이 스크립트는 현재 코드(filter 28-dim 수정됨)로 실행되므로
#         진짜 "old code" 재현은 git stash/checkout 필요
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.45
source .env && python train_robo.py \
  --dataset_dir=ph \
  --config.model_cls=FasterEXPOLearner \
  --env_name=square $@ \
  --use_image_obs=True \
  --batch_size=512 \
  --save_video=True \
  --utd_ratio=4 \
  --checkpoint_model=True \
  --eval_interval=5000 \
  --eval_episodes=20 \
  --chunk_size=8 \
  --exec_horizon=4 \
  --config.ne_samples=1 \
  --config.ne_samples_train=1 \
  --config.actor_num_blocks=3 \
  --run_tag=ne1utd4old \
  --wandb_run_group=ac8e4_ablation
