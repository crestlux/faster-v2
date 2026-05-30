#!/usr/bin/env bash
# Ablation C — ne=8 효과 격리: utd=4, blocks=3 유지, ne_samples만 1→8
#   wandb name: square_ph_image_ac8e4_ne8utd4_<timestamp>__s42
#   wandb group: ac8e4_ablation
#   B vs C: ne=1 vs ne=8 (코드/utd 통제) → edit_actor 후보 수 효과
#   C vs D: ne=8 상태에서 utd=4 vs utd=8+capacity 효과
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.45
source .env && python train_robo.py \
  --config=faster/agents/faster_expo_learner.py \
  --dataset_dir=ph \
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
  --config.ne_samples=8 \
  --config.ne_samples_train=8 \
  --config.actor_num_blocks=3 \
  --noshare_encoder \
  --run_tag=ne8utd4 \
  --wandb_run_group=ac8e4_ablation
