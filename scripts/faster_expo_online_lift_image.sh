#!/usr/bin/env bash
# lift (Lift, horizon=400) — BASELINE: chunk_size=1. image + proprioception, FASTER-EXPO.
#   가장 쉬운 task → 예산 작게. 목적: image+proprio 확장 검증(sanity) → 표준 EXPO ne_samples=1.
#   (_ac.sh는 chunk8·ne=8 — 다른 목적. chunk만 통제하려면 --config.ne_samples=8로 맞출 것.)
# online step은 _ac.sh와 동일 값(=정책 결정/update/wandb step 축 일치; ac는 결정당 4 env step 실행).
# 2026-06 fixes: image-scale fix, online-actor eval, actor_encoder_lr=1e-4, EXPO-FT augmentation.
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4
export XLA_FLAGS="--xla_gpu_strict_conv_algorithm_picker=false"
source .env && python train_robo.py \
  --config=faster/agents/faster_expo_learner.py \
  --dataset_dir=ph \
  --env_name=lift $@ \
  --use_image_obs=True \
  --batch_size=64 \
  --save_video=True \
  --utd_ratio=20 \
  --checkpoint_model=True \
  --checkpoint_keep=3 \
  --eval_interval=5000 \
  --eval_episodes=20 \
  --offline_eval_interval=5000 \
  --pretrain_steps=20000 \
  --max_steps=50000 \
  --chunk_size=1 \
  --config.ne_samples=1 \
  --config.ne_samples_train=1 \
  --config.actor_num_blocks=4 \
  --config.actor_encoder_lr=1e-4 \
  --config.r_action_scale=0.2 \
  --augment_obs=True \
  --state_proj_dim=64 \
  "--config.filter_critic_hidden_dims=(512,512,512)" \
  --noshare_encoder \
  --run_tag=chunk1_baseline_enclr1e4_aug \
  --wandb_run_group=lift_ac8e4_ablation
