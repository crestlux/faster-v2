#!/usr/bin/env bash
# square (NutAssemblySquare, horizon=400) — BASELINE: chunk_size=1 (single action)
#   image + proprioception, FASTER-EXPO. 목적: image+proprio 확장이 원본 low-dim FASTER만큼
#   학습되는지 검증(sanity). 그래서 표준 EXPO 설정 ne_samples=1(=원본 default)을 쓴다.
#
# 2026-06 수정 반영: 이미지 스케일 fix, eval/rollout이 online actor 사용,
#   actor 이미지 인코더 LR 노출(1e-5 frozen → 1e-4), EXPO-FT augmentation on.
#
# online step 정렬(사용자 선택): max_steps/eval_interval을 _ac.sh와 **동일 값**으로 둠.
#   의미: 정책 결정 횟수·gradient update 횟수·wandb step축이 일치(그래프 직접 overlay).
#   단, _ac.sh는 결정당 exec_horizon(4) env step을 실행하므로 동일 max_steps에서 env 상호작용은 4배
#   (= ac가 더 많은 실데이터·더 긴 wallclock). return/length는 env-step 기준이라 그대로 비교 가능.
# baseline(ne=1) vs _ac.sh(chunk8, ne=8)는 chunk_size뿐 아니라 ne도 달라 "다른 목적의 실험"이다.
#   chunk_size만 통제한 깔끔한 ablation을 원하면 baseline을 --config.ne_samples=8로 맞춰 돌리면 됨.
# step 수는 시작점 — 필요시 $@로 조정(예: --pretrain_steps=100000).
# 메모리: batch=256 peak ~6.6GB. 단독 실행이면 MEM_FRACTION을 0.9로 올려도 됨.
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4
export XLA_FLAGS="--xla_gpu_strict_conv_algorithm_picker=false"
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
  "--config.filter_critic_hidden_dims=(512,512,512)" \
  --noshare_encoder \
  --run_tag=chunk1_baseline_enclr1e4_aug \
  --wandb_run_group=square_ac8e4_ablation
