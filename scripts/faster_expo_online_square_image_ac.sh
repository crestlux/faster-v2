#!/usr/bin/env bash
# square (NutAssemblySquare, horizon=400) — ACTION CHUNK: chunk_size=8, exec_horizon=4
#   image + proprioception, FASTER-EXPO (EXPO-FT style chunk critic/edit).
#
# 2026-06 수정 반영: 이미지 스케일 fix, eval/rollout이 online actor 사용(critic/filter 학습과 일관),
#   actor 인코더 LR 노출(1e-4), EXPO-FT augmentation on,
#   chunk critic backup: 단일 γ (논문 Eq.5, chunk_discount_mode=per_chunk 기본 — 성능 목적).
#     (ablation으로 per-env-step γ 고정을 원하면 --config.chunk_discount_mode=per_step)
#
# online step 정렬(사용자 선택): max_steps/eval_interval을 _image.sh(baseline)와 **동일 값**으로 둠.
#   → 정책 결정 횟수·gradient update 횟수·wandb step축 일치(그래프 직접 overlay).
#   주의: ac는 결정당 exec_horizon(4) env step 실행 → 동일 max_steps에서 env 상호작용 4배(더 많은 실데이터),
#   wallclock도 더 김. (sample-efficiency를 env-step 기준으로 보려면 ac max_steps를 ÷4 하면 됨: $@로 조정.)
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
  --chunk_size=8 \
  --exec_horizon=4 \
  --config.ne_samples=8 \
  --config.ne_samples_train=8 \
  --config.actor_num_blocks=4 \
  --config.actor_encoder_lr=1e-4 \
  --config.r_action_scale=0.05 \
  --augment_obs=True \
  --state_proj_dim=64 \
  "--config.filter_critic_hidden_dims=(512,512,512)" \
  --noshare_encoder \
  --run_tag=ac8e4_enclr1e4_aug \
  --wandb_run_group=square_ac8e4_ablation
