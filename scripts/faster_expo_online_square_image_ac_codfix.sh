#!/usr/bin/env bash
# Ablation B — CODE FIX ONLY: ne=1, utd=4, blocks=3 (OLD 하이퍼파라미터)
#   wandb name: square_ph_image_ac8e4_ne1utd4codefix_<timestamp>__s42
#   wandb group: ac8e4_ablation
#   A vs B: filter 28-dim 코드 수정 + share_encoder=False 효과만
#           (A = 구 코드, B = 새 코드, 나머지 HP 동일)
#
# 코드 변경 사항 (새 코드 기준):
#   - share_encoder=False: 논문 방식, critic 별도 encoder
#   - filter_act_dim=critic_action_dim=28 (exec 4×7): exec portion만 필터 스코어링
#   - target_critic subsample_ensemble 버그 수정
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
  --config.ne_samples=1 \
  --config.ne_samples_train=1 \
  --config.actor_num_blocks=3 \
  --noshare_encoder \
  --run_tag=ne1utd4codefix \
  --wandb_run_group=ac8e4_ablation
