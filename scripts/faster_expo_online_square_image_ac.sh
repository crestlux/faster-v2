#!/usr/bin/env bash
# ne=8 + utd=8 + blocks=4 + filter512
#   wandb name: square_ph_image_ac8e4_ne8utd8full_<timestamp>__s42
#   wandb group: ac8e4_ablation
#
# 코드 현황 (2026-05-30 수정 반영):
#   - 보상 버그 수정: make_chunk_dataset가 terminal 청크(reward/done/mask) 유지
#   - Filter critic: 전체 노이즈(action_dim=56)로 조건화 (exec 28 trim 제거)
#   - Temperature: unscaled residual 엔트로피 + target=-14 → α가 0으로 붕괴 안 함
#   - Critic 인코더: 공유 trunk 단일 ResNet18(~11M) + ensemble head, raw obs로 일관 사용
#                    (이전: 앙상블원마다 인코더 10개 → 추론 시 bypass되던 비일관 제거)
#
# 메모리/설정 (shared-encoder critic, 실측):
#   batch=256 → peak ~6.6GB (이전 10-encoder critic 대비 ~14GB에서 대폭 감소)
#   _image.sh와 동일 GPU 동시 실행: 각 MEM_FRACTION=0.4(9.6GB) → 합 0.8(19.2GB) < 24GB
#   utd_ratio=8: 메모리 중립(순차 루프, mini-batch=batch_size). 샘플효율은 utd로 조절
#   pretrain_steps=10000 / max_steps=150000 유지
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
  --chunk_size=8 \
  --exec_horizon=4 \
  --config.ne_samples=8 \
  --config.ne_samples_train=8 \
  --config.actor_num_blocks=4 \
  "--config.filter_critic_hidden_dims=(512,512,512)" \
  --noshare_encoder \
  --run_tag=ne8utd8full_b256_fixed \
  --wandb_run_group=ac8e4_ablation
