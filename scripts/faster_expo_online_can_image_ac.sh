#!/usr/bin/env bash
# can (PickPlaceCan, horizon=400) — ACTION CHUNK: chunk_size=8, exec_horizon=4. image + proprioception.
#   online step(max_steps/eval_interval)을 baseline과 동일 값으로 둠(=결정/update/wandb step 축 일치).
#   주의: 동일 max_steps에서 ac는 4× env step 실행(더 많은 실데이터·더 긴 wallclock).
# 2026-06 fixes: image-scale fix, online-actor eval, actor_encoder_lr=1e-4, EXPO-FT augmentation,
#   chunk backup: 단일 γ (논문 Eq.5, chunk_discount_mode=per_chunk 기본; ablation은 --config.chunk_discount_mode=per_step).
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4
export XLA_FLAGS="--xla_gpu_strict_conv_algorithm_picker=false"
source .env && python train_robo.py \
  --config=faster/agents/faster_expo_learner.py \
  --dataset_dir=ph \
  --env_name=can $@ \
  --use_image_obs=True \
  --batch_size=64 \
  --save_video=True \
  --utd_ratio=20 \
  --checkpoint_model=True \
  --checkpoint_keep=3 \
  --eval_interval=5000 \
  --eval_episodes=20 \
  --offline_eval_interval=5000 \
  --pretrain_steps=30000 \
  --max_steps=60000 \
  --chunk_size=8 \
  --exec_horizon=4 \
  --config.ne_samples=8 \
  --config.ne_samples_train=8 \
  --config.actor_num_blocks=4 \
  --config.actor_encoder_lr=1e-4 \
  --config.r_action_scale=0.1 \
  --augment_obs=True \
  --state_proj_dim=64 \
  "--config.filter_critic_hidden_dims=(512,512,512)" \
  --noshare_encoder \
  --run_tag=ac8e4_enclr1e4_aug \
  --wandb_run_group=can_ac8e4_ablation
