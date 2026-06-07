#!/usr/bin/env bash
# lift ph image chunk1 — EXPO-FT deploy: 8 base + 8 stochastic edit, deterministic top-Q with
# PESSIMISTIC selection (full online ensemble min) to reject OOD-overestimated edits (no collapse).
# Training backup keeps ne_samples_train=1 (cheap). Offline checkpoints + eval every 1000.
set -e
export WANDB_MODE=offline MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.55
export XLA_FLAGS="--xla_gpu_strict_conv_algorithm_picker=false"
export PYTHONPATH=/home/crestlux/faster_crestlux/faster-v2
cd /home/crestlux/faster_crestlux/faster-v2
nohup /home/crestlux/miniforge3/envs/faster/bin/python -u train_robo.py \
  --config=faster/agents/faster_expo_learner.py \
  --dataset_dir=ph --env_name=lift \
  --use_image_obs=True --batch_size=64 --utd_ratio=20 \
  --save_video=False --checkpoint_model=True --checkpoint_keep=8 \
  --eval_interval=1000 --eval_episodes=20 \
  --offline_eval_interval=1000 --pretrain_steps=4000 --max_steps=8000 --start_training=1000 \
  --chunk_size=1 \
  --config.ne_samples=8 --config.ne_samples_train=1 \
  --config.n_base_deploy=8 \
  --config.actor_num_blocks=4 --config.actor_encoder_lr=1e-4 --config.r_action_scale=0.2 \
  --augment_obs=True --state_proj_dim=64 \
  "--config.filter_critic_hidden_dims=(512,512,512)" \
  --noshare_encoder \
  --project_name=faster_diag --run_tag=expoft_lift_chunk1 --wandb_run_group=diag \
  > _diag_logs/lift_expoft.log 2>&1 &
echo "LIFT_EXPOFT_PID=$!"
