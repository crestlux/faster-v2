#!/usr/bin/env bash
# square (NutAssemblySquare) image+proprio, chunk1 baseline — DIAGNOSTIC run.
# Goal: confirm the historically-0% hard task now climbs off 0% after the fixes.
# Checkpointed (every 2000) so _eval_modes.py can run the raw/filter/full ablation.
# pretrain 20000 (eval every 1000 -> read signal by ~5000); online 10000.
set -e
export WANDB_MODE=offline MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export XLA_FLAGS="--xla_gpu_strict_conv_algorithm_picker=false"
export PYTHONPATH=/home/crestlux/faster_crestlux/faster-v2
cd /home/crestlux/faster_crestlux/faster-v2
nohup /home/crestlux/miniforge3/envs/faster/bin/python -u train_robo.py \
  --config=faster/agents/faster_expo_learner.py \
  --dataset_dir=ph --env_name=square \
  --use_image_obs=True --batch_size=256 --utd_ratio=20 \
  --save_video=False --checkpoint_model=True --checkpoint_keep=12 \
  --eval_interval=1000 --eval_episodes=20 \
  --offline_eval_interval=3000 --pretrain_steps=6000 --max_steps=20000 --start_training=1000 \
  --chunk_size=1 \
  --config.ne_samples=8 --config.ne_samples_train=1 \
  --config.n_base_deploy=8 \
  --config.actor_num_blocks=4 --config.actor_encoder_lr=1e-4 --config.r_action_scale=0.05 \
  --augment_obs=True --state_proj_dim=64 \
  "--config.filter_critic_hidden_dims=(512,512,512)" \
  --noshare_encoder \
  --project_name=faster_diag --run_tag=diag_square_chunk1 --wandb_run_group=diag \
  > _diag_logs/square_chunk1.log 2>&1 &
echo "SQUARE_PID=$!"
