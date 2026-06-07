#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl DISPLAY= JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=offline
export ROBOMIMIC_DATASETS_PATH=/home/crestlux/faster_crestlux/faster-v2/datasets/robomimic/low_dim
export PYTHONPATH=/home/crestlux/faster_crestlux/faster-v2
cd /home/crestlux/faster_crestlux/faster-v2
python train_robo.py \
  --env_name=square --dataset_dir=ph --config.model_cls=FasterEXPOLearner \
  --batch_size=256 --max_steps=200000 --eval_interval=10000 --eval_episodes=50 \
  --wandb_run_group=official_baseline
