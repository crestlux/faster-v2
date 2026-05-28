#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.45
source .env && python train_robo.py \
  --dataset_dir=ph \
  --config.model_cls=FasterEXPOLearner \
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
  --resume_from=exp/square_ph_image_ac8e4_20260528_203952__s42/checkpoints/checkpoint_10000.msgpack \
  --resume_step=10000 \
  --wandb_resume_id=9erdj4yo
