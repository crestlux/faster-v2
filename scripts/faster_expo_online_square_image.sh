#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
source .env && python train_robo.py \
  --dataset_dir=ph \
  --config.model_cls=FasterEXPOLearner \
  --env_name=square $@ \
  --use_image_obs=True \
  --batch_size=128 \
  --save_video=True \
  --utd_ratio=5 \
  --checkpoint_model=True \
  --eval_interval=10000