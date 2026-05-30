#!/usr/bin/env bash
# [OBSOLETE ABLATION - edit_actor now inherently restricts to exec portion]
#
# Originally tested: limiting edit_actor to exec_horizon steps only (mask non-exec to 0).
# Current architecture: edit_actor outputs critic_action_dim=28 (exec only) by design,
# so masking non-exec dims is no longer needed - the architecture enforces it.
#
# If needed for intra-exec masking (e.g., mask some DOFs within exec horizon):
#   residual_action_mask must be (critic_action_dim,) = (28,)-shaped, all values in [0, 1].
#   Example below masks the last 14 dims (steps 3-4 of exec horizon) within exec portion:
#
# "--config.residual_action_mask=(1,1,1,1,1,1,1, 1,1,1,1,1,1,1, 0,0,0,0,0,0,0, 0,0,0,0,0,0,0)"
#
# Keeping script as reference; runs without mask (equivalent to no mask).
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
  --utd_ratio=8 \
  --checkpoint_model=True \
  --eval_interval=5000 \
  --eval_episodes=20 \
  --chunk_size=8 \
  --exec_horizon=4 \
  --config.ne_samples=8 \
  --config.ne_samples_train=8 \
  "--config.filter_critic_hidden_dims=(512,512,512)"
