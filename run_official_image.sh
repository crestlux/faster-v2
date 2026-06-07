#!/usr/bin/env bash
# Image square on the clean official core (image-only, chunk_size=1).
# Recipe: ResNet18 + <POOL> pooling + 64-d proprio projection (9-d proprio: eef_pos +
# eef_quat + gripper_qpos; NO privileged object; NO velocity) + train-time image augmentation.
# Critic owns ONE shared TD-trained encoder (RedQ-style, share_encoder=False), warm-started
# from the actor; filter/edit actors are pure MLPs on flat actor features (no extra ResNets).
# All RL math is the official's (target_actor deploy, r_action_scale log-prob correction, N=8,
# ne_samples=1, r_action_scale=0.15, actor_tau=0.001, num_blocks=3, filter_at_eval, zscore).
#
# Usage: run_official_image.sh <GPU> <POOL=spatial_softmax|gap> <BATCH> <RUN_TAG>
set -euo pipefail
GPU="${1:-0}"
POOL="${2:-spatial_softmax}"
BATCH="${3:-256}"
RUN_TAG="${4:-img_${POOL}_b${BATCH}}"
export CUDA_VISIBLE_DEVICES="$GPU" MUJOCO_GL=egl DISPLAY= JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_strict_conv_algorithm_picker=false"
export WANDB_MODE=online
export ROBOMIMIC_DATASETS_PATH=/home/crestlux/faster_crestlux/faster-v2/datasets/robomimic
export PYTHONPATH=/home/crestlux/faster_crestlux/faster-v2
cd /home/crestlux/faster_crestlux/faster-v2
python train_robo.py \
  --env_name=square --dataset_dir=ph --config.model_cls=FasterEXPOLearner \
  --use_image_obs=True --vision_pool="$POOL" --state_proj_dim=64 --num_kp=32 --augment_obs=True \
  --config.actor_encoder_lr="${ACT_ENC_LR:-1e-4}" --config.critic_encoder_lr=1e-4 \
  --batch_size="$BATCH" --utd_ratio="${UTD:-5}" \
  --max_steps=250000 --start_training=5000 --pretrain_steps="${PRETRAIN:-20000}" \
  --eval_interval=10000 --offline_eval_interval="${OFF_EVAL:-2000}" --eval_episodes=50 --skip_initial_eval=True \
  --checkpoint_model=False --save_video=False --tqdm=True \
  --project_name=faster_image --wandb_run_group=official_image --wandb_tags="image,square,$POOL"
