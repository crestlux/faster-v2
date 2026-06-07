#!/usr/bin/env bash
# Stage 2 — IMAGE BUDGET PROBE: does more budget lift image square past the ~12% plateau?
# = Arm A's validated image config (SpatialSoftmax, r_scale=0.05, enc_lr=1e-4, state_proj=64, aug)
#   + batch 64->128 (toward the official 256 spec) + velocity proprio + longer steps (250k).
# Question: image online asymptote rises above 12% with budget (-> budget artifact, fair test),
#           or stays ~12% (-> real image bottleneck beyond budget). Gate at ~100k online steps.
# batch=128 chosen for FEASIBILITY: batch256 image would be ~2 weeks; 128 reaches the gate in ~2-3d.
export CUDA_VISIBLE_DEVICES=1
export PROPRIO_VELOCITY=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_strict_conv_algorithm_picker=false"
export WANDB_MODE=offline
source .env && python train_robo.py \
  --config=faster/agents/faster_expo_learner.py \
  --dataset_dir=ph --env_name=square \
  --use_image_obs=True \
  --batch_size=128 --utd_ratio=20 \
  --pretrain_steps=40000 --max_steps=250000 --start_training=5000 \
  --chunk_size=1 \
  --config.ne_samples=1 --config.ne_samples_train=1 \
  --config.r_action_scale=0.05 \
  --config.actor_num_blocks=4 --config.actor_encoder_lr=1e-4 \
  --eval_interval=10000 --eval_episodes=50 --offline_eval_interval=20000 \
  --augment_obs=True --state_proj_dim=64 \
  --vision_pool=spatial_softmax --num_kp=32 \
  "--config.filter_critic_hidden_dims=(512,512,512)" \
  --noshare_encoder \
  --checkpoint_model=True --checkpoint_keep=3 --save_video=False \
  --run_tag=image_probe_b128_softmax_vel --wandb_run_group=image_proper_recipe
