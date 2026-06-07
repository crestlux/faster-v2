#!/usr/bin/env bash
# Disconnect-proof watchdog for Stage1 (low-dim integrity) + Stage2 (image probe).
# Runs in tmux, independent of Claude. Logs status every 5 min to logs/watchdog_status.log,
# detects crash/complete, and auto-relaunches Stage2 at batch=64 ONCE if it OOMs at batch=128.
cd /home/crestlux/faster_crestlux/faster-v2 || exit 1
ST=logs/watchdog_status.log
S2_RELAUNCHED=0
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$ST"; }
log "================ watchdog started (pid $$) ================"

relaunch_stage2_b64(){
  log "Stage2 -> RELAUNCHING at batch=64 (OOM fallback)"
  CUDA_VISIBLE_DEVICES=1 PROPRIO_VELOCITY=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_FLAGS="--xla_gpu_strict_conv_algorithm_picker=false" WANDB_MODE=offline \
  nohup python train_robo.py \
    --config=faster/agents/faster_expo_learner.py --dataset_dir=ph --env_name=square \
    --use_image_obs=True --batch_size=64 --utd_ratio=20 \
    --pretrain_steps=40000 --max_steps=250000 --start_training=5000 --chunk_size=1 \
    --config.ne_samples=1 --config.ne_samples_train=1 --config.r_action_scale=0.05 \
    --config.actor_num_blocks=4 --config.actor_encoder_lr=1e-4 \
    --eval_interval=10000 --eval_episodes=50 --offline_eval_interval=20000 \
    --augment_obs=True --state_proj_dim=64 --vision_pool=spatial_softmax --num_kp=32 \
    "--config.filter_critic_hidden_dims=(512,512,512)" --noshare_encoder \
    --checkpoint_model=True --checkpoint_keep=3 --save_video=False \
    --run_tag=image_probe_b64_softmax_vel --wandb_run_group=image_proper_recipe \
    > logs/stage2_image_b64_$(date +%m%d_%H%M).log 2>&1 &
}

while true; do
  # ---- Stage 1 (low-dim integrity) ----
  L1=$(ls -t logs/stage1_lowdim_*.log 2>/dev/null | head -1)
  D1=$(ls -td exp/square_ph_lowdim_*lowdim_evaltarget_official* 2>/dev/null | head -1)
  if pgrep -f "train_robo.py.*lowdim_evaltarget" >/dev/null; then
    st1=$(awk -F',' '$1=="training"&&$2=="actor_loss"{s=$4} END{print s+0}' "$D1/train.csv" 2>/dev/null)
    tr1=$(awk -F',' '$1=="evaluation"&&$2=="return"{printf "%.2f ",$3}' "$D1/eval.csv" 2>/dev/null | awk '{n=NF; for(i=(n>4?n-3:1);i<=n;i++)printf "%s ",$i}')
    e1=$(awk -F',' '$1=="evaluation"&&$2=="return"{m=$3} END{printf "%.2f",m+0}' "$D1/eval.csv" 2>/dev/null)
    n1=$(grep -ciE "nan|inf" "$D1/train.csv" 2>/dev/null)
    log "Stage1 ALIVE step=$st1 eval_last3=[ $tr1] nan=$n1"
    [ "$n1" -gt 0 ] 2>/dev/null && log "  !!! ALERT Stage1 NaN/Inf=$n1"
    # plateau alert: past 80k but latest eval still <0.55 (official ~0.8 by 100k)
    awk "BEGIN{exit !($st1>80000 && $e1<0.55)}" && log "  !!! ALERT Stage1 step=$st1 eval=$e1 <0.55 — UNDER official trajectory (possible fork/config gap, not reaching ~0.8)"
  else
    if grep -qE "200000/200000|100%\|" "$L1" 2>/dev/null; then log "Stage1 DONE (max_steps)"; else log "  !!! ALERT Stage1 DOWN/CRASHED (inspect $L1)"; fi
  fi

  # ---- Stage 2 (image probe), with guarded OOM relaunch ----
  L2=$(ls -t logs/stage2_image_*.log 2>/dev/null | head -1)
  D2=$(ls -td exp/square_ph_image_*image_probe_b*softmax_vel* 2>/dev/null | head -1)
  if pgrep -f "train_robo.py.*image_probe_b" >/dev/null; then
    off=$(tr '\r' '\n' < "$L2" 2>/dev/null | grep -oE "[0-9]+/40000" | tail -1)
    onl=$(tr '\r' '\n' < "$L2" 2>/dev/null | grep -oE "[0-9]+/250000" | tail -1)
    tr2=$(awk -F',' '$1=="evaluation"&&$2=="return"{printf "%.2f ",$3}' "$D2/eval.csv" 2>/dev/null | awk '{n=NF; for(i=(n>4?n-3:1);i<=n;i++)printf "%s ",$i}')
    nan2=$(grep -ciE "nan|inf" "$D2/train.csv" 2>/dev/null)
    log "Stage2 ALIVE off=$off onl=$onl eval_last3=[ $tr2] nan=$nan2"
    [ "$nan2" -gt 0 ] 2>/dev/null && log "  !!! ALERT Stage2 NaN/Inf=$nan2"
  else
    if grep -qiE "RESOURCE_EXHAUSTED|out of memory|OOM|XlaRuntimeError.*memory" "$L2" 2>/dev/null && [ "$S2_RELAUNCHED" = 0 ]; then
      S2_RELAUNCHED=1; relaunch_stage2_b64
    elif grep -qE "250000/250000" "$L2" 2>/dev/null; then log "Stage2 DONE (reached max_steps)";
    else log "Stage2 DOWN/CRASHED (inspect $L2)"; fi
  fi

  # ---- GPUs + disk ----
  g=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader -i 0,1 2>/dev/null | tr '\n' ' | ')
  fr=$(df -h . | awk 'NR==2{print $4}')
  log "GPU[$g] disk_free=$fr"
  [ "$(df . | awk 'NR==2{print $4}')" -lt 12000000 ] && log "!!! DISK LOW (<12G) — manual cleanup needed"

  sleep 300
done
