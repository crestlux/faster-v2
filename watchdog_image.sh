#!/usr/bin/env bash
# Disconnect-proof watchdog for the two official-core image-square runs:
#   GPU0 = spatial_softmax, GPU1 = gap (single-variable ablation).
# Logs status every 5 min to logs/watchdog_image.log: offline/online step, recent eval
# returns, NaN/Inf checks, OOM/crash detection, GPU memory, disk.
cd /home/crestlux/faster_crestlux/faster_official || exit 1
mkdir -p logs
ST=logs/watchdog_image.log
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$ST"; }
log "================ image watchdog started (pid $$) ================"

report_run(){
  # $1 = human label, $2 = expected vision_pool, $3 = log file
  local label="$1" pool="$2" lf="$3"
  # find the newest _image exp dir whose flags.json vision_pool matches $pool
  local d="" cand
  for cand in $(ls -td exp/*image* 2>/dev/null); do
    if grep -q "\"vision_pool\": \"$pool\"" "$cand/flags.json" 2>/dev/null; then d="$cand"; break; fi
  done
  if [ -z "$d" ]; then log "  [$label] no exp dir yet"; return; fi
  local off onl ev nan
  off=$(awk -F',' '$1=="offline-training"&&$2=="q"{c++} END{print c+0}' "$d/train.csv" 2>/dev/null)
  onl=$(awk -F',' '$1=="training"&&$2=="critic_loss"{c++} END{print c+0}' "$d/train.csv" 2>/dev/null)
  ev=$(awk -F',' '$1=="evaluation"&&$2=="return"{printf "%.2f ",$3}' "$d/eval.csv" 2>/dev/null | awk '{n=NF; for(i=(n>4?n-3:1);i<=n;i++)printf "%s ",$i}')
  local offev
  offev=$(awk -F',' '$1=="offline-evaluation"&&$2=="return"{printf "%.2f ",$3}' "$d/eval.csv" 2>/dev/null | awk '{n=NF; for(i=(n>4?n-3:1);i<=n;i++)printf "%s ",$i}')
  nan=$(grep -ciE "nan|inf" "$d/train.csv" 2>/dev/null)
  log "  [$label/$pool] off_logs=$off onl_logs=$onl off_eval=[ $offev] eval=[ $ev] nan=$nan dir=$(basename $d)"
  [ "${nan:-0}" -gt 0 ] 2>/dev/null && log "    !!! ALERT $label NaN/Inf=$nan"
  if grep -qiE "RESOURCE_EXHAUSTED|out of memory|XlaRuntimeError" "$lf" 2>/dev/null; then log "    !!! ALERT $label OOM (inspect $lf)"; fi
}

LOOP=0
# Current offline runs to mirror to wandb (project faster_image) every 30 min.
# Future runs launch with WANDB_MODE=online and don't need this.
WANDB_SYNC_GLOB="wandb/offline-run-20260607_0302*"
WANDB_PROJECT_NAME="faster_image"
while true; do
  LOOP=$((LOOP+1))
  G0=$(cat /tmp/img_gpu0_pid 2>/dev/null); G1=$(cat /tmp/img_gpu1_pid 2>/dev/null)
  if ps -p "${G0:-0}" >/dev/null 2>&1; then a0=ALIVE; else a0=DOWN; fi
  if ps -p "${G1:-0}" >/dev/null 2>&1; then a1=ALIVE; else a1=DOWN; fi
  log "GPU0(spatial_softmax)=$a0  GPU1(gap)=$a1"
  report_run "GPU0" "spatial_softmax" "$(cat /tmp/img_gpu0_log 2>/dev/null)"
  report_run "GPU1" "gap" "$(cat /tmp/img_gpu1_log 2>/dev/null)"
  g=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader -i 0,1 2>/dev/null | tr '\n' '|')
  fr=$(df -h . | awk 'NR==2{print $4}')
  log "GPU[$g] disk_free=$fr"
  [ "$(df . | awk 'NR==2{print $4}')" -lt 12000000 ] && log "!!! DISK LOW (<12G)"
  # Mirror the current offline runs to wandb every 6th loop (~30 min).
  if [ $((LOOP % 6)) -eq 1 ]; then
    for d in $WANDB_SYNC_GLOB; do
      [ -d "$d" ] || continue
      timeout 150 wandb sync -p "$WANDB_PROJECT_NAME" "$d" >/dev/null 2>&1 && log "  wandb-sync OK $(basename $d)" || log "  wandb-sync FAIL $(basename $d)"
    done
  fi
  sleep 300
done
