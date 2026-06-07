#!/usr/bin/env bash
# Quick live status of the two image-square runs. Use:
#   bash status.sh            # one snapshot
#   watch -n 30 bash status.sh # auto-refresh every 30s
cd /home/crestlux/faster_crestlux/faster_official || exit 1
echo "==================== $(date '+%m-%d %H:%M:%S') ===================="
for row in "GPU0 spatial_softmax /tmp/img_gpu0_pid" "GPU1 gap /tmp/img_gpu1_pid"; do
  set -- $row; gpu="$1"; pool="$2"; pidf="$3"
  pid=$(cat "$pidf" 2>/dev/null)
  # wrapper alive? and is its python child alive?
  alive=DOWN; ps -p "${pid:-0}" >/dev/null 2>&1 && alive=ALIVE
  # newest exp dir for this pool
  dir=""
  for c in $(ls -td exp/*image* 2>/dev/null); do
    grep -q "\"vision_pool\": \"$pool\"" "$c/flags.json" 2>/dev/null && { dir="$c"; break; }
  done
  if [ -z "$dir" ]; then echo "[$gpu/$pool] $alive  (no exp dir yet — loading)"; continue; fi
  offstep=$(awk -F',' '$1=="offline-training"&&$2=="q"{s=$4} END{print s+0}' "$dir/train.csv" 2>/dev/null)
  onstep=$(awk -F',' '$1=="training"&&$2=="critic_loss"{s=$4} END{print s+0}' "$dir/train.csv" 2>/dev/null)
  q=$(awk -F',' '$1=="offline-training"&&$2=="q"{v=$3} END{printf "%.3f",v}' "$dir/train.csv" 2>/dev/null)
  nan=$(grep -ciE "nan|inf" "$dir/train.csv" 2>/dev/null)
  # last 6 offline-eval and eval returns as value@step
  offev=$(awk -F',' '$1=="offline-evaluation"&&$2=="return"{printf "%.2f@%dk ",$3,$4/1000}' "$dir/eval.csv" 2>/dev/null | awk '{n=NF;for(i=(n>6?n-5:1);i<=n;i++)printf "%s ",$i}')
  ev=$(awk -F',' '$1=="evaluation"&&$2=="return"{printf "%.2f@%dk ",$3,$4/1000}' "$dir/eval.csv" 2>/dev/null | awk '{n=NF;for(i=(n>6?n-5:1);i<=n;i++)printf "%s ",$i}')
  echo "[$gpu/$pool] $alive  offline_step=$offstep  online_step=$onstep  Q=$q  nan=$nan"
  echo "        offline_eval(succ@step): $offev"
  [ -n "$ev" ] && echo "        online_eval(succ@step):  $ev"
done
echo "GPU:  $(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader -i 0,1 2>/dev/null | tr '\n' '  ')"
echo "disk: $(df -h . | awk 'NR==2{print $4}') free"
echo "(offline_step advances every 1000; offline_eval logged every 2000; success rate = mean over 50 episodes)"
