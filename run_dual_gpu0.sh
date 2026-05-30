#!/bin/bash
# Run two experiments simultaneously on GPU 0, each with 45% memory
# Experiment A: single action (chunk_size=1)
# Experiment B: action chunk (chunk_size=N, exec_horizon=N)
#
# Usage: bash run_dual_gpu0.sh [chunk_size] [exec_horizon] [env_name] [dataset_dir]
# Example: bash run_dual_gpu0.sh 5 5 can ph

CHUNK_SIZE=${1:-5}
EXEC_HORIZON=${2:-5}
ENV_NAME=${3:-can}
DATASET_DIR=${4:-ph}
CONFIG_FILE=${5:-faster/agents/faster_idql_learner.py}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR=exp/dual_${TIMESTAMP}
mkdir -p "$LOG_DIR"

echo "=== Dual GPU0 experiment: ${ENV_NAME} ${DATASET_DIR} ==="
echo "  Exp A: chunk_size=1 (single action)"
echo "  Exp B: chunk_size=${CHUNK_SIZE} exec_horizon=${EXEC_HORIZON}"
echo "  Config: ${CONFIG_FILE}"
echo "  Logs: ${LOG_DIR}"
echo ""

# Experiment A — single action (chunk_size=1)
CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 \
python train_robo.py \
  --config "$CONFIG_FILE" \
  --env_name "$ENV_NAME" \
  --dataset_dir "$DATASET_DIR" \
  --chunk_size 1 \
  --noshare_encoder \
  --run_tag "chunk1" \
  --log_dir "${LOG_DIR}/chunk1" \
  > "${LOG_DIR}/chunk1.log" 2>&1 &
PID_A=$!
echo "Experiment A (chunk=1) started: PID=${PID_A}"

# Brief stagger so JAX compilations don't collide
sleep 10

# Experiment B — action chunk
CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 \
python train_robo.py \
  --config "$CONFIG_FILE" \
  --env_name "$ENV_NAME" \
  --dataset_dir "$DATASET_DIR" \
  --chunk_size "$CHUNK_SIZE" \
  --exec_horizon "$EXEC_HORIZON" \
  --noshare_encoder \
  --run_tag "chunk${CHUNK_SIZE}e${EXEC_HORIZON}" \
  --log_dir "${LOG_DIR}/chunk${CHUNK_SIZE}" \
  > "${LOG_DIR}/chunk${CHUNK_SIZE}.log" 2>&1 &
PID_B=$!
echo "Experiment B (chunk=${CHUNK_SIZE} exec=${EXEC_HORIZON}) started: PID=${PID_B}"

echo ""
echo "Monitor logs:"
echo "  tail -f ${LOG_DIR}/chunk1.log"
echo "  tail -f ${LOG_DIR}/chunk${CHUNK_SIZE}.log"
echo "GPU memory: nvidia-smi"
echo ""
echo "To stop: kill ${PID_A} ${PID_B}"

wait $PID_A $PID_B
echo "Both experiments completed."
