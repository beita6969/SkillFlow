#!/bin/bash
# restart_sglang.sh — kill all SGLang servers and relaunch with same config
#
# Usage:
#   ./scripts/restart_sglang.sh                  # restart with current (non-deterministic) config
#   ./scripts/restart_sglang.sh --deterministic  # restart with --enable-deterministic-inference
#   ./scripts/restart_sglang.sh --quick          # skip warmup wait
#
# Default config (matches what's currently running):
#   port 8100 (GPU 0): supervisor + theta_ckpt110 LoRA
#   port 8101 (GPU 1): m_exec
#   port 8102 (GPU 2): m_exec
#   port 8103 (GPU 3): m_exec

set -u

REPO="${SKILLFLOW_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYBIN="${SKILLFLOW_PYTHON:-python}"
MODEL="${SKILLFLOW_BASE_MODEL:-Qwen/Qwen3.5-9B}"
LORA_PATH="${SKILLFLOW_LORA_PATH:-$REPO/outputs/skillflow_general/checkpoint_step_0110/supervisor_lora/theta}"
API_KEY="${SGLANG_API_KEY:-EMPTY}"
LOG_DIR=/tmp
TS=$(date +%Y%m%d_%H%M%S)
SUPERVISOR_MEM=${SGLANG_SUPERVISOR_MEM:-0.90}
MEXEC_MEM=${SGLANG_MEXEC_MEM:-0.92}
CUDA_GRAPH_MAX_BS=${SGLANG_CUDA_GRAPH_MAX_BS:-}
EXTRA_SGLANG_ARGS=()
if [ -n "$CUDA_GRAPH_MAX_BS" ]; then
  EXTRA_SGLANG_ARGS+=(--cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS")
fi

DETERMINISTIC=""
SKIP_WARMUP=0
for arg in "$@"; do
  case "$arg" in
    --deterministic)
      DETERMINISTIC="--enable-deterministic-inference"
      echo "[mode] deterministic inference ENABLED (slower 30-50%)"
      ;;
    --quick)
      SKIP_WARMUP=1
      echo "[mode] skipping warmup wait"
      ;;
  esac
done

# ─────────────────────────────────────────────────────────────────
# Step 1: kill existing SGLang servers
# ─────────────────────────────────────────────────────────────────
echo ""
echo "================ STEP 1: KILL ================"
# Restrict matches to real Python SGLang launchers. Plain `pgrep -f
# sglang.launch_server` can accidentally match wrapper/diagnostic shells whose
# command text contains that string.
PIDS=$(pgrep -f "$PYBIN -m sglang.launch_server" || true)
if [ -n "$PIDS" ]; then
  echo "[kill] found SGLang launchers: $PIDS"
  for p in $PIDS; do
    cmdline=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | head -c 120)
    echo "  killing PID=$p ($cmdline...)"
  done
  pkill -TERM -f "$PYBIN -m sglang.launch_server" || true
  sleep 5
  REMAINING=$(pgrep -f "$PYBIN -m sglang.launch_server" || true)
  if [ -n "$REMAINING" ]; then
    echo "[kill] some still alive, force-killing: $REMAINING"
    pkill -KILL -f "$PYBIN -m sglang.launch_server" || true
    sleep 3
  fi
else
  echo "[kill] no SGLang servers running"
fi

# Wait for GPU memory to fully release
echo "[gpu] waiting 15s for GPU memory release..."
sleep 15
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader 2>/dev/null | head -4
if command -v nvidia-smi >/dev/null 2>&1; then
  BUSY_GPUS=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null | awk -F, '$2+0 > 1000 {print $1 ":" $2 "MiB"}')
  if [ -n "$BUSY_GPUS" ]; then
    echo "ERROR: GPU memory did not release after killing SGLang; refusing to half-start. Busy GPUs: $BUSY_GPUS" >&2
    exit 1
  fi
fi

# ─────────────────────────────────────────────────────────────────
# Step 2: backup old logs
# ─────────────────────────────────────────────────────────────────
echo ""
echo "================ STEP 2: BACKUP LOGS ================"
mkdir -p "$LOG_DIR/sglang_logs_archive"
for log in sglang_supervisor.log sglang_mexec_8101.log sglang_mexec_8102.log sglang_mexec_8103.log; do
  if [ -f "$LOG_DIR/$log" ]; then
    mv "$LOG_DIR/$log" "$LOG_DIR/sglang_logs_archive/${log}.${TS}"
    echo "  archived $log"
  fi
done

# ─────────────────────────────────────────────────────────────────
# Step 3: launch supervisor (GPU 0, with LoRA)
# ─────────────────────────────────────────────────────────────────
echo ""
echo "================ STEP 3: LAUNCH SUPERVISOR (port 8100, GPU 0) ================"
CUDA_VISIBLE_DEVICES=0 setsid nohup $PYBIN -m sglang.launch_server \
    --model-path "$MODEL" \
    --port 8100 \
    --api-key "$API_KEY" \
    --served-model-name skillflow_eval \
    --enable-lora \
    --lora-paths "theta_ckpt110=$LORA_PATH" \
    --max-lora-rank 64 \
    --max-loras-per-batch 1 \
    --max-loaded-loras 1 \
    --lora-target-modules q_proj k_proj v_proj o_proj \
    --mem-fraction-static "$SUPERVISOR_MEM" \
    --context-length 32768 \
    --schedule-policy lpm \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --trust-remote-code \
    $DETERMINISTIC \
    "${EXTRA_SGLANG_ARGS[@]}" \
    > "$LOG_DIR/sglang_supervisor.log" 2>&1 &
SUPERVISOR_PID=$!
echo "[launch] supervisor PID=$SUPERVISOR_PID → $LOG_DIR/sglang_supervisor.log"

# ─────────────────────────────────────────────────────────────────
# Step 4: launch 3 m_exec workers (GPU 1/2/3, no LoRA)
# ─────────────────────────────────────────────────────────────────
echo ""
echo "================ STEP 4: LAUNCH M_EXEC × 3 ================"
declare -A MEXEC_PIDS
for i in 1 2 3; do
  port=$((8100 + i))
  gpu=$i
  CUDA_VISIBLE_DEVICES=$gpu setsid nohup $PYBIN -m sglang.launch_server \
      --model-path "$MODEL" \
      --port $port \
      --api-key "$API_KEY" \
      --served-model-name m_exec \
      --mem-fraction-static "$MEXEC_MEM" \
      --context-length 32768 \
      --schedule-policy lpm \
      --reasoning-parser qwen3 \
      --tool-call-parser qwen3_coder \
      --trust-remote-code \
      $DETERMINISTIC \
      "${EXTRA_SGLANG_ARGS[@]}" \
      > "$LOG_DIR/sglang_mexec_${port}.log" 2>&1 &
  pid=$!
  MEXEC_PIDS[$port]=$pid
  echo "[launch] m_exec port=$port GPU=$gpu PID=$pid → $LOG_DIR/sglang_mexec_${port}.log"
done

# ─────────────────────────────────────────────────────────────────
# Step 5: wait for warmup (poll /v1/models on each)
# ─────────────────────────────────────────────────────────────────
if [ "$SKIP_WARMUP" = "0" ]; then
  echo ""
  echo "================ STEP 5: WAIT FOR READY ================"
  echo "(polling each server's /v1/models endpoint, model loading typically 60-120s)"
  for port in 8100 8101 8102 8103; do
    echo -n "  port $port: waiting"
    for attempt in $(seq 1 60); do
      resp=$(timeout 3 curl -s -o /dev/null -w "%{http_code}" \
                  -H "Authorization: Bearer $API_KEY" \
                  "http://localhost:$port/v1/models" 2>/dev/null || echo "000")
      if [ "$resp" = "200" ]; then
        echo " READY (after ${attempt}×5s)"
        break
      fi
      echo -n "."
      sleep 5
      if [ "$attempt" = "60" ]; then
        echo " TIMEOUT (>5min)"
        echo "  check: tail $LOG_DIR/sglang_$([ $port = 8100 ] && echo supervisor || echo mexec_$port).log"
      fi
    done
  done

  echo ""
  echo "================ STEP 5b: POST-READY STABILITY CHECK ================"
  echo "(sleeping 20s, then re-checking every endpoint to catch immediate post-ready exits)"
  sleep 20
  ALL_READY=1
  for port in 8100 8101 8102 8103; do
    resp=$(timeout 5 curl -s -o /dev/null -w "%{http_code}" \
                -H "Authorization: Bearer $API_KEY" \
                "http://localhost:$port/v1/models" 2>/dev/null || echo "000")
    if [ "$resp" = "200" ]; then
      echo "  port $port: STABLE"
    else
      echo "  port $port: UNHEALTHY (HTTP $resp)"
      ALL_READY=0
    fi
  done
fi

# ─────────────────────────────────────────────────────────────────
# Step 6: report final status
# ─────────────────────────────────────────────────────────────────
echo ""
echo "================ STEP 6: STATUS ================"
echo "live SGLang processes:"
pgrep -af "$PYBIN -m sglang.launch_server" | grep -v grep | awk '{printf "  PID=%s  port=%s\n", $1, ($0 ~ /--port 8100/ ? "8100" : ($0 ~ /--port 8101/ ? "8101" : ($0 ~ /--port 8102/ ? "8102" : ($0 ~ /--port 8103/ ? "8103" : "?")))) }' || true
echo ""
echo "GPU usage:"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null

echo ""
echo "================ DONE ================"
echo "Logs:"
echo "  supervisor: $LOG_DIR/sglang_supervisor.log"
echo "  m_exec 8101: $LOG_DIR/sglang_mexec_8101.log"
echo "  m_exec 8102: $LOG_DIR/sglang_mexec_8102.log"
echo "  m_exec 8103: $LOG_DIR/sglang_mexec_8103.log"
[ -n "$DETERMINISTIC" ] && echo "Mode: DETERMINISTIC (slower but reproducible)" || echo "Mode: standard (high throughput, batch-dependent outputs)"
echo "Mem fractions: supervisor=$SUPERVISOR_MEM, m_exec=$MEXEC_MEM"
[ -n "$CUDA_GRAPH_MAX_BS" ] && echo "CUDA graph max batch size: $CUDA_GRAPH_MAX_BS"

if [ "${ALL_READY:-1}" != "1" ]; then
  echo "ERROR: at least one SGLang endpoint failed post-ready stability check" >&2
  exit 1
fi
