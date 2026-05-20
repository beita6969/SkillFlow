#!/bin/bash
REPO="${SKILLFLOW_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LOG="${SKILLFLOW_TRAINING_LOG:-$REPO/outputs/skillflow_general/training_log.jsonl}"
PYTHON="${SKILLFLOW_PYTHON:-python}"
MONITOR="${SKILLFLOW_MONITOR:-$REPO/scripts/monitor_evolution.py}"
LAST_EVOLUTION_STEP=-1

while true; do
    if [ ! -f "$LOG" ]; then
        sleep 60
        continue
    fi

    LATEST=$(tail -1 "$LOG" 2>/dev/null | $PYTHON -c "import sys,json; print(json.load(sys.stdin)['step'])" 2>/dev/null)

    if [ -z "$LATEST" ]; then
        sleep 60
        continue
    fi

    CURRENT_EVOL=$((LATEST / 10 * 10))

    if [ "$CURRENT_EVOL" -gt "$LAST_EVOLUTION_STEP" ] && [ "$CURRENT_EVOL" -gt 0 ] && [ "$LATEST" -ge "$((CURRENT_EVOL + 2))" ]; then
        echo ""
        echo "============================================"
        echo "  检测到进化 @ Step $CURRENT_EVOL (当前 Step $LATEST)"
        echo "  $(date '+%Y-%m-%d %H:%M:%S')"
        echo "============================================"
        $PYTHON "$MONITOR"
        LAST_EVOLUTION_STEP=$CURRENT_EVOL
    fi

    sleep 60
done
