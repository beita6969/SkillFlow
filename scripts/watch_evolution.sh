#!/bin/bash
# 监控训练进化 — 每60秒检查一次，检测到新进化步时输出报告
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

    # 获取最新 step
    LATEST=$(tail -1 "$LOG" 2>/dev/null | $PYTHON -c "import sys,json; print(json.load(sys.stdin)['step'])" 2>/dev/null)

    if [ -z "$LATEST" ]; then
        sleep 60
        continue
    fi

    # 检测进化步 (step % 10 == 0 且 step > 0)
    CURRENT_EVOL=$((LATEST / 10 * 10))

    if [ "$CURRENT_EVOL" -gt "$LAST_EVOLUTION_STEP" ] && [ "$CURRENT_EVOL" -gt 0 ] && [ "$LATEST" -ge "$((CURRENT_EVOL + 2))" ]; then
        # 进化后至少过了2步，数据足够做对比
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
