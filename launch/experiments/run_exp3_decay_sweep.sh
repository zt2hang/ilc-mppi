#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# 实验 3：遗忘因子 γ 参数扫描（圆形赛道）
#
# 逐一运行 γ ∈ {0.980, 0.990, 0.995, 0.999, 1.000}，对比：
#   - 收敛速度（达到目标 RMSE 所需的圈数）
#   - 稳态残余误差
#   - 偏置 RMS 饱和水平
#
# 场景：圆形 (radius=10m)，μ=0.3
# 预计每组 15 圈 × 35s ≈ 9 分钟，5 组共 ~ 50 分钟
#
# 用法：
#   bash run_exp3_decay_sweep.sh                     # 扫描全部 5 个 γ 值
#   bash run_exp3_decay_sweep.sh  --laps 20          # 每组 20 圈
#   bash run_exp3_decay_sweep.sh  --values 0980,0999 # 只跑指定 γ 值
#   bash run_exp3_decay_sweep.sh  --dry-run
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# 自动 source catkin workspace
WS_SETUP="${HOME}/planner_code/mppi_swerve_drive_ros_ex/devel/setup.bash"
[[ -f "$WS_SETUP" ]] && source "$WS_SETUP"

EXP_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_DIR="${EXP_DIR}/config"
LOG_ROOT="${HOME}/log/experiments"
LAP_TIME=35
NUM_LAPS=15
SETTLE=15
COOLDOWN=10
DRY_RUN=false
DECAY_VALUES="0980,0990,0995,0999,1000"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --laps)   NUM_LAPS="$2"; shift 2 ;;
    --values) DECAY_VALUES="$2"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    *) echo "未知参数: $1"; exit 1 ;;
  esac
done

DRIVE_SEC=$((NUM_LAPS * LAP_TIME))
TS=$(date +%Y%m%d_%H%M%S)

IFS=',' read -ra DECAYS <<< "$DECAY_VALUES"
TOTAL_GROUPS=${#DECAYS[@]}
TOTAL_MIN=$(( (SETTLE + DRIVE_SEC + COOLDOWN) * TOTAL_GROUPS / 60 ))

cat <<EOF

╔══════════════════════════════════════════════════════════╗
║  实验 3：遗忘因子 γ 参数扫描                             ║
╠══════════════════════════════════════════════════════════╣
║  赛道：圆形 (radius=10m)    摩擦系数：μ=0.3            ║
║  扫描值：γ ∈ {$(echo "${DECAYS[*]}" | sed 's/ /, /g')}
║  每组 ${NUM_LAPS} 圈 (~$((DRIVE_SEC/60))min) × ${TOTAL_GROUPS} 组 → 总计 ~${TOTAL_MIN} 分钟
╚══════════════════════════════════════════════════════════╝

EOF

run_idx=0
for decay in "${DECAYS[@]}"; do
  run_idx=$((run_idx + 1))
  # 标准化为 4 位零填充（兼容 --values 980 和 --values 0980）
  decay=$(printf '%04d' "$decay")
  config_file="${CONFIG_DIR}/decay_${decay}.yaml"
  if [ ! -f "$config_file" ]; then
    # γ=0.995 is the default in ilc_enabled.yaml
    config_file="${CONFIG_DIR}/ilc_enabled.yaml"
  fi

  TAG="exp3_decay_${decay}_${TS}"
  LOG_DIR="${LOG_ROOT}/${TAG}"
  mkdir -p "$LOG_DIR"

  gamma_display="0.${decay:1}"
  [[ "$decay" == "1000" ]] && gamma_display="1.000"

  echo "──────────────────────────────────────────────────────────"
  echo "  [${run_idx}/${TOTAL_GROUPS}] γ = ${gamma_display}  → ${LOG_DIR}"
  echo "──────────────────────────────────────────────────────────"

  if $DRY_RUN; then
    echo "  [DRY-RUN] 跳过。"
    continue
  fi

  roslaunch "${EXP_DIR}/exp3_decay_sweep.launch" \
    "decay_config:=${config_file}" \
    "eval_tag_suffix:=decay_${decay}" \
    eval_log_dir:="$LOG_DIR" \
    > "${LOG_DIR}/roslaunch_stdout.log" 2>&1 &
  PID=$!
  trap 'echo ""; kill -INT $PID 2>/dev/null || true; wait $PID 2>/dev/null; exit 130' INT

  sleep "$SETTLE"

  elapsed=0
  while [ $elapsed -lt $DRIVE_SEC ]; do
    if ! kill -0 "$PID" 2>/dev/null; then break; fi
    sleep 10; elapsed=$((elapsed + 10))
    printf "\r  [γ=%s] %3d%% | %ds / %ds | ~第 %d 圈" \
      "$gamma_display" $((elapsed * 100 / DRIVE_SEC)) "$elapsed" "$DRIVE_SEC" $((elapsed / LAP_TIME))
  done

  echo ""
  kill -INT "$PID" 2>/dev/null || true
  wait "$PID" 2>/dev/null || true
  echo "  [$(date +%H:%M:%S)] γ=${gamma_display} 完成。"

  if [ $run_idx -lt $TOTAL_GROUPS ]; then
    echo "  冷却 ${COOLDOWN}s..."
    sleep "$COOLDOWN"
  fi
done

echo ""
echo "══════════════════════════════════════════════════════════"
echo "  实验 3 全部完成 (${TOTAL_GROUPS} 组)。"
echo ""
echo "  画图命令:"
COLLECTED_DIRS=""
for decay in "${DECAYS[@]}"; do
  decay=$(printf '%04d' "$decay")
  d="${LOG_ROOT}/exp3_decay_${decay}_${TS}"
  [ -d "$d" ] && COLLECTED_DIRS="${COLLECTED_DIRS} ${d}"
done
echo "    python3 ${EXP_DIR}/plot_exp3_decay_sweep.py${COLLECTED_DIRS}"
echo "══════════════════════════════════════════════════════════"
