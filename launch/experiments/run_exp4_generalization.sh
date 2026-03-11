#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# 实验 4：多赛道泛化验证
#
# 在四种不同赛道上验证 ILC 的泛化能力：
#   circular  — 恒定曲率 (radius=10m, μ=0.3)
#   racetrack — 直道+弯道混合 (straight=20m, radius=3m, μ=0.1)
#   figure8   — 反向弯道交替 (radius=5m, μ=0.1)
#   square    — 急转弯 (side=20m, μ=0.1)
#
# 预计每个赛道 8 圈，共 ~ 30 分钟
#
# 用法：
#   bash run_exp4_generalization.sh                              # 全部 4 种赛道
#   bash run_exp4_generalization.sh --scenarios circular,figure8 # 指定赛道
#   bash run_exp4_generalization.sh --laps 12                    # 每赛道 12 圈
#   bash run_exp4_generalization.sh --dry-run
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# 自动 source catkin workspace
WS_SETUP="${HOME}/planner_code/mppi_swerve_drive_ros_ex/devel/setup.bash"
[[ -f "$WS_SETUP" ]] && source "$WS_SETUP"

EXP_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_ROOT="${HOME}/log/experiments"
SETTLE=15
COOLDOWN=10
NUM_LAPS=8
DRY_RUN=false
SCENARIOS="circular,racetrack,figure8,square"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --laps)      NUM_LAPS="$2"; shift 2 ;;
    --scenarios) SCENARIOS="$2"; shift 2 ;;
    --dry-run)   DRY_RUN=true; shift ;;
    *) echo "未知参数: $1"; exit 1 ;;
  esac
done

IFS=',' read -ra SCENE_LIST <<< "$SCENARIOS"
TS=$(date +%Y%m%d_%H%M%S)

# 每种赛道的预估圈时（秒）和摩擦系数
declare -A LAP_TIMES=( [circular]=35 [racetrack]=50 [figure8]=45 [square]=60 )
declare -A FRICTIONS=( [circular]=0.3 [racetrack]=0.10 [figure8]=0.10 [square]=0.10 )

TOTAL_GROUPS=${#SCENE_LIST[@]}

cat <<EOF

╔══════════════════════════════════════════════════════════╗
║  实验 4：多赛道泛化验证                                  ║
╠══════════════════════════════════════════════════════════╣
║  赛道：$(echo "${SCENE_LIST[*]}" | sed 's/ /, /g')
║  每赛道 ${NUM_LAPS} 圈，共 ${TOTAL_GROUPS} 组
╚══════════════════════════════════════════════════════════╝

EOF

run_idx=0
for scenario in "${SCENE_LIST[@]}"; do
  run_idx=$((run_idx + 1))
  lap_t=${LAP_TIMES[$scenario]:-45}
  friction=${FRICTIONS[$scenario]:-0.10}
  drive_sec=$((NUM_LAPS * lap_t))

  TAG="exp4_gen_${scenario}_${TS}"
  LOG_DIR="${LOG_ROOT}/${TAG}"
  mkdir -p "$LOG_DIR"

  echo "──────────────────────────────────────────────────────────"
  echo "  [${run_idx}/${TOTAL_GROUPS}] 赛道: ${scenario}  μ=${friction}  ${NUM_LAPS}圈 ~$((drive_sec/60))min"
  echo "  → ${LOG_DIR}"
  echo "──────────────────────────────────────────────────────────"

  if $DRY_RUN; then
    echo "  [DRY-RUN] 跳过。"
    continue
  fi

  roslaunch "${EXP_DIR}/exp4_generalization.launch" \
    "scenario:=${scenario}" \
    "wheel_friction:=${friction}" \
    eval_log_dir:="$LOG_DIR" \
    > "${LOG_DIR}/roslaunch_stdout.log" 2>&1 &
  PID=$!
  trap 'echo ""; kill -INT $PID 2>/dev/null || true; wait $PID 2>/dev/null; exit 130' INT

  sleep "$SETTLE"

  elapsed=0
  while [ $elapsed -lt $drive_sec ]; do
    if ! kill -0 "$PID" 2>/dev/null; then break; fi
    sleep 10; elapsed=$((elapsed + 10))
    printf "\r  [%s] %3d%% | %ds / %ds | ~第 %d 圈" \
      "$scenario" $((elapsed * 100 / drive_sec)) "$elapsed" "$drive_sec" $((elapsed / lap_t))
  done

  echo ""
  kill -INT "$PID" 2>/dev/null || true
  wait "$PID" 2>/dev/null || true
  echo "  [$(date +%H:%M:%S)] ${scenario} 完成。"

  if [ $run_idx -lt $TOTAL_GROUPS ]; then
    echo "  冷却 ${COOLDOWN}s..."
    sleep "$COOLDOWN"
  fi
done

echo ""
echo "══════════════════════════════════════════════════════════"
echo "  实验 4 全部完成 (${TOTAL_GROUPS} 种赛道)。"
echo ""
echo "  画图命令:"
COLLECTED_DIRS=""
for scenario in "${SCENE_LIST[@]}"; do
  d="${LOG_ROOT}/exp4_gen_${scenario}_${TS}"
  [ -d "$d" ] && COLLECTED_DIRS="${COLLECTED_DIRS} ${d}"
done
echo "    python3 ${EXP_DIR}/plot_exp4_generalization.py${COLLECTED_DIRS}"
echo "══════════════════════════════════════════════════════════"
