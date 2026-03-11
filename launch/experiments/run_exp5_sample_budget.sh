#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# 实验 5：采样数预算 — ILC 作为计算资源的替代品
#
# 对比 K ∈ {256, 512, 1000, 2000, 4000} × {ILC ON, ILC OFF} 共 10 组
# 场景：racetrack (straight=20m, radius=3m)，μ=0.10
# 预计每组 10 圈 × 50s ≈ 8 分钟，10 组共 ~ 90 分钟
#
# 核心问题：在采样数受限时 (K=256,512)，ILC 先验能否弥补采样不足？
#
# 用法：
#   bash run_exp5_sample_budget.sh                        # 全部 10 组
#   bash run_exp5_sample_budget.sh --laps 15              # 每组 15 圈
#   bash run_exp5_sample_budget.sh --K 256,1000           # 只跑指定 K 值
#   bash run_exp5_sample_budget.sh --ilc on               # 只跑 ILC ON
#   bash run_exp5_sample_budget.sh --dry-run
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# 自动 source catkin workspace
WS_SETUP="${HOME}/planner_code/mppi_swerve_drive_ros_ex/devel/setup.bash"
[[ -f "$WS_SETUP" ]] && source "$WS_SETUP"

EXP_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_DIR="${EXP_DIR}/config"
LOG_ROOT="${HOME}/log/experiments"
LAP_TIME=50
NUM_LAPS=10
SETTLE=15
COOLDOWN=10
DRY_RUN=false
K_VALUES="0256,0512,1000,2000,4000"
ILC_FILTER=""  # "on" | "off" | "" (both)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --laps)    NUM_LAPS="$2"; shift 2 ;;
    --K)       K_VALUES="$2"; shift 2 ;;
    --ilc)     ILC_FILTER="$2"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    *) echo "未知参数: $1"; exit 1 ;;
  esac
done

DRIVE_SEC=$((NUM_LAPS * LAP_TIME))
TS=$(date +%Y%m%d_%H%M%S)

IFS=',' read -ra KS <<< "$K_VALUES"

# Build run list
RUNS=()
for k in "${KS[@]}"; do
  for ilc in on off; do
    if [[ -n "$ILC_FILTER" && "$ILC_FILTER" != "$ilc" ]]; then continue; fi
    RUNS+=("${k}:${ilc}")
  done
done

TOTAL_GROUPS=${#RUNS[@]}
TOTAL_MIN=$(( (SETTLE + DRIVE_SEC + COOLDOWN) * TOTAL_GROUPS / 60 ))

cat <<EOF

╔══════════════════════════════════════════════════════════╗
║  实验 5：采样数预算分析                                  ║
╠══════════════════════════════════════════════════════════╣
║  赛道：跑道形 (straight=20m, radius=3m)  μ=0.10        ║
║  K 值：$(echo "${KS[*]}" | sed 's/ /, /g')
║  每组 ${NUM_LAPS} 圈 × ${TOTAL_GROUPS} 组 → 总计 ~${TOTAL_MIN} 分钟
╚══════════════════════════════════════════════════════════╝

EOF

run_idx=0
for run_spec in "${RUNS[@]}"; do
  run_idx=$((run_idx + 1))
  k="${run_spec%%:*}"
  ilc="${run_spec##*:}"

  # 标准化 K 值为 4 位零填充（兼容 --K 256 和 --K 0256）
  k=$(printf '%04d' "$k")

  # Determine config file
  if [[ "$k" == "2000" ]]; then
    if [[ "$ilc" == "on" ]]; then
      config_file="${CONFIG_DIR}/ilc_enabled.yaml"
    else
      config_file="${CONFIG_DIR}/ilc_disabled.yaml"
    fi
  else
    config_file="${CONFIG_DIR}/samples_${k}_ilc_${ilc}.yaml"
  fi

  ilc_zh="开启"; [[ "$ilc" == "off" ]] && ilc_zh="关闭"
  TAG="exp5_K${k}_ilc_${ilc}_${TS}"
  LOG_DIR="${LOG_ROOT}/${TAG}"
  mkdir -p "$LOG_DIR"

  echo "──────────────────────────────────────────────────────────"
  echo "  [${run_idx}/${TOTAL_GROUPS}] K=${k}  ILC=${ilc_zh}  ${NUM_LAPS}圈 ~$((DRIVE_SEC/60))min"
  echo "  config: $(basename "$config_file")"
  echo "  → ${LOG_DIR}"
  echo "──────────────────────────────────────────────────────────"

  if $DRY_RUN; then
    echo "  [DRY-RUN] 跳过。"
    continue
  fi

  roslaunch "${EXP_DIR}/exp5_sample_budget.launch" \
    "sample_config:=${config_file}" \
    "eval_tag_suffix:=K${k}_ilc_${ilc}" \
    eval_log_dir:="$LOG_DIR" \
    > "${LOG_DIR}/roslaunch_stdout.log" 2>&1 &
  PID=$!
  trap 'echo ""; kill -INT $PID 2>/dev/null || true; wait $PID 2>/dev/null; exit 130' INT

  sleep "$SETTLE"

  elapsed=0
  while [ $elapsed -lt $DRIVE_SEC ]; do
    if ! kill -0 "$PID" 2>/dev/null; then break; fi
    sleep 10; elapsed=$((elapsed + 10))
    printf "\r  [K=%s ILC=%s] %3d%% | %ds / %ds | ~第 %d 圈" \
      "$k" "$ilc" $((elapsed * 100 / DRIVE_SEC)) "$elapsed" "$DRIVE_SEC" $((elapsed / LAP_TIME))
  done

  echo ""
  kill -INT "$PID" 2>/dev/null || true
  wait "$PID" 2>/dev/null || true
  echo "  [$(date +%H:%M:%S)] K=${k} ILC=${ilc_zh} 完成。"

  if [ $run_idx -lt $TOTAL_GROUPS ]; then
    echo "  冷却 ${COOLDOWN}s..."
    sleep "$COOLDOWN"
  fi
done

echo ""
echo "══════════════════════════════════════════════════════════"
echo "  实验 5 全部完成 (${TOTAL_GROUPS} 组)。"
echo ""
echo "  画图命令:"
COLLECTED_DIRS=""
for run_spec in "${RUNS[@]}"; do
  k="${run_spec%%:*}"; ilc="${run_spec##*:}"
  k=$(printf '%04d' "$k")
  d="${LOG_ROOT}/exp5_K${k}_ilc_${ilc}_${TS}"
  [ -d "$d" ] && COLLECTED_DIRS="${COLLECTED_DIRS} ${d}"
done
echo "    python3 ${EXP_DIR}/plot_exp5_sample_budget.py${COLLECTED_DIRS}"
echo "══════════════════════════════════════════════════════════"
