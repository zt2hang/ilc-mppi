#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# 实验 7：空间扰动补偿 — ILC 学习空间分布干扰的独特能力
#
# 注入 d_vy(s) = A·sin(2π·n·s/L) 形式的空间扰动，模拟路面倾斜/局部侧风
# 对比四种模式：
#   1. ILC off     (纯 MPPI，无在线估计/反馈)
#   2. ILC on      (标准 ILC 参数)
#   3. ILC aggressive (激进 ILC 参数)
#   4. Full stack  (无 ILC，但开启在线估计+反馈)
#
# 场景：racetrack (straight=20m, radius=3m)，μ=0.10
# 扰动：A=0.10 m/s, n=3 (可配置)
# 预计每组 20 圈 × 50s ≈ 17 分钟，4 组共 ~ 70 分钟
#
# 用法：
#   bash run_exp7_spatial_disturbance.sh                        # 全部 4 组
#   bash run_exp7_spatial_disturbance.sh --laps 25              # 每组 25 圈
#   bash run_exp7_spatial_disturbance.sh --amp 0.15 --wave 4    # 调扰动强度
#   bash run_exp7_spatial_disturbance.sh --only ilc_aggressive  # 只跑一组
#   bash run_exp7_spatial_disturbance.sh --dry-run
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

WS_SETUP="${HOME}/planner_code/mppi_swerve_drive_ros_ex/devel/setup.bash"
[[ -f "$WS_SETUP" ]] && source "$WS_SETUP"

EXP_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_DIR="${EXP_DIR}/config"
LOG_ROOT="${HOME}/log/experiments"
LAP_TIME=50
NUM_LAPS=20
SETTLE=15
COOLDOWN=10
DRY_RUN=false
AMP=0.10
WAVE=3.0
ONLY=""  # "ilc_off" | "ilc_on" | "ilc_aggressive" | "full_stack" | "" (all)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --laps)    NUM_LAPS="$2"; shift 2 ;;
    --amp)     AMP="$2"; shift 2 ;;
    --wave)    WAVE="$2"; shift 2 ;;
    --only)    ONLY="$2"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    *) echo "未知参数: $1"; exit 1 ;;
  esac
done

DRIVE_SEC=$((NUM_LAPS * LAP_TIME))
TS=$(date +%Y%m%d_%H%M%S)

# Run groups: config_basename : display_name
declare -A GROUP_LABELS
GROUP_LABELS["disturbance_ilc_off"]="ILC关闭(纯MPPI)"
GROUP_LABELS["disturbance_ilc_on"]="ILC标准"
GROUP_LABELS["disturbance_ilc_aggressive"]="ILC激进"
GROUP_LABELS["disturbance_full_stack"]="全栈补偿(无ILC)"

# Ordered run list
GROUP_ORDER=("disturbance_ilc_off" "disturbance_ilc_on" "disturbance_ilc_aggressive" "disturbance_full_stack")

RUNS=()
for g in "${GROUP_ORDER[@]}"; do
  if [[ -n "$ONLY" && "$ONLY" != "$g" ]]; then continue; fi
  RUNS+=("$g")
done

TOTAL_GROUPS=${#RUNS[@]}
TOTAL_MIN=$(( (SETTLE + DRIVE_SEC + COOLDOWN) * TOTAL_GROUPS / 60 ))

cat <<EOF

╔══════════════════════════════════════════════════════════════╗
║  实验 7：空间扰动补偿                                       ║
╠══════════════════════════════════════════════════════════════╣
║  赛道：跑道形 (straight=20m, radius=3m)  μ=0.10            ║
║  扰动：A=${AMP} m/s, n=${WAVE} cycles/lap                  ║
║  模式：$(echo "${RUNS[*]}" | sed 's/ /, /g')
║  每组 ${NUM_LAPS} 圈 × ${TOTAL_GROUPS} 组 → 总计 ~${TOTAL_MIN} 分钟
╚══════════════════════════════════════════════════════════════╝

EOF

run_idx=0
for group in "${RUNS[@]}"; do
  run_idx=$((run_idx + 1))
  config_file="${CONFIG_DIR}/${group}.yaml"
  display_name="${GROUP_LABELS[$group]}"

  TAG="exp7_${group}_A${AMP}_n${WAVE}_${TS}"
  LOG_DIR="${LOG_ROOT}/${TAG}"
  mkdir -p "$LOG_DIR"

  echo "──────────────────────────────────────────────────────────"
  echo "  [${run_idx}/${TOTAL_GROUPS}] ${display_name}  ${NUM_LAPS}圈 ~$((DRIVE_SEC/60))min"
  echo "  config: $(basename "$config_file")"
  echo "  disturbance: A=${AMP} m/s, n=${WAVE}"
  echo "  → ${LOG_DIR}"
  echo "──────────────────────────────────────────────────────────"

  if $DRY_RUN; then
    echo "  [DRY-RUN] 跳过。"
    continue
  fi

  if [[ ! -f "$config_file" ]]; then
    echo "  [ERROR] 配置不存在: $config_file — 跳过"
    continue
  fi

  roslaunch "${EXP_DIR}/exp7_spatial_disturbance.launch" \
    "sample_config:=${config_file}" \
    "disturbance_amplitude:=${AMP}" \
    "disturbance_wavenumber:=${WAVE}" \
    "eval_tag_suffix:=${group}" \
    eval_log_dir:="$LOG_DIR" \
    > "${LOG_DIR}/roslaunch_stdout.log" 2>&1 &
  PID=$!
  trap 'echo ""; kill -INT $PID 2>/dev/null || true; wait $PID 2>/dev/null; exit 130' INT

  sleep "$SETTLE"

  elapsed=0
  while [ $elapsed -lt $DRIVE_SEC ]; do
    if ! kill -0 "$PID" 2>/dev/null; then break; fi
    sleep 10; elapsed=$((elapsed + 10))
    printf "\r  [%s] %3d%% | %ds / %ds | ~第 %d 圈" \
      "$display_name" $((elapsed * 100 / DRIVE_SEC)) "$elapsed" "$DRIVE_SEC" $((elapsed / LAP_TIME))
  done

  echo ""
  kill -INT "$PID" 2>/dev/null || true
  wait "$PID" 2>/dev/null || true
  echo "  [$(date +%H:%M:%S)] ${display_name} 完成。"

  if [ $run_idx -lt $TOTAL_GROUPS ]; then
    echo "  冷却 ${COOLDOWN}s..."
    sleep "$COOLDOWN"
  fi
done

echo ""
echo "══════════════════════════════════════════════════════════"
echo "  实验 7 全部完成 (${TOTAL_GROUPS} 组)。"
echo ""
echo "  画图命令:"
COLLECTED_DIRS=""
for group in "${RUNS[@]}"; do
  d="${LOG_ROOT}/exp7_${group}_A${AMP}_n${WAVE}_${TS}"
  [ -d "$d" ] && COLLECTED_DIRS="${COLLECTED_DIRS} ${d}"
done
echo "    python3 ${EXP_DIR}/plot_exp7_spatial_disturbance.py${COLLECTED_DIRS}"
echo "══════════════════════════════════════════════════════════"
