#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# 实验 6：采样效率 — ILC 先验在低采样数下的决定性优势
#
# 核心论证：ILC + K=64 ≈ 纯 MPPI + K=1000
#
# 对比 K ∈ {32, 64, 128, 256, 512, 1000, 2000} × {ILC off, ILC on, ILC aggressive}
# 共 21 组（隔离模式：estimator frozen, feedback off）
#
# 场景：racetrack (straight=20m, radius=3m)，μ=0.10
# 预计每组 15 圈 × 50s ≈ 12.5 分钟，21 组共 ~ 270 分钟
#
# 用法：
#   bash run_exp6_sample_efficiency.sh                      # 全部 21 组
#   bash run_exp6_sample_efficiency.sh --laps 20            # 每组 20 圈
#   bash run_exp6_sample_efficiency.sh --K 32,64,128        # 只跑指定 K 值
#   bash run_exp6_sample_efficiency.sh --ilc off            # 只跑 ILC OFF
#   bash run_exp6_sample_efficiency.sh --ilc aggressive     # 只跑 aggressive 版
#   bash run_exp6_sample_efficiency.sh --dry-run
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

WS_SETUP="${HOME}/planner_code/mppi_swerve_drive_ros_ex/devel/setup.bash"
[[ -f "$WS_SETUP" ]] && source "$WS_SETUP"

EXP_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_DIR="${EXP_DIR}/config"
LOG_ROOT="${HOME}/log/experiments"
LAP_TIME=50
NUM_LAPS=8
SETTLE=15
COOLDOWN=10
DRY_RUN=false
K_VALUES="0032,0064,0128,0256,0512,1000,2000"
ILC_FILTER=""  # "on" | "off" | "aggressive" | "" (all three)

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

# Build run list: off, on, aggressive for each K
RUNS=()
for k in "${KS[@]}"; do
  for ilc in off on aggressive; do
    if [[ -n "$ILC_FILTER" && "$ILC_FILTER" != "$ilc" ]]; then continue; fi
    RUNS+=("${k}:${ilc}")
  done
done

TOTAL_GROUPS=${#RUNS[@]}
TOTAL_MIN=$(( (SETTLE + DRIVE_SEC + COOLDOWN) * TOTAL_GROUPS / 60 ))

cat <<EOF

╔══════════════════════════════════════════════════════════════╗
║  实验 6：采样效率分析（隔离 ILC 贡献）                       ║
╠══════════════════════════════════════════════════════════════╣
║  赛道：跑道形 (straight=20m, radius=3m)  μ=0.10            ║
║  K 值：$(echo "${KS[*]}" | sed 's/ /, /g')
║  ILC 模式：off / on / aggressive
║  隔离：estimator frozen, feedback off
║  每组 ${NUM_LAPS} 圈 × ${TOTAL_GROUPS} 组 → 总计 ~${TOTAL_MIN} 分钟
╚══════════════════════════════════════════════════════════════╝

EOF

run_idx=0
for run_spec in "${RUNS[@]}"; do
  run_idx=$((run_idx + 1))
  k="${run_spec%%:*}"
  ilc="${run_spec##*:}"

  k=$(printf '%04d' "$((10#$k))")

  # Determine config file
  if [[ "$ilc" == "aggressive" ]]; then
    config_file="${CONFIG_DIR}/isolated_${k}_ilc_aggressive.yaml"
  else
    config_file="${CONFIG_DIR}/isolated_${k}_ilc_${ilc}.yaml"
  fi

  case "$ilc" in
    on)         ilc_zh="ILC标准" ;;
    off)        ilc_zh="ILC关闭" ;;
    aggressive) ilc_zh="ILC激进" ;;
  esac

  TAG="exp6_K${k}_ilc_${ilc}_${TS}"
  LOG_DIR="${LOG_ROOT}/${TAG}"
  mkdir -p "$LOG_DIR"

  echo "──────────────────────────────────────────────────────────"
  echo "  [${run_idx}/${TOTAL_GROUPS}] K=${k}  ${ilc_zh}  ${NUM_LAPS}圈 ~$((DRIVE_SEC/60))min"
  echo "  config: $(basename "$config_file")"
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

  roslaunch "${EXP_DIR}/exp6_sample_efficiency.launch" \
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
    printf "\r  [K=%s %s] %3d%% | %ds / %ds | ~第 %d 圈" \
      "$k" "$ilc_zh" $((elapsed * 100 / DRIVE_SEC)) "$elapsed" "$DRIVE_SEC" $((elapsed / LAP_TIME))
  done

  echo ""
  kill -INT "$PID" 2>/dev/null || true
  wait "$PID" 2>/dev/null || true
  echo "  [$(date +%H:%M:%S)] K=${k} ${ilc_zh} 完成。"

  if [ $run_idx -lt $TOTAL_GROUPS ]; then
    echo "  冷却 ${COOLDOWN}s..."
    sleep "$COOLDOWN"
  fi
done

echo ""
echo "══════════════════════════════════════════════════════════"
echo "  实验 6 全部完成 (${TOTAL_GROUPS} 组)。"
echo ""
echo "  画图命令:"
COLLECTED_DIRS=""
for run_spec in "${RUNS[@]}"; do
  k="${run_spec%%:*}"; ilc="${run_spec##*:}"
  k=$(printf '%04d' "$((10#$k))")
  d="${LOG_ROOT}/exp6_K${k}_ilc_${ilc}_${TS}"
  [ -d "$d" ] && COLLECTED_DIRS="${COLLECTED_DIRS} ${d}"
done
echo "    python3 ${EXP_DIR}/plot_exp6_sample_efficiency.py${COLLECTED_DIRS}"
echo "══════════════════════════════════════════════════════════"
