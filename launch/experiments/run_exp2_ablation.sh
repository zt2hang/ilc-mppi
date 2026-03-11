#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# 实验 2：ILC 消融实验（跑道赛道）
#
# 分两次运行：先 ILC ON，再 ILC OFF，对比跟踪精度差异
# 场景：racetrack (straight=20m, radius=3m)，μ=0.10
# 预计每组 10 圈，每圈约 50s → 每组 ~ 8 分钟，共 ~ 18 分钟
#
# 用法：
#   bash run_exp2_ablation.sh                 # 运行两组（默认各 10 圈）
#   bash run_exp2_ablation.sh --laps 15       # 每组 15 圈
#   bash run_exp2_ablation.sh --only on       # 仅运行 ILC ON
#   bash run_exp2_ablation.sh --only off      # 仅运行 ILC OFF
#   bash run_exp2_ablation.sh --dry-run       # 仅打印
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# 自动 source catkin workspace
WS_SETUP="${HOME}/planner_code/mppi_swerve_drive_ros_ex/devel/setup.bash"
[[ -f "$WS_SETUP" ]] && source "$WS_SETUP"

EXP_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_ROOT="${HOME}/log/experiments"
LAP_TIME=50
NUM_LAPS=10
SETTLE=15
COOLDOWN=10
DRY_RUN=false
ONLY=""  # "on" | "off" | "" (both)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --laps)    NUM_LAPS="$2"; shift 2 ;;
    --only)    ONLY="$2"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    *) echo "未知参数: $1"; exit 1 ;;
  esac
done

DRIVE_SEC=$((NUM_LAPS * LAP_TIME))
TS=$(date +%Y%m%d_%H%M%S)

# ─── 通用运行函数 ────────────────────────────────────────────────────────────
run_one() {
  local ilc_state="$1"  # "on" or "off"
  local launch_file="${EXP_DIR}/exp2_ablation_ilc_${ilc_state}.launch"
  local TAG="exp2_ablation_ilc_${ilc_state}_${TS}"
  local LOG_DIR="${LOG_ROOT}/${TAG}"
  mkdir -p "$LOG_DIR"

  local ilc_zh; [[ "$ilc_state" == "on" ]] && ilc_zh="开启" || ilc_zh="关闭"

  cat <<EOF

┌──────────────────────────────────────────────────────────┐
│  实验 2 (ILC ${ilc_zh})：消融实验                          │
│  赛道：跑道形 (straight=20m, radius=3m)  μ=0.10         │
│  目标圈数：${NUM_LAPS} 圈               预计时长：~$(( (SETTLE + DRIVE_SEC) / 60 )) 分钟        │
│  日志：${LOG_DIR}
└──────────────────────────────────────────────────────────┘

EOF

  if $DRY_RUN; then
    echo "[DRY-RUN] 跳过。"
    return 0
  fi

  echo "[$(date +%H:%M:%S)] 启动 roslaunch (ILC ${ilc_zh})..."
  roslaunch "$launch_file" \
    eval_log_dir:="$LOG_DIR" \
    > "${LOG_DIR}/roslaunch_stdout.log" 2>&1 &
  local PID=$!
  trap 'echo ""; echo "[中断] 正在关闭..."; kill -INT $PID 2>/dev/null || true; wait $PID 2>/dev/null; exit 130' INT

  sleep "$SETTLE"
  echo "[$(date +%H:%M:%S)] 开始跟踪 — ${NUM_LAPS} 圈 (~${DRIVE_SEC}s)"

  local elapsed=0
  while [ $elapsed -lt $DRIVE_SEC ]; do
    if ! kill -0 "$PID" 2>/dev/null; then
      echo "[$(date +%H:%M:%S)] roslaunch 已退出。"; break
    fi
    sleep 10; elapsed=$((elapsed + 10))
    printf "\r  [ILC %s] %3d%% | %ds / %ds | ~第 %d 圈" \
      "$ilc_state" $((elapsed * 100 / DRIVE_SEC)) "$elapsed" "$DRIVE_SEC" $((elapsed / LAP_TIME))
  done

  echo ""
  echo "[$(date +%H:%M:%S)] 关闭 roslaunch..."
  kill -INT "$PID" 2>/dev/null || true
  wait "$PID" 2>/dev/null || true
  echo "[$(date +%H:%M:%S)] ILC ${ilc_zh} 完成 → ${LOG_DIR}/"
}

# ─── 执行 ────────────────────────────────────────────────────────────────────
if [[ -z "$ONLY" || "$ONLY" == "on" ]]; then
  run_one "on"
  if [[ -z "$ONLY" ]]; then
    echo ""
    echo "[$(date +%H:%M:%S)] 冷却 ${COOLDOWN}s..."
    sleep "$COOLDOWN"
  fi
fi

if [[ -z "$ONLY" || "$ONLY" == "off" ]]; then
  run_one "off"
fi

# Collect log directories for plotting
DIR_ON="${LOG_ROOT}/exp2_ablation_ilc_on_${TS}"
DIR_OFF="${LOG_ROOT}/exp2_ablation_ilc_off_${TS}"

echo ""
echo "══════════════════════════════════════════════════════════"
echo "  实验 2 全部完成。"
echo ""
echo "  生成的数据文件:"
for d in "$DIR_ON" "$DIR_OFF"; do
  [ -d "$d" ] && find "$d" -name '*.csv' -printf '    %p\n' 2>/dev/null
done
echo ""
echo "  画图命令:"
echo "    python3 ${EXP_DIR}/plot_exp2_ablation.py ${DIR_ON} ${DIR_OFF}"
echo "══════════════════════════════════════════════════════════"
