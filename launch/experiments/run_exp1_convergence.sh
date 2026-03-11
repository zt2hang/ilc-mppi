#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# 实验 1：ILC 收敛性验证（圆形赛道）
#
# 场景：radius=10m 的圆形闭合路径，μ=0.3，ILC 开启
# 目标：观察横向 RMSE / 偏置 RMS 随圈数的下降曲线
# 预计运行 15 圈，每圈约 35 s → 总计约 10 分钟
#
# 用法：
#   bash run_exp1_convergence.sh              # 运行 15 圈（默认 600s）
#   bash run_exp1_convergence.sh  --laps 20   # 运行 20 圈
#   bash run_exp1_convergence.sh  --dry-run   # 仅打印，不实际启动
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# 自动 source catkin workspace
WS_SETUP="${HOME}/planner_code/mppi_swerve_drive_ros_ex/devel/setup.bash"
[[ -f "$WS_SETUP" ]] && source "$WS_SETUP"

EXP_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_ROOT="${HOME}/log/experiments"
LAP_TIME=35           # 每圈大约秒数（radius=10, v≈1.8）
NUM_LAPS=15
SETTLE=15
DRY_RUN=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --laps)   NUM_LAPS="$2"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    *) echo "未知参数: $1"; exit 1 ;;
  esac
done

DRIVE_SEC=$((NUM_LAPS * LAP_TIME))
TOTAL_SEC=$((SETTLE + DRIVE_SEC))
TAG="exp1_convergence_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_ROOT}/${TAG}"
mkdir -p "$LOG_DIR"

cat <<EOF

╔══════════════════════════════════════════════════════════╗
║  实验 1：ILC 收敛性验证                                 ║
╠══════════════════════════════════════════════════════════╣
║  赛道：圆形 (radius=10m)    摩擦系数：μ=0.3            ║
║  ILC：开启 (γ=0.995)       采样数：K=2000              ║
║  目标圈数：${NUM_LAPS} 圈               预计时长：~$((TOTAL_SEC/60)) 分钟          ║
║  日志目录：${LOG_DIR}
╚══════════════════════════════════════════════════════════╝

EOF

if $DRY_RUN; then
  echo "[DRY-RUN] 不会实际启动 roslaunch，退出。"
  exit 0
fi

echo "[$(date +%H:%M:%S)] 启动 roslaunch..."
roslaunch "${EXP_DIR}/exp1_convergence.launch" \
  eval_log_dir:="$LOG_DIR" \
  > "${LOG_DIR}/roslaunch_stdout.log" 2>&1 &
PID=$!
trap 'echo ""; echo "[中断] 正在关闭 roslaunch (PID=$PID)..."; kill -INT $PID 2>/dev/null || true; wait $PID 2>/dev/null; echo "已停止。"; exit 130' INT

echo "[$(date +%H:%M:%S)] Gazebo 初始化中，等待 ${SETTLE}s..."
sleep "$SETTLE"

echo "[$(date +%H:%M:%S)] 开始跟踪 — 预计 ${NUM_LAPS} 圈 (~${DRIVE_SEC}s)"
echo ""

elapsed=0
lap_est=0
while [ $elapsed -lt $DRIVE_SEC ]; do
  # 检查 roslaunch 是否仍在运行
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "[$(date +%H:%M:%S)] roslaunch 已退出（可能仿真结束或出错）。"
    break
  fi
  sleep 10
  elapsed=$((elapsed + 10))
  lap_est=$((elapsed / LAP_TIME))
  pct=$((elapsed * 100 / DRIVE_SEC))
  printf "\r  [进度] %3d%% | 已运行 %ds / %ds | ~第 %d 圈 / %d 圈" \
    "$pct" "$elapsed" "$DRIVE_SEC" "$lap_est" "$NUM_LAPS"
done

echo ""
echo ""
echo "[$(date +%H:%M:%S)] 时间到，正在关闭 roslaunch..."
kill -INT "$PID" 2>/dev/null || true
wait "$PID" 2>/dev/null || true

echo "[$(date +%H:%M:%S)] 实验 1 完成！"
echo "  日志目录: ${LOG_DIR}/"
echo ""
echo "  生成的数据文件:"
find "$LOG_DIR" -name '*.csv' -printf '    %p\n' 2>/dev/null || echo "    (等待节点写入)"
echo ""
echo "  画图命令:"
echo "    python3 ${EXP_DIR}/plot_exp1_convergence.py ${LOG_DIR}"
