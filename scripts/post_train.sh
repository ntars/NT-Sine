#!/bin/zsh
# 训练收尾流水线 —— 训练一停，一条命令跑完「取曲线 → 评测 → 边界 → 对比 → 汇总报告」
#
# 为什么固化：这几步串行依赖上一步的产物（曲线 → 评测 → 边界 → 对比），
# 参数与顺序原本散在清单的追加记录里，每次运行都得重新翻、容易用错口径
# （比如把贪心解码的结果和采样解码的结果混在一张表里比）。
# 固化之后不会漏跑、不会错口径，产物文件名也固定下来供后续步骤引用。
#
# 用法：
#   bash scripts/post_train.sh                # 二轮（默认，向后兼容）
#   ROUND=3 bash scripts/post_train.sh        # 三轮
#   bash scripts/post_train.sh --force        # 全部重跑
#
# 退出码：0 = 全部完成；3 = 训练仍在进行（不是错误）；1 = 某步失败
#
# ⚠️ 为什么用 ROUND 而不是让调用方覆盖单个环境变量：
# 产物路径如果逐项覆盖，漏掉一个就会写到上一轮的文件上——二轮的评测 JSON
# 与报告是交付证据，被三轮覆盖就等于证据被毁且无 diff 可查。故整组连坐。

set -e
cd "$(dirname "$0")/.."

PY=/Users/ntars/.workbuddy/binaries/python/envs/default/bin/python
ROUND=${ROUND:-2}
FORCE=0
[ "${1:-}" = "--force" ] && FORCE=1

# 产物路径按轮次整组声明，避免各处手写字符串写岔、也避免跨轮覆盖
case "$ROUND" in
  2)
    ROUND_LABEL="二轮"
    LOG=${LOG:-runs/train_sine_v3.log}
    ADAPTER=${ADAPTER:-runs/sine-lora}
    CURVE=runs/curve_v3.txt
    M_MAIN=runs/stage_matrix_v2.json
    M_SAMPLE=runs/stage_matrix_v2_sampled.json
    BOUND=runs/boundary_v2.json
    COMPARE=runs/compare_v1_v2.md
    REPORT=runs/round2_report.md
    M_PREV=runs/stage_matrix_v1.json
    LABEL_PREV="首轮 v1"
    LABEL_CUR="二轮 v2"
    B_PREV=""
    ;;
  3)
    ROUND_LABEL="三轮"
    LOG=${LOG:-runs/train_sine_r3.log}
    ADAPTER=${ADAPTER:-runs/sine-lora-r3}
    CURVE=runs/curve_r3.txt
    M_MAIN=runs/stage_matrix_v3.json
    M_SAMPLE=runs/stage_matrix_v3_sampled.json
    BOUND=runs/boundary_v3.json
    COMPARE=runs/compare_v2_v3.md
    REPORT=runs/round3_report.md
    M_PREV=runs/stage_matrix_v2.json
    LABEL_PREV="二轮 v2"
    LABEL_CUR="三轮 v3"
    B_PREV=runs/boundary_v2.json
    ;;
  *)
    echo "[收尾] 不认识的 ROUND=$ROUND（只支持 2 / 3）" >&2
    exit 1
    ;;
esac

need() {  # 需要重跑就返回真：产物缺失或强制重跑
  [ "$FORCE" -eq 1 ] || [ ! -s "$1" ]
}

# ── 第 0 步：训练必须已经退出 ──
# 日志被 mlx_lm 写成了带 ANSI 颜色码的富文本（`\033[38;5;244m` 之类），
# 不剥掉的话 "行首空白 + 步数" 这类锚点正则一行也匹配不到。
strip_ansi() { perl -pe 's/\e\[[\d;]*m//g'; }

if pgrep -f mlx_lm.lora > /dev/null; then
  LAST=$(strip_ansi < "$LOG" 2>/dev/null | grep -E '^ +[0-9]+ +(val |[0-9]\.)' | tail -1)
  echo "[收尾] 训练仍在进行，本次不取结果。最新进度：${LAST:-（日志暂无迭代行）}"
  exit 3
fi

if [ ! -s "$ADAPTER/adapters.safetensors" ]; then
  echo "[收尾] 训练已退出但找不到 $ADAPTER/adapters.safetensors，先查日志再重跑" >&2
  exit 1
fi

# ── 第 1 步：提取训练曲线 ──
# val 行形如 `   200    val 2.672    534.22s`；train 行形如 `    10    3.284 ▼`
if need "$CURVE"; then
  {
    echo "# 二轮训练曲线（$LOG）"
    echo
    echo "## 验证（每 200 步一次）"
    strip_ansi < "$LOG" | grep -E '^ +[0-9]+ +val ' || true
    echo
    echo "## 训练损失（每 10 步采样）"
    strip_ansi < "$LOG" | grep -E '^ +[0-9]+ +[0-9]\.[0-9]{3} ' || true
  } > "$CURVE"
  echo "[曲线] 已写入 $CURVE"
fi

VAL_TABLE=$(strip_ansi < "$LOG" | grep -E '^ +[0-9]+ +val ' || true)
BEST_LINE=$(echo "$VAL_TABLE" | awk '{print $3" "$1}' | sort -n | head -1)
BEST_VAL=$(echo "$BEST_LINE" | cut -d' ' -f1)
BEST_ITER=$(echo "$BEST_LINE" | cut -d' ' -f2)
LAST_VAL=$(echo "$VAL_TABLE" | tail -1 | awk '{print $3}')
LAST_ITER=$(echo "$VAL_TABLE" | tail -1 | awk '{print $1}')

echo "[曲线] 验证记录 $(echo "$VAL_TABLE" | grep -c . || true) 次｜最低 val ${BEST_VAL:-缺} @ ${BEST_ITER:-缺} 步｜末次 val ${LAST_VAL:-缺} @ ${LAST_ITER:-缺} 步"
# 末步比最低点高 5% 以上就提醒：说明后段在过拟合，评测仍按末步走（与首轮口径一致），
# 但判定时要写明，必要时改评 0000500_*.safetensors 里对应的那一份。
if [ -n "$BEST_VAL" ] && [ -n "$LAST_VAL" ]; then
  "$PY" - "$BEST_VAL" "$LAST_VAL" <<'PYEOF' || true
import sys
best, last = float(sys.argv[1]), float(sys.argv[2])
if last > best * 1.05:
    print(f"[曲线] 注意：末步 val 比最低点高 {(last/best-1)*100:.1f}%，后段有过拟合迹象")
else:
    print("[曲线] 末步 val 与最低点差距在 5% 内，按末步评测即可")
PYEOF
fi

# ── 第 2 步：阶段矩阵，双口径分开跑 ──
# 主口径走贪心解码（temp=0），与首轮严格可比；采样口径模拟线上体验作补充。
# 两者绝不能混在一张表里比大小。
if need "$M_MAIN"; then
  echo "[评测] 主口径（贪心，与 v1 可比）…"
  "$PY" scripts/evaluate.py --adapter "$ADAPTER" --out "$M_MAIN"
fi

if need "$M_SAMPLE"; then
  echo "[评测] 补充口径（线上采样参数 temp 0.72 / top_p 0.9）…"
  "$PY" scripts/evaluate.py --adapter "$ADAPTER" --temp 0.72 --top-p 0.9 --out "$M_SAMPLE"
fi

# ── 第 3 步：边界红线 ──
if need "$BOUND"; then
  echo "[边界] 诱导探针…"
  "$PY" scripts/boundary_check.py --adapter "$ADAPTER" --verbose --out "$BOUND"
fi

# ── 第 4 步：两轮对比 ──
# 表头轮次名由 LABEL_PREV / LABEL_CUR 决定：工具默认写「首轮/二轮」，
# 三轮复用时不传就会把二轮标成首轮，标错轮次的对比表比没有更糟。
# 首轮没有 boundary JSON（当时脚本还不支持 --out），故 B_PREV 为空，对比表会标注"缺"
if need "$COMPARE"; then
  if [ ! -s "$M_PREV" ]; then
    echo "[对比] 缺 $M_PREV，无法对比" >&2
    exit 1
  fi
  echo "[对比] 生成对比表（$LABEL_PREV → $LABEL_CUR）…"
  B_PREV_ARG=""
  [ -n "$B_PREV" ] && [ -s "$B_PREV" ] && B_PREV_ARG="--b1 $B_PREV"
  "$PY" scripts/compare_rounds.py --v1 "$M_PREV" --v2 "$M_MAIN" \
    --label1 "$LABEL_PREV" --label2 "$LABEL_CUR" \
    ${=B_PREV_ARG} --b2 "$BOUND" --out "$COMPARE"
fi

# ── 第 5 步：把散落的数汇成一份报告，供判定与文档回填 ──
{
  echo "# ${ROUND_LABEL}训练验收报告"
  echo
  echo "> 由 \`scripts/post_train.sh\` 自动汇总，数字全部取自 runs/ 下的原始产物，未手工转录。"
  echo
  echo "## 一、训练曲线"
  echo
  echo "- 验证次数：$(echo "$VAL_TABLE" | grep -c .)"
  echo "- 最低 val：**$BEST_VAL**（$BEST_ITER 步）"
  echo "- 末次 val：**$LAST_VAL**（$LAST_ITER 步）"
  echo
  echo '```'
  echo "$VAL_TABLE"
  echo '```'
  echo
  echo "## 二、两轮对比（主口径）"
  echo
  cat "$COMPARE"
  echo
  echo "## 三、产物索引"
  echo
  echo "| 文件 | 内容 |"
  echo "|---|---|"
  echo "| \`$CURVE\` | 训练与验证曲线原始行 |"
  echo "| \`$M_MAIN\` | 阶段矩阵（贪心，主口径） |"
  echo "| \`$M_SAMPLE\` | 阶段矩阵（采样，补充口径，不可与主口径混比） |"
  echo "| \`$BOUND\` | 六条边界诱导探针的回复与命中 |"
  echo "| \`$COMPARE\` | 两轮逐项升降对比 |"
} > "$REPORT"

echo
echo "[完成] 报告：$REPORT"
echo "[完成] 下一步：按报告做迭代判定，再跑端到端回归"
