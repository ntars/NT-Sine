#!/bin/zsh
# Sine 训练启动器 —— 专为 Qwen3-4B-Instruct-2507-4bit（MLX 4bit）定制
#
# 每个参数都有实测或基座事实依据，不要凭感觉改：
#   基座 36 层 / hidden 2560 / 词表 151936 / eos=<|im_end|>（读 config.json 得来）
#   --num-layers 16  → 只训最后 16 层（覆盖 44%），可训练参数约 7.3M（0.18%）
#   --max-seq-length 1024 → 训练集 p100 实测 894 token，留 130 余量。
#     不用 896：只差 2 token 时 mlx-lm 会截断尾部（可能切掉 <|im_end|>），
#     模型会学到"句子没说完就停"的坏习惯。
#   --batch-size 1   → 16GB 统一内存下 batch 2 会触发系统 jetsam 杀进程
#   --grad-checkpoint → 4B 不开必被杀（旧链路实测）
#   --save-every 500 → 每 500 步落盘一次（约 35 分钟），兼顾 SSD 寿命与崩溃损失
#   --learning-rate 1e-4 → 4B 上实测收敛平稳（loss 3.8 → 0.4，无发散）
#   --val-batches -1 → 验证集小，全量评估（每轮约 550 秒，实测值）
#   --steps-per-eval 200 → 每 200 步验证一次。实测 2026-10-23 二轮首启用 50：
#     240 步里训练本体只占 380 秒，4 次全量验证却吃掉 2200 秒（九成时间在验证）。
#     按 50 步一验跑满 3200 步要 64 次验证 ≈ 9.8 小时；改 200 步一验只需 16 次 ≈ 2.4 小时，
#     曲线分辨率仍够看住过拟合（每 200 步约 0.24 个 epoch）。
#   --mask-prompt    → 只在她的话上算损失（对话微调的正确姿势）
#
# 用法：
#   bash scripts/train_sine.sh              # 默认 2000 步
#   ITERS=3000 bash scripts/train_sine.sh   # 覆盖步数
#
# 长任务必须脱离进程组发射（否则会被环境周期性杀掉）：
#   python -c "import subprocess; log=open('runs/train_sine.log','a'); \
#   subprocess.Popen(['bash','scripts/train_sine.sh'], stdout=log, stderr=subprocess.STDOUT, \
#   stdin=subprocess.DEVNULL, start_new_session=True)"

set -e
cd "$(dirname "$0")/.."

PY=/Users/ntars/.workbuddy/binaries/python/envs/default/bin/python
LOCAL_BASE="/Users/ntars/Documents/Github/Sine 大模型/基座模型/Qwen3-4B-Instruct-2507-4bit"
MODEL=${MODEL:-"$LOCAL_BASE"}
[ -f "$MODEL/config.json" ] || MODEL=mlx-community/Qwen3-4B-Instruct-2507-4bit
OUT=${OUT:-runs/sine-lora}
ITERS=${ITERS:-2000}

mkdir -p "$OUT" runs
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HUB_DISABLE_XET=1

# ── 训练前内存体检 ──
# 注意 page size 是 16384（Apple Silicon 为 16KB 页，不是 4KB）——
# 按 4096 算会把可用内存低估 4 倍，误判成"内存不足"。
# 阈值 3000MB：macOS 的文件缓存（active/inactive）在压力下会被回收，
# 真正的硬指标是训练中是否触发 jetsam；这里只拦"明显不够"的情况。
PAGE=16384
FREE_MB=$(vm_stat | awk -v ps="$PAGE" '/Pages free|Pages inactive/ {gsub(/\./,"",$3); s+=$3} END {printf "%.0f", s*ps/1048576}')
echo "[体检] 估算可回收内存 ${FREE_MB}MB（训练峰值预计 8-10GB，系统缓存会按需释放）"
if [ "$FREE_MB" -lt 3000 ]; then
  echo "[体检] 可回收内存过低，请关闭浏览器等占内存程序后再训练" >&2
  exit 1
fi
if pgrep -f mlx_lm.server > /dev/null; then
  echo "[体检] 检测到推理服务在运行，训练期间不允许并行推理（会抢内存导致系统卡顿）" >&2
  exit 1
fi

# ── 训练集存在性检查（只认新构建器的产物） ──
if [ ! -f data_train/train.jsonl ]; then
  echo "[体检] 缺少 data_train/train.jsonl，请先运行 scripts/build_dataset.py" >&2
  exit 1
fi
if [ ! -f data_train/valid.jsonl ]; then
  echo "[体检] 缺少验证集 valid.jsonl（没有验证集就看不出过拟合，拒绝裸训）" >&2
  exit 1
fi

ARGS=(
  --model "$MODEL"
  --train
  --data data_train
  --batch-size 1
  --num-layers 16
  --iters "$ITERS"
  --learning-rate 1e-4
  --save-every 500
  --steps-per-eval 200
  --val-batches -1
  --adapter-path "$OUT"
  --seed 42
  --mask-prompt
  --grad-checkpoint
  --max-seq-length 1024
)

# ── 监督循环：被杀/崩溃后从最新 checkpoint 续训，最多 60 次 ──
for attempt in $(seq 1 60); do
  RESUME=$(ls -1 "$OUT"/*adapters.safetensors 2>/dev/null | sort | tail -1 || true)
  if [ -n "$RESUME" ]; then
    echo "[监督器] 第 ${attempt} 次启动，从 $RESUME 续训"
    "$PY" -m mlx_lm.lora "${ARGS[@]}" --resume-adapter-file "$RESUME" && break
  else
    echo "[监督器] 第 ${attempt} 次启动，从头训练"
    "$PY" -m mlx_lm.lora "${ARGS[@]}" && break
  fi
  echo "[监督器] 训练退出（被杀/崩溃），5 秒后从最新存档续训"
  sleep 5
done

echo "训练收尾，adapter 在 $OUT"
ls -la "$OUT"
