#!/bin/zsh
# NT-Sine LoRA 训练（mlx-lm，Apple Silicon 原生）· 带崩溃自恢复的监督版
# 前置：pip install mlx-lm；数据已由 build_training.py 生成到 data_train/train.jsonl
# 网络受限时：export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1
set -e
cd "$(dirname "$0")/.."

PY=/Users/ntars/.workbuddy/binaries/python/envs/default/bin/python
LOCAL_BASE="/Users/ntars/Documents/Github/Sine 大模型/基座模型/Qwen3-4B-Instruct-2507-4bit"
# 优先用本地基座副本（已归档在 Sine 大模型/），不存在再走镜像拉取
MODEL=${MODEL:-"$LOCAL_BASE"}
if [ ! -f "$MODEL/config.json" ]; then
  MODEL=mlx-community/Qwen3-4B-Instruct-2507-4bit
fi
OUT=runs/sine-lora
mkdir -p "$OUT"

export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HUB_DISABLE_XET=1

# mask-prompt：只在玖玖的回复 token 上算损失（对话微调的正确姿势）
# grad-checkpoint：4B 必须开，否则 16GB 显存扛不住（1.5B 时代就因此被杀过）
# batch 2 + max-seq 896：实测样本中位约 600 token/最大 739（真 tokenizer 测量），
#   无截断且显存占用减半，降低 16GB 机器被系统杀掉的概率
# save-every 200：崩溃最多丢 200 步；监督循环自动从最新 checkpoint 续训
BASE_ARGS=(
  --model "$MODEL"
  --train
  --data data_train
  --batch-size 2
  --num-layers 16
  --iters 3100
  --learning-rate 1e-4
  --save-every 200
  --adapter-path "$OUT"
  --seed 42
  --mask-prompt
  --grad-checkpoint
  --max-seq-length 896
)

# 注意：实测本环境会周期性杀掉整个后台任务进程组（疑似 16GB 内存 jetsam），
# 监督循环重试次数给到 60：每次被杀后 5 秒自动从最新 checkpoint 续训，
# 200 步存档保证最多丢 200 步（约 13 分钟）进度。
for attempt in $(seq 1 60); do
  # 找最新 checkpoint（大数字编号优先，如 0001200_adapters.safetensors）
  RESUME=$(ls -1 "$OUT"/*adapters.safetensors 2>/dev/null | sort | tail -1 || true)
  if [ -n "$RESUME" ]; then
    echo "[监督器] 第 ${attempt} 次启动，从 $RESUME 续训"
    "$PY" -m mlx_lm.lora "${BASE_ARGS[@]}" --resume-adapter-file "$RESUME" && break
  else
    echo "[监督器] 第 ${attempt} 次启动，从头训练"
    "$PY" -m mlx_lm.lora "${BASE_ARGS[@]}" && break
  fi
  echo "[监督器] 训练退出（被杀/崩溃），5 秒后从最新存档续训"
  sleep 5
done

echo "训练收尾，adapter 保存在 $OUT"
ls -la "$OUT"
