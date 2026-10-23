#!/bin/zsh
# 用 MLX 起一个 OpenAI 兼容服务（与 Ollama 同协议），加载**融合模型**。
# 服务端挂载微调权重，app 只需改 base_url 即可切换，无需改代码。
#
# 用法：bash scripts/serve_mlx.sh [融合模型目录] [端口]
#   融合模型目录 默认 models/sine-fused（由 mlx_lm fuse 产出）
#   端口         默认 8081（Ollama 默认 11434，互不干扰）
#
# 注意：不要改回 --adapter-path 直挂 adapter 的旧用法——实测该模式在请求
# 携带模型路径时会静默退回未微调基座（回复变客服腔）。必须先 fuse 再起服务。

set -e
cd "$(dirname "$0")/.."

PY=/Users/ntars/.workbuddy/binaries/python/envs/default/bin/python
MODEL=${1:-models/sine-fused}
PORT=${2:-8081}

export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HUB_DISABLE_XET=1

if [ ! -f "$MODEL/config.json" ]; then
  echo "融合模型不存在：$MODEL" >&2
  echo "先执行融合：BASE=\"\${SINE_BASE:-\$HOME/Documents/Github/Sine 大模型/基座模型/Qwen3-4B-Instruct-2507-4bit}\"; python -m mlx_lm fuse --model \"\$BASE\" --adapter-path runs/sine-lora --save-path models/sine-fused" >&2
  exit 1
fi

echo "启动 MLX 推理服务：$MODEL，监听 127.0.0.1:$PORT"
echo "把 config/settings.yaml 的 ollama.base_url 改为 http://127.0.0.1:$PORT 即可接入（不要带 /v1，生成层会自己拼）"
"$PY" -m mlx_lm.server --model "$MODEL" --host 127.0.0.1 --port "$PORT"
