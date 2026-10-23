#!/usr/bin/env python3
"""端到端检查：用微调后的 MLX 服务跑完整链路（世界书→情感→生成→过滤→节奏→落库）。

前置：bash scripts/serve_mlx.sh runs/nt-sine-lora-v3 8081
运行：python scripts/e2e_check.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import SETTINGS  # noqa: E402

# 指向 MLX 融合模型服务（与 config/settings.yaml 的 ollama.model 对齐；
# 必须用 fuse 产物直接起服务，勿用 --adapter-path 直挂——会静默退回未微调基座）
MODEL_ID = "/Users/ntars/Documents/Github/NT-Sine/models/sine-fused"
SETTINGS["ollama"]["base_url"] = "http://127.0.0.1:8081"  # 不要带 /v1，生成层会自己拼
SETTINGS["ollama"]["model"] = MODEL_ID

# 用临时数据库：端到端检查绝不能污染真实对话历史
import tempfile  # noqa: E402
_TMP = tempfile.mkdtemp(prefix="nt-sine-e2e-")
SETTINGS["app"]["db_path"] = f"{_TMP}/e2e.db"
SETTINGS["app"]["emotion_state_path"] = f"{_TMP}/emotion.json"

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

PROBES = [
    "你今天真好看，我忍不住说出口了",
    "刚才有个女同事跟我聊得挺开心的，别多想啊",
    "我今天好累，加班到十点才下班",
    "宝宝~我今天超想你，快来见我一面",
]


def main() -> int:
    with TestClient(app) as client:
        health = client.get("/health").json()
        print(f"健康检查：{health}")
        for text in PROBES:
            r = client.post("/api/chat", json={"message": text}).json()
            segs = " / ".join(s["text"] for s in r.get("segments", []))
            print(f"\n用户：{text}")
            print(f"玖玖：{r.get('reply')}")
            print(f"分条：{segs}")
            print(f"场景档位：{r.get('scene_mode')} | 世界书 token：{r.get('worldbook_tokens')}")
            print(f"过滤器命中：{r.get('filtered') or '无'} | 总延迟：{r.get('delay_s')}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())