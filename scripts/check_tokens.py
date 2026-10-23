#!/usr/bin/env python3
"""用真 tokenizer 量训练样本的 token 长度分布（定 max-seq 用）。

运行：python scripts/check_tokens.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    from mlx_lm import load
    # 本地基座优先（与 scripts/train_lora.sh 同构），不存在再走镜像拉取
    local_base = "/Users/ntars/Documents/Github/Sine 大模型/基座模型/Qwen3-4B-Instruct-2507-4bit"
    model = local_base if Path(local_base, "config.json").exists() \
        else "mlx-community/Qwen3-4B-Instruct-2507-4bit"
    _, tok = load(model, lazy=True)

    def count(text: str) -> int:
        return len(tok.encode(text, add_special_tokens=False))

    first = json.loads((ROOT / "data_train" / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])
    sys_content = first["messages"][0]["content"]
    print("系统提示含【当前场景】:", "【当前场景】" in sys_content, "| token:", count(sys_content))

    lens = []
    for ln in (ROOT / "data_train" / "train.jsonl").read_text(encoding="utf-8").splitlines():
        d = json.loads(ln)
        lens.append(sum(count(m["content"]) + 4 for m in d["messages"]))
    lens.sort()
    n = len(lens)
    for p in (50, 90, 95, 100):
        print(f"p{p}: {lens[min(n - 1, n * p // 100)]} token")
    print("样本数:", n)

    # 训练/服务提示一致性检查（借鉴 deepseek-recipe 的"协议对齐"动机）：
    # 训练时模型看到的系统提示，必须与线上实际拼出来的大致同构，
    # 否则模型在推理时面对"没见过的提示"——v1 就是这么翻车的。
    import tempfile
    from app.context import build_messages
    from app.emotion.engine import EmotionEngine
    from app.memory.store import MemoryStore
    from app.worldbook.engine import WorldbookEngine
    from app.emotion.engine import RELATION_STAGES, load_stage_cards
    from scripts.build_dataset import build_system

    probe = "我今天好累，加班到十点才下班"
    tmp = tempfile.mkdtemp(prefix="nt-sine-parity-")
    wb = WorldbookEngine()
    # 服务侧模拟「破冰期」，与训练样本的阶段卡对齐
    emotion = EmotionEngine(Path(tmp) / "e.json")
    emotion.state.active_days, emotion.state.intimacy, emotion.state.trust = 3, 15, 30
    prod = build_messages(probe, wb.scan(probe, []), emotion,
                          MemoryStore(Path(tmp) / "t.db"), [])[0]["content"]
    train = build_train_system_prompt("他很难过", "thawing")
    print(f"\n训练提示 {count(train)} token ｜ 服务提示 {count(prod)} token ｜ 差值 {count(prod) - count(train)}")
    print("训练含【当前场景】:", "【当前场景】" in train,
          "｜训练含【关系阶段】:", "【关系阶段】" in train,
          "｜服务含【关系阶段】:", "【关系阶段】" in prod,
          "｜服务含 L4 场景卡:", any(a.entry.layer == "L4" for a in wb.scan(probe, []).entries))
    if abs(count(prod) - count(train)) > 350:
        print("⚠️ 提示长度差异过大：检查训练与线下的系统提示是否同构")
    return 0


if __name__ == "__main__":
    sys.exit(main())