#!/usr/bin/env python3
"""训练后评估：双模式人设抽检。

模式 real（默认，贴近线上）：走完整链路——世界书按触发词注入 → 情感状态拼上下文
  → 用 build_messages 组装后生成。评估的是"生产环境实际会发生什么"。
模式 bare（对照）：只给训练系统提示（无场景卡），用于观察模型裸能力。

用法：python scripts/evaluate.py --adapter runs/sine-lora [--mode real|bare|both]
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mlx_lm import load, stream_generate  # noqa: E402
from mlx_lm.sample_utils import make_sampler  # noqa: E402

PROBES = [
    "你今天真好看，我忍不住说出口了",
    "刚才有个女同事跟我聊得挺开心的，别多想啊",
    "我今天好累，加班到十点才下班",
    "宝宝~我今天超想你，快来见我一面",
]


def _base_system_prompt() -> str:
    """人格核心（硬约束 + L1 + L6），与训练一致。"""
    system = (ROOT / "config" / "prompts" / "system.txt").read_text(encoding="utf-8").strip()
    l1 = json.loads((ROOT / "data" / "worldbook" / "l1_anchor.json").read_text(encoding="utf-8"))
    l6 = json.loads((ROOT / "data" / "worldbook" / "l6_boundary.json").read_text(encoding="utf-8"))
    return system + "\n\n" + "\n\n".join(e["content"] for e in l1["entries"] + l6["entries"])


def run(model, tokenizer, messages, max_tokens=120, temperature=0.7,
        deterministic: bool = False) -> str:
    """生成一段回复。

    deterministic=True 时用贪心解码（temp=0）+ 固定随机种子——
    借鉴 DeepInfer 的「确定性采样」设计：同输入必得同输出，
    否则多版本对比里混进了采样噪声，结论不可复现。
    """
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    if deterministic:
        import mlx.core as mx
        mx.random.seed(20261022)
        sampler = make_sampler(temp=0.0)
    else:
        sampler = make_sampler(temp=temperature, top_p=0.9)
    out = ""
    for resp in stream_generate(model, tokenizer, prompt, max_tokens=max_tokens, sampler=sampler):
        out += resp.text
    # 剥离思考痕迹后再返回：保证多版本对比只比回复本体（复用生产过滤器，单一真源）
    from app.output.filters import strip_think
    return strip_think(out)


def real_chain_messages(user_text: str, tmp: str) -> tuple[list[dict], list[str]]:
    """走线上同款链路：世界书扫描 → 情感状态 → build_messages。返回消息与激活条目。"""
    from app.context import build_messages
    from app.emotion.engine import EmotionEngine
    from app.memory.store import MemoryStore
    from app.worldbook.engine import WorldbookEngine

    wb = WorldbookEngine()
    emotion = EmotionEngine(Path(tmp) / "emotion.json")
    store = MemoryStore(Path(tmp) / "eval.db")
    assembly = wb.scan(user_text, [])
    msgs = build_messages(user_text, assembly, emotion, store, [])
    activated = [a.entry.id for a in assembly.entries]
    store.close()
    return msgs, activated


def main() -> int:
    ap = argparse.ArgumentParser()
    # 默认本地 Qwen3-4B 基座（Sine 3.4 锁定；1.5B 通道已废弃，本地副本已不存在）
    # 与 scripts/train_lora.sh 的 LOCAL_BASE 同源，避免误加载旧基座导致层数不匹配
    ap.add_argument("--model", default="/Users/ntars/Documents/Github/Sine 大模型/基座模型/Qwen3-4B-Instruct-2507-4bit")
    ap.add_argument("--adapter", default="runs/sine-lora")
    ap.add_argument("--mode", default="both", choices=["real", "bare", "both"])
    ap.add_argument("--deterministic", action="store_true",
                    help="确定性评测（贪心+固定种子），多版本对比可复现")
    args = ap.parse_args()
    if not Path(args.adapter, "adapters.safetensors").exists():
        print(f"adapter 不存在：{args.adapter}", file=sys.stderr)
        return 1

    model, tokenizer = load(args.model, adapter_path=args.adapter)

    if args.mode in ("real", "both"):
        print("=" * 64)
        print("真实链路（世界书按触发词注入 → 情感状态 → 组装 → 生成）")
        print("=" * 64)
        with tempfile.TemporaryDirectory() as tmp:
            for p in PROBES:
                msgs, activated = real_chain_messages(p, tmp)
                reply = run(model, tokenizer, msgs, deterministic=args.deterministic)
                print(f"\n用户：{p}")
                print(f"激活条目：{activated}")
                print(f"玖玖：{reply}")

    if args.mode in ("bare", "both"):
        print("\n" + "=" * 64)
        print("裸提示对照（无场景卡，训练外的分布）")
        print("=" * 64)
        system = _base_system_prompt()
        for p in PROBES:
            reply = run(model, tokenizer,
                        [{"role": "system", "content": system}, {"role": "user", "content": p}],
                        deterministic=args.deterministic)
            print(f"\n用户：{p}")
            print(f"玖玖：{reply}")

    print("\n" + "=" * 64)
    print("日记探针（目标：独白体，不汇报流水账，嘴硬收尾）")
    print("=" * 64)
    from app.diary.author import ANGLES, _SYSTEM as DIARY_SYSTEM
    diary_msgs = [
        {"role": "system", "content": DIARY_SYSTEM},
        {"role": "user", "content": f"今晚写这个角度：{ANGLES[0]}\n写吧。"},
    ]
    print(run(model, tokenizer, diary_msgs, max_tokens=300, temperature=0.85))
    return 0


if __name__ == "__main__":
    sys.exit(main())