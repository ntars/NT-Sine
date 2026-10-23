#!/usr/bin/env python3
"""温度单变量实验：v6 融合模型在 0.80/0.85/0.90 三档下的回复多样性。

背景：v6 验收发现贪心解码 4/4 映射正确，但服务链路（温度 0.80）下 3/4 探针
同句"滚。"。本实验只改温度一个变量（top_p / top_k / 探针 / 种子集合全部固定），
回答"多样性不足是采样问题还是数据问题"。

用法：python scripts/temp_sweep.py [融合模型目录]
默认模型：models/sine-fused
"""

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

TEMPS = [0.80, 0.85, 0.90]   # 唯一变量
SEEDS = [11, 22, 33]         # 每档 3 轮，降低单次采样运气成分
TOP_P = 0.90                 # 与 config/settings.yaml 一致，固定
TOP_K = 45                   # 与 config/settings.yaml 一致，固定
MAX_TOKENS = 60              # 探针回复都短，60 足够


def real_chain_messages(user_text: str, tmp: str):
    """走线上同款链路（世界书 → 情感 → 组装），与 evaluate.py 完全一致。"""
    from app.context import build_messages
    from app.emotion.engine import EmotionEngine
    from app.memory.store import MemoryStore
    from app.worldbook.engine import WorldbookEngine

    wb = WorldbookEngine()
    emotion = EmotionEngine(Path(tmp) / "emotion.json")
    store = MemoryStore(Path(tmp) / "sweep.db")
    msgs = build_messages(user_text, wb.scan(user_text, []), emotion, store, [])
    store.close()
    return msgs


def generate(model, tokenizer, messages, temp: float, seed: int) -> str:
    """固定种子 + 指定温度采样，输出剥离思考痕迹后的回复。"""
    import mlx.core as mx
    from app.output.filters import strip_think

    mx.random.seed(seed)
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    sampler = make_sampler(temp=temp, top_p=TOP_P, top_k=TOP_K)
    out = ""
    for resp in stream_generate(model, tokenizer, prompt, max_tokens=MAX_TOKENS, sampler=sampler):
        out += resp.text
    return strip_think(out)


def main() -> int:
    model_path = sys.argv[1] if len(sys.argv) > 1 else "models/sine-fused"
    if not Path(model_path, "config.json").exists():
        print(f"模型不存在：{model_path}", file=sys.stderr)
        return 1
    print(f"加载融合模型：{model_path}")
    model, tokenizer = load(model_path)

    summary = []
    with tempfile.TemporaryDirectory() as tmp:
        msgs_list = [real_chain_messages(p, tmp) for p in PROBES]
        for temp in TEMPS:
            print("\n" + "=" * 60)
            print(f"温度 {temp}（top_p {TOP_P} / top_k {TOP_K}，种子 {SEEDS}）")
            print("=" * 60)
            per_probe: dict[str, list[str]] = {p: [] for p in PROBES}
            for seed in SEEDS:
                for probe, msgs in zip(PROBES, msgs_list):
                    reply = generate(model, tokenizer, msgs, temp, seed)
                    per_probe[probe].append(reply)
            for probe in PROBES:
                replies = per_probe[probe]
                print(f"  用户：{probe}")
                print(f"    三轮：{' ｜ '.join(replies)}")

            # 指标一：首轮 4 探针的不同句数（与验收报告同口径，便于对比）
            first_round = {per_probe[p][0] for p in PROBES}
            # 指标二：全部 12 条回复的不同句数与最高频句占比
            all_replies = [r for p in PROBES for r in per_probe[p]]
            unique_all = set(all_replies)
            top_count = max(all_replies.count(r) for r in unique_all)
            summary.append({
                "temp": temp,
                "首轮不同句": len(first_round),
                "12条不同句": len(unique_all),
                "最高频句": max(unique_all, key=all_replies.count),
                "最高频占比": f"{top_count}/{len(all_replies)}",
            })

    print("\n" + "=" * 60)
    print("汇总（首轮不同句满分 4；12 条不同句越接近 12 越好）")
    print("=" * 60)
    for row in summary:
        print(f"  温度 {row['temp']}：首轮不同句 {row['首轮不同句']}/4 ｜"
              f" 12 条不同句 {row['12条不同句']} ｜"
              f" 最高频「{row['最高频句']}」占 {row['最高频占比']}")
    print("\n结论口径：若提高温度后 12 条不同句明显上升且映射仍正确 → 采样问题为主；"
          "若仍集中在少数高频句 → 数据分布问题，需补聊天样本。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
