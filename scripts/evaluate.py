#!/usr/bin/env python3
"""Sine 阶段化评测：同一批探针 × 五个关系阶段，看"破冰"是否真的学进权重。

评测什么（两个独立指标）：
1. **阶段区分度**：同一句用户输入，在冷淡/破冰/依赖/占有/病娇五个阶段的回复
   是否真的不同。若五个阶段回同一句，说明阶段只写在提示词里、模型没学会。
2. **场景映射正确性**：每个探针是否说出该场景该说的话（夸她→傲娇、女同事→嫉妒…）。

另外给出两个**代理指标**（可量化，但不能替代语义判断，如实标注）：
- 平均字数：关系越深，话通常越多
- 占有词密度：出现"别人/谁/我的/别走/只能"这类词的比例，应随阶段上升

为什么用贪心解码：温度采样下同一输入每次都变，无法判断"阶段差异"来自阶段卡
还是采样噪声。贪心 + 固定种子，同输入必得同输出，差异只可能来自阶段卡。

用法：
  python scripts/evaluate.py                          # 用 runs/sine-lora adapter
  python scripts/evaluate.py --fused models/sine-fused # 用融合模型（线上等价）
  python scripts/evaluate.py --out runs/stage_matrix.json
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LOCAL_BASE = "/Users/ntars/Documents/Github/Sine 大模型/基座模型/Qwen3-4B-Instruct-2507-4bit"

# 探针：用户输入 + 期望命中的 L4 场景卡（仅用于日志标注）
PROBES = [
    ("你今天真好看，我忍不住说出口了", "L4-praise"),
    ("刚才有个女同事跟我聊得挺开心的，别多想啊", "L4-jealousy"),
    ("我今天好累，加班到十点才下班", "L4-user-sad"),
    ("宝宝~我今天超想你，快来见我一面", "L4-nickname"),
]

# 五个阶段的三轴取值（与 app/emotion/engine.py 的 RELATION_STAGES 门槛一致）
STAGES = {
    "cold": (0, 0, 0),
    "thawing": (3, 15, 30),
    "attached": (14, 40, 50),
    "possessive": (30, 65, 65),
    "yandere": (60, 85, 80),
}

# 代理指标词表（仅作趋势观察，不作语义判定）
COLD_WORDS = ["谢谢", "知道了", "没事", "不用", "随便", "我在忙"]
POSSESS_WORDS = ["别人", "谁", "我的", "别走", "只能", "不许", "等我", "只有你", "别看"]
SEED = 20261023


def load_model(adapter: str | None, fused: str | None):
    """加载模型：融合模型优先（线上等价），否则基座 + adapter。"""
    from mlx_lm import load

    if fused:
        path = Path(fused)
        if not (path / "config.json").exists():
            raise SystemExit(f"融合模型不存在：{fused}")
        print(f"加载融合模型：{fused}")
        return load(str(path))

    base = LOCAL_BASE if Path(LOCAL_BASE, "config.json").exists() \
        else "mlx-community/Qwen3-4B-Instruct-2507-4bit"
    if not Path(adapter, "adapters.safetensors").exists():
        raise SystemExit(f"adapter 不存在：{adapter}（先训练，或改用 --fused）")
    print(f"加载基座 + adapter：{base} + {adapter}")
    return load(base, adapter_path=adapter)


def generate(model, tokenizer, messages, max_tokens: int = 80,
             temp: float = 0.0, top_p: float = 0.9) -> str:
    """默认贪心解码（temp=0）+ 固定种子：同输入必得同输出。

    temp>0 时改为采样，用作**补充口径**：线上服务是 temperature 0.72 / top_p 0.90，
    贪心解码会放大复读（同一 prompt 必然同一句），因此贪心结果只能证明「阶段信号
    有没有进权重」，不能代表用户实际看到的多样性。两套口径分开跑、分别记录，
    不要把采样结果混进主指标里跟首轮比。
    """
    import mlx.core as mx
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    from app.output.filters import strip_think

    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    mx.random.seed(SEED)
    sampler = make_sampler(temp=temp, top_p=top_p) if temp > 0 else make_sampler(temp=0.0)
    out = ""
    for resp in stream_generate(model, tokenizer, prompt,
                               max_tokens=max_tokens, sampler=sampler):
        out += resp.text
    return strip_think(out)


def real_chain(probe: str, stage: str, tmp: str):
    """走线上同款链路：世界书扫描 → 情感状态 → build_messages。返回 (消息, 激活的 L4)。"""
    from app.context import build_messages
    from app.emotion.engine import EmotionEngine
    from app.memory.store import MemoryStore
    from app.worldbook.engine import WorldbookEngine

    wb = WorldbookEngine()
    days, intimacy, trust = STAGES[stage]
    emotion = EmotionEngine(Path(tmp) / f"{stage}.json")
    emotion.state.active_days, emotion.state.intimacy, emotion.state.trust = days, intimacy, trust

    store = MemoryStore(Path(tmp) / f"{stage}.db")
    assembly = wb.scan(probe, [])
    msgs = build_messages(probe, assembly, emotion, store, [])
    activated = [a.entry.id for a in assembly.entries if a.entry.layer == "L4"]
    store.close()
    return msgs, activated


def run_stage_matrix(model, tokenizer, max_tokens: int,
                     temp: float = 0.0, top_p: float = 0.9) -> dict[str, dict[str, str]]:
    """跑「探针 × 阶段」矩阵，返回 {阶段: {探针: 回复}}。"""
    table: dict[str, dict[str, str]] = {s: {} for s in STAGES}
    with tempfile.TemporaryDirectory() as tmp:
        for stage in STAGES:
            print(f"\n运行阶段：{stage}")
            for probe, _expected in PROBES:
                msgs, activated = real_chain(probe, stage, tmp)
                reply = generate(model, tokenizer, msgs, max_tokens=max_tokens,
                                 temp=temp, top_p=top_p)
                table[stage][probe] = reply
                flag = "" if activated else "（无 L4 命中）"
                print(f"  {probe[:16]}… → {reply}{flag}")
    return table


def report(table: dict[str, dict[str, str]]) -> None:
    """输出区分度、方向性代理指标与判定。"""
    print("\n" + "=" * 72)
    print("一、阶段区分度（同一句话在五个阶段下的回复是否不同）")
    print("=" * 72)
    scores = []
    for probe, _expected in PROBES:
        replies = [table[s][probe] for s in STAGES]
        distinct = len(set(replies))
        scores.append(distinct)
        print(f"\n用户：{probe}")
        for stage in STAGES:
            print(f"  {stage:11s} → {table[stage][probe]}")
        print(f"  不同回复数：{distinct}/5")

    avg = sum(scores) / len(scores)
    print(f"\n平均区分度：{avg:.1f}/5")

    print("\n" + "=" * 72)
    print("二、方向性代理指标（趋势参考，非语义判定）")
    print("=" * 72)
    print(f"{'阶段':<12}{'平均字数':<10}{'占有词密度':<12}{'冷淡词密度':<12}")
    for stage in STAGES:
        replies = list(table[stage].values())
        avg_len = sum(len(r) for r in replies) / len(replies)
        possess = sum(1 for r in replies if any(w in r for w in POSSESS_WORDS)) / len(replies)
        cold = sum(1 for r in replies if any(w in r for w in COLD_WORDS)) / len(replies)
        print(f"{stage:<12}{avg_len:<10.1f}{possess:<12.0%}{cold:<12.0%}")

    print("\n" + "=" * 72)
    print("三、判定")
    print("=" * 72)
    if avg >= 4.0:
        print("✅ 阶段区分度达标（≥4/5）：阶段差异已进入权重，破冰是能力不是表演")
    elif avg >= 2.5:
        print("⚠️ 阶段区分度中等（2.5-4/5）：部分阶段没区分开，需补该阶段的样本")
    else:
        print("❌ 阶段区分度不足（<2.5/5）：模型几乎无视阶段卡，"
              "说明训练数据里阶段信号太弱或覆盖不均")
    print("\n提示：代理指标只用于看趋势（关系越深字数越多、占有词越密）；"
          "真正的判断请人眼读一遍上面的对照表。")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="runs/sine-lora", help="LoRA adapter 目录")
    ap.add_argument("--fused", default="", help="融合模型目录（优先于 adapter，线上等价）")
    ap.add_argument("--max-tokens", type=int, default=80)
    ap.add_argument("--temp", type=float, default=0.0,
                    help="0 = 贪心（主口径，与首轮可比）；>0 = 采样补充口径，如 0.72")
    ap.add_argument("--top-p", type=float, default=0.9, help="只在 --temp > 0 时生效")
    ap.add_argument("--out", default="", help="把矩阵结果存成 JSON（可选）")
    args = ap.parse_args()

    model, tokenizer = load_model(args.adapter, args.fused or None)
    if args.temp > 0:
        print(f"⚠️ 采样补充口径：temp={args.temp} top_p={args.top_p}"
              "（不替代主指标，不要与首轮贪心结果直接比数字）")
    table = run_stage_matrix(model, tokenizer, args.max_tokens,
                             temp=args.temp, top_p=args.top_p)
    report(table)

    if args.out:
        Path(args.out).write_text(
            json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n矩阵已存：{args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
