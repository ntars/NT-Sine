#!/usr/bin/env python3
"""Sine 训练集构建器（针对 Qwen3-4B-Instruct-2507-4bit 定制，全新实现）。

为什么重写而不是改旧脚本：
1. 旧链路是「台词库 → drafts → clean → train」四级中间文件，每级都可能格式漂移；
   这里一次完成「组装 + 切样本」，只有台词库一个数据源头。
2. 旧链路按**字符数**估算长度，中文标点与 ChatML 模板会带来约 10% 误差；
   这里逐条用**真 tokenizer** 编码后测量，超长直接丢弃并计数。
3. 旧链路没有验证集，mlx-lm 只能裸训、无法判断过拟合；这里按**对话**划分
   训练/验证（不是按样本随机分——同一段对话的不同轮分到两边会造成泄漏）。
4. 旧链路不清理控制标签；Qwen3 系实测会自吐 `<think>` / `<tool_call>`，
   混进监督目标等于教模型"说话要带标签"，这里在写入前统一剥离。
5. 系统提示按基座真实的 ChatML 结构组装：人格硬约束 + L1 + L6 +
   【关系阶段】卡 + 【当前场景】卡（与线上 app/context.py 注入同源）。

基座事实（取自 config.json / tokenizer_config.json，非估计）：
  36 层 / hidden 2560 / 4bit(gs64) / 词表 151936 / eos = <|im_end|> / 无 bos

用法：
  python scripts/build_dataset.py --count 300 --max-seq 896
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data_factory.scene_defs import STAGE_SCENES, STAGE_WEIGHTS, SCENE_ENTRY  # noqa: E402
from app.emotion.engine import RELATION_STAGES, load_stage_cards  # noqa: E402

BANKS_DIR = ROOT / "data_factory" / "banks"
OUT_DIR = ROOT / "data_train"
LOCAL_BASE = Path("/Users/ntars/Documents/Github/Sine 大模型/基座模型/Qwen3-4B-Instruct-2507-4bit")

# 用户起手式（与场景无关，模拟真实开口）
OPENERS = [
    "在吗", "我下班了", "我到家了", "吃了吗", "今天好冷啊", "你在干嘛",
    "我回来了", "刚忙完", "今天好累", "外面下雨了", "你睡了吗", "我睡不着",
]

# Qwen3 系的自吐控制块：监督目标里绝不能出现
_CONTROL_BLOCK = re.compile(r"<think>[\s\S]*?</think>|<tool_call>[\s\S]*?</tool_call>|</?think>|</?tool_call>")


# ── 系统提示（与线上同源）──
def base_system() -> str:
    """人格硬约束 + L1 角色锚点 + L6 边界规则。"""
    system = (ROOT / "config" / "prompts" / "system.txt").read_text(encoding="utf-8").strip()
    parts = [system]
    for fname in ("l1_anchor.json", "l6_boundary.json"):
        raw = json.loads((ROOT / "data" / "worldbook" / fname).read_text(encoding="utf-8"))
        parts.extend(e["content"] for e in raw["entries"])
    return "\n\n".join(parts)


def scene_card(scene: str) -> str:
    """该场景对应的 L4 世界书条目正文（与推理时按触发词注入的内容同一份）。"""
    entry_id = SCENE_ENTRY.get(scene)
    if not entry_id:
        return ""
    for f in sorted((ROOT / "data" / "worldbook").glob("*.json")):
        raw = json.loads(f.read_text(encoding="utf-8"))
        for e in raw.get("entries", []):
            if e["id"] == entry_id:
                return e["content"]
    return ""


def build_system(scene: str, stage: str, _cards: dict, _stage_names: dict) -> str:
    """完整训练系统提示：人格 + 关系阶段卡 + 当前场景卡。"""
    parts = [base_system()]
    card = (_cards.get(stage) or {}).get("card", "")
    if card:
        parts.append(f"【关系阶段】{_stage_names.get(stage, stage)}\n{card.strip()}")
    sc = scene_card(scene)
    if sc:
        parts.append(f"【当前场景】\n{sc}")
    return "\n\n".join(parts)


# ── 台词库 ──
def load_banks() -> dict[tuple[str, str, str, str], list[str]]:
    """(stage, scene, beat, role) → 台词列表；同库内自动去重。"""
    banks: dict[tuple[str, str, str, str], list[str]] = defaultdict(list)
    for stage_dir in sorted(p for p in BANKS_DIR.iterdir() if p.is_dir()):
        for f in sorted(stage_dir.glob("*.jsonl")):
            for line in f.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                key = (stage_dir.name, f.stem, rec["beat"], rec["role"])
                if rec["t"] not in banks[key]:
                    banks[key].append(rec["t"])
    return banks


def assemble(stage: str, banks: dict, rng: random.Random,
             usage: Counter, max_uses: int) -> dict | None:
    """在指定阶段内组装一段对话（按该阶段的节拍序列推进，不洗牌）。"""
    stage_scenes = STAGE_SCENES.get(stage, {})
    usable = [s for s in stage_scenes if any(k[0] == stage and k[1] == s for k in banks)]
    if not usable:
        return None
    scene = rng.choice(usable)
    beats = stage_scenes[scene]

    msgs: list[dict] = [{"role": "user", "content": rng.choice(OPENERS)}]
    used_here: Counter = Counter()

    def pick(beat: str, role: str) -> str | None:
        pool = banks.get((stage, scene, beat, role))
        if not pool:
            return None
        for _ in range(12):
            line = rng.choice(pool)
            if used_here[line] < 2 and usage[line] < max_uses:
                used_here[line] += 1
                usage[line] += 1
                return line
        return None

    for beat in beats:
        if beat == "greet":
            continue  # 开场由 OPENERS 承担
        reply = pick(beat, "jiujiu")
        if reply is None:
            continue
        msgs.append({"role": "assistant", "content": reply})
        follow = pick(beat, "user")
        if follow is not None:
            msgs.append({"role": "user", "content": follow})

    if msgs[-1]["role"] != "assistant":
        tail = pick(beats[-1], "jiujiu")
        if tail is None:
            return None
        msgs.append({"role": "assistant", "content": tail})

    if sum(1 for m in msgs if m["role"] == "assistant") < 2:
        return None
    return {"stage": stage, "scene": scene, "messages": msgs}


# ── 切样本（真 tokenizer 测量）──
def to_samples(conv: dict, system: str, count_tokens, max_seq: int) -> tuple[list[dict], int]:
    """每轮她的回复切一个样本；返回 (样本列表, 因超长丢弃数)。"""
    samples: list[dict] = []
    dropped = 0
    history: list[dict] = [{"role": "system", "content": system}]
    for m in conv["messages"]:
        clean = _CONTROL_BLOCK.sub("", m["content"]).strip()
        if not clean:
            continue
        msg = {"role": m["role"], "content": clean}
        if m["role"] == "assistant":
            cand = history + [msg]
            # 用真 tokenizer 逐条量：模板开销（<|im_start|>role 等）也计入
            total = sum(len(count_tokens(x["content"])) + 8 for x in cand)
            if total <= max_seq:
                samples.append({"messages": cand})
            else:
                dropped += 1
        history.append(msg)
    return samples, dropped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=300, help="目标对话段数")
    ap.add_argument("--max-seq", type=int, default=896, help="单样本最大 token（含模板开销）")
    ap.add_argument("--max-uses", type=int, default=8, help="同一句台词最多复用次数")
    ap.add_argument("--valid-ratio", type=float, default=0.08, help="验证集对话占比")
    ap.add_argument("--seed", type=int, default=20261023)
    args = ap.parse_args()

    if not BANKS_DIR.exists():
        print(f"台词库不存在：{BANKS_DIR}", file=sys.stderr)
        return 1

    # 真 tokenizer：本地基座优先
    from mlx_lm import load
    model_id = str(LOCAL_BASE) if (LOCAL_BASE / "config.json").exists() else "mlx-community/Qwen3-4B-Instruct-2507-4bit"
    _, tokenizer = load(model_id, lazy=True)
    count_tokens = lambda s: tokenizer.encode(s, add_special_tokens=False)  # noqa: E731

    cards = load_stage_cards()
    stage_names = {s["key"]: s["name"] for s in RELATION_STAGES}
    banks = load_banks()
    print(f"台词库：{sum(len(v) for v in banks.values())} 条，"
          f"{len({k[0] for k in banks})} 个阶段 × {len({k[1] for k in banks})} 个场景")

    rng = random.Random(args.seed)
    usage: Counter = Counter()
    quota = {s: max(1, round(args.count * w)) for s, w in STAGE_WEIGHTS.items()}

    convs: list[dict] = []
    for stage, target in quota.items():
        got, miss = 0, 0
        while got < target and miss < 500:
            conv = assemble(stage, banks, rng, usage, args.max_uses)
            if conv is None:
                miss += 1
                continue
            convs.append(conv)
            got += 1

    # 按对话划分（防止同一段对话的训练/验证样本互相泄漏）
    rng.shuffle(convs)
    n_valid = max(1, int(len(convs) * args.valid_ratio))
    valid_convs, train_convs = convs[:n_valid], convs[n_valid:]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stats = {}
    for name, part in (("train", train_convs), ("valid", valid_convs)):
        out_path = OUT_DIR / f"{name}.jsonl"
        n_samples = n_dropped = 0
        stage_counter: Counter = Counter()
        scene_counter: Counter = Counter()
        lens: list[int] = []
        with out_path.open("w", encoding="utf-8") as f:
            for conv in part:
                system = build_system(conv["scene"], conv["stage"], cards, stage_names)
                samples, dropped = to_samples(conv, system, count_tokens, args.max_seq)
                n_dropped += dropped
                for s in samples:
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")
                    n_samples += 1
                    lens.append(sum(len(count_tokens(x["content"])) + 8 for x in s["messages"]))
                stage_counter[conv["stage"]] += 1
                scene_counter[conv["scene"]] += 1
        lens.sort()
        p = lambda q: lens[min(len(lens) - 1, len(lens) * q // 100)] if lens else 0  # noqa: E731
        stats[name] = (n_samples, n_dropped, dict(stage_counter), len(part), p(50), p(95), p(100))
        print(f"\n[{name}] 样本 {n_samples} 条｜对话 {len(part)} 段｜超长丢弃 {n_dropped}｜"
              f"token p50 {p(50)} / p95 {p(95)} / p100 {p(100)}")
        print(f"[{name}] 阶段分布 {dict(stage_counter)}")
        print(f"[{name}] 场景 top5 {dict(scene_counter.most_common(5))}")

    print(f"\n输出：{OUT_DIR / 'train.jsonl'} 与 {OUT_DIR / 'valid.jsonl'}")
    if stats["train"][6] > args.max_seq:
        print("⚠️ 训练集 p100 超过 max-seq，建议提高 --max-seq 或缩短台词")
    return 0


if __name__ == "__main__":
    sys.exit(main())
