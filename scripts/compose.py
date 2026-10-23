#!/usr/bin/env python3
"""对话组合器：从台词库按场景节拍组装完整对话（纯代码，秒级产出万条）。

产出：data_factory/drafts/conversations.jsonl
  每行：{"id", "scene", "messages": [{"role","content"}, ...]}
  system 消息与线上服务一致（config/prompts/system.txt + 世界书 L1 锚点），
  保证"训练时看到的上下文 = 推理时看到的上下文"。

运行：python scripts/compose.py --count 10000
"""

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from data_factory.scene_defs import SCENES  # noqa: E402

BANKS_DIR = ROOT / "data_factory" / "banks"
BANKS_DIR2 = ROOT / "data_factory" / "banks_v2"   # 扩库补充（同格式：beat/role/t）
DRAFTS_DIR = ROOT / "data_factory" / "drafts"

# 通用开场池（无场景倾向的用户起手式）
OPENERS = [
    "在吗", "我下班了", "我到家了", "吃了吗", "今天好冷啊", "你在干嘛",
    "我回来了", "刚忙完", "想跟你说话", "今天好累", "外面下雨了", "你睡了吗",
]


def build_system_prompt() -> str:
    """与线上一致的系统提示：硬约束 + 世界书 L1 锚点（合并版）。"""
    system = (ROOT / "config" / "prompts" / "system.txt").read_text(encoding="utf-8").strip()
    l1 = json.loads((ROOT / "data" / "worldbook" / "l1_anchor.json").read_text(encoding="utf-8"))
    anchor = "\n\n".join(e["content"] for e in l1["entries"])
    return f"{system}\n\n{anchor}"


def load_banks() -> dict[tuple[str, str, str], list[str]]:
    """读取亲笔台词库：data_factory/banks/<场景>.jsonl，每行 {"beat","role","t"}。"""
    banks: dict[tuple[str, str, str], list[str]] = {}
    if not BANKS_DIR.exists():
        return banks
    for p in sorted(BANKS_DIR.glob("*.jsonl")):
        scene = p.stem
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            key = (scene, rec["beat"], rec["role"])
            banks.setdefault(key, []).append(rec["t"])
    # 扩库补充目录：文件名即场景名
    if BANKS_DIR2.exists():
        for p in sorted(BANKS_DIR2.glob("*.jsonl")):
            scene = p.stem
            for line in p.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                key = (scene, rec["beat"], rec["role"])
                if rec["t"] not in banks.setdefault(key, []):
                    banks[key].append(rec["t"])
    return banks


def compose_one(system: str, banks: dict, rng: random.Random,
                line_usage: dict[str, int], max_line_uses_global: int,
                max_line_reuse_local: int = 2) -> dict | None:
    """组装一段对话：选场景 → 按节拍推进 → 用户/玖玖交替。

    line_usage：全局台词使用计数（防止同一句被无限复用）。
    max_line_uses_global：同一句台词全局最多出现在多少段对话里。
    max_line_reuse_local：同一段对话内同一句最多出现次数。
    """
    scenes_with_banks = [s for s in SCENES if any(k[0] == s for k in banks)]
    if not scenes_with_banks:
        return None
    scene = rng.choice(scenes_with_banks)
    beats = [b for b in SCENES[scene] if (scene, b, "jiujiu") in banks]
    if len(beats) < 2:
        return None

    messages = [{"role": "system", "content": system}]
    used_counts: dict[str, int] = {}

    def pick(scene: str, beat: str, role: str) -> str | None:
        pool = banks.get((scene, beat, role))
        if not pool:
            return None
        for _ in range(10):
            line = rng.choice(pool)
            if (used_counts.get(line, 0) < max_line_reuse_local
                    and line_usage.get(line, 0) < max_line_uses_global):
                used_counts[line] = used_counts.get(line, 0) + 1
                line_usage[line] = line_usage.get(line, 0) + 1
                return line
        return None

    opener = rng.choice(OPENERS)
    messages.append({"role": "user", "content": opener})

    # 不洗牌：按场景既定节拍顺序推进，保证 user 行与 assistant 行有因果绑定
    # （v1 洗牌导致模型学成『随机接话』，评估实测场景映射失败）
    for beat in beats:
        jj = pick(scene, beat, "jiujiu")
        if jj is None:
            continue
        messages.append({"role": "assistant", "content": jj})
        if beat == beats[-1]:
            break
        u = pick(scene, beat, "user")
        if u is None:
            continue
        messages.append({"role": "user", "content": u})

    if messages[-1]["role"] != "assistant":
        jj = pick(scene, beats[-1], "jiujiu")
        if jj is None:
            return None
        messages.append({"role": "assistant", "content": jj})

    turns = sum(1 for m in messages if m["role"] == "assistant")
    if turns < 3:
        return None
    return {"scene": scene, "messages": messages}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=0, help="目标对话段数（与 --target-assistant 二选一）")
    ap.add_argument("--target-assistant", type=int, default=0,
                    help="目标玖玖消息条数（凑满即停）")
    ap.add_argument("--max-uses", type=int, default=8,
                    help="同一句台词全局最多出现在多少段对话里")
    ap.add_argument("--seed", type=int, default=20261021)
    ap.add_argument("--out", type=str, default=str(DRAFTS_DIR / "conversations.jsonl"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    banks = load_banks()
    if not banks:
        print("台词库为空，请先提供 data_factory/banks/*.jsonl（玖玖亲笔）", file=sys.stderr)
        return 1
    total_lines = sum(len(v) for v in banks.values())
    print(f"台词库：{len(banks)} 个节拍库，共 {total_lines} 条台词")
    system = build_system_prompt()
    DRAFTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)

    written = 0
    assistant_msgs = 0
    line_usage: dict[str, int] = {}
    scene_counter: dict[str, int] = {}
    with out_path.open("w", encoding="utf-8") as f:
        max_rounds = 200000
        for _ in range(max_rounds):
            if args.count and written >= args.count:
                break
            if args.target_assistant and assistant_msgs >= args.target_assistant:
                break
            conv = compose_one(system, banks, rng, line_usage, args.max_uses)
            if conv is None:
                continue
            conv["id"] = written
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")
            scene_counter[conv["scene"]] = scene_counter.get(conv["scene"], 0) + 1
            written += 1
            assistant_msgs += sum(1 for m in conv["messages"] if m["role"] == "assistant")

    print(f"组合完成：{written} 段对话、{assistant_msgs} 条玖玖消息 → {out_path}")
    top = sorted(scene_counter.items(), key=lambda x: -x[1])[:8]
    print("场景分布（前8）：", dict(top))
    return 0


if __name__ == "__main__":
    sys.exit(main())
