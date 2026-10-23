#!/usr/bin/env python3
"""训练格式转换：对话 → mlx-lm 微调 JSONL。

每条玖玖消息生成一个训练样本：
  系统提示 + 之前的全部对话 + 本轮玖玖回复（作为监督目标）

训练专用系统提示（与线上服务的差别是故意的）：
  线上 system = 硬约束 + L1 + L6 + L2画像 + L5场景 + L4场景卡 + 召回（运行时动态生成）
  训练 system = 硬约束 + L1 + L6 + 【当前场景】对应的 L4 条目
理由：L2/L5/召回 是运行时数据，让模型背诵会导致推理时"幻觉画像"；
但 L4 场景卡必须出现在训练样本里——推理时 worldbook 按触发词注入它，
训练时不给模型看，模型就学不会"读场景条目→说对应的话"（v1 实测失败）。
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLEAN = ROOT / "data_factory" / "clean" / "conversations.jsonl"


def _base_system_prompt() -> str:
    """人格核心：硬约束 + L1 锚点 + L6 边界。"""
    system = (ROOT / "config" / "prompts" / "system.txt").read_text(encoding="utf-8").strip()
    l1 = json.loads((ROOT / "data" / "worldbook" / "l1_anchor.json").read_text(encoding="utf-8"))
    l6 = json.loads((ROOT / "data" / "worldbook" / "l6_boundary.json").read_text(encoding="utf-8"))
    return system + "\n\n" + "\n\n".join(e["content"] for e in l1["entries"] + l6["entries"])


def _load_entry_index() -> dict[str, str]:
    """扫描全部世界书文件，建 id → content 索引。"""
    index: dict[str, str] = {}
    for f in sorted((ROOT / "data" / "worldbook").glob("*.json")):
        raw = json.loads(f.read_text(encoding="utf-8"))
        items = raw if isinstance(raw, list) else raw.get("entries", [])
        for e in items:
            index[e["id"]] = e["content"]
    return index


def build_train_system_prompt(scene: str) -> str:
    """人格核心 + 该场景对应的 L4 场景卡（模拟推理时的 worldbook 注入）。"""
    sys.path.insert(0, str(ROOT))
    from data_factory.scene_defs import SCENE_ENTRY

    base = _base_system_prompt()
    entry_id = SCENE_ENTRY.get(scene)
    content = _load_entry_index().get(entry_id) if entry_id else None
    if not content:
        return base
    return f"{base}\n\n【当前场景】\n{content}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=str(ROOT / "data_train" / "train.jsonl"))
    ap.add_argument("--max-context", type=int, default=1536,
                    help="单样本最大字符数（超出丢弃，防止超长上下文）")
    args = ap.parse_args()

    if not CLEAN.exists():
        print("找不到 clean/conversations.jsonl，先运行 filter_dataset.py", file=sys.stderr)
        return 1
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = skipped_long = 0
    with out_path.open("w", encoding="utf-8") as out:
        for ln in CLEAN.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            conv = json.loads(ln)
            # 每段对话用自己场景的 L4 条目构造系统提示
            train_system = build_train_system_prompt(conv.get("scene", ""))
            history: list[dict] = []
            for m in conv["messages"]:
                if m["role"] == "system":
                    history.append({"role": "system", "content": train_system})
                    continue
                if m["role"] == "assistant":
                    sample = {"messages": history + [m]}
                    text_len = sum(len(x["content"]) for x in sample["messages"])
                    if text_len <= args.max_context:
                        out.write(json.dumps(sample, ensure_ascii=False) + "\n")
                        n += 1
                    else:
                        skipped_long += 1
                    history.append(m)
                else:
                    history.append(m)

    # 并入日记样本（compose_diary.py 产出）：模型需要同时学会"聊天"与"写日记"
    diary_path = ROOT / "data_train" / "diary.jsonl"
    diary_n = 0
    if diary_path.exists():
        with out_path.open("a", encoding="utf-8") as out:
            for ln in diary_path.read_text(encoding="utf-8").splitlines():
                if ln.strip():
                    out.write(ln + "\n")
                    diary_n += 1
        n += diary_n

    print(f"训练样本：{n} 条（聊天 {n - diary_n} + 日记 {diary_n}）→ {out_path}（超长跳过 {skipped_long}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
