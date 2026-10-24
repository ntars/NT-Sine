#!/usr/bin/env python3
"""台词库多样性度量：动作 1 的验收尺子。

为什么需要它：2.30 判定的核心量化事实是「每格台词独立率 p50 仅 24%，破冰/被夸
128 个样本只有 19 句不同的话」，也就是模型在「19 句池子里挑一句」，贪心解码必然
每次挑同一句——复读是数据结构的必然后果。要把这条从「结论」变成「可跟踪的指标」，
就得有一把固定口径的尺子，扩写前后各量一次。

**两个口径必须分清，不能混着说**（这是本项目最容易糊弄自己的地方）：
  - 库内独立率 = 该格唯一台词数 / 该格台词总数。只说明「写的时候有没有重复」，
    现在全库都已接近 100%，看着很漂亮，但它**不等于**训练时模型看到的多样性。
  - 样本级独立率 = 该格唯一台词数 / 该格在训练集里出现的次数。样本可以被反复
    采样（同一个池子被 30 段对话共用），所以这个数才会掉到 24%。
本脚本默认两个都算：库内独立率从台词文件直接算；样本级独立率按训练集的抽样方式
估算（每格池大小 ÷ 该格预期样本数），并明确标注是估算值。

用法：
  python scripts/bank_diversity.py            # 全库一览
  python scripts/bank_diversity.py --check    # 只看是否达到扩写目标，不达标退出码 1
  python scripts/bank_diversity.py --stage cold
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data_factory.scene_defs import STAGE_SCENES, STAGE_WEIGHTS  # noqa: E402

BANKS_DIR = ROOT / "data_factory" / "banks"

# 动作 1 的目标：每（阶段 × 场景）从 25 条扩到 60 条
TARGET_PER_CELL = 60
# 扩写前的基线（2.30 记录）：样本级独立率中位数
BASELINE_SAMPLE_RATE = 0.24


def read_cell(path: Path) -> list[dict]:
    """读一格台词；坏行直接报文件名与行号，不静默跳过。"""
    rows: list[dict] = []
    if not path.exists():
        return rows
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        import json

        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"格式错误：{path.relative_to(ROOT)} 第 {i} 行（{exc.msg}）") from exc
    return rows


def cell_stats(path: Path) -> dict:
    """一格的统计：总条数、唯一句数、库内独立率、她的台词条数。

    独立率只统计 **她的台词**（role=jiujiu）：用户那句是场景铺垫，
    模型要学的是她怎么回，把用户台词算进来会把指标稀释得好看而不真实。
    """
    rows = read_cell(path)
    hers = [r["t"] for r in rows if r.get("role") == "jiujiu"]
    unique = len(set(hers))
    return {
        "rows": len(rows),
        "hers": len(hers),
        "unique": unique,
        "rate": (unique / len(hers)) if hers else 0.0,
        "dups": len(hers) - unique,
    }


def collect(stage_filter: str | None = None) -> list[dict]:
    """按 scene_defs 的定义遍历（以定义为准，目录里多写的文件也算出来）。"""
    out: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for stage, scenes in STAGE_SCENES.items():
        if stage_filter and stage != stage_filter:
            continue
        for scene in scenes:
            path = BANKS_DIR / stage / f"{scene}.jsonl"
            seen.add((stage, scene))
            stats = cell_stats(path)
            stats.update({"stage": stage, "scene": scene, "exists": path.exists()})
            out.append(stats)
    for stage_dir in sorted(p for p in BANKS_DIR.iterdir() if p.is_dir()):
        if stage_filter and stage_dir.name != stage_filter:
            continue
        for f in sorted(stage_dir.glob("*.jsonl")):
            if (stage_dir.name, f.stem) in seen:
                continue
            stats = cell_stats(f)
            stats.update({"stage": stage_dir.name, "scene": f.stem, "exists": True,
                          "orphan": True})
            out.append(stats)
    return out


def expected_samples(stage: str, total_cells_in_stage: int, total_convs: int) -> int:
    """估算该阶段一格的训练样本数（用于算样本级独立率）。

    算法照 build_dataset 的做法：先按权重把总对话数分给五个阶段，再在同一阶段的
    各场景间随机挑，于是「一格拿到的对话数 ≈ 该阶段对话数 ÷ 该阶段场景数」。
    每段对话产出约 2 个样本（assemble 保证至少 2 句她的台词）。
    """
    weight = STAGE_WEIGHTS.get(stage, 0)
    stage_convs = max(1, round(total_convs * weight))
    per_cell_convs = max(1, stage_convs // max(1, total_cells_in_stage))
    return per_cell_convs * 2


def report(rows: list[dict], total_convs: int = 300) -> None:
    cells_by_stage: dict[str, int] = defaultdict(int)
    for r in rows:
        cells_by_stage[r["stage"]] += 1

    print("| 阶段 | 场景 | 条数 | 她的台词 | 唯一 | 库内独立率 | 估算样本级独立率 |")
    print("|---|---|---:|---:|---:|---:|---:|")
    rates: list[float] = []
    for r in sorted(rows, key=lambda x: (x["stage"], x["scene"])):
        exp = expected_samples(r["stage"], cells_by_stage[r["stage"]], total_convs)
        sample_rate = min(1.0, r["unique"] / exp) if exp else 0.0
        if r["hers"]:
            rates.append(sample_rate)
        flag = "" if r["exists"] else "（缺文件）"
        print(f"| {r['stage']} | {r['scene']}{flag} | {r['rows']} | {r['hers']} | "
              f"{r['unique']} | {r['rate'] * 100:.0f}% | {sample_rate * 100:.0f}% |")

    rates.sort()
    p50 = rates[len(rates) // 2] * 100 if rates else 0.0
    print(f"\n估算样本级独立率中位数：**{p50:.0f}%**（扩写前基线 {BASELINE_SAMPLE_RATE * 100:.0f}%）")
    total_rows = sum(r["rows"] for r in rows)
    total_dups = sum(r["dups"] for r in rows)
    print(f"全库 {total_rows} 条，其中她的台词重复 {total_dups} 条")
    short = [r for r in rows if r["rows"] < TARGET_PER_CELL]
    if short:
        print(f"\n未到 {TARGET_PER_CELL} 条的格子 {len(short)} 个：")
        for r in short:
            print(f"  {r['stage']}/{r['scene']}：{r['rows']} 条（缺 {TARGET_PER_CELL - r['rows']}）")
    else:
        print(f"\n全部格子均已达 {TARGET_PER_CELL} 条。")


def main() -> int:
    ap = argparse.ArgumentParser(description="台词库多样性度量")
    ap.add_argument("--stage", default=None, help="只看某个阶段")
    ap.add_argument("--convs", type=int, default=300, help="训练对话段数（默认同 build_dataset）")
    ap.add_argument("--check", action="store_true", help="只判断是否达标，不达标退出码 1")
    args = ap.parse_args()

    rows = collect(args.stage)
    if not rows:
        print("没有找到任何台词格", file=sys.stderr)
        return 1

    if args.check:
        short = [r for r in rows if r["rows"] < TARGET_PER_CELL]
        dups = [r for r in rows if r["dups"] > 0]
        for r in short:
            print(f"待补：{r['stage']}/{r['scene']} {r['rows']}/{TARGET_PER_CELL}")
        for r in dups:
            print(f"有重复：{r['stage']}/{r['scene']} 重复 {r['dups']} 条")
        if short or dups:
            return 1
        print(f"全部 {len(rows)} 格均达 {TARGET_PER_CELL} 条且无重复")
        return 0

    report(rows, args.convs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
