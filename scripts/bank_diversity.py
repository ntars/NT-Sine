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
**估算**（有放回抽样的期望唯一率），并明确标注是估算值。

**口径修正（2026-10-24，三轮验收时发现）**：估算公式此前写成
`min(1.0, 唯一数 / 预期样本数)`，这是「无放回、且池子够大」时的**上界**，不是期望值。
池子一旦大于样本数，这个式子必然等于 1.0，于是全库都显示 100%——
看着像问题解决了，其实只是把尺子顶到了头。真实情况是：从 63 条池子里有放回抽
24 次，期望唯一率约 84%（会重复抽到），不是 100%。现改为原方案里本该用的
抽样模型 `U × (1 − (1 − 1/U)^N) / N`（U = 池大小、N = 样本数），并用实测数据
校验过：U=19、N=128 时该式给 14.8%，与二轮实测的 15% 吻合。

**要最准的数就用 `--measured`**：直接读 `data_train/train.jsonl`，按 system 里的
阶段名分组，数「最后一个 assistant 的原文」有多少种，不做任何估算假设。

**为什么还要按节拍看**（2026-10-24 修正）：`assemble()` 是**逐节拍**从池子里
`rng.choice` 的，所以真正的抽样单位不是「整格 60 条」，而是
「(阶段, 场景, 节拍, 角色) 这个池子有多少条」。一格凑到 60 条、但四个节拍各分到
15 条，和四个节拍各 4 条是完全不同的两件事——后者她在那个位置只会说 4 种话。
故本脚本提供两把尺子：
  - `--check`：整格口径（历史口径，保留用于对比）
  - `--beats`：节拍口径（**扩写的验收闸门**），每池 ≥ BEATS_MIN 条

用法：
  python scripts/bank_diversity.py            # 全库一览
  python scripts/bank_diversity.py --check    # 整格口径是否达标，不达标退出码 1
  python scripts/bank_diversity.py --beats    # 节拍口径是否达标，不达标退出码 1
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
# 节拍口径的目标：每个 (阶段, 场景, 节拍) 她的池子至少这么多条。
# 15 这个数的来历：`assemble()` 对每条台词有「同一段对话里最多用 2 次」的约束，
# 池子小于 15 时，12 次重抽的窗口里很容易全撞到已用过的，`pick()` 直接放弃该节拍；
# 池子到 15 条时，一段对话 3–4 个节拍各抽一条，重复率才能压到句均 2 次以内。
BEATS_MIN = 15
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


def beat_pools(stage_filter: str | None = None) -> list[dict]:
    """按节拍统计每一格的池子大小——这才是 `assemble()` 真正的抽样单位。

    返回每个 (阶段, 场景, 节拍) 一条记录：
      hers / unique：该节拍下她的台词条数与去重后的句数
      usr：同一节拍里用户追问的条数（不参与达标判定，只作参考——
           用户的台词不进监督目标，它少只影响上下文花样，不影响她学什么）
      missing_def：定义里写了这个节拍、实际文件里一条都没有（最严重的缺口）
    """
    out: list[dict] = []
    for stage, scenes in STAGE_SCENES.items():
        if stage_filter and stage != stage_filter:
            continue
        # 防御：STAGE_SCENES 的正常形状是 {阶段: {场景: [节拍…]}}，
        # 若被替换成别的形状（单测里会这么干），跳过而不炸。
        if not isinstance(scenes, dict):
            continue
        for scene, beats in scenes.items():
            path = BANKS_DIR / stage / f"{scene}.jsonl"
            rows = read_cell(path)
            for beat in beats:
                if beat == "greet":
                    continue  # 开场由 OPENERS 承担，台词库里没有 greet 池
                hers = [r["t"] for r in rows
                        if r.get("beat") == beat and r.get("role") == "jiujiu"]
                usr = [r for r in rows
                       if r.get("beat") == beat and r.get("role") == "user"]
                out.append({
                    "stage": stage, "scene": scene, "beat": beat,
                    "hers": len(hers), "unique": len(set(hers)), "usr": len(usr),
                    "missing_def": len(hers) == 0,
                })
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


def expected_unique_rate(pool: int, draws: int) -> float:
    """从 `pool` 条台词里有放回抽 `draws` 次，期望的「唯一句数 / 抽数」。

    这条式子是本脚本的口径要害，别改成 `min(1, pool/draws)`：
    后者是「无放回」的上界，池子一大就恒等于 1.0，指标直接失效。
    有放回抽样下每次抽不到重复的概率递减，唯一率必然低于上界——
    公式 `U × (1 − (1 − 1/U)^N) / N` 就是它的期望值。
    """
    if pool <= 0 or draws <= 0:
        return 0.0
    if pool == 1:
        return 1.0 / draws
    return pool * (1.0 - (1.0 - 1.0 / pool) ** draws) / draws


def report(rows: list[dict], total_convs: int = 300) -> None:
    cells_by_stage: dict[str, int] = defaultdict(int)
    for r in rows:
        cells_by_stage[r["stage"]] += 1

    print("| 阶段 | 场景 | 条数 | 她的台词 | 唯一 | 库内独立率 | 估算样本级独立率 |")
    print("|---|---|---:|---:|---:|---:|---:|")
    rates: list[float] = []
    for r in sorted(rows, key=lambda x: (x["stage"], x["scene"])):
        exp = expected_samples(r["stage"], cells_by_stage[r["stage"]], total_convs)
        sample_rate = expected_unique_rate(r["unique"], exp) if exp else 0.0
        if r["hers"]:
            rates.append(sample_rate)
        flag = "" if r["exists"] else "（缺文件）"
        print(f"| {r['stage']} | {r['scene']}{flag} | {r['rows']} | {r['hers']} | "
              f"{r['unique']} | {r['rate'] * 100:.0f}% | {sample_rate * 100:.0f}% |")

    rates.sort()
    p50 = rates[len(rates) // 2] * 100 if rates else 0.0
    print(f"\n估算样本级独立率中位数：**{p50:.0f}%**（扩写前基线 {BASELINE_SAMPLE_RATE * 100:.0f}%）")
    print("（估算口径按有放回抽样模型算期望值，不是上界；要真实数字用 `--measured`）")
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


def beats_report(pools: list[dict]) -> None:
    """节拍口径一览：按缺口从大到小排，先看最薄的池子。"""
    pools = sorted(pools, key=lambda p: (p["hers"], p["stage"], p["scene"], p["beat"]))
    print("| 阶段 | 场景 | 节拍 | 她的台词 | 唯一 | 用户追问 |")
    print("|---|---|---|---:|---:|---:|")
    for p in pools:
        print(f"| {p['stage']} | {p['scene']} | {p['beat']} | "
              f"{p['hers']} | {p['unique']} | {p['usr']} |")
    short = [p for p in pools if p["hers"] < BEATS_MIN]
    print(f"\n节拍池共 {len(pools)} 个，未达 {BEATS_MIN} 条的 {len(short)} 个，"
          f"合计需补 {sum(BEATS_MIN - p['hers'] for p in short)} 条")
    if short:
        sizes = sorted(p["hers"] for p in pools)
        print(f"池大小：最小 {sizes[0]}、中位 {sizes[len(sizes) // 2]}、最大 {sizes[-1]}")


def beats_check(pools: list[dict]) -> int:
    """节拍口径闸门：任一池 < BEATS_MIN 或存在重复句即返回 1。"""
    short = [p for p in pools if p["hers"] < BEATS_MIN]
    dups = [p for p in pools if p["unique"] < p["hers"]]
    for p in short:
        print(f"待补：{p['stage']}/{p['scene']}/{p['beat']} "
              f"{p['hers']}/{BEATS_MIN}（缺 {BEATS_MIN - p['hers']}）")
    for p in dups:
        print(f"有重复：{p['stage']}/{p['scene']}/{p['beat']} "
              f"重复 {p['hers'] - p['unique']} 条")
    if short or dups:
        total = sum(BEATS_MIN - p["hers"] for p in short)
        print(f"\n共 {len(short)} 个池未达标，需补 {total} 条")
        return 1
    print(f"全部 {len(pools)} 个节拍池均达 {BEATS_MIN} 条且无重复")
    return 0


def measured_report(stage_filter: str | None = None) -> int:
    """实测训练集里的样本多样性——不做任何估算，直接数真东西。

    做法：读 `data_train/train.jsonl`，从 system 里取【关系阶段】后面的阶段名分组，
    每条样本取「最后一个 assistant 的原文」作为她的回复，数有多少种不同的回复。
    这是模型真正看到的东西，比任何估算都靠得住。
    """
    import json
    import re

    path = ROOT / "data_train" / "train.jsonl"
    if not path.exists():
        print(f"找不到训练集：{path}，先跑 scripts/build_dataset.py", file=sys.stderr)
        return 1

    pat = re.compile(r"【关系阶段】([^\n]+)")
    stat: dict[str, dict] = defaultdict(lambda: {"n": 0, "uniq": set(), "cnt": Counter()})
    total = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        msgs = json.loads(line)["messages"]
        sysmsg = next((m["content"] for m in msgs if m["role"] == "system"), "")
        found = pat.search(sysmsg)
        stage = found.group(1).strip() if found else "未知"
        if stage_filter and stage != stage_filter:
            continue
        replies = [m["content"] for m in msgs if m["role"] == "assistant"]
        if not replies:
            continue
        last = replies[-1]
        stat[stage]["n"] += 1
        stat[stage]["uniq"].add(last)
        stat[stage]["cnt"][last] += 1
        total += 1

    if not total:
        print("训练集里没有可用样本", file=sys.stderr)
        return 1

    print("| 阶段 | 样本 | 唯一回复 | 样本级独立率 | 句均强化次数 | 最高频句（次数） |")
    print("|---|---:|---:|---:|---:|---|")
    tot_uniq = 0
    for stage, d in sorted(stat.items(), key=lambda kv: -kv[1]["n"]):
        uniq = len(d["uniq"])
        tot_uniq += uniq
        top, times = d["cnt"].most_common(1)[0]
        print(f"| {stage} | {d['n']} | {uniq} | {uniq / d['n'] * 100:.1f}% | "
              f"{d['n'] / uniq:.2f} | {top}（{times}） |")
    print(f"\n整体：{total} 样本 / {tot_uniq} 唯一 = {tot_uniq / total * 100:.1f}%，"
          f"句均强化 {total / tot_uniq:.2f} 次（目标 ≤2.5）")
    print(f"对照：扩写前样本级独立率中位数 {BASELINE_SAMPLE_RATE * 100:.0f}%")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="台词库多样性度量")
    ap.add_argument("--stage", default=None, help="只看某个阶段")
    ap.add_argument("--convs", type=int, default=300, help="训练对话段数（默认同 build_dataset）")
    ap.add_argument("--check", action="store_true", help="只判断整格口径是否达标，不达标退出码 1")
    ap.add_argument("--beats", action="store_true",
                    help="按节拍口径检查（真正的抽样单位），不达标退出码 1")
    ap.add_argument("--beats-report", action="store_true", help="打印节拍池一览表")
    ap.add_argument("--measured", action="store_true",
                    help="直接读 data_train 实测样本级独立率（不估算）")
    args = ap.parse_args()

    if args.measured:
        return measured_report(args.stage)

    if args.beats or args.beats_report:
        pools = beat_pools(args.stage)
        if not pools:
            print("没有找到任何节拍池", file=sys.stderr)
            return 1
        if args.beats_report:
            beats_report(pools)
            return 0
        return beats_check(pools)

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
