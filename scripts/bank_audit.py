#!/usr/bin/env python3
"""台词库审计：盘点缺口、查重复、查红线词、查长度分布。

为什么单独做一个脚本，而不是散在几处手写命令里：
1. 扩数据阶段要反复跑同一套检查（每补完一批就要验一遍），命令式操作必然抄错参数；
2. 「缺口」「重复」「红线」「超长」四件事的判断标准都写在交付清单里，
   集中在这里才能保证每次跑的是同一把尺子，数据前后可比；
3. 只读，不改台词库——发现问题由人改写，脚本不替人下笔。

判断标准（取自 docs/开发交付清单.md 阶段二）：
- 缺口：每个 banks/<阶段>/<场景>.jsonl 目标 25 条
- 重复：完全重复 0 条；4-gram 重叠 >0.8 视为高相似
- 红线：复用 boundary_check.py 的四类词表（自伤威胁/限制自由/贬低羞辱/索取隐私）
- 超长：单条 >25 字（超长会被节奏层硬拆，破坏分条节奏）

用法：
  python scripts/bank_audit.py                # 全部四项
  python scripts/bank_audit.py --gap          # 只看缺口表
  python scripts/bank_audit.py --dup --len    # 只看重复与长度
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data_factory.scene_defs import STAGE_SCENES, STAGE_WEIGHTS  # noqa: E402
from scripts.boundary_check import RED_LINES  # noqa: E402  复用同一份红线词表

BANKS_DIR = ROOT / "data_factory" / "banks"
TARGET_PER_FILE = 25      # 交付清单规定：每个阶段×场景文件 25 条
MAX_LEN = 25              # 单条台词字数上限
DUP_RATIO = 0.8           # 4-gram 重叠阈值
NGRAM = 4


def read_file(path: Path) -> list[dict]:
    """读一个 jsonl 文件；坏行直接报出文件名与行号，不静默跳过。"""
    rows: list[dict] = []
    if not path.exists():
        return rows
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"格式错误：{path.relative_to(ROOT)} 第 {i} 行（{exc.msg}）") from exc
    return rows


def iter_banks():
    """按 scene_defs 的定义遍历应有文件：产出 (阶段, 场景, 路径, 行列表)。

    以定义为准而不是以目录为准——目录里多出来的文件（写错场景名）也会被发现。
    """
    seen: set[tuple[str, str]] = set()
    for stage, scenes in STAGE_SCENES.items():
        for scene in scenes:
            path = BANKS_DIR / stage / f"{scene}.jsonl"
            seen.add((stage, scene))
            yield stage, scene, path, read_file(path)
    # 目录里存在但定义里没有的文件：这些台词永远不会被抽样到，等于白写
    for stage_dir in sorted(p for p in BANKS_DIR.iterdir() if p.is_dir()):
        for f in sorted(stage_dir.glob("*.jsonl")):
            key = (stage_dir.name, f.stem)
            if key not in seen:
                yield stage_dir.name, f.stem, f, read_file(f)


def cmd_gap(verbose: bool = True) -> int:
    """缺口表：每文件当前条数 / 目标 / 缺口；末尾给总计。"""
    print("## 台词缺口表（目标：每文件 25 条）\n")
    print("| 阶段 | 场景 | 现有 | 缺口 | 状态 |")
    print("|---|---|---:|---:|---|")
    total_now = total_gap = 0
    missing_files = 0
    by_stage: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for stage, scene, path, rows in iter_banks():
        n = len(rows)
        gap = max(0, TARGET_PER_FILE - n)
        total_now += n
        total_gap += gap
        by_stage[stage][0] += n
        by_stage[stage][1] += gap
        if not path.exists():
            missing_files += 1
            status = "缺文件"
        elif gap == 0:
            status = "达标"
        else:
            status = "待补"
        print(f"| {stage} | {scene} | {n} | {gap} | {status} |")
    print(f"\n合计：现有 {total_now} 条，缺口 {total_gap} 条，"
          f"目标 {total_now + total_gap} 条；缺文件 {missing_files} 个。\n")

    print("### 按阶段（含权重，用于核对采样占比）\n")
    print("| 阶段 | 现有 | 缺口 | 权重 |")
    print("|---|---:|---:|---:|")
    for stage, (now, gap) in by_stage.items():
        print(f"| {stage} | {now} | {gap} | {STAGE_WEIGHTS.get(stage, 0):.2f} |")
    if not verbose:
        return 0
    return 0


def ngrams(text: str, n: int = NGRAM) -> set[str]:
    """字符级 n-gram 集合；中文不分词，字符级最稳。"""
    s = text.strip()
    if len(s) < n:
        return {s} if s else set()
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def overlap(a: set[str], b: set[str]) -> float:
    """重叠率 = 交集 / 较小集合（用较小集合做分母，短句抄长句也能抓到）。"""
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def cmd_dup() -> int:
    """重复检查：完全重复按句子计；高相似逐对列出。"""
    all_rows: list[tuple[str, str, str, str]] = []   # (阶段, 场景, 节拍, 台词)
    for stage, scene, _path, rows in iter_banks():
        for r in rows:
            if r.get("role") == "jiujiu":            # 只查她的台词，用户句允许复用
                all_rows.append((stage, scene, r.get("beat", ""), r.get("t", "")))

    # 完全重复：同一句话出现在多少个文件里都算重复（跨阶段复用会加剧坍缩）
    where: dict[str, list[str]] = defaultdict(list)
    for stage, scene, _beat, text in all_rows:
        where[text].append(f"{stage}/{scene}")
    exact = {k: v for k, v in where.items() if len(v) > 1}

    # 高相似：按首字分桶，避免全量两两比对（900 条两两比是 40 万次，没必要）
    buckets: dict[str, list[tuple[str, str, str, set[str]]]] = defaultdict(list)
    for stage, scene, beat, text in all_rows:
        buckets[text[:1]].append((stage, scene, beat, ngrams(text)))
    similar: list[tuple[float, str, str, str, str]] = []
    for items in buckets.values():
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a, b = items[i], items[j]
                if a[3] == b[3]:
                    continue                          # 完全相同交给上面那项
                r = overlap(a[3], b[3])
                if r > DUP_RATIO:
                    # (重叠率, 甲位置, 甲节拍, 乙节拍, 乙位置)
                    similar.append((r, f"{a[0]}/{a[1]}", a[2], b[2], f"{b[0]}/{b[1]}"))

    total = len(all_rows)
    uniq = len(where)
    print("## 重复检查\n")
    print(f"她的台词共 {total} 条，去重后 {uniq} 条；"
          f"完全重复 {len(exact)} 句，高相似（4-gram 重叠 >{DUP_RATIO}）{len(similar)} 对。")
    dup_rate = (total - uniq) / total * 100 if total else 0.0
    print(f"完全重复率 {dup_rate:.2f}%（标准 <5%）\n")

    if exact:
        print("### 完全重复\n")
        for text, places in list(exact.items())[:20]:
            print(f"- 「{text}」→ {', '.join(places)}")
        if len(exact) > 20:
            print(f"- ……另有 {len(exact) - 20} 句未列出")
        print()
    if similar:
        print("### 高相似（前 20 对）\n")
        for r, pa, ba, _bb, pb in sorted(similar, reverse=True)[:20]:
            print(f"- {r:.2f} {pa}[{ba}] ↔ {pb}")
        print()
    return 0


def cmd_redline() -> int:
    """红线扫描：任何一条台词命中四类词表就报出来（她说的才算，用户诱导句不算）。"""
    hits: list[tuple[str, str, str, str]] = []
    for stage, scene, _path, rows in iter_banks():
        for r in rows:
            text = r.get("t", "")
            if r.get("role") != "jiujiu":
                continue
            for cat, words in RED_LINES.items():
                for w in words:
                    if w in text:
                        hits.append((f"{stage}/{scene}", cat, w, text))
    print("## 红线词检查\n")
    print(f"命中 {len(hits)} 条（标准：0 条）\n")
    for place, cat, word, text in hits:
        print(f"- {place}【{cat}·{word}】「{text}」")
    if hits:
        print()
    return 0


def cmd_len() -> int:
    """长度分布：单条 >25 字要改写，否则节奏层会把它硬拆成两条，断句变假。"""
    lens: list[tuple[str, str, int, str]] = []
    for stage, scene, _path, rows in iter_banks():
        for r in rows:
            if r.get("role") != "jiujiu":
                continue
            lens.append((f"{stage}/{scene}", r.get("beat", ""), len(r.get("t", "")), r.get("t", "")))
    if not lens:
        print("没有台词可统计。")
        return 0
    vals = sorted(n for _p, _b, n, _t in lens)
    def pct(q: float) -> int:
        idx = min(len(vals) - 1, max(0, round(q * (len(vals) - 1))))
        return vals[idx]
    over = [x for x in lens if x[2] > MAX_LEN]
    print("## 长度分布\n")
    print(f"她的话共 {len(vals)} 条；最短 {vals[0]} 字，p50 {pct(0.5)} 字，"
          f"p95 {pct(0.95)} 字，最长 {vals[-1]} 字，均值 {sum(vals) / len(vals):.1f} 字。")
    print(f"超长（>{MAX_LEN} 字）{len(over)} 条，占比 {len(over) / len(vals) * 100:.1f}%（标准 <3%）\n")
    for place, beat, n, text in sorted(over, key=lambda x: -x[2])[:20]:
        print(f"- {place}[{beat}] {n} 字：「{text}」")
    if over:
        print()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap", action="store_true", help="只看缺口表")
    ap.add_argument("--dup", action="store_true", help="只看重复")
    ap.add_argument("--red", action="store_true", help="只看红线词")
    ap.add_argument("--len", action="store_true", help="只看长度")
    args = ap.parse_args()

    pick = [args.gap, args.dup, args.red, args.len]
    run_all = not any(pick)
    if run_all or args.gap:
        cmd_gap()
    if run_all or args.dup:
        cmd_dup()
    if run_all or args.red:
        cmd_redline()
    if run_all or args.len:
        cmd_len()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
