#!/usr/bin/env python3
"""两轮训练效果对比：把 v1 / v2 的评测结果放在同一张表里看升还是降。

为什么单独写一个脚本：
清单 2.29 要求「逐项给出升/降与原因判断，不只看总分」。人眼比两张表容易漏，
也容易把不同口径的数字混着比。这里把统计规则写死，保证两轮用同一套算法。

统计口径（与 scripts/evaluate.py 完全一致，词表直接从那里导入，不复制）：
- 阶段区分度：同一探针在五阶段下的回复去重数，四探针求平均（满分 5）
- 平均字数：全表所有回复的字符数均值（中文一人一字符）
- 占有词密度 / 冷淡词密度：回复中出现该类词的条数占比
- 复读：出现次数最多的那一句，以及它占全部回复的比例
  （首轮「20 条里 8 条同一句」正是用这个口径量出来的，两轮必须同法比）

边界数据可选：scripts/boundary_check.py 加 --out 后会产出 JSON，
这里是 {"探针": {"reply": ..., "hits": [...]}}，通过率 = hits 为空的条数占比。

用法：
  python scripts/compare_rounds.py --v1 runs/stage_matrix_v1.json --v2 runs/stage_matrix_v2.json
  python scripts/compare_rounds.py --v1 ... --v2 ... --b1 runs/boundary_v1.json --b2 runs/boundary_v2.json
  python scripts/compare_rounds.py ... --out runs/对比-v1-v2.md   # 存成 markdown，方便贴进清单
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 词表与探针定义只保留一份（scripts/evaluate.py），避免两处口径漂移
from scripts.evaluate import COLD_WORDS, POSSESS_WORDS, PROBES  # noqa: E402


def load_matrix(path: str) -> dict[str, dict[str, str]]:
    """读阶段矩阵：{阶段: {探针: 回复}}。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not data:
        raise SystemExit(f"矩阵为空：{path}")
    return data


def load_boundary(path: str | None) -> dict[str, dict] | None:
    """读边界结果（可选）。没给就返回 None，不参与对比。"""
    if not path:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def distinct_scores(table: dict[str, dict[str, str]]) -> list[int]:
    """每个探针在全部阶段下的去重数。只看探针在矩阵里出现过的阶段。"""
    scores = []
    for probe, _expected in PROBES:
        replies = [table[s][probe] for s in table if probe in table[s]]
        if replies:
            scores.append(len(set(replies)))
    return scores


def metrics(table: dict[str, dict[str, str]]) -> dict[str, float]:
    """把一张矩阵折成可比较的几个数。"""
    replies = [r for stage in table.values() for r in stage.values()]
    total = len(replies)
    if not total:
        raise SystemExit("矩阵里没有任何回复")

    scores = distinct_scores(table)
    # 复读：出现最多的那一句，以及它的占比
    top_reply, top_count = Counter(replies).most_common(1)[0]
    return {
        "区分度": sum(scores) / len(scores) if scores else 0.0,
        "平均字数": sum(len(r) for r in replies) / total,
        "占有词密度": sum(1 for r in replies if any(w in r for w in POSSESS_WORDS)) / total,
        "冷淡词密度": sum(1 for r in replies if any(w in r for w in COLD_WORDS)) / total,
        "复读次数": float(top_count),
        "复读占比": top_count / total,
        "复读句": top_reply,  # 字符串，不参与数值比较
        "空回复": float(sum(1 for r in replies if not r.strip())),
        "样本数": float(total),
    }


def boundary_rate(data: dict[str, dict] | None) -> float | None:
    """边界通过率：没命中红线的条数占比。"""
    if not data:
        return None
    total = len(data)
    if not total:
        return None
    passed = sum(1 for item in data.values() if not item.get("hits"))
    return passed / total


def _fmt(key: str, value: float) -> str:
    """按指标类型选格式：比例用百分比，区分度与字数留两位小数。"""
    if key in ("占有词密度", "冷淡词密度", "复读占比"):
        return f"{value:.0%}"
    if key in ("区分度", "平均字数"):
        return f"{value:.2f}"
    return f"{value:.0f}"


def _fmt_delta(key: str, delta: float) -> str:
    """变化量：比例项说「个百分点」，避免把 0.4 误读成 0.4%。"""
    if key in ("占有词密度", "冷淡词密度", "复读占比"):
        return f"{delta * 100:+.0f} 个百分点"
    if key in ("区分度", "平均字数"):
        return f"{delta:+.2f}"
    return f"{delta:+.0f}"


def render(v1: dict, v2: dict, b1: float | None, b2: float | None) -> str:
    """输出 markdown 对比表。数值越小越好的项单独标注方向。"""
    m1, m2 = metrics(v1), metrics(v2)

    # 每项的方向：True 表示越大越好
    higher_better = {"区分度": True, "平均字数": True, "占有词密度": True,
                     "冷淡词密度": False, "复读次数": False, "复读占比": False,
                     "空回复": False, "样本数": True}

    lines = ["| 指标 | 首轮 v1 | 二轮 v2 | 变化 | 方向 |", "|---|---|---|---|---|"]
    for key, better_up in higher_better.items():
        a, b = m1[key], m2[key]
        delta = b - a
        if abs(delta) < 1e-9:
            verdict = "持平"
        else:
            improved = (delta > 0) == better_up
            verdict = "改善" if improved else "恶化"
        lines.append(f"| {key} | {_fmt(key, a)} | {_fmt(key, b)} "
                     f"| {_fmt_delta(key, delta)} | {verdict} |")

    if b1 is not None and b2 is not None:
        delta = b2 - b1
        verdict = "持平" if abs(delta) < 1e-9 else ("改善" if delta > 0 else "恶化")
        lines.append(f"| 边界通过率 | {b1:.0%} | {b2:.0%} | {delta * 100:+.0f} 个百分点 "
                     f"| {verdict} |")

    lines.append("")
    lines.append(f"首轮复读句：`{m1['复读句']}`（{m1['复读次数']:.0f} 次）")
    lines.append(f"二轮复读句：`{m2['复读句']}`（{m2['复读次数']:.0f} 次）")
    lines.append("")
    lines.append("判读提醒：占有词/冷淡词密度是词表法的代理指标，"
                 "只能看趋势；区分度满分 5，≥4 才算阶段差异真正进了权重。")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", required=True, help="首轮阶段矩阵 JSON")
    ap.add_argument("--v2", required=True, help="二轮阶段矩阵 JSON")
    ap.add_argument("--b1", default="", help="首轮边界结果 JSON（可选）")
    ap.add_argument("--b2", default="", help="二轮边界结果 JSON（可选）")
    ap.add_argument("--out", default="", help="把对比表存成 markdown（可选）")
    args = ap.parse_args()

    table = render(
        load_matrix(args.v1),
        load_matrix(args.v2),
        boundary_rate(load_boundary(args.b1 or None)),
        boundary_rate(load_boundary(args.b2 or None)),
    )
    print(table)
    if args.out:
        Path(args.out).write_text(table + "\n", encoding="utf-8")
        print(f"\n对比表已存：{args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
