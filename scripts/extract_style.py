#!/usr/bin/env python3
"""从原型小说提取语言风格统计（只输出统计与样本清单，不复制正文进训练集）。

输出：
  data_factory/style_profile.json  — 风格画像（句长分布、标点习惯、高频语气词计数）
  data_factory/heroine_samples.txt — 夏柠柒台词样本（供数据工厂提示词参考，不入训练集）

运行：python scripts/extract_style.py <小说路径>
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data_factory"

QUOTE_RE = re.compile(r"[“\"]([^”\"]{2,80})[”\"]")


def main() -> int:
    if len(sys.argv) < 2:
        print("用法: python scripts/extract_style.py <小说txt路径>", file=sys.stderr)
        return 2
    novel = Path(sys.argv[1])
    text = novel.read_text(encoding="utf-8")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    quotes: list[str] = []
    heroine_samples: list[str] = []
    for i, ln in enumerate(lines):
        qs = QUOTE_RE.findall(ln)
        quotes.extend(qs)
        # 启发式：本行或相邻行提到夏柠柒 → 视为她的台词
        window = "\n".join(lines[max(0, i - 1): i + 2])
        if qs and "夏柠柒" in window:
            heroine_samples.extend(qs)

    lens = sorted(len(q) for q in quotes)
    n = len(lens)
    stats = {
        "总行数": len(lines),
        "对白总句数": n,
        "句长_均值": round(sum(lens) / max(1, n), 1),
        "句长_中位": lens[n // 2] if n else 0,
        "句长_p10": lens[n // 10] if n else 0,
        "句长_p90": lens[n * 9 // 10] if n else 0,
        "含省略号占比": round(sum("…" in q or "..." in q for q in quotes) / max(1, n), 3),
        "含问号占比": round(sum("？" in q or "?" in q for q in quotes) / max(1, n), 3),
        "单句≤10字占比": round(sum(l <= 10 for l in lens) / max(1, n), 3),
    }
    particles = Counter()
    for q in quotes:
        for p in ["哼", "啧", "哦", "呵", "喂", "喵", "笨", "滚", "闭嘴", "有病",
                  "傻子", "变态", "……", "！"]:
            if p in q:
                particles[p] += 1
    stats["高频成分计数"] = dict(particles.most_common(20))

    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "style_profile.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # 样本最多留 300 条，打乱前先切片
    sample = heroine_samples[:300]
    (OUT_DIR / "heroine_samples.txt").write_text("\n".join(sample), encoding="utf-8")
    stats["夏柠柒台词样本数"] = len(heroine_samples)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
