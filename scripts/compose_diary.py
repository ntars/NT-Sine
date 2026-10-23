#!/usr/bin/env python3
"""日记语料重组器：把手写核心 + 手写片段扩成 N 条独白样本。

素材来源：全部由作者亲笔（data_factory/diary_core*.jsonl 手写独白本体、
data_factory/diary_parts.jsonl 手写开场/细节/收尾片段），
本脚本只做**重组**：open + core + detail + close → 一条完整独白。
不使用任何模型生成，风格统一且可复现。

产出：
  data_factory/diary_pool.jsonl   —— 独白池（angle + text）
  data_train/diary.jsonl          —— mlx-lm 训练格式（含 system/用户/目标）

质检：长度区间、句内复读（同一 2-6 字片段重复出现）、禁词、去重。

运行：python scripts/compose_diary.py --target 5000
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.diary.author import ANGLES, _SYSTEM as DIARY_SYSTEM  # noqa: E402
sys.path.insert(0, str(ROOT / 'scripts'))
from filter_dataset import find_repeat as has_repeat  # noqa: E402

CORE_FILES = ["diary_core.jsonl", "diary_core2.jsonl"]
PARTS_FILE = "diary_parts.jsonl"
BANNED = ["AI", "助手", "模型", "作为语言", "我不能"]


# 高频词白名单：这些片段重复出现属于正常行文，不算复读
_COMMON = {
    "今天", "明天", "昨天", "什么", "没有", "自己", "一点", "真的", "好像", "不是",
    "就是", "因为", "所以", "已经", "还没", "一直", "有点", "这个", "那个", "一直",
}


def has_repeat(text: str, min_len: int = 3, max_len: int = 6) -> str | None:
    """检测句内复读：同一 3-6 字片段出现两次（不含高频词与相邻叠词）。"""
    for n in range(min_len, max_len + 1):
        for i in range(len(text) - n * 2 + 1):
            frag = text[i:i + n]
            if frag in _COMMON:
                continue
            if text.count(frag) >= 2:
                # 相邻叠字不算病态；间隔至少 1 个字才算
                if text.find(frag, i + n + 1) != -1:
                    return frag
    return None


def load_core() -> dict[str, list[str]]:
    cores: dict[str, list[str]] = {}
    for name in CORE_FILES:
        p = ROOT / "data_factory" / name
        for ln in p.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            r = json.loads(ln)
            cores.setdefault(r["angle"], []).append(r["text"])
    return cores


def load_parts() -> dict[str, dict[str, list[str]]]:
    parts: dict[str, dict[str, list[str]]] = {}
    p = ROOT / "data_factory" / PARTS_FILE
    for ln in p.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        r = json.loads(ln)
        parts.setdefault(r["angle"], {}).setdefault(r["kind"], []).append(r["text"])
    return parts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20261022)
    ap.add_argument("--min-chars", type=int, default=50)
    ap.add_argument("--max-chars", type=int, default=320)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    cores = load_core()
    parts = load_parts()
    hand_written = sum(len(v) for v in cores.values())
    print(f"手写独白 {hand_written} 条（{len(cores)} 个角度）+ 手写片段 {sum(len(k) for v in parts.values() for k in v.values())} 条")

    # 逐角度枚举组合
    pool: list[tuple[str, str]] = []
    seen: set[str] = set()
    for angle in ANGLES:
        angle_cores = cores.get(angle, [])
        angle_parts = parts.get(angle, {})
        opens = angle_parts.get("open", [])
        details = angle_parts.get("detail", [])
        closes = angle_parts.get("close", [])
        if not angle_cores:
            continue
        # 三段必配（开场 + 主体 + 收尾），细节可加可不加、也可加两条
        for core in angle_cores:
            for op in (opens or [""]):
                for det in (details or [""]):
                    for det2 in (details or [""]):
                        for cl in (closes or [""]):
                            body = op + core + det + (det2 if det2 != det else "")
                            text = body + cl
                            if text in seen:
                                continue
                            seen.add(text)
                            pool.append((angle, text))

    rng.shuffle(pool)
    # 质检过滤
    kept: list[tuple[str, str]] = []
    dropped = {"过长": 0, "过短": 0, "复读": 0, "禁词": 0}
    for angle, text in pool:
        if len(text) > args.max_chars:
            dropped["过长"] += 1
            continue
        if len(text) < args.min_chars:
            dropped["过短"] += 1
            continue
        if any(b in text for b in BANNED):
            dropped["禁词"] += 1
            continue
        rep = has_repeat(text)
        if rep:
            dropped["复读"] += 1
            continue
        kept.append((angle, text))
        if len(kept) >= args.target:
            break

    # 输出
    pool_path = ROOT / "data_factory" / "diary_pool.jsonl"
    pool_path.write_text("".join(json.dumps({"angle": a, "text": t}, ensure_ascii=False) + "\n"
                                  for a, t in kept), encoding="utf-8")
    train_path = ROOT / "data_train" / "diary.jsonl"
    train_path.parent.mkdir(parents=True, exist_ok=True)
    with train_path.open("w", encoding="utf-8") as f:
        for angle, text in kept:
            sample = {
                "messages": [
                    {"role": "system", "content": DIARY_SYSTEM},
                    {"role": "user", "content": f"今晚写这个角度：{angle}\n写吧。"},
                    {"role": "assistant", "content": text},
                ]
            }
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    by_angle: dict[str, int] = {}
    for a, _ in kept:
        by_angle[a] = by_angle.get(a, 0) + 1
    print(f"独白样本 {len(kept)} 条 → {train_path}")
    print("过滤统计：", dropped)
    print("角度分布：", dict(sorted(by_angle.items(), key=lambda x: -x[1])))
    return 0


if __name__ == "__main__":
    sys.exit(main())