#!/usr/bin/env python3
"""数据集质量过滤：人设红线、长度、去重，产出干净对话集。

检查项（对应方案 §12 的人设口径）：
1. 玖玖台词禁用：波浪号、软语气词（呢/哒/呜 结尾）、暴露设定词
2. 长度：玖玖 ≤60 字，用户 ≤80 字
3. 对话至少 3 条玖玖消息
4. 完全重复的对话只留一份

运行：python scripts/filter_dataset.py
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DRAFTS = ROOT / "data_factory" / "drafts" / "conversations.jsonl"
CLEAN = ROOT / "data_factory" / "clean" / "conversations.jsonl"

BANNED_IN_JJ = ["~", "哒", "呜", "AI", "ai", "世界书", "提示词",
                "模型", "设定", "助手", "抱歉", "对不起我是"]
# 软撒娇式"呢"结尾（"好呢/行呢"这类）；反问式"想什么呢/外套呢"是冷硬语气，允许
SOFT_N_ENDINGS = ("好呢", "行呢", "是呢", "来呢", "知道呢", "喜欢呢")

# 句内复读：同一 3-6 字片段重复出现（v2 实测出现"白玖玖……白玖玖"）
_COMMON_REPEAT = {
    "今天", "明天", "昨天", "什么", "没有", "自己", "一点", "真的", "好像", "不是",
    "就是", "因为", "所以", "已经", "还没", "有点", "这个", "那个", "怎么",
}


_INTENSIFIERS = ("真是", "都", "就", "也", "明明", "简直", "一下", "一口", "才")


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"[。！？!?…]", text) if s.strip()]


def find_repeat(text: str, min_len: int = 3, max_len: int = 6) -> str | None:
    """句内重复（病态复读，如"我叫白玖玖。你又没听我的，白玖玖"）→ 返回片段。

    跨句重复默认合法——那是她的修辞："我瞎了眼。真是瞎了眼。"
    "这个我可以尝一口。就一口。" 只有在两句之间没有强调词时才判病态。
    """
    # 1) 句内重复：一律算病态
    for s in _sentences(text):
        for n in range(min_len, max_len + 1):
            for i in range(len(s) - n * 2 + 1):
                frag = s[i:i + n]
                if frag in _COMMON_REPEAT:
                    continue
                if s.count(frag) >= 2 and s.find(frag, i + n + 1) != -1:
                    return frag
    # 2) 跨句重复：无强调词衔接才算病态
    for n in range(min_len, max_len + 1):
        for m in re.finditer(r"(?=(" + "." * n + r"))", text):
            pass
        for frag in {text[i:i + n] for i in range(len(text) - n + 1)}:
            if frag in _COMMON_REPEAT:
                continue
            idxs = [m.start() for m in re.finditer(re.escape(frag), text)]
            if len(idxs) >= 2:
                between = text[idxs[0] + n:idxs[1]]
                if not any(w in between for w in _INTENSIFIERS):
                    return frag
    return None


