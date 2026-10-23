"""用户侧记忆检索：让她"记得主人说过什么"（方案 §6.4 预留位的落地）。

两路注入，全部走系统提示区，不进对话记录：
1. 滚动画像（L2 动态版）：SQLite profile 表聚合，每次对话必带
2. 关联召回（L4 记忆事件层）：用户当前消息 × 存量记忆的关键词暴力检索，
   规模 < 万条时毫秒级够用；过万再换向量检索（升级位，不改接口）

诚实声明：这是关键词召回，不是语义理解——"想吃花生"能召回"他对花生过敏"，
但"今天过桥路过"召不回"我提过那座桥"。这是已知边界，向量检索是升级方案。
"""

from __future__ import annotations

import re

from app.memory.store import MemoryStore

_CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")
_ASCII_WORD = re.compile(r"[a-zA-Z]{2,}")

# 停用二元组：高频功能词（2-gram 检索的噪声源）
_STOP_PHRASES = {
    "今天", "明天", "昨天", "现在", "时候", "地方", "事情", "东西", "大家",
    "怎么", "什么", "可以", "没有", "不是", "这个", "那个", "还是", "就是",
    "但是", "然后", "因为", "所以", "一个", "我们", "你们", "他们",
}


def _bigrams(text: str) -> set[str]:
    """提取 2-gram 检索集：中文滑窗二元组 + 英文整词。

    记忆内容通常是无标点的完整短句，整词级匹配失效，
    2-gram 是中文无分词场景下最稳的召回口径。停用二元组剔除。
    """
    grams: set[str] = set()
    for run in _CJK_RUN.findall(text):
        if len(run) == 1:
            continue
        for i in range(len(run) - 1):
            grams.add(run[i:i + 2])
    grams.update(w.lower() for w in _ASCII_WORD.findall(text))
    return {g for g in grams if g not in _STOP_PHRASES}


def retrieve_relevant(user_text: str, store: MemoryStore, limit: int = 4) -> list[dict]:
    """暴力 2-gram 召回：重叠数 ≥1 即候选，按 (重叠数, 提及次数) 降序取前 limit 条。"""
    query = _bigrams(user_text)
    if not query:
        return []
    scored: list[tuple[int, dict]] = []
    for kind in ("fact", "preference", "event", "promise", "pending", "trait", "topic"):
        for m in store.memories_by_kind(kind, limit=60):
            hit = len(query & _bigrams(m.content))
            if hit >= 1:
                scored.append((hit, {
                    "kind": m.kind,
                    "content": m.content,
                    "mentions": m.mentions,
                }))
    scored.sort(key=lambda x: (-x[0], -x[1]["mentions"]))
    return [s for _, s in scored[:limit]]


def user_profile_block(store: MemoryStore) -> str:
    """L2 动态画像：把 SQLite 聚合的知识翻译成"她记得的事"。≤180 字。"""
    profile = store.all_profile()
    parts: list[str] = []
    for key, label in (("facts", "关于他的事实"), ("preferences", "他的偏好")):
        value = profile.get(key, "")
        if value:
            parts.append(f"{label}：{value}")
    timeline = store.recent_timeline(3)
    for t in timeline:
        parts.append(f"你们一起的事：{t['title']}")

    if not parts:
        return ("你对他目前几乎一无所知。没记下的事不许编造，也不许明知故问；"
                "知道的事绝不装作不知道。")
    return "你认识他（按共同生活慢慢记下的）：" + "；".join(parts)


def recall_block(recalled: list[dict]) -> str:
    """召回块：把相关记忆列给她，让她"想起"而不是"被喂答案"。"""
    if not recalled:
        return ""
    lines = []
    kind_label = {"fact": "事实", "preference": "偏好", "event": "共同经历",
                  "promise": "承诺", "pending": "他没说出口的事",
                  "trait": "他的习惯/性格", "topic": "最近聊过"}
    for r in recalled:
        lines.append(f"（{kind_label.get(r['kind'], r['kind'])}）{r['content']}")
    return "关于他，你想起了这些：\n" + "\n".join(lines)
