"""记忆抽取：每 3-5 轮用本地模型轻量抽取一次（方案 §6.3）。

四类目标：事实 / 偏好 / 事件 / 承诺。只抽结构化短句，绝不生成总结段落
（总结段落是幻觉的来源）。抽取失败静默跳过——主链路优先。
"""

from __future__ import annotations

import json

from app.config import SETTINGS
from app.generation.client import Generator
from app.memory.store import MemoryStore

_SYSTEM = (
    "从对话里抽取值得记住的信息，只输出 JSON 数组，每项形如："
    '{"kind":"fact|preference|event|promise|pending|trait|topic","content":"30字内的短句","detail":{}}。'
    "pending = 他犹豫着没说出口的事；trait = 你观察到的他的性格/习惯（如\"常熬夜\"）；"
    "topic = 最近聊过的话题（滚动）。"
    # 人称约定：这些短句会被主动消息直接引用（"你上次说的X，别忘了"），
    # 用第三人称"他"引出来就穿帮了，所以统一用"你"。
    "content 里指代对方一律用\"你\"，不要用\"他\"，例如\"你周五要去面试\"。"
    "没有值得记的就输出 []。不要总结、不要感想、不要任何其他文字。"
)

# 每类记忆的容量上限（借鉴 ai-companion-kit memory.js：FIFO 淘汰，防账本无限膨胀）
KIND_CAPS = {"fact": 60, "preference": 40, "trait": 40, "topic": 20, "pending": 12,
             "event": 40, "promise": 30}


class MemoryExtractor:
    def __init__(self, store: MemoryStore, generator: Generator | None = None):
        self.store = store
        self.generator = generator
        self.cfg = SETTINGS["memory"]
        self._turn_counter = 0

    def maybe_extract(self, recent_messages: list[dict]) -> int:
        """每 N 轮抽一次；返回本次写入的记忆条数。Ollama 不可用时返回 0。"""
        self._turn_counter += 1
        if self._turn_counter % int(self.cfg["extract_every_n_turns"]) != 0:
            return 0
        if self.generator is None:
            return 0
        rounds = recent_messages[-int(self.cfg["extract_recent_rounds"]):]
        dialogue = "\n".join(f"{m['role']}: {m['content']}" for m in rounds)
        data = self.generator.chat_json(
            [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": dialogue}],
            num_ctx=2048,
        )
        if not isinstance(data, list):
            return 0
        count = 0
        for item in data:
            if not isinstance(item, dict):
                continue
            kind = item.get("kind", "")
            content = str(item.get("content", "")).strip()
            if kind not in KIND_CAPS or not content:
                continue
            detail = item.get("detail") if isinstance(item.get("detail"), dict) else {}
            self.store.add_memory(kind, content, source=dialogue[-200:], detail=detail)
            if kind == "event":
                self.store.add_timeline(content, mood_tag="event")
            count += 1
        self.prune()
        return count

    def prune(self) -> None:
        """按 KIND_CAPS 淘汰最旧的低价值记忆（FIFO，mention 多的不动）。"""
        for kind, cap in KIND_CAPS.items():
            kept = self.store.memories_by_kind(kind, limit=10_000)
            if len(kept) > cap:
                # memories_by_kind 按 last_ts 降序，砍掉尾部最旧的
                for m in kept[cap:]:
                    self.store.update_memory_status(m.id, "dropped")


def build_profile(store: MemoryStore) -> dict[str, str]:
    """用户画像滚动更新：SQLite 聚合即可，不调模型（方案 §6.4 改动一）。"""
    facts = [m.content for m in store.memories_by_kind("fact", limit=10)]
    prefs = [m.content for m in store.memories_by_kind("preference", limit=10)]
    if facts:
        store.set_profile("facts", "；".join(facts))
    if prefs:
        store.set_profile("preferences", "；".join(prefs))
    return store.all_profile()
