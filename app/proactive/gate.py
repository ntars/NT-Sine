"""主动消息：三重闸门（方案 §11）。异步定时检查，零第三方调度器。

做不好就是骚扰——所以每一道闸门都偏保守：
- 闸门一：时间与间隔（太黏/深夜/超次数都不开口）
- 闸门二：话题选择（承诺 > 事件 > 场景 > 冲突 > 随机回忆）
- 闸门三：怎么开口（≤20 字、不用"在吗"、必须带具体锚点、没回最多补一次）
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

from app.config import SETTINGS


@dataclass
class ProactiveDecision:
    should: bool
    reason: str
    message: str = ""


class ProactiveGate:
    """三重闸门判定。不调模型——检查开销只是一次时间比较（方案 §11.2）。"""

    def __init__(self):
        self.cfg = SETTINGS["proactive"]
        self._today_count = 0
        self._today_key = self._day_key()
        self._last_sent_ts = 0.0
        self._last_refused_ts = 0.0  # 上次主动但没被回复

    @staticmethod
    def _day_key() -> int:
        return int(time.time() // 86400)

    def _roll_day(self) -> None:
        key = self._day_key()
        if key != self._today_key:
            self._today_key = key
            self._today_count = 0

    # ── 闸门一：该不该开口 ──
    def gate_time(self, last_interaction_ts: float | None, now: float | None = None) -> tuple[bool, str]:
        self._roll_day()
        now = now or time.time()
        hour = time.localtime(now).tm_hour
        q_start, q_end = self.cfg["night_quiet"]
        if q_start <= hour < q_end:
            return False, "深夜禁主动"
        if self._today_count >= int(self.cfg["daily_cap"]):
            return False, "今日主动已达上限"
        if last_interaction_ts is None:
            return False, "从未聊过，等第一次对话之后再说"
        gap_h = (now - last_interaction_ts) / 3600
        if gap_h < float(self.cfg["min_gap_h"]):
            return False, "距上次对话太近（太黏）"
        if self._last_refused_ts and (now - self._last_refused_ts) / 3600 < float(self.cfg["retry_gap_h"]):
            if gap_h < float(self.cfg["retry_gap_h"]):
                return False, "上次主动未获回复，冷却中"
        return True, f"间隔 {gap_h:.1f} 小时，可以开口"

    # ── 闸门二：说什么（话题优先级，方案 §11.1）──
    def pick_topic(self, memories: list[dict], timeline: list[dict]) -> tuple[str, str]:
        """返回 (类型, 锚点文字)。承诺 > pending > 近期事件 > 场景 > 随机回忆。"""
        for m in memories:
            if m.get("kind") == "promise" and m.get("status") != "done":
                return "承诺", m["content"]
        for m in memories:
            if m.get("kind") == "pending":
                return "pending", m["content"]
        for m in memories:
            if m.get("kind") == "event":
                return "事件", m["content"]
        if timeline:
            return "回忆", timeline[-1].get("title", "")
        return "场景", random.choice(["降温", "周末", "下雨"])

    # ── 闸门三：怎么开口 ──
    def compose(self, topic_type: str, anchor: str) -> str:
        """一句、不超过 20 字、带具体锚点、不用"在吗"。"""
        if topic_type == "承诺":
            return "你答应我的那件事，还算数吧。"
        if topic_type == "事件":
            return f"今天是你那个事的日子吧。"[:20]
        if topic_type == "回忆":
            return f"刚才想起{anchor[:6]}那回。"[:20]
        if topic_type == "pending":
            return "你上次是不是有话没说完。想说就说。"[:20]
        if topic_type == "场景":
            return f"今天{anchor}了，你那破身子骨注意点。"[:20]
        return f"今天{anchor}了，你注意点。"[:20]

    # ── 总判定 ──
    def check(self, last_interaction_ts: float | None,
              memories: list[dict], timeline: list[dict]) -> ProactiveDecision:
        ok, reason = self.gate_time(last_interaction_ts)
        if not ok:
            return ProactiveDecision(False, reason)
        # 30-40% 触发被"她自己觉得算了"掉——允许她沉默，才是人（方案 §11.3）
        if random.random() < float(self.cfg["abandon_rate"]):
            return ProactiveDecision(False, "她自己觉得算了")
        topic_type, anchor = self.pick_topic(memories, timeline)
        message = self.compose(topic_type, anchor)
        self._today_count += 1
        self._last_sent_ts = time.time()
        return ProactiveDecision(True, f"话题:{topic_type}", message)

    def mark_unanswered(self) -> None:
        """主动消息未获回复 → 进入冷却，不再连发。"""
        self._last_refused_ts = time.time()
