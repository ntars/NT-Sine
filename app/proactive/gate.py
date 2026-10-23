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
        self._last_refused_ts = 0.0  # 上次主动但没被回复的时刻
        self._pending = False        # 上一条主动还没等到回应

    @staticmethod
    def _day_key(ts: float | None = None) -> int:
        """本地日历日，形如 2026031（年 ×1000 + 当年第几天）。

        不能拿 `时间戳 // 86400` 当日界：那是 UTC 的日界，东八区要等到
        早上 8 点才翻页，于是"每日最多主动 3 次"的重置点落在早八点——
        凌晨那几次会被算进前一天，配额和 user 眼里的"今天"对不上。
        """
        t = time.localtime(time.time() if ts is None else ts)
        return t.tm_year * 1000 + t.tm_yday

    def _roll_day(self, now: float | None = None) -> None:
        """跨过本地零点就把"今天发了几次"清零。now 由调用方传，便于测试固定日期。"""
        key = self._day_key(now)
        if key != self._today_key:
            self._today_key = key
            self._today_count = 0

    # ── 闸门一：该不该开口 ──
    def gate_time(self, last_interaction_ts: float | None, now: float | None = None) -> tuple[bool, str]:
        now = now or time.time()
        self._roll_day(now)
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
            # 事件类必须把锚点摆出来。原来这里是一句对所有事件都一样的固定话，
            # 等于闸门三自己定的"必须带具体锚点"没落地——她"记得"这件事就废了。
            if not anchor:
                return "你上次提的那件事，怎么样了。"
            return f"{anchor[:12]}，别忘了。"[:20]
        if topic_type == "回忆":
            return f"刚才想起{anchor[:6]}那回。"[:20]
        if topic_type == "pending":
            return "你上次是不是有话没说完。想说就说。"[:20]
        if topic_type == "场景":
            return f"今天{anchor}了，你那破身子骨注意点。"[:20]
        return f"今天{anchor}了，你注意点。"[:20]

    # ── 总判定 ──
    def check(self, last_interaction_ts: float | None,
              memories: list[dict], timeline: list[dict],
              now: float | None = None, commit: bool = True) -> ProactiveDecision:
        """now 只在测试里显式传入，线上沿用当前时刻。

        不能省掉这个参数：闸门一里"深夜禁主动"排在最前面，若测试拿真实时间戳跑，
        结论就随"跑测试时是几点"变化——下午跑通过、凌晨跑失败。机器时钟不该
        决定测试结果，所以时刻由调用方给进来，默认值保持线上原行为。

        commit=False 是"只看不发"：调试台要能查「现在会不会开口」，但不能真把
        配额用掉。check 内部会累加今日次数、记发出时刻、挂上待回应标记，甚至
        因为上次没回而刷一次冷却——调试一次就等于真发了一次，还顺手让接下来
        六小时不再开口。所以只读模式改为算完把四项状态原样放回。
        """
        snapshot = (self._today_count, self._last_sent_ts,
                    self._last_refused_ts, self._pending)
        try:
            return self._check(last_interaction_ts, memories, timeline, now)
        finally:
            if not commit:
                (self._today_count, self._last_sent_ts,
                 self._last_refused_ts, self._pending) = snapshot

    def _check(self, last_interaction_ts: float | None,
               memories: list[dict], timeline: list[dict],
               now: float | None) -> ProactiveDecision:
        now = now or time.time()
        # 上一条主动发出后对方一直没回 → 先记一次"未获回复"，再走闸门。
        # 标记只记一次：每次检查都刷新冷却时间，冷却就永远不会过期。
        if self._pending and (last_interaction_ts is None or last_interaction_ts < self._last_sent_ts):
            self.mark_unanswered(now)
        elif self._pending:
            self._pending = False   # 已经回了，解除待回应状态

        ok, reason = self.gate_time(last_interaction_ts, now)
        if not ok:
            return ProactiveDecision(False, reason)
        # 30-40% 触发被"她自己觉得算了"掉——允许她沉默，才是人（方案 §11.3）
        if random.random() < float(self.cfg["abandon_rate"]):
            return ProactiveDecision(False, "她自己觉得算了")
        topic_type, anchor = self.pick_topic(memories, timeline)
        message = self.compose(topic_type, anchor)
        self._today_count += 1
        self._last_sent_ts = now
        self._pending = True          # 等她这次有没有回
        return ProactiveDecision(True, f"话题:{topic_type}", message)

    def mark_unanswered(self, now: float | None = None) -> None:
        """主动消息未获回复 → 进入冷却，不再连发。"""
        self._last_refused_ts = now or time.time()
        self._pending = False
