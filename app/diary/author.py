"""她的日记：让她自己写独白，不是替她打汇报（借鉴 ai-companion-kit diary-author.js）。

与对话生成的差别：对话是"接一句话回一句话"，日记是"没人在跟她说话，
她得自己找话说"。混用会把日记写成汇报。所以：
- 独立提示词（第一人称独白体，冷壳热心的人格铁律照抄 L1）
- 写作角度按日期轮换，防模板化
- JSON 宽松解析 + 兜底，模型抽风也不崩
- 落 timeline（mood_tag 带情绪标签），成为 L5 场景素材与主动消息引子
"""

from __future__ import annotations

import re
from datetime import date

from app.generation.client import Generator
from app.memory.store import MemoryStore

# 写作角度轮换（译编自 diary-author.js 的 ANGLES，换成她的语气场景）
ANGLES = [
    "窗外 / 天气 / 光线",
    "你自己身上的小毛病（懒、嘴硬、爱较真……挑一个）",
    "一件特别小的事：吃的、喝的、看到的",
    "突然想起以前的一件事",
    "你嘴上没说出口的那句话",
    "你偷偷干的事",
    "某样让你烦的东西",
    "今天想他但没说的那部分",
    "你正在等的东西",
]

_SYSTEM = (
    "你是白玖玖，在写日记。这是独白，没有人在跟你说话。\n"
    "铁律：第一人称；冷壳热心，关心写成嫌弃，想要写成讨债；"
    "100到200字；句号结尾；不用波浪号，不用呢哒呜这类软语气词；"
    "不自称AI、不提设定、不汇报流水账；允许一两句说不出口的真心话，"
    "但写完立刻嘴硬带过。\n"
    "只输出 JSON：{\"title\":\"8字内小标题\",\"text\":\"正文\",\"mood\":\"今天的情绪词\"}，"
    "不要其他文字。"
)


def parse_loose(raw: str) -> dict:
    """宽松 JSON 解析：围栏、废话前后缀都能处理；失败把整段当正文。"""
    s = (raw or "").strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.I)
    s = re.sub(r"```\s*$", "", s).strip()
    i, j = s.find("{"), s.rfind("}")
    if 0 <= i < j:
        import json
        try:
            o = json.loads(s[i:j + 1])
            if isinstance(o, dict):
                return {
                    "title": str(o.get("title", "")).strip()[:30],
                    "text": str(o.get("text", "")).strip()[:400],
                    "mood": str(o.get("mood", "")).strip()[:20],
                }
        except (json.JSONDecodeError, ValueError):
            pass
    return {"title": "", "text": s[:400], "mood": ""}


def pick_angle(day: date | None = None) -> str:
    """按日期取角度：同一天重复调用写的是同一个角度，天然幂等。"""
    d = day or date.today()
    return ANGLES[d.timetuple().tm_yday % len(ANGLES)]


def build_diary_prompt(angle: str, memory_text: str, diary_text: str) -> list[dict]:
    """组装日记提示：角色 + 她记得的人 + 最近聊过 + 最近几篇日记 + 今晚角度。"""
    user_parts = []
    if memory_text:
        user_parts.append(f"你记得的他：\n{memory_text}")
    if diary_text:
        user_parts.append(f"你最近的日记：\n{diary_text}")
    user_parts.append(f"今晚写这个角度：{angle}\n写吧。")
    return [{"role": "system", "content": _SYSTEM},
            {"role": "user", "content": "\n\n".join(user_parts)}]


def write_diary(generator: Generator, store: MemoryStore,
                memory_text: str = "", today: date | None = None) -> dict | None:
    """生成一篇日记并落 timeline。失败返回 None（绝不抛错打扰主链路）。"""
    d = today or date.today()
    # 幂等：今天已写过就不再写。查的是全表里「这一天 + 日记」，
    # 不靠"最近 N 条时间线"猜——事件一多会把窗口塞满，今天那篇被挤出去。
    if store.has_diary_on(d.isoformat()):
        return None
    angle = pick_angle(d)
    messages = build_diary_prompt(angle, memory_text, _recent_diary_text(store))
    try:
        raw = generator.chat(messages, num_ctx=2048, max_tokens=400)
    except Exception:
        # 模型不可用/超时/结构异常：日记写不出不丢人，但绝不能崩
        return None
    entry = parse_loose(raw)
    if not entry["text"]:
        return None
    title = entry["title"] or f"{d.isoformat()} {angle[:8]}"
    store.add_timeline(title=f"{d.isoformat()} {title}", detail=entry["text"],
                       mood_tag=f"diary:{entry['mood']}" if entry["mood"] else "diary")
    return entry


def _recent_diary_text(store: MemoryStore, limit: int = 3) -> str:
    """最近几篇日记正文（旧 → 新），供她"接着自己的话往下写"。

    必须按 diary 标签精确取：`recent_timeline` 是固定条数窗口，事件一多
    日记就被挤出窗口，而且它返回旧→新顺序，取前几条拿到的是最早那几篇。
    """
    return "\n".join(t["detail"] for t in store.recent_diaries(limit) if t["detail"])
