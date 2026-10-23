"""拟人化节奏层（方案 §6.5）：分条、停顿、打字时长模拟。

这一层对"像人"的贡献大于任何采样参数微调——实现成本极低（纯本地规则）。
服务端不 sleep，只输出"节奏计划"，由前端按 delay 播放。
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

from app.config import SETTINGS


@dataclass
class Segment:
    text: str
    typing_s: float      # 这一条模拟的打字时长（"正在输入…"持续时间）
    pause_s: float       # 发送后到下一条之间的停顿


def _split_segments(text: str, mood: float, max_segments: int) -> list[str]:
    """把一段回复拆成 8-25 字的分条。情绪低落 → 条少而长；高涨 → 条多而短。"""
    cfg = SETTINGS["rhythm"]
    seg_min, seg_max = int(cfg["segment_min"]), int(cfg["segment_max"])
    text = text.strip()
    if not text:
        return []

    # 按句子边界拆，再按情绪合并/拆分
    sentences = [s for s in re_split_sentences(text) if s]
    segments: list[str] = []
    buf = ""
    for s in sentences:
        if buf and len(buf) + len(s) > seg_max:
            segments.append(buf)
            buf = s
        else:
            buf += s
    if buf:
        segments.append(buf)

    # 分条下限：短条单独发出去像打字机卡壳，先攒着并在下一条里；末尾剩下的并回上一条。
    # 例外：整段话本身就短于 seg_min 时不做任何合并（合并也没得可并）
    if len(text) >= seg_min:
        merged: list[str] = []
        carry = ""
        for s in segments:
            s = carry + s
            carry = ""
            if len(s) < seg_min:
                carry = s
                continue
            merged.append(s)
        if carry:
            if merged:
                merged[-1] += carry
            else:
                merged.append(carry)
        segments = merged

    # 情绪调制：低落合并、高涨拆分
    if mood < -0.3 and len(segments) > 1:
        merged = [segments[0]]
        for s in segments[1:]:
            if len(merged[-1]) + len(s) <= seg_max * 1.3:
                merged[-1] += s
            else:
                merged.append(s)
        segments = merged

    # 超长单条按 seg_max 硬拆；拆完清理首尾空白
    final: list[str] = []
    for s in segments:
        s = s.strip()
        while len(s) > seg_max * 1.5:
            final.append(s[:seg_max].strip())
            s = s[seg_max:]
        if s.strip():
            final.append(s.strip())
    return final[:max_segments]


def re_split_sentences(text: str) -> list[str]:
    """按中文句末标点切句，标点跟在句前；括号（动作/神态）内部不切。"""
    out: list[str] = []
    buf = ""
    depth = 0
    for ch in text:
        buf += ch
        if ch in "（(":
            depth += 1
        elif ch in "）)" and depth > 0:
            depth -= 1
        elif ch in "。！？!?…；;" and depth == 0:
            out.append(buf)
            buf = ""
    if buf:
        out.append(buf)
    return out


def plan_delivery_now(text: str, mood: float = 0.0) -> list[Segment]:
    """生产入口：自动取本机当前小时，深夜才会真的变慢。

    单独包一层的原因：底层 plan_delivery 把 hour 做成参数是为了可测，
    但真实调用方（服务端）如果忘了穿小时，深夜变速等于没做——这里兜住。
    """
    return plan_delivery(text, mood=mood, hour=time.localtime().tm_hour)


def plan_delivery(text: str, mood: float = 0.0, hour: int | None = None) -> list[Segment]:
    """生成节奏计划：每条的模拟打字时长 + 条间停顿。"""
    cfg = SETTINGS["rhythm"]
    rnd = random.Random()  # 每次调用真随机，避免节奏被"看穿"
    cps_min, cps_max = float(cfg["chars_per_sec_min"]), float(cfg["chars_per_sec_max"])
    p_min, p_max = float(cfg["pause_min_s"]), float(cfg["pause_max_s"])

    night_start, night_end = cfg["night_hours"]
    is_night = hour is not None and (night_start <= hour < night_end)
    slow = float(cfg["night_slow_factor"]) if is_night else 1.0
    if mood < -0.3:
        slow *= 1.4  # 情绪低落：不想说话，更慢
    elif mood > 0.4:
        slow *= 0.85  # 情绪高涨：话多手快

    # 长停顿参数：偶尔停 3-9 秒（走神/被叫住/话到嘴边又咽回去）
    long_chance = float(cfg.get("long_pause_chance", 0.0))
    long_min, long_max = cfg.get("long_pause_range_s", [3.0, 9.0])

    segments = _split_segments(text, mood, int(cfg["max_segments"]))
    typing_before = cfg["typing_before_first_s"]
    result: list[Segment] = []
    for i, seg in enumerate(segments):
        cps = rnd.uniform(cps_min, cps_max) / slow
        typing_s = max(0.5, len(seg) / cps)
        if i == 0:
            typing_s = max(typing_s, rnd.uniform(*typing_before))
        # 停顿：0.6-2.5s 随机；低落拉长、高涨缩短；偶尔 3-9 秒长停顿（走神/咽回去）
        span = (p_max - p_min) * (1.3 if mood < -0.3 else 0.8 if mood > 0.4 else 1.0)
        pause_s = min(4.0, rnd.uniform(p_min, p_min + span))
        if i < len(segments) - 1 and rnd.random() < long_chance:
            pause_s = rnd.uniform(long_min, long_max)
        result.append(Segment(text=seg, typing_s=round(typing_s, 2),
                              pause_s=round(pause_s if i < len(segments) - 1 else 0.0, 2)))
    return result
