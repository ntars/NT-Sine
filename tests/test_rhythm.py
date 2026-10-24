"""节奏层：分条的完整性、条数上限与打字时长。

守着一个用户能直接看见的坑（2026-10-24 实测）：分条超过上限时用切片硬截，
等于把她说的话吞掉一截——108 字的回复只吐出 84 字，播完前四条就没了，
听感上是"话说一半"。条数上限是为了不像客服发通知，不该以丢内容为代价。
"""

from __future__ import annotations

from app.config import SETTINGS
from app.output.rhythm import _split_segments, plan_delivery, plan_delivery_now

MAX_SEG = int(SETTINGS["rhythm"]["max_segments"])
SEG_MIN = int(SETTINGS["rhythm"]["segment_min"])


def test_long_reply_keeps_every_character():
    """超长回复分条后，拼回来的字数必须与原文一致。"""
    text = "水在桌上，你记得喝。" * 6 + "今天风大，出门多穿一件。" * 4
    segs = _split_segments(text, 0.0, MAX_SEG)

    assert len(segs) <= MAX_SEG, "条数上限要守住"
    assert "".join(segs) == text, f"分条丢字：原文 {len(text)}，拼回 {sum(len(s) for s in segs)}"


def test_segment_count_capped():
    """再长也不超过上限——多出来的并进最后一条，不是多发几条。"""
    segs = _split_segments("他说了一句话。" * 30, 0.0, MAX_SEG)
    assert len(segs) == MAX_SEG


def test_short_reply_stays_one_segment():
    """本来就短的话不该被拆成好几条。"""
    for text in ("水在桌上。", "……嗯。"):
        segs = _split_segments(text, 0.0, MAX_SEG)
        assert segs == [text], segs


def test_short_segment_merges_into_neighbour():
    """低于 segment_min 的碎片要并进相邻条，不能单独发出去像打字机卡壳。"""
    segs = _split_segments("嗯。水在桌上，你记得喝，凉的别喝。", 0.0, MAX_SEG)
    assert len(segs) == 1, segs
    assert len(segs[0]) >= SEG_MIN


def test_typing_time_grows_with_length():
    """同一个随机种子下，长句的打字时长必须比短句长。"""
    import random

    random.seed(7)
    short = plan_delivery("水在桌上。", mood=0.0, hour=14)[0]
    random.seed(7)
    long = plan_delivery("你今天回来得晚，饭在锅里，自己热一下再吃。", mood=0.0, hour=14)[0]

    assert long.typing_s > short.typing_s, (short.typing_s, long.typing_s)
    assert short.typing_s >= 0.5


def test_last_segment_has_no_pause():
    """最后一条说完就结束，不该再挂一段停顿。"""
    segs = plan_delivery("水在桌上。今天风大。出门多穿件。", mood=0.0, hour=14)
    assert segs[-1].pause_s == 0.0
    assert all(s.pause_s >= 0 for s in segs)


def test_night_is_slower():
    """深夜整体变慢：同一段话，夜里的打字时长要明显长于白天。

    口径说明（2026-10-24 修正）：语速在 4-7 字/秒之间随机，单次波动幅度
    1.75 倍，大于夜间倍率 1.6 倍——所以"抽一次比大小"在数学上就不可能稳定，
    原测试用 `random.seed(11)` 想固定随机，但 `plan_delivery` 内部是独立的
    `random.Random()` 实例，全局种子对它无效，这条测试一直在碰运气（实测
    曾出现 day=2.4 / night=2.87 失败）。改为注入固定随机源 + 取 200 次
    采样的中位数：产品上用户感知的也是整体快慢，不是某一条的毫秒数。
    """
    import random

    text = "水在桌上，你记得喝。"
    day = sorted(plan_delivery(text, mood=0.0, hour=14, rnd=random.Random(i))[0].typing_s
                 for i in range(200))
    night = sorted(plan_delivery(text, mood=0.0, hour=3, rnd=random.Random(i))[0].typing_s
                   for i in range(200))
    assert night[100] > day[100] * 1.3, (day[100], night[100])


def test_rhythm_random_source_is_injectable():
    """同一个种子必须给出同一份计划——这是上面那条测试能被复现的前提。"""
    import random

    a = plan_delivery("水在桌上。今天风大。出门多穿件。", mood=0.0, hour=14, rnd=random.Random(7))
    b = plan_delivery("水在桌上。今天风大。出门多穿件。", mood=0.0, hour=14, rnd=random.Random(7))
    assert [(s.text, s.typing_s, s.pause_s) for s in a] == [(s.text, s.typing_s, s.pause_s) for s in b]


def test_plan_delivery_now_reads_local_hour():
    """生产入口自己取小时，避免调用方忘了传导致深夜变速失效。"""
    segs = plan_delivery_now("水在桌上。", mood=0.0)
    assert segs and segs[0].text
