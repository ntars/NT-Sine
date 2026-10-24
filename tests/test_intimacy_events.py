"""亲密事件与关系跨段：守住「关系会变」这条理念在引擎侧真的成立。

背景（2026-10-24 实测到）：`note_intimacy_event()` 是亲密值唯一的写入口，
但全仓没有任何地方调用它。于是亲密值永远停在初始的 20，而五个阶段的门槛是
15 / 40 / 65 / 85 —— 「依赖」「占有」「病娇」这三个阶段（五分之三）在真实
链路上永远到不了。

实测对照坐实了这一点：连续 30 个自然日、每天 20 轮最正向的对话，好感与信任
都顶到 100，亲密值仍是 20，阶段死锁在「破冰」。也就是说「关系会变」在产品
里只兑现了前两级，后三级是死的。

这组用例守两件事：
1. 事件能被认出来，且同一类事件说了五遍只记一次（不能又变成刷屏刷出来的跨越）
2. 光有事件或光有时间都不够，三轴同时满足才跨段——顺序与门槛不能被绕过
"""

from __future__ import annotations

import pytest

from app.emotion import events
from app.emotion.engine import EmotionEngine


@pytest.fixture
def engine(tmp_path):
    return EmotionEngine(tmp_path / "emotion.json")


# ── 事件识别 ─────────────────────────────────────────────


def test_detect_confession():
    """「我喜欢你」必须认成告白——这是最典型的跨段事件。"""
    got = events.detect("其实我喜欢你很久了")
    assert got is not None
    assert got[0] == "confess"


def test_plain_praise_is_not_an_event():
    """夸她好看不等于告白。宁缺勿滥：认不准就不给，否则跨段信号会贬值。"""
    assert events.detect("你今天真好看") is None
    assert events.detect("谢谢你") is None
    assert events.detect("在吗") is None


def test_negated_confession_is_not_an_event():
    """「我不喜欢你」是拒绝，不是告白——否定语境必须挡掉。"""
    assert events.detect("我不喜欢你这样说话") is None


def test_praise_about_words_is_not_a_confession():
    """「我很喜欢你刚才说的话」在夸一句话，不是告白。

    跨段是产品里最贵的信号，这类误记会让关系无缘无故往前跳一级。
    """
    assert events.detect("我很喜欢你刚才说的话") is None
    assert events.detect("我喜欢你这样") is None


def test_rides_are_not_gifts():
    """「我送你回家」是接送，不是送礼。"""
    assert events.detect("我送你回家吧") is None
    assert events.detect("我送你去车站") is None


def test_gift_phrasings():
    """两种语序都要认：送你 / 给你买了。"""
    assert events.detect("这个送给你")[0] == "gift"
    assert events.detect("我给你买了蛋糕")[0] == "gift"


def test_detect_returns_first_event_only():
    """一句话里既有告白又有送礼，也只算一次，不叠 buff。"""
    got = events.detect("我喜欢你，这个送给你")
    assert got is not None and got[0] == "confess"


def test_all_kinds_have_description():
    """每条规则都要有说明文案——它会写进状态、供投影与日志使用。"""
    assert events.kinds(), "规则表不能为空"


# ── 冷却与数值 ───────────────────────────────────────────


def test_event_raises_intimacy(engine):
    before = engine.state.intimacy
    assert engine.note_intimacy_event("他说他喜欢你", kind="confess") is True
    assert engine.state.intimacy == before + 15.0


def test_same_kind_event_counts_once_within_cooldown(engine):
    """连说五轮「我喜欢你」，亲密值只能涨一次。

    没有这层冷却，跨段就退回成「刷轮数」——说五遍顶半个月，正是设计要防的事。
    """
    before = engine.state.intimacy
    results = [engine.note_intimacy_event("他说他喜欢你", kind="confess") for _ in range(5)]
    assert results == [True, False, False, False, False], results
    assert engine.state.intimacy == before + 15.0


def test_different_kinds_each_count(engine):
    """不同类事件各自记账：告白、拉钩、送礼是三件不同的事。"""
    before = engine.state.intimacy
    assert engine.note_intimacy_event("他说他喜欢你", kind="confess") is True
    assert engine.note_intimacy_event("你们拉了钩", kind="promise") is True
    assert engine.note_intimacy_event("他送了东西给你", kind="gift") is True
    assert engine.state.intimacy == before + 45.0


def test_event_survives_restart(engine, tmp_path):
    """亲密值与冷却记录都要落盘，重启后不能归零重来。"""
    engine.note_intimacy_event("他说他喜欢你", kind="confess")
    intimacy = engine.state.intimacy

    again = EmotionEngine(tmp_path / "emotion.json")
    assert again.state.intimacy == intimacy
    # 重启后立刻再说同一类事件，仍处于冷却期，不重复记账
    assert again.note_intimacy_event("他说他喜欢你", kind="confess") is False


# ── 跨段判定 ─────────────────────────────────────────────


def test_stage_reaches_attached_with_events(engine):
    """14 天 + 两次不同事件 → 依赖。这是原先永远到不了的那一级。"""
    engine.state.active_days = 14
    engine.note_intimacy_event("他说他喜欢你", kind="confess")
    engine.note_intimacy_event("你们拉了钩", kind="promise")
    engine.state.trust = 60.0  # 信任由日常互动累积，这里直接摆到门槛之上
    assert engine.relation_stage["key"] == "attached", engine.state


def test_stage_does_not_skip_on_events_alone(engine):
    """亲密值刷满但只聊过两天，只能停在破冰——三轴必须同时满足。"""
    engine.state.active_days = 2
    for kind in ("confess", "promise", "stay", "together", "gift", "reconcile"):
        engine.note_intimacy_event("事件", kind=kind)
    assert engine.state.intimacy >= 85
    assert engine.relation_stage["key"] == "cold", "天数不够却跳级了"


def test_positive_chat_alone_cannot_reach_attached(engine):
    """回归用例：光靠聊得开心到不了「依赖」。

    用改动前会失败的路径断言——把 events 的调用摘掉，这条用例就必须红。
    """
    engine.state.active_days = 30
    engine.state.trust = 90.0
    engine.state.affection = 90.0
    # 不喂任何亲密事件，只喂普通正向话
    for _ in range(20):
        engine.update("谢谢你，你真好", "……嗯。")
    assert engine.state.intimacy == 20.0
    assert engine.relation_stage["key"] == "thawing", engine.state


def test_full_curve_is_reachable(tmp_path):
    """五个阶段在参数上都够得着——门槛不能有一个是高不可攀的死数。

    逐个阶段摆出「刚好达标」的状态，确认判定给出对应阶段；这是对
    「后三级永远到不了」的结构性检查，不依赖任何具体对话内容。
    """
    e = EmotionEngine(tmp_path / "e.json")
    cases = [
        ("cold", 0, 0, 0),
        ("thawing", 3, 15, 30),
        ("attached", 14, 40, 50),
        ("possessive", 30, 65, 65),
        ("yandere", 60, 85, 80),
    ]
    for key, days, intimacy, trust in cases:
        e.state.active_days = days
        e.state.intimacy = intimacy
        e.state.trust = trust
        assert e.relation_stage["key"] == key, (key, e.state)

    # 而且这些门槛是靠事件累加能走到的：初始 20 + 5 次不同事件 = 95 ≥ 85
    e2 = EmotionEngine(tmp_path / "e2.json")
    for kind in ("confess", "promise", "stay", "together", "gift"):
        assert e2.note_intimacy_event("事件", kind=kind) is True
    assert e2.state.intimacy >= 85


def test_colloquial_stay_phrasings_are_recognised():
    """「一直在一起 / 永远陪着你」这类最典型的长期承诺不能漏。

    回归用例（2026-10-24）：第一版 stay 规则只写了「留在/陪着」的固定语序，
    实测把「我们会一直在一起」「我永远陪着你」两句最像告白的话全漏掉——
    漏识别等于用户说了最重的一句、关系却纹丝不动。
    """
    for text in ("我们会一直在一起", "我永远陪着你", "我会一直在你身边",
                 "我陪着你", "我留在你身边", "我不会走", "我不走"):
        got = events.detect(text)
        assert got is not None and got[0] == "stay", (text, got)


def test_mundane_together_phrasings_are_not_events():
    """普通叙述不能被认成亲密事件——「我在你身边工作」不是承诺。"""
    assert events.detect("我在你身边工作") is None
    assert events.detect("我跟她在一起") is None
    assert events.detect("大家一起吃饭") is None


def test_zanmen_counts_as_together():
    """「咱们」是口语里最常见的自称双数，第一版只认「我们/咱俩」。"""
    got = events.detect("咱们一起去看海吧")
    assert got is not None and got[0] == "together", got


def test_confession_with_forever_modifier():
    """「我永远爱你」同样是告白，修饰语不能把规则挡在外面。"""
    got = events.detect("我永远爱你")
    assert got is not None and got[0] == "confess", got
