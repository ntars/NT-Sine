"""情感引擎：冲突的进入与解除。

守着一个把"关系会变"直接掐死的死结（2026-10-24 实测到）：冲突期间 mood 被
锁在 -0.15 以下，而解除冲突的条件正是"mood 回正"，两边对夹，冲突一旦发生
就再也解不开——此后每一轮上下文里都塞着"你们之间还横着一件事"，她永远带刺，
好感与信任只跌不涨，关系阶段再也不可能推进。

另有一条同源的误判：求和句里往往带着核按钮词（"我们别分手"），按子串直判
会把一句求和当成新的决裂。
"""

from __future__ import annotations

import pytest

from app.emotion.engine import EmotionEngine


@pytest.fixture
def engine(tmp_path):
    return EmotionEngine(tmp_path / "emotion.json")


def test_conflict_blocks_mood(engine):
    """非道歉轮：冲突期间 mood 必须压在负区，不能一句话就翻篇。"""
    engine.update("我们分手吧")
    assert engine.state.conflict, "核按钮词没进冲突态"

    engine.update("随便你怎么想")
    assert engine.state.mood <= -0.15, engine.state.mood
    assert engine.state.conflict, "没道歉就把冲突消掉了"


def test_conflict_can_be_resolved_by_apology(engine):
    """道歉之后冲突必须能解除——这是原先解不开的那一环。"""
    engine.update("我们分手吧，你别再找我了")
    assert engine.state.conflict

    engine.update("对不起，我错了，别生气好吗")

    assert engine.state.conflict == "", "道歉了冲突还挂着，关系永远回不来"
    assert engine.state.mood > -0.15, engine.state.mood


def test_conflict_resolves_within_a_couple_of_apologies(engine):
    """连着两轮示好一定解得开，不能要求用户道歉十次。"""
    engine.update("我们分手吧")
    engine.update("对不起")
    engine.update("真的对不起，我错了")
    assert engine.state.conflict == "", engine.state.conflict


def test_reconciliation_sentence_is_not_a_new_conflict(engine):
    """「我们别分手」是求和，不是新的决裂。"""
    assert EmotionEngine.signals("我们别分手好吗")[1] is False
    assert EmotionEngine.signals("分手吧")[1] is True
    assert EmotionEngine.signals("我不想理你了")[1] is True, "「不想理你」本身就是决裂句"
    assert EmotionEngine.signals("别生气，我不是那个意思")[1] is False


def test_conflict_cleared_leaves_no_trace_in_projection(engine):
    """和解后 L3 投影里不能再出现"横着一件事"。"""
    engine.update("我们分手吧")
    assert "横着一件事" in engine.project()

    engine.update("对不起，我错了")
    assert "横着一件事" not in engine.project(), engine.project()
