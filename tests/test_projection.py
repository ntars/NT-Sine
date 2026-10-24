"""L3 状态投影：预算不够时整句取舍，绝不拦腰切断。

守的是一个"看着没事、实际在最有戏的一刻崩掉"的缺陷（2026-10-24 实测到）。

原先 `project()` 末尾是 `return text[:100]`，按字符硬切。温和状态下拼出来
不到 100 字，切不到东西，所以一切正常；但只要同时出现「有冲突 + 高好感 +
低信任 + 疲惫」——恰好是产品里情绪最浓的那一刻——拼出来 150 字，被切在
第 100 个字：

    ……你不太敢信他的话，会忍不住想试探。你

结尾停在孤零零的一个「你」，模型会把它当作下一句的开头；而「你困了，
打字变慢，句子变短」整句永远进不了提示。半句话比没有这句话更糟。

这组用例守四件事：
1. 任何状态下输出都以句号收尾，长度不超过预算（没有半截句）
2. 低负载时疲惫描述照常进得去（不因修截断而误伤）
3. 「不安」与「不太敢信」是同一件事的两种说法，同时成立时只留一句
4. 用户原话式的超长冲突被压缩，不把整份状态挤满
"""

from __future__ import annotations

import pytest

from app.emotion.engine import _PROJECTION_BUDGET, EmotionEngine


@pytest.fixture
def engine(tmp_path):
    return EmotionEngine(tmp_path / "emotion.json")


def _extreme(engine):
    """产品里情绪最浓的一刻：有裂痕、极依赖、不敢信、已经熬到没力气。"""
    s = engine.state
    s.conflict = "你昨天答应陪他吃饭，结果加班到十点"
    s.mood = -0.5
    s.affection = 90.0
    s.trust = 20.0
    s.energy = 8.0
    return engine


def test_extreme_state_is_not_cut_mid_sentence(engine):
    """回归用例：极端状态曾经在这里被切成「……想试探。你」。"""
    text = _extreme(engine).project()
    assert text.endswith("。"), f"结尾不是完整句子：{text!r}"
    assert not text.endswith("你"), text
    assert len(text) <= _PROJECTION_BUDGET, len(text)


def test_projection_always_ends_with_full_sentence(engine):
    """扫一遍各档组合，任何一格都不许出现半截句。"""
    for mood in (-0.9, -0.2, 0.0, 0.6):
        for affection in (10.0, 40.0, 60.0, 90.0):
            for trust in (10.0, 45.0, 90.0):
                for energy in (5.0, 40.0, 80.0):
                    s = engine.state
                    s.conflict = "我们分手吧" if mood < -0.5 else ""
                    s.mood, s.affection, s.trust, s.energy = mood, affection, trust, energy
                    text = engine.project()
                    assert len(text) <= _PROJECTION_BUDGET, (text, len(text))
                    assert text == "" or text.endswith("。"), (text, s.to_dict())


def test_tiredness_still_reaches_projection_when_there_is_room(engine):
    """低负载状态必须保留疲惫描述——修截断不能顺手把它一起丢掉。

    精力直接决定打字快慢，这句是模型判断"该短该慢"的唯一文字依据。
    """
    s = engine.state
    s.mood = 0.0
    s.affection = 20.0
    s.trust = 80.0
    s.energy = 10.0
    text = engine.project()
    assert "困了" in text, text


def test_extreme_state_keeps_tiredness_and_drops_plain_distrust(engine):
    """预算不够时先丢「不太敢信他」，保住「她困了」。

    这是一条明确的取舍：精力在别处没有出口（打字快慢由时段决定，不看精力），
    这句被挤掉，「她会累」就等于没实现；而「不太敢信」在多数场景下已被
    「不安」那句覆盖，且用户更难直接察觉。
    """
    text = _extreme(engine).project()
    assert "困了" in text, f"疲惫描述被挤掉了：{text}"
    assert "会忍不住想试探" not in text, f"低优先级的句子没让位：{text}"
    assert len(text) <= _PROJECTION_BUDGET, len(text)


def test_insecurity_supersedes_plain_distrust(engine):
    """「怕他走又不信他」与「不太敢信他」同时成立时只留一句，不啰嗦也不重复。"""
    s = engine.state
    s.intimacy = 60.0   # 不安的前提：亲密已高
    s.trust = 20.0      # 低信任：两句的触发条件同时满足
    s.affection = 80.0
    assert engine.insecure is True
    text = engine.project()
    assert "怕他走" in text, text
    assert "会忍不住想试探" not in text, f"同一件事说了两遍：{text}"


def test_very_long_conflict_is_compressed(engine):
    """冲突是用户原话，可能一整段；压到 40 字加省略号，别把情绪全挤出去。"""
    s = engine.state
    s.conflict = "你昨天明明说好陪我去医院复查的，结果我在门口等了一个多小时，" \
                 "打你电话也不接，后来才知道你跟同事去吃饭了，我一句话都没说"
    s.mood = -0.6
    s.affection = 80.0
    s.trust = 40.0
    text = engine.project()
    assert len(text) <= _PROJECTION_BUDGET, len(text)
    assert "…" in text, f"超长冲突没被压缩：{text}"
    assert text.endswith("。"), text
