"""关系阶段（破冰曲线）回归测试。

验证要点：
1. 三轴共判：时间兜底（只耗时间不聊天不推进）、亲密值门槛、信任门槛
2. 逐级不跳：某一轴暴涨也不能越级
3. 互动天数按自然日去重：同一天聊多少轮都只算一天
4. 阶段句能注入上下文，且不污染 L3 情绪投影
"""

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.context import build_messages  # noqa: E402
from app.emotion.engine import RELATION_STAGES, EmotionEngine  # noqa: E402
from app.memory.store import MemoryStore  # noqa: E402
from app.worldbook.engine import WorldbookEngine  # noqa: E402


def _engine(tmp: str) -> EmotionEngine:
    return EmotionEngine(Path(tmp) / "emotion.json")


def test_initial_stage_is_cold():
    """新用户：没互动天数、亲密 20、信任 40 → 冷淡。"""
    with tempfile.TemporaryDirectory() as tmp:
        assert _engine(tmp).relation_stage["key"] == "cold"


def test_stage_requires_all_three_axes():
    """三轴缺一不可：天数够了但亲密不够，仍停在上一级。"""
    with tempfile.TemporaryDirectory() as tmp:
        e = _engine(tmp)
        e.state.active_days = 30      # 天数到"占有"门槛
        e.state.intimacy = 10         # 亲密低于破冰门槛 15
        e.state.trust = 80
        assert e.relation_stage["key"] == "cold"


def test_stage_never_skips():
    """逐级满足：亲密 90 但只互动过 1 天，不能跳到病娇。"""
    with tempfile.TemporaryDirectory() as tmp:
        e = _engine(tmp)
        e.state.active_days = 1
        e.state.intimacy = 90
        e.state.trust = 90
        assert e.relation_stage["key"] == "cold"

        e.state.active_days = 3
        assert e.relation_stage["key"] == "thawing"


def test_stage_advances_step_by_step():
    """三轴一起放开时，能逐级走到病娇。"""
    with tempfile.TemporaryDirectory() as tmp:
        e = _engine(tmp)
        e.state.active_days, e.state.intimacy, e.state.trust = 14, 40, 50
        assert e.relation_stage["key"] == "attached"
        e.state.active_days, e.state.intimacy, e.state.trust = 30, 65, 65
        assert e.relation_stage["key"] == "possessive"
        e.state.active_days, e.state.intimacy, e.state.trust = 60, 85, 80
        assert e.relation_stage["key"] == "yandere"


def test_active_days_counted_once_per_day():
    """同一天聊多轮，互动天数只 +1（防止熬夜刷出病娇）。"""
    with tempfile.TemporaryDirectory() as tmp:
        e = _engine(tmp)
        for _ in range(5):
            e.update("我今天好累", "……去睡。")
        assert e.state.active_days == 1
        assert e.state.first_meet_ts > 0


def test_stage_sentence_present_and_short():
    """每个阶段都有话术句，且长度可控（注入系统提示，不宜过长）。"""
    with tempfile.TemporaryDirectory() as tmp:
        e = _engine(tmp)
        for stage in RELATION_STAGES:
            e.state.active_days = stage["active_days"]
            e.state.intimacy = stage["intimacy"]
            e.state.trust = stage["trust"]
            sentence = e.stage_sentence()
            assert sentence, f"{stage['key']} 缺话术句"
            assert len(sentence) <= 40, f"{stage['key']} 话术句过长：{len(sentence)}"


def test_stage_injected_into_system_prompt():
    """阶段句要进系统提示，且不挤掉 L3 情绪投影。"""
    with tempfile.TemporaryDirectory() as tmp:
        e = _engine(tmp)
        e.state.active_days, e.state.intimacy, e.state.trust = 14, 40, 50
        store = MemoryStore(Path(tmp) / "t.db")
        wb = WorldbookEngine()
        probe = "我今天好累"
        msgs = build_messages(probe, wb.scan(probe, []), e, store, [])
        store.close()

        system = msgs[0]["content"]
        assert "你们的关系" in system
        assert e.stage_sentence() in system
        # L3 情绪投影仍在最后一条 user 消息尾部（分层注入，互不冲突）
        assert "你此刻的状态" in msgs[-1]["content"]
