"""阶段三缺陷修复的回归测试（对应交付清单 3.1-3.8）。

每条测试都先于实现写下（红→绿），失败即代表该缺陷仍在：
- 分条下限：模型偶尔只吐「嗯。」这种两字短条，节奏层必须并进相邻条，
  否则前端会出现"打字机卡壳"式的碎条
- 深夜变速：0-6 点她打字更慢（困），这个参数此前没被调用方传进去
- 主动冷却：主动消息发出后对方没回，最多少补一句就停，绝不连发
- 档位字段：plan() 返回的档位必须与世界书扫描结果一致，不能写死 normal
- 窗口语义：滑动窗口的单位是"轮"（一轮=用户+她），不是"条"
- 复读豁免：刻意重复是情绪表达（「我说了，不行。不行。」），不该被判复读
- 否定语境：「不太开心」是负面，不能因为出现"开心"就算正面

运行：pytest tests/test_fixes_stage3.py -q
"""

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.emotion.engine import EmotionEngine
from app.memory.store import MemoryStore
from app.orchestrator import Orchestrator
from app.output.filters import filter_output
from app.output.rhythm import plan_delivery
from app.proactive.gate import ProactiveGate

SEG_MIN = 8  # 与 config/settings.yaml 的 rhythm.segment_min 一致


# ── 3.1 / 3.2 分条下限 ──
def test_segment_min_merges_short():
    """两字短条必须并入相邻条，不能单独成条。"""
    segs = plan_delivery("嗯。我今天在公司楼下看到一只很胖的橘猫，它一直跟着我走了两条街。", mood=0.0, hour=14)
    assert segs, "不该产出空计划"
    short = [s.text for s in segs if len(s.text) < SEG_MIN]
    assert not short, f"存在过短分条：{short}"


def test_segment_min_keeps_normal_split():
    """正常的两个长句仍应保持两条，别把下限做成"全部合并"。"""
    text = "今天外面下了好大的雨，我在楼下站了十分钟。你那边带伞了没有。"
    segs = plan_delivery(text, mood=0.0, hour=14)
    assert len(segs) >= 2, [s.text for s in segs]


# ── 3.3 深夜变速 ──
def _mean_typing(hour: int, rounds: int = 200) -> float:
    """把整轮时间伪装成指定小时，测的是生产入口（而不是底层函数）。"""
    import app.output.rhythm as rhythm

    real = time.localtime

    class _Fake:
        @staticmethod
        def now():
            return real()

    vals = []
    original = rhythm.time.localtime
    rhythm.time.localtime = lambda *_a, **_k: time.struct_time(
        (2026, 10, 23, hour, 0, 0, 4, 296, 0))
    try:
        for _ in range(rounds):
            segs = rhythm.plan_delivery_now("今天外面下雨了，你带伞没有。", mood=0.0)
            vals.extend(s.typing_s for s in segs)
    finally:
        rhythm.time.localtime = original
    return sum(vals) / len(vals)


def test_night_typing_slower():
    """0-6 点打字时长应约等于白天的 night_slow_factor 倍（1.6）。"""
    day, night = _mean_typing(14), _mean_typing(3)
    ratio = night / day
    assert 1.4 < ratio < 1.8, f"夜间/白天打字时长比 {ratio:.2f}，未按 1.6 变速"


# ── 3.4 主动冷却 ──
def _afternoon_ts() -> float:
    """今天 15:00 的时间戳——避开 1-6 点禁主动时段，让闸门只考冷却逻辑。"""
    lt = time.localtime()
    return time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, 15, 0, 0, 0, 0, -1))


def test_proactive_cooldown_after_unanswered():
    """上次主动发出且 1 小时没回 → 6 小时冷却内不再开口。"""
    now = _afternoon_ts()
    gate = ProactiveGate()
    gate._last_sent_ts = now - 3600
    gate._last_refused_ts = now - 3600
    ok, reason = gate.gate_time(last_interaction_ts=now - 20 * 3600, now=now)
    assert not ok
    assert "冷却" in reason, reason


def test_proactive_cooldown_expires():
    """冷却 6 小时后可以再开口一次。"""
    now = _afternoon_ts()
    gate = ProactiveGate()
    gate._last_sent_ts = now - 7 * 3600
    gate._last_refused_ts = now - 7 * 3600
    ok, reason = gate.gate_time(last_interaction_ts=now - 20 * 3600, now=now)
    assert ok, reason


def test_proactive_auto_marks_unanswered():
    """check() 发现上一条主动无人回应时，必须自己记一次未回复。"""
    gate = ProactiveGate()
    gate.cfg = dict(gate.cfg, abandon_rate=0.0)   # 关掉随机放弃，只看冷却
    gate._last_sent_ts = time.time() - 3600
    gate._pending = True
    decision = gate.check(time.time() - 10 * 3600, [], [])
    assert not decision.should
    assert "冷却" in decision.reason, decision.reason
    assert not gate._pending, "未回复应清掉待回应标记，避免每次检查都刷新冷却"


# ── 3.5 档位字段 ──
def test_turnplan_tier_matches_scan():
    """plan() 的档位必须取世界书扫描结果，不能写死。"""
    plan = Orchestrator().plan("今天风好大", scene_mode="deep")
    assert plan.scene_mode == "deep"


def test_turnplan_default_tier():
    plan = Orchestrator().plan("今天风好大")
    assert plan.scene_mode == "normal"


# ── 3.6 窗口语义 ──
def test_recent_window_is_turns(tmp_path):
    """recent_turns(10) 应取最近 10 轮 = 20 条消息，按旧→新排列。"""
    store = MemoryStore(tmp_path / "t.db")
    for i in range(30):
        store.add_message("user" if i % 2 == 0 else "assistant", f"第{i}条")
    rows = store.recent_turns(10)
    assert len(rows) == 20, len(rows)
    assert rows[-1]["content"] == "第29条"
    assert rows[0]["content"] == "第10条"


def test_recent_window_short_history(tmp_path):
    """历史不足时全量返回，不补空。"""
    store = MemoryStore(tmp_path / "t2.db")
    for i in range(7):
        store.add_message("user", f"第{i}条")
    assert len(store.recent_turns(10)) == 7


# ── 3.7 复读豁免 ──
def test_repeat_emphasis_exempt():
    """刻意重复（带强调标记）不该被判复读。"""
    history = ["不行。"]
    result = filter_output("我说了，不行。不行。", history=history)
    assert "复读" not in result.triggered, result.triggered


def test_repeat_collapse_flagged():
    """真正的一字不差复读必须拦下。"""
    line = "……你别把这事说出去。"
    result = filter_output(line, history=[line])
    assert "复读" in result.triggered, result.triggered


# ── 3.8 否定语境 ──
def test_negation_marks_negative():
    """「不太开心」应判为负向。"""
    delta, _conflict, _apology = EmotionEngine.signals("我今天不太开心")
    assert delta < 0, delta


def test_degree_word_amplifies(tmp_path):
    """程度副词应放大强度：「特别难过」比「难过」更负。"""
    plain, _c1, _a1 = EmotionEngine.signals("我难过")
    strong, _c2, _a2 = EmotionEngine.signals("我特别难过")
    assert strong < plain, (strong, plain)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
