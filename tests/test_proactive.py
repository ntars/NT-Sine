"""主动消息闸门：时间口径与冷却通路。

守着的坑（2026-10-24 线上实测定到）：她主动发的那句话也落在 messages 表里。
主动闸门原先拿「最后一条消息」的时刻判「他有没有回」，于是她自己的话
被当成回应，待回应标记被清掉，冷却永不触发——表现就是「他没回，她照样
每隔两小时发一次」。

闸门自身的单元测试（test_fixes_stage3.py）一直是绿的，因为它手动摆
_pending 与 _last_sent_ts，绕过了「时间戳从哪儿来」这一层。缺的正是这里：
从 store 取到的时刻到底代表谁在说话。
"""

from __future__ import annotations

import time
from datetime import datetime

import pytest

from app.memory.store import MemoryStore
from app.proactive.gate import ProactiveGate


class _FrozenClock:
    """拨表替身：只顶替 store 模块里的 time，不动全局模块。

    消息表的生产写入用 time.time() 记时刻，测试要摆出「几小时前」的对话
    就得能指定时刻。替换整个 time 模块（而不是 time.time 属性）可避免
    影响 pytest 自身的计时。
    """

    def __init__(self, ts: float) -> None:
        self.ts = ts

    def time(self) -> float:
        return self.ts


def _at(hour: int, minute: int = 0, day: int = 24) -> float:
    """2026-10-24 某个钟点的本地时间戳。日期固定，测试不随运行日漂移。"""
    return datetime(2026, 10, day, hour, minute).timestamp()


def _insert(store: MemoryStore, monkeypatch, role: str, content: str, ts: float) -> None:
    """在指定时刻往消息表塞一条。"""
    monkeypatch.setattr("app.memory.store.time", _FrozenClock(ts))
    store.add_message(role, content)


@pytest.fixture
def store(tmp_path):
    return MemoryStore(tmp_path / "proactive.db")


def test_last_user_ts_ignores_her_own_message(store, monkeypatch):
    """last_user_ts 只认他说的那句，不受她自己更晚的消息影响。"""
    _insert(store, monkeypatch, "user", "今天加班到十点", _at(9))
    _insert(store, monkeypatch, "assistant", "哦。", _at(12))

    assert store.last_interaction_ts() == pytest.approx(_at(12)), "最后一条消息应是她那句"
    assert store.last_user_ts() == pytest.approx(_at(9)), "他的时刻不该被她的话顶掉"


def test_last_user_ts_empty_returns_none(store):
    """从没聊过时返回 None，让闸门给「等第一次对话之后再说」。"""
    assert store.last_user_ts() is None
    assert store.last_interaction_ts() is None


def test_cooldown_survives_her_own_proactive_message(store, monkeypatch):
    """她主动发完、他没回、再到点检查：必须进冷却，不能再发。

    同一条路径跑两遍，只为把缺陷钉死：
    - 取 last_user_ts（修复后）→ 认出「他没回」→ 冷却
    - 取 last_interaction_ts（修复前）→ 误认为「他回了」→ 照发
    后者若哪天重新绿了，说明有人把取值改回去了。
    """
    t_he_spoke = _at(9)    # 09:00 他说话
    t_she_sent = _at(12)   # 12:00 她主动发（间隔 3 小时 > min_gap 2 小时）
    t_check = _at(15)      # 15:00 他没回，循环再来一次

    # ── 第一步：他 09:00 说过话，她 12:00 主动发一次 ──
    _insert(store, monkeypatch, "user", "今天加班到十点", t_he_spoke)
    gate = ProactiveGate()
    gate.cfg = dict(gate.cfg, abandon_rate=0.0)  # 关掉随机放弃，只看冷却
    first = gate.check(t_he_spoke, [], [], now=t_she_sent)
    assert first.should, first.reason
    _insert(store, monkeypatch, "assistant", first.message, t_she_sent)

    # ── 第二步（修复后）：用他真正的时刻判定 ──
    second = gate.check(store.last_user_ts(), [], [], now=t_check)
    assert not second.should, f"他没回却又要开口：{second.reason}"
    assert "冷却" in second.reason, second.reason

    # ── 第三步（对照，修复前）：同状态改用最后一条消息的时刻 ──
    legacy = ProactiveGate()
    legacy.cfg = dict(legacy.cfg, abandon_rate=0.0)
    legacy._last_sent_ts = t_she_sent
    legacy._pending = True
    legacy._today_count = gate._today_count
    wrong = legacy.check(store.last_interaction_ts(), [], [], now=t_check)
    assert wrong.should, "若这里不再放行，说明对照失效，测试已抓不住原缺陷"


def test_his_reply_releases_cooldown(store, monkeypatch):
    """他真回了：待回应标记解除，间隔够了可以再发一次。"""
    t_he_spoke = _at(9)
    t_she_sent = _at(12)
    t_he_replied = _at(14)  # 他回了

    _insert(store, monkeypatch, "user", "今天加班到十点", t_he_spoke)
    gate = ProactiveGate()
    gate.cfg = dict(gate.cfg, abandon_rate=0.0)
    first = gate.check(t_he_spoke, [], [], now=t_she_sent)
    assert first.should, first.reason
    _insert(store, monkeypatch, "assistant", first.message, t_she_sent)
    _insert(store, monkeypatch, "user", "刚看到", t_he_replied)

    # 15:00 检查：他 1 小时前才说过话，还没到 min_gap（2 小时）
    early = gate.check(store.last_user_ts(), [], [], now=_at(15))
    assert not early.should and "太近" in early.reason, early.reason
    assert not gate._pending, "他回了就该解除待回应标记"

    # 18:00 检查：间隔满 4 小时，可以再开口
    later = gate.check(store.last_user_ts(), [], [], now=_at(18))
    assert later.should, later.reason
