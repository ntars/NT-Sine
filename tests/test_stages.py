"""关系阶段（破冰曲线）回归测试。

验证要点：
1. 三轴共判：时间兜底（只耗时间不聊天不推进）、亲密值门槛、信任门槛
2. 逐级不跳：某一轴暴涨也不能越级
3. 互动天数按自然日去重：同一天聊多少轮都只算一天
4. 阶段句能注入上下文，且不污染 L3 情绪投影
"""

import sys
import tempfile
import time
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
    """默认位置：阶段卡在 system 里，且不挤掉 L3 情绪投影。

    为什么默认在 system：2026-10-24 用 `diagnose_stage.py position` 在同一份权重上
    实测过「挪到 user 尾部」这条路——被夸探针 1/5 对 1/5、加班探针 3/5 对 2/5，
    前移没有收益，其中一例还更差。所以回到 system，并把这条结论钉在测试里，
    免得下次又有人凭直觉把它挪走。
    """
    with tempfile.TemporaryDirectory() as tmp:
        e = _engine(tmp)
        e.state.active_days, e.state.intimacy, e.state.trust = 14, 40, 50
        store = MemoryStore(Path(tmp) / "t.db")
        wb = WorldbookEngine()
        probe = "我今天好累"
        msgs = build_messages(probe, wb.scan(probe, []), e, store, [])
        store.close()

        system = msgs[0]["content"]
        assert "【关系阶段】" in system
        assert e.stage_card() in system
        # L3 情绪投影仍在最后一条 user 消息尾部（分层注入，互不冲突）
        assert "你此刻的状态" in msgs[-1]["content"]
        assert "【关系阶段】" not in msgs[-1]["content"], "同一份卡不能两边都注入"


def test_stage_signal_user_moves_card_to_user_message():
    """`stage_signal="user"` 时阶段卡要真的搬到 user 消息里：这是 A/B 的开关。

    诊断脚本靠它把「位置」这一个变量单独拎出来；开关失灵，测的就不是位置了。
    前缀必须拼在用户原话之后，不能把他的话顶掉。
    """
    with tempfile.TemporaryDirectory() as tmp:
        e = _engine(tmp)
        e.state.active_days, e.state.intimacy, e.state.trust = 14, 40, 50
        store = MemoryStore(Path(tmp) / "t.db")
        wb = WorldbookEngine()
        probe = "我今天好累"
        msgs = build_messages(probe, wb.scan(probe, []), e, store, [],
                              stage_signal="user")
        store.close()

        system = msgs[0]["content"]
        user_msg = msgs[-1]["content"]
        assert e.stage_card() in user_msg
        assert "【关系阶段】" not in system
        assert user_msg.startswith(probe), "用户原话必须完整保留在最前面"


# ── 通道交互与精力恢复（2026-10-24 补）──
def test_trust_loss_pulls_mood_down():
    """信任被明显消耗时，当下情绪要同步走低（通道交互）。

    早前这行判断写在 s.trust 更新之前：prev_trust 记下后中间没有任何改动，
    差值恒为 0，这条通路从来没触发过。这里用「你很烦，滚」钉住它——
    该输入让信任掉 4 点，情绪应在正常衰减之外再被压 0.05。
    """
    with tempfile.TemporaryDirectory() as tmp:
        e = _engine(tmp)
        e.state.mood = 0.5
        e.state.affection = 40.0
        e.state.trust = 40.0
        e.update("你很烦，滚")

        assert e.state.trust < 36.5, f"信任应掉 4 点，实际 {e.state.trust}"
        # 只按 delta 走的话 mood 会停在 0.3325；接通通道交互才是 0.2825
        assert e.state.mood < 0.31, f"信任掉 4 点却没压情绪，通道交互没生效：{e.state.mood}"
        assert abs(e.state.mood - 0.2825) < 0.005, e.state.mood


def test_energy_recovers_after_long_gap():
    """隔了几小时再聊，精力要回满；连着聊才继续衰减。

    精力原本只按 session_start 到现在的小时数算，而 session_start 只在进程启动
    时初始化——服务连跑几小时后 energy 恒为 0，之后每轮都注入"你困了"，
    隔夜回来也还是困的。
    """
    with tempfile.TemporaryDirectory() as tmp:
        e = _engine(tmp)
        e.state.session_start = time.time() - 6 * 3600
        e.state.energy = 0.0
        e.state.last_turn_ts = time.time() - 3 * 3600   # 上一轮是三小时前
        e.update("在吗")
        assert e.state.energy > 70, f"隔了三小时再聊，精力该回满，实际 {e.state.energy}"

        e.state.session_start = time.time() - 3 * 3600
        e.state.last_turn_ts = time.time() - 60          # 刚聊过，算同一场会话
        e.update("在吗")
        assert e.state.energy < 40, f"同一场会话里聊了三小时，精力该降下来，实际 {e.state.energy}"
