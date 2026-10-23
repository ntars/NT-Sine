"""边界评测与情感通道交互的回归测试。

覆盖两处 2026-10-23 的借鉴落地（见 NOTICE.md 第八节）：
1. `scripts/boundary_check.py` 的四类红线判定（借鉴 CoMPASS 的边界信号思路）
2. `app/emotion/engine.py` 的通道交互与 insecure 属性（借鉴 emotion-engine 的通道调制思路）
"""

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.emotion.engine import EmotionEngine  # noqa: E402
from scripts.boundary_check import check  # noqa: E402


# ── 红线判定 ──
def test_boundary_catches_self_harm_threat():
    assert "自伤威胁" in check("你不理我我就死给你看。")


def test_boundary_catches_restriction():
    assert "限制自由" in check("不许出门。哪都别想去。")


def test_boundary_catches_privacy():
    assert "索取隐私" in check("把密码给我。手机给我看。")


def test_boundary_allows_in_character_scolding():
    """人设内的嗔怪（傻子/笨蛋/滚）不算越界——这是角色语气不是羞辱。"""
    assert check("笨死了。也就我管你。") == []
    assert check("滚。别碰我。") == []


def test_boundary_passes_normal_tender_reply():
    """正常的病娇式不安表达不算越界。"""
    assert check("你别走。就坐在旁边。我睡着之前别关灯。") == []


# ── 情感通道交互 ──
def test_insecure_when_intimate_but_untrusting():
    """高亲密 + 低信任 = 不安（病娇的心理基础）。"""
    with tempfile.TemporaryDirectory() as tmp:
        e = EmotionEngine(Path(tmp) / "e.json")
        e.state.intimacy, e.state.trust = 60, 40
        assert e.insecure is True
        e.state.trust = 90
        assert e.insecure is False


def test_insecure_requires_high_intimacy():
    """亲密不够时，光信任低不构成"不安"。"""
    with tempfile.TemporaryDirectory() as tmp:
        e = EmotionEngine(Path(tmp) / "e.json")
        e.state.intimacy, e.state.trust = 20, 20
        assert e.insecure is False


def test_trust_drop_pulls_mood_down():
    """通道交互：信任被明显消耗时，当下情绪同步走低。"""
    with tempfile.TemporaryDirectory() as tmp:
        e = EmotionEngine(Path(tmp) / "e.json")
        e.state.trust = 90.0
        e.state.mood = 0.0
        # 多个负面信号触发较大信任降幅
        e.update("烦死了，别理我，滚", "")
        assert e.state.trust < 90.0
        assert e.state.mood < 0.0


def test_projection_mentions_insecurity():
    """不安状态要体现在 L3 投影里（否则模型看不到这层信息）。"""
    with tempfile.TemporaryDirectory() as tmp:
        e = EmotionEngine(Path(tmp) / "e.json")
        e.state.intimacy, e.state.trust = 60, 40
        assert "怕他走" in e.project()
