"""阶段塌缩诊断脚本的契约测试。

这些用例不加载模型（那是分钟级的事，放进回归会拖垮每次改动），
只钉住「诊断口径本身别漂」：

- 阶段的五档与三轴取值必须与线上引擎一致，否则诊断出的分数不对应真实运行
- 极简提示必须覆盖全部五档，缺一档会静默少测一格
- 装配出来的消息必须是 system + user 两条，且带阶段卡
- 默认探针必须真的命中 L4-praise，否则测的不是「被夸」这一格

为什么值得测：诊断脚本的结论会写进交付清单当根因依据。
脚本口径悄悄变了、结论却还按旧的引用，比不诊断更糟。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.emotion.engine import RELATION_STAGES  # noqa: E402
from scripts.diagnose_stage import (  # noqa: E402
    MINIMAL_HINT,
    PROBE,
    STAGES,
    build_stage_messages,
)


def test_stage_keys_match_engine():
    """诊断脚本的五档名字必须与线上引擎一致（引擎里是列表，按 key 取）。"""
    assert set(STAGES) == {stage["key"] for stage in RELATION_STAGES}


def test_stage_thresholds_match_engine():
    """三轴取值必须落在引擎的同一档位，否则评的不是真实阶段。"""
    by_key = {stage["key"]: stage for stage in RELATION_STAGES}
    for stage, (days, intimacy, trust) in STAGES.items():
        expected = by_key[stage]
        assert (days, intimacy, trust) == (
            expected["active_days"], expected["intimacy"], expected["trust"]
        ), f"{stage} 的三轴取值与引擎不一致"


def test_minimal_hint_covers_all_stages():
    """极简提示缺一档，就会静默少测一格。"""
    assert set(MINIMAL_HINT) == set(STAGES)
    assert all(hint.strip() for hint in MINIMAL_HINT.values())


def test_stage_messages_shape():
    """装配结果应是五份 system + user，且 system 里带当前阶段名。

    默认位置是 system（实测前移到 user 没有收益，见 app/context.py 文件头）；
    这条断言守的是「诊断跑的确实是线上那套装配」，位置变了就得跟着变。
    """
    msgs = build_stage_messages(PROBE)
    assert set(msgs) == set(STAGES)
    for stage, messages in msgs.items():
        assert [m["role"] for m in messages] == ["system", "user"]
        assert "【关系阶段】" in messages[0]["content"]
        assert messages[1]["content"].startswith(PROBE)


def test_position_mode_both_positions_differ():
    """位置开关要真的改变装配结果，否则 position 模式测的是同一份东西。"""
    in_user = build_stage_messages(PROBE, stage_signal="user")
    in_system = build_stage_messages(PROBE, stage_signal="system")
    assert "【关系阶段】" in in_user["cold"][1]["content"]
    assert "【关系阶段】" not in in_user["cold"][0]["content"]
    assert "【关系阶段】" in in_system["cold"][0]["content"]
    assert "【关系阶段】" not in in_system["cold"][1]["content"]


def test_probe_hits_l4_praise():
    """默认探针必须命中 L4-praise，否则诊断的就不是「被夸」这一格。"""
    from app.worldbook.engine import WorldbookEngine

    assembly = WorldbookEngine().scan(PROBE, [])
    activated = [e.entry.id for e in assembly.entries if e.entry.layer == "L4"]
    assert "L4-praise" in activated
