"""世界书引擎回归测试（pytest）。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from app.worldbook.engine import WorldbookEngine, estimate_tokens


@pytest.fixture(scope="module")
def wb() -> WorldbookEngine:
    return WorldbookEngine()


def test_estimate_tokens_cjk():
    assert estimate_tokens("你好世界") == 4
    assert estimate_tokens("") == 0


def test_load_and_layers(wb):
    layers = {e.layer for e in wb.entries}
    assert {"L1", "L2", "L4", "L6"} <= layers


def test_always_layers_present(wb):
    a = wb.scan("随便说点什么", [])
    layer_set = {x.entry.layer for x in a.entries}
    assert {"L1", "L2", "L6"} <= layer_set


def test_praise_trigger(wb):
    a = wb.scan("你今天真好看", [])
    assert "L4-praise" in [x.entry.id for x in a.entries]


def test_recursion_only_l4(wb):
    for e in wb.entries:
        if e.recursive:
            assert e.layer == "L4"


def test_budget_cap(wb):
    a = wb.scan("下雨了，你不是说好来接我，结果放我鸽子，我很生气", [])
    assert a.total_tokens <= int(wb.cfg["total_budget"])


def test_whole_word_disabled():
    # 引擎层强制关闭整词匹配；settings 若被改成 true 应直接抛错
    from app.config import SETTINGS
    original = SETTINGS["worldbook"]["whole_word_match"]
    try:
        SETTINGS["worldbook"]["whole_word_match"] = True
        with pytest.raises(ValueError):
            WorldbookEngine()
    finally:
        SETTINGS["worldbook"]["whole_word_match"] = original


# ── 压制记录的口径（2026-10-24 审计补）──
def test_suppressed_only_lists_matched_entries(tmp_path):
    """被压制名单只该记「真的命中但优先级被压」的卡。

    早前实现把压制判定写在匹配判定之前，于是同一层里所有低优先级卡都被
    记成"被压制"——实测「今天下雨了，我在等伞」会把 L4-drunk / L4-money /
    L4-cat 这些毫无关系的卡全列进去。调试台的数字因此不可信，
    而"可测量"要求每个数字都对应一件真事。
    """
    import json

    data = [
        {"id": "L4-a", "layer": "L4", "priority": 60, "recursive": True,
         "triggers": ["下雨"], "content": "下雨了，记着带伞。"},
        {"id": "L4-b", "layer": "L4", "priority": 50, "recursive": True,
         "triggers": ["伞"], "content": "伞就挂在门口。"},
        {"id": "L4-c", "layer": "L4", "priority": 40, "recursive": True,
         "triggers": ["猫"], "content": "猫又趴在窗台上。"},
    ]
    (tmp_path / "l4.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    eng = WorldbookEngine(tmp_path)
    a = eng.scan("下雨了", [])

    ids = [x.entry.id for x in a.entries]
    assert "L4-a" in ids, ids
    assert "L4-b" in a.suppressed, f"命中但优先级更低，应记为压制：{a.suppressed}"
    assert "L4-c" not in a.suppressed, f"完全没触发的卡不该进压制名单：{a.suppressed}"


# ── 压制记录的口径（2026-10-24 审计补）──
