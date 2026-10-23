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
