"""v3 三项修复的回归测试：L4 互斥、句内复读检测、日记语料重组。"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.worldbook.engine import WorldbookEngine
from scripts.compose_diary import has_repeat
from scripts.filter_dataset import find_repeat


def _l4_ids(assembly):
    return [a.entry.id for a in assembly.entries if a.entry.layer == "L4"]


def test_l4_exclusive_single_card():
    """多卡同时命中时只保留优先级最高的一张（防串味）。"""
    wb = WorldbookEngine()
    a = wb.scan("我今天好累，加班到十点才下班", [])
    ids = _l4_ids(a)
    assert ids == ["L4-user-sad"], ids  # user-sad(65) > leaving(55)
    assert "L4-leaving" in a.suppressed


def test_l4_exclusive_keeps_single_hit():
    wb = WorldbookEngine()
    a = wb.scan("拉钩，一百年不许变", [])
    assert _l4_ids(a) == ["L4-promise"], _l4_ids(a)


def test_l4_exclusive_nickname():
    wb = WorldbookEngine()
    a = wb.scan("宝宝~我今天超想你", [])
    assert _l4_ids(a) == ["L4-nickname"], _l4_ids(a)


def test_repeat_detector_catches_duplication():
    assert find_repeat("……我叫白玖玖。你又没听我的，白玖玖。") is not None


def test_repeat_detector_ignores_common_words():
    assert find_repeat("今天今天你今天去哪儿") is None  # "今天"属高频词，不算复读
    assert find_repeat("笨死了。也就我管你。") is None


def test_diary_repeat_detector():
    assert has_repeat("我把蛋藏底下了。我把蛋藏底下了。") is not None
    assert has_repeat("今天煮了面，多下了一个蛋，蛋黄没熟。") is None


def test_diary_corpus_shape():
    """重组后的日记语料：条数达标、无复读、角度覆盖 8 类。"""
    pool = Path("data_factory/diary_pool.jsonl")
    if not pool.exists():
        return  # 尚未生成
    items = [json.loads(l) for l in pool.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(items) >= 1000, len(items)
    angles = {i["angle"] for i in items}
    assert len(angles) >= 8, angles
    for i in items[:200]:
        assert 50 <= len(i["text"]) <= 320, i["text"][:40]
        assert has_repeat(i["text"]) is None, i["text"][:60]


def test_diary_training_file_exists():
    p = Path("data_train/diary.jsonl")
    if not p.exists():
        return
    first = json.loads(p.read_text(encoding="utf-8").splitlines()[0])
    roles = [m["role"] for m in first["messages"]]
    assert roles == ["system", "user", "assistant"], roles
    assert "写吧" in first["messages"][1]["content"]
