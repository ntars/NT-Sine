"""用户侧记忆检索回归测试（方案 §6.4）。"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.context import build_messages
from app.emotion.engine import EmotionEngine
from app.memory.retrieval import recall_block, retrieve_relevant, user_profile_block
from app.memory.store import MemoryStore
from app.worldbook.engine import WorldbookEngine


def _make_store(tmp: str) -> MemoryStore:
    store = MemoryStore(Path(tmp) / "t.db")
    store.add_memory("fact", "他对花生过敏", source="测试")
    store.add_memory("preference", "他喜欢喝美式咖啡", source="测试")
    store.add_memory("event", "上周你们一起看了电影", source="测试")
    store.add_memory("promise", "他答应周五来接她", source="测试")
    store.add_memory("pending", "他犹豫着没说的事是想换工作", source="测试")
    store.set_profile("facts", "他对花生过敏；他住城东")
    store.set_profile("preferences", "他喜欢喝美式咖啡")
    return store


def test_recall_pending(tmp_path=None):
    with tempfile.TemporaryDirectory() as tmp:
        store = _make_store(tmp)
        hits = retrieve_relevant("我想换个工作，但又不敢跟家里说", store)
        assert any("换工作" in h["content"] for h in hits), hits


def test_recall_allergy(tmp_path=None):
    with tempfile.TemporaryDirectory() as tmp:
        store = _make_store(tmp)
        hits = retrieve_relevant("我买了花生糕给你吃", store)
        assert any("花生" in h["content"] for h in hits), hits


def test_recall_single_keyword(tmp_path=None):
    with tempfile.TemporaryDirectory() as tmp:
        store = _make_store(tmp)
        hits = retrieve_relevant("咖啡要加糖吗", store)
        assert any("咖啡" in h["content"] for h in hits), hits


def test_recall_no_false_positive(tmp_path=None):
    with tempfile.TemporaryDirectory() as tmp:
        store = _make_store(tmp)
        assert retrieve_relevant("今天天气真好啊", store) == []


def test_profile_block_empty(tmp_path=None):
    with tempfile.TemporaryDirectory() as tmp:
        store = MemoryStore(Path(tmp) / "t.db")
        block = user_profile_block(store)
        assert "不许编造" in block


def test_profile_block_filled(tmp_path=None):
    with tempfile.TemporaryDirectory() as tmp:
        store = _make_store(tmp)
        block = user_profile_block(store)
        assert "花生过敏" in block and "美式咖啡" in block


def test_build_messages_injects_user_memory(tmp_path=None):
    with tempfile.TemporaryDirectory() as tmp:
        store = _make_store(tmp)
        wb = WorldbookEngine()
        emotion = EmotionEngine(Path(tmp) / "e.json")
        assembly = wb.scan("我买了花生糕给你吃", [])
        msgs = build_messages("我买了花生糕给你吃", assembly, emotion, store, [])
        system = msgs[0]["content"]
        assert "花生过敏" in system, "召回应注入系统提示"


def test_recall_block_format(tmp_path=None):
    with tempfile.TemporaryDirectory() as tmp:
        store = _make_store(tmp)
        hits = retrieve_relevant("我买了花生糕给你吃", store)
        text = recall_block(hits)
        assert text.startswith("关于他，你想起了这些：")
        assert recall_block([]) == ""
