"""日记模块回归测试（借鉴 ai-companion-kit diary/diary-author.js）。"""

import sys
import tempfile
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.diary.author import (build_diary_prompt, parse_loose, pick_angle,
                              write_diary)
from app.memory.extractor import KIND_CAPS, MemoryExtractor
from app.memory.store import MemoryStore


class FakeGenerator:
    """假生成器：返回固定内容，不依赖模型。"""

    def __init__(self, raw: str):
        self.raw = raw

    def chat(self, messages, **kwargs):
        return self.raw


def test_parse_loose_plain():
    r = parse_loose('{"title":"今天","text":"面又煮糊了。","mood":"烦"}')
    assert r == {"title": "今天", "text": "面又煮糊了。", "mood": "烦"}


def test_parse_loose_fenced_and_junk():
    r = parse_loose('好的，这是日记：\n```json\n{"title":"梨","text":"那只梨没回来。"}\n```\n以上。')
    assert r["title"] == "梨" and r["text"] == "那只梨没回来。"


def test_parse_loose_fallback():
    r = parse_loose("模型抽风输出一段废话")
    assert r["text"] == "模型抽风输出一段废话" and r["title"] == ""


def test_parse_loose_rejects_non_json():
    r = parse_loose("没有花括号的普通文本")
    assert r["text"] == "没有花括号的普通文本"


def test_angle_stable_within_day():
    d = date(2026, 10, 21)
    assert pick_angle(d) == pick_angle(d)
    angles = {pick_angle(date(2026, 1, i)) for i in range(1, 10)}
    assert len(angles) > 1  # 不同天不同角度（轮换生效）


def test_build_prompt_has_angle():
    msgs = build_diary_prompt("窗外 / 天气 / 光线", "他对花生过敏", "昨天的日记")
    assert msgs[0]["role"] == "system"
    assert "独白" in msgs[0]["content"]
    assert "窗外" in msgs[1]["content"] and "花生过敏" in msgs[1]["content"]


def test_write_diary_flow(tmp_path=None):
    with tempfile.TemporaryDirectory() as tmp:
        store = MemoryStore(Path(tmp) / "t.db")
        gen = FakeGenerator('{"title":"降温","text":"降温了。他肯定又忘了穿秋裤。","mood":"嘴硬"}')
        entry = write_diary(gen, store, memory_text="他对花生过敏",
                            today=date(2026, 10, 21))
        assert entry and entry["title"] == "降温"
        tl = store.recent_timeline(5)
        assert any(t["mood_tag"].startswith("diary") for t in tl)
        # 幂等：同一天再写返回 None
        again = write_diary(gen, store, memory_text="", today=date(2026, 10, 21))
        assert again is None


def test_write_diary_model_down(tmp_path=None):
    with tempfile.TemporaryDirectory() as tmp:
        store = MemoryStore(Path(tmp) / "t.db")

        class Boom:
            def chat(self, *a, **k):
                raise RuntimeError("模型不在")

        assert write_diary(Boom(), store, today=date(2026, 10, 22)) is None


def test_extractor_kind_caps(tmp_path=None):
    """超过容量上限的旧记忆被淘汰（借鉴 memory.js 的 FIFO 上限）。"""
    with tempfile.TemporaryDirectory() as tmp:
        store = MemoryStore(Path(tmp) / "t.db")
        for i in range(KIND_CAPS["topic"] + 5):
            store.add_memory("topic", f"聊过的话题{i}", source="测试")
        MemoryExtractor(store, None).prune()
        kept = store.memories_by_kind("topic", limit=10_000)
        assert len(kept) == KIND_CAPS["topic"], len(kept)


def test_new_kinds_recalled(tmp_path=None):
    with tempfile.TemporaryDirectory() as tmp:
        from app.memory.retrieval import retrieve_relevant
        store = MemoryStore(Path(tmp) / "t.db")
        store.add_memory("trait", "他常熬夜到三点", source="测试")
        store.add_memory("topic", "聊过养猫的话题", source="测试")
        hits = retrieve_relevant("最近总熬夜，顺便想养只猫", store)
        contents = " ".join(h["content"] for h in hits)
        assert "熬夜" in contents or "猫" in contents, hits


# ── 取哪几篇 / 今天写过没有（2026-10-24 审计补）──
def _seed_diaries(store, n: int) -> None:
    for i in range(1, n + 1):
        store.add_timeline(f"2026-10-{10 + i:02d} 第{i}篇", detail=f"日记正文{i}",
                           mood_tag="diary")


def test_recent_diary_text_takes_latest(tmp_path):
    """喂给模型的必须是最近几篇，不是最老几篇。

    角度轮换的意义就是「别跟刚写的重样」，喂最老的等于白轮换。
    早前按 timeline 旧→新顺序取前 3 条，拿到的正是最早那三篇。
    """
    from app.diary.author import _recent_diary_text

    store = MemoryStore(tmp_path / "d.db")
    _seed_diaries(store, 6)
    text = _recent_diary_text(store, limit=3)
    assert "日记正文4" in text and "日记正文6" in text, text
    assert "日记正文1" not in text, f"不该把最早那篇喂进去：{text}"


def test_diary_idempotent_when_events_flood_window(tmp_path):
    """事件很多时，同一天也不能写第二篇日记。

    幂等判断原本靠 recent_timeline(20) 里找「今天」——记忆抽取每 4 轮就落一条
    event，事件一多就把 20 条窗口塞满，今天的日记被挤出去，于是又写一篇。
    """
    store = MemoryStore(tmp_path / "d2.db")
    today = date(2026, 10, 24)
    store.add_timeline(f"{today.isoformat()} 已经写过", detail="写过了", mood_tag="diary")
    for i in range(25):
        store.add_timeline(f"他说的第{i}件事", mood_tag="event")

    gen = FakeGenerator('{"title":"新的","text":"又写了一篇。","mood":"烦"}')
    entry = write_diary(gen, store, memory_text="", today=today)
    assert entry is None, "今天已经写过，不该再写"


# ── 取哪几篇 / 今天写过没有（2026-10-24 审计补）──
def _seed_diaries(store, n: int) -> None:
    for i in range(1, n + 1):
        store.add_timeline(f"2026-10-{10 + i:02d} 第{i}篇", detail=f"日记正文{i}",
                           mood_tag="diary")


def test_recent_diary_text_takes_latest(tmp_path):
    """喂给模型的必须是最近几篇，不是最老几篇。

    角度轮换的意义就是「别跟刚写的重样」，喂最老的等于白轮换。
    早前按 timeline 旧→新顺序取前 3 条，拿到的正是最早那三篇。
    """
    from app.diary.author import _recent_diary_text

    store = MemoryStore(tmp_path / "d.db")
    _seed_diaries(store, 6)
    text = _recent_diary_text(store, limit=3)
    assert "日记正文4" in text and "日记正文6" in text, text
    assert "日记正文1" not in text, f"不该把最早那篇喂进去：{text}"


def test_diary_idempotent_when_events_flood_window(tmp_path):
    """事件很多时，同一天也不能写第二篇日记。

    幂等判断原本靠 recent_timeline(20) 里找「今天」——记忆抽取每 4 轮就落一条
    event，事件一多就把 20 条窗口塞满，今天的日记被挤出去，于是又写一篇。
    """
    store = MemoryStore(tmp_path / "d2.db")
    today = date(2026, 10, 24)
    store.add_timeline(f"{today.isoformat()} 已经写过", detail="写过了", mood_tag="diary")
    for i in range(25):
        store.add_timeline(f"他说的第{i}件事", mood_tag="event")

    gen = FakeGenerator('{"title":"新的","text":"又写了一篇。","mood":"烦"}')
    entry = write_diary(gen, store, memory_text="", today=today)
    assert entry is None, "今天已经写过，不该再写"
