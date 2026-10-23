"""记忆写入侧回归：抽取窗口与时间线去重。

召回（读取侧）的用例在 test_retrieval.py；这里守的是写进去的那一端——
窗口取多长、同一件事会不会被重复记。

两个坑都是"测试全绿但线上不对"的形态：
- 配置项写"6 轮"，实现按 6 条取，实际只喂了 3 轮；
- 记忆表按内容去重，时间线没有，同一件事反复记会把最近几条占满。
"""

from __future__ import annotations

import pytest

from app.memory.extractor import MemoryExtractor, build_profile
from app.memory.store import MemoryStore


class RecordingGenerator:
    """假抽取器：记下收到的提示，返回预设结果。"""

    def __init__(self, result=None) -> None:
        self.result = result if result is not None else []
        self.prompts: list[str] = []

    def chat_json(self, messages, num_ctx=2048):  # noqa: ANN001
        self.prompts.append(messages[-1]["content"])
        return self.result


@pytest.fixture
def store(tmp_path):
    return MemoryStore(tmp_path / "memory.db")


def _dialogue(rounds: int) -> list[dict]:
    """造 rounds 轮对话，每轮一条 user + 一条 assistant。"""
    msgs: list[dict] = []
    for i in range(rounds):
        msgs.append({"role": "user", "content": f"他说的第{i}句"})
        msgs.append({"role": "assistant", "content": f"她回的第{i}句"})
    return msgs


def test_extract_window_counts_turns_not_messages(store):
    """extract_recent_rounds=6 就该喂 6 轮（12 条），不是 6 条。"""
    gen = RecordingGenerator()
    ex = MemoryExtractor(store, gen)
    ex._turn_counter = int(ex.cfg["extract_every_n_turns"]) - 1  # 下一次调用正好命中周期

    ex.maybe_extract(_dialogue(10))

    assert gen.prompts, "到了抽取周期却没调用模型"
    prompt = gen.prompts[0]
    # 最近 6 轮 = 第 4~9 轮，第 3 轮（更早）不该出现
    assert "他说的第4句" in prompt, prompt
    assert "他说的第9句" in prompt, prompt
    assert "他说的第3句" not in prompt, "窗口多喂了更早的轮次"
    assert prompt.count("他说的第") == 6, f"应恰好 6 轮，实际 {prompt.count('他说的第')}"


def test_extract_writes_memory_and_timeline(store):
    """抽到 event 时，记忆表与时间线各记一份。"""
    gen = RecordingGenerator([{"kind": "event", "content": "你周五要去面试"}])
    ex = MemoryExtractor(store, gen)
    ex._turn_counter = int(ex.cfg["extract_every_n_turns"]) - 1

    count = ex.maybe_extract(_dialogue(3))

    assert count == 1
    assert [m.content for m in store.memories_by_kind("event")] == ["你周五要去面试"]
    assert [t["title"] for t in store.recent_timeline(5)] == ["你周五要去面试"]


def test_extract_skips_unknown_kind_and_empty_content(store):
    """模型偶尔会吐出 kind 或 content 不合规的项，跳过而不是写脏数据。"""
    gen = RecordingGenerator([
        {"kind": "乱写", "content": "这条不该进库"},
        {"kind": "fact", "content": "   "},
        {"kind": "fact", "content": "你养了一只猫"},
        "不是对象",
    ])
    ex = MemoryExtractor(store, gen)
    ex._turn_counter = int(ex.cfg["extract_every_n_turns"]) - 1

    assert ex.maybe_extract(_dialogue(2)) == 1
    assert [m.content for m in store.memories_by_kind("fact")] == ["你养了一只猫"]


def test_timeline_dedupes_same_event_within_a_day(store, freeze):
    """同一件事一天内只记一条：否则"上次聊到"永远停在它上面。"""
    freeze("app.memory.store.time", 1_790_000_000.0)
    store.add_timeline("你周五要去面试", mood_tag="event")
    store.add_timeline("你周五要去面试", mood_tag="event")
    store.add_timeline("你养了一只猫", mood_tag="event")

    titles = [t["title"] for t in store.recent_timeline(10)]
    assert titles == ["你周五要去面试", "你养了一只猫"], titles


def test_timeline_keeps_same_event_next_day(store, freeze):
    """隔天再提同一件事，是新的经历，得记。"""
    day = 1_790_000_000.0
    freeze("app.memory.store.time", day)
    store.add_timeline("你周五要去面试", mood_tag="event")
    freeze("app.memory.store.time", day + 86400 * 2)
    store.add_timeline("你周五要去面试", mood_tag="event")

    assert len(store.recent_timeline(10)) == 2


def test_timeline_diary_entries_not_merged(store, freeze):
    """日记标题带日期，同日同标题去重不会误伤连续两天。"""
    freeze("app.memory.store.time", 1_790_000_000.0)
    store.add_timeline("2026-10-23 他加班", mood_tag="diary")
    store.add_timeline("2026-10-24 他加班", mood_tag="diary")
    assert len(store.recent_diaries(5)) == 2


def test_profile_reflects_recent_facts(store):
    """画像只由 SQLite 聚合，不调模型；写了事实就该出现在画像里。"""
    store.add_memory("fact", "他对花生过敏")
    store.add_memory("preference", "他喜欢喝美式")
    profile = build_profile(store)

    assert "花生" in profile["facts"]
    assert "美式" in profile["preferences"]
