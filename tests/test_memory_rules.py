"""记忆抽取的兜底通路：模型抽不出来时，规则必须接得住。

为什么要这层：实测五轮真实对话（「我叫小玖」「我下周五要去杭州出差」）后，
memories 表 **0 条**。原因是抽取靠同一份微调模型输出 JSON 数组，而它被训成
「只吐一句短台词」，连「输出 JSON」这条指令都做不到——直接回「……我去倒水。」。
`chat_json` 解析失败返回 None，`maybe_extract` 静默返回 0，于是「不健忘」这条
产品第一理念在真实链路上完全不通，而且从接口上看不出任何异常。

这组用例钉三件事：
1. 高置信度的中文事实模式能抽出来（称呼、行程、职业、住处、偏好、习惯）
2. 抽不出来的时候返回空表，不编造（宁缺勿滥，幻觉比遗忘更糟）
3. `maybe_extract` 在模型不可用或返回非 JSON 时，仍然落库
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.memory.extractor import MemoryExtractor, extract_by_rules  # noqa: E402
from app.memory.store import MemoryStore  # noqa: E402


def _kinds(items: list[dict]) -> dict[str, str]:
    """把结果整理成 {kind: content}，方便逐条断言。"""
    return {it["kind"]: it["content"] for it in items}


def test_rule_extracts_name():
    """「我叫小玖」是最典型的事实，必须抽到，且用「你」不用「他」。"""
    items = extract_by_rules("user: 我叫小玖，在做一个模型项目。")
    assert items
    joined = " ".join(it["content"] for it in items)
    assert "小玖" in joined
    assert "他" not in joined


def test_rule_extracts_trip():
    """带时间的行程要能识别成 event，主动消息才有东西可提。"""
    items = extract_by_rules("user: 我下周五要去杭州出差三天，那边有个技术评审。")
    events = [it for it in items if it["kind"] == "event"]
    assert events, f"没抽出 event：{items}"
    assert "杭州" in events[0]["content"]


def test_rule_extracts_preference():
    """偏好类：喜欢 / 讨厌 都算。"""
    like = _kinds(extract_by_rules("user: 我特别喜欢喝美式。"))
    assert "preference" in like and "美式" in like["preference"]

    hate = _kinds(extract_by_rules("user: 我讨厌下雨天。"))
    assert "preference" in hate and "下雨" in hate["preference"]


def test_rule_extracts_trait():
    """习惯类：熬夜、早起这类可反复观察到的行为。"""
    items = extract_by_rules("user: 我最近天天熬夜改代码，眼睛疼。")
    traits = [it for it in items if it["kind"] == "trait"]
    assert traits, f"没抽出 trait：{items}"
    assert "熬夜" in " ".join(t["content"] for t in traits)


def test_rule_ignores_chitchat():
    """闲聊不含可记事实时返回空表——宁可不记，也不能编。"""
    assert extract_by_rules("user: 嗯。") == []
    assert extract_by_rules("user: 好的，那我先睡了。") == []


def test_rule_ignores_questions():
    """疑问句不是事实。

    实测漏了这一层，会把「你还记得我刚才说我叫什么吗？」抽成事实「你叫什么吗」，
    脏数据留在表里会被反复召回，比漏记更麻烦。
    """
    assert extract_by_rules("user: 你还记得我刚才说我叫什么吗？") == []
    assert extract_by_rules("user: 我最近在做什么呢，你猜。") == []
    assert extract_by_rules("user: 我喜欢什么，你说说看。") == []


def test_rule_content_capped():
    """每条 content 不超过 30 字，与模型抽取同一约束（超长会撑爆提示预算）。"""
    long_text = "user: 我下周五要去一个特别特别远的地方出差大概要待很久很久还要转两次飞机"
    for item in extract_by_rules(long_text):
        assert len(item["content"]) <= 30, item


def test_maybe_extract_falls_back_when_model_breaks(tmp_path):
    """模型返回非 JSON 时，兜底规则仍要把记忆落库。

    这是本轮真实故障的回归守卫：`chat_json` 返回 None 时，
    旧实现直接 return 0，五轮对话一条记忆都没写。
    """

    class BrokenGenerator:
        """模拟被训坏了的模型：对任何抽取请求都回一句台词。"""

        def chat_json(self, messages, num_ctx=2048):
            return None

    store = MemoryStore(tmp_path / "m.db")
    ex = MemoryExtractor(store, BrokenGenerator())
    # 把触发间隔压到 1，免得为了凑轮数写一堆填充对话
    ex.cfg = dict(ex.cfg)
    ex.cfg["extract_every_n_turns"] = 1
    ex.cfg["extract_recent_rounds"] = 6

    messages = [
        {"role": "user", "content": "我叫小玖，在做一个模型项目。"},
        {"role": "assistant", "content": "……我去倒水。"},
    ]
    count = ex.maybe_extract(messages)
    assert count >= 1, "模型失效时兜底通路没有生效"
    assert store.memories_by_kind("fact"), "记忆没有落库"
    store.close()


def test_maybe_extract_prefers_model_result(tmp_path):
    """模型正常时用模型的，不能被规则覆盖（规则只做兜底）。"""

    class GoodGenerator:
        def chat_json(self, messages, num_ctx=2048):
            return [{"kind": "fact", "content": "你叫小玖", "detail": {}}]

    store = MemoryStore(tmp_path / "m.db")
    ex = MemoryExtractor(store, GoodGenerator())
    ex.cfg = dict(ex.cfg)
    ex.cfg["extract_every_n_turns"] = 1
    ex.cfg["extract_recent_rounds"] = 6

    count = ex.maybe_extract([
        {"role": "user", "content": "我叫小玖，我喜欢喝美式。"},
        {"role": "assistant", "content": "……嗯。"},
    ])
    assert count == 1, "模型给了一条，结果条数应与之相符"
    contents = [m.content for m in store.memories_by_kind("fact")]
    assert contents == ["你叫小玖"]
    store.close()
