"""HTTP 层回归：五层装配、失败兜底、调试端点。

为什么单独建这个文件：用标准库 trace 实测过一轮行覆盖，`app/main.py` 是唯一
0% 的模块——原有的 70 项测试全打在引擎层（情感、世界书、记忆、节奏、过滤），
而「五层怎么串起来、出错时怎么兜」这一层始终没被守住。

这里用假生成器替掉真实模型服务，不依赖 GPU、不连推理端口，
所以训练期间也能跑（训练期间禁止的是并行推理，不是跑单测）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import SETTINGS  # noqa: E402
from app.generation.client import GenerationError  # noqa: E402

DEFAULT_REPLY = "水在桌上。"


class FakeGenerator:
    """假生成器：按队列吐台词，可指定第几次调用抛错。

    行为对齐真实 `Generator`：`chat()` 失败抛 `GenerationError`，
    `chat_json()` 抽取失败返回 None（不打断主链路）。
    """

    def __init__(self) -> None:
        self.queue: list[str] = []
        self.fail_at: set[int] = set()
        self.calls = 0

    def chat(self, messages, num_ctx=None, max_tokens=None):  # noqa: ANN001
        self.calls += 1
        if self.calls in self.fail_at:
            raise GenerationError("模拟模型服务不可用")
        return self.queue.pop(0) if self.queue else DEFAULT_REPLY

    def chat_json(self, messages, num_ctx=2048):  # noqa: ANN001
        return None


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """每个用例一套干净的库与情感状态，生成层换成假件。"""
    import app.main as m

    monkeypatch.setitem(SETTINGS["app"], "db_path", str(tmp_path / "api.db"))
    monkeypatch.setitem(SETTINGS["app"], "emotion_state_path", str(tmp_path / "emotion.json"))
    monkeypatch.setattr(m, "Generator", FakeGenerator)

    # 两个后台循环在 lifespan 里第一轮就会立刻执行（不是等一个间隔）。
    # 深夜跑测试会真的触发写日记，给断言带噪声；这里换成空转协程，
    # 循环本身的行为由 gate / diary 的函数级测试覆盖。
    async def _idle(_state):  # noqa: ANN001, ANN202
        return

    monkeypatch.setattr(m, "_proactive_loop", _idle)
    monkeypatch.setattr(m, "_diary_loop", _idle)
    monkeypatch.setattr(m, "STATE", None)

    with TestClient(m.app) as c:
        yield c


def test_health_returns_version(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["version"]


def test_chat_returns_all_contract_fields(client):
    """前端契约：reply / segments / delay_s / scene_mode / worldbook_tokens。"""
    r = client.post("/api/chat", json={"message": "我今天好累，加班到十点才下班"})
    assert r.status_code == 200
    body = r.json()
    assert body["reply"]
    assert body["segments"], "分条不能为空，前端靠它做逐条上屏"
    assert body["segments"][0]["text"]
    assert body["delay_s"] >= 0
    assert body["scene_mode"] in ("daily", "normal", "longrun", "deep")
    assert body["worldbook_tokens"] > 0, "世界书没注入说明装配断了"


def test_chat_persists_both_sides(client):
    import app.main as m

    client.post("/api/chat", json={"message": "我今天好累"})
    turns = m.STATE.store.recent_turns(5)
    roles = [t["role"] for t in turns]
    assert roles == ["user", "assistant"], roles
    assert turns[1]["content"] == DEFAULT_REPLY


def test_chat_fallback_not_written_into_history(client):
    """生成失败用角色内兜底句，且兜底句不进历史。

    为什么必须这样：兜底句一旦入库，下一轮"复读"过滤器会把它判成重复，
    触发重生成→再失败→再兜底，形成死循环（端到端实测踩到过）。
    """
    import app.main as m

    m.STATE.generator.fail_at = {1, 2}  # 首次与重生成各失败一次
    r = client.post("/api/chat", json={"message": "在吗"})
    assert r.status_code == 200
    body = r.json()
    assert body["reply"] == "……我有点累，先不说了。"

    turns = m.STATE.store.recent_turns(5)
    assert [t["role"] for t in turns] == ["user"], "兜底句不该写进历史"


def test_chat_retries_once_when_repeating(client):
    """与最近一轮高度重复时重生成一次，第二次的文本才落库。

    同时守住"复读留痕"：重生成成功后，`filtered` 仍要写明这一轮判过复读，
    情感层的降级计数也要 +1——否则模型状态不稳这件事被彻底抹掉，
    线上排查时看不到任何痕迹。
    """
    import app.main as m

    client.post("/api/chat", json={"message": "说句话"})
    m.STATE.generator.queue = [DEFAULT_REPLY, "碗放着，我来洗。"]
    r = client.post("/api/chat", json={"message": "再说一句"})
    body = r.json()
    assert m.STATE.generator.calls == 3, "第一轮 1 次 + 本轮重生成 1 次"
    assert body["reply"] == "碗放着，我来洗。"
    assert "复读（已重生成）" in body["filtered"], body["filtered"]
    assert m.STATE.emotion.state.degraded_turns >= 1


def test_chat_rejects_empty_message(client):
    r = client.post("/api/chat", json={"message": ""})
    assert r.status_code == 422, "空消息应在入口被挡，不进模型"


def test_state_endpoint_exposes_five_dimensions(client):
    client.post("/api/chat", json={"message": "今天有点难过"})
    body = client.get("/api/state").json()
    for key in ("mood", "affection", "trust", "energy", "intimacy"):
        assert key in body["emotion"], f"情感状态缺 {key}"
    assert body["turn_count"] == 1
    assert isinstance(body["profile"], dict)


def test_debug_worldbook_returns_activated_entries(client):
    body = client.get("/api/debug/worldbook", params={"message": "你今天真好看"}).json()
    ids = [a["id"] for a in body["activated"]]
    assert "L4-praise" in ids, ids
    assert body["total_tokens"] > 0
    assert "dropped" in body


def test_add_memory_validates_kind(client):
    assert client.post("/api/memory", params={"kind": "乱写", "content": "x"}).status_code == 400
    ok = client.post("/api/memory", params={"kind": "fact", "content": "他养了一只猫"})
    assert ok.status_code == 200
    assert isinstance(ok.json()["id"], int)


def test_proactive_check_returns_decision(client):
    body = client.get("/api/proactive/check").json()
    assert set(body) == {"should", "reason", "message"}
    assert isinstance(body["should"], bool)


def test_uninitialized_state_returns_503():
    """服务尚未初始化时装配层要明确报 503，而不是抛 AttributeError。"""
    import app.main as m

    original = m.STATE
    try:
        m.STATE = None
        # 不进 with，lifespan 不跑，STATE 保持 None
        r = TestClient(m.app).get("/api/state")
        assert r.status_code == 503
    finally:
        m.STATE = original
