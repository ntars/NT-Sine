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


def _afternoon() -> float:
    """今天的下午三点。主动闸门里"深夜禁主动"排在最前，用例必须避开凌晨。"""
    import time

    lt = time.localtime()
    return time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, 15, 0, 0, 0, 0, -1))


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


def test_chat_empty_reply_falls_back(client):
    """模型只吐思考块时前端不能拿到空白：兜底一句，且兜底句不入库。

    2026-10-24 审计补的用例。剥掉思考块后剩空串，早前会原样往下走——
    分条为空、前端播一段「什么都没说」。README 记的「约 8% 空返回」里，
    有一部分不是模型没输出，而是输出被剥成了空。
    """
    import app.main as m

    m.STATE.generator.queue = ["<think>\n\n</think>\n\n", "<think>\n\n</think>\n\n"]
    r = client.post("/api/chat", json={"message": "在吗"})
    assert r.status_code == 200
    body = r.json()
    assert body["reply"] == "……我有点累，先不说了。", body["reply"]
    assert "空回复兜底" in body["filtered"], body["filtered"]
    assert body["segments"], "兜底之后必须有分条，前端才有东西可播"

    turns = m.STATE.store.recent_turns(5)
    assert [t["role"] for t in turns] == ["user"], "兜底句不该写进历史"


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


def test_proactive_check_is_readonly(client, freeze):
    """调试接口连查多次，今日配额与待回应标记都不能变。

    闸门 check() 内部会累加今日次数、记发出时刻、挂待回应标记；调试台若
    直接调，查一次就等于真发了一次——查三次当天不再开口，还会因为"没回"
    进入六小时冷却。所以接口必须走只读模式。

    时钟钉在下午三点：闸门一里"深夜禁主动"排在最前，跑这条用例时若是凌晨，
    返回的理由会是深夜而不是我们要验的配额。

    **两侧时钟都要钉**：只钉闸门、让写库用真实时钟，则在真实时刻晚于 15:00 时
    `now - last_interaction` 会是负数，被判成"距上次对话太近"，用例下午跑必红。
    所以把写库时钟一并钉到两小时前，与真实时刻彻底脱钩。
    """
    import app.main as m

    now = _afternoon()
    freeze("app.proactive.gate.time", now)
    freeze("app.memory.store.time", now - 7200)
    gate = m.STATE.proactive
    gate.cfg = dict(gate.cfg, min_gap_h=0, abandon_rate=0.0)
    client.post("/api/chat", json={"message": "今天加班到十点"})

    first = client.get("/api/proactive/check").json()
    assert first["should"] is True, first["reason"]
    for _ in range(4):
        again = client.get("/api/proactive/check").json()
        assert again["should"] is True, f"只读查询把配额吃掉了：{again['reason']}"

    assert gate._today_count == 0, gate._today_count
    assert gate._pending is False
    assert gate._last_sent_ts == 0.0


def test_proactive_check_ignores_her_own_message(client, insert_at, freeze):
    """她主动说过一句、他没回：接口要给「冷却中」，不能当成他回了。

    这是线上同一个缺陷的第二个入口——后台循环改了取值口径，调试接口当时
    还留在 last_interaction_ts 上，于是同一时刻两处给出相反结论。
    """
    import app.main as m

    now = _afternoon()
    freeze("app.proactive.gate.time", now)
    gate = m.STATE.proactive
    gate.cfg = dict(gate.cfg, min_gap_h=0, abandon_rate=0.0)
    store = m.STATE.store

    insert_at(store, "user", "我回来了", now - 5 * 3600)            # 他五小时前说话
    insert_at(store, "assistant", "外面降温了。", now - 3 * 3600)   # 她三小时前主动发
    gate._last_sent_ts = now - 3 * 3600
    gate._pending = True

    body = client.get("/api/proactive/check").json()
    assert body["should"] is False, body
    assert "冷却" in body["reason"], body["reason"]


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


def test_chat_records_intimacy_event(client):
    """走完整 HTTP 链路：一句告白必须真的把亲密值推上去。

    引擎层已经有 `note_intimacy_event` 的用例，但那是直接调函数——它证明
    「函数是对的」，不证明「主链路接上了」。这正是本次缺陷的形态：函数写好了、
    没有任何调用方，接口上完全看不出异常。
    """
    import app.main as m

    before = m.STATE.emotion.state.intimacy
    r = client.post("/api/chat", json={"message": "其实我喜欢你很久了"})
    assert r.status_code == 200
    assert m.STATE.emotion.state.intimacy == before + 15.0


def test_chat_repeated_confession_counts_once(client):
    """同一类事件连说三轮，只记一次——防「刷屏刷出跨段」。"""
    import app.main as m

    before = m.STATE.emotion.state.intimacy
    for _ in range(3):
        client.post("/api/chat", json={"message": "我喜欢你"})
    assert m.STATE.emotion.state.intimacy == before + 15.0


def test_state_exposes_relation_stage(client):
    """状态接口要直接给出关系阶段，界面与调试台不必自己推断。"""
    body = client.get("/api/state").json()
    assert body["stage"]["key"] in ("cold", "thawing", "attached", "possessive", "yandere")
    assert body["stage"]["name"]


def test_messages_endpoint_polls_incrementally(client):
    """增量取消息：带 last_id 轮询只拿新增的那几条，且顺序是旧 → 新。

    这条端点补的是「她主动开口却没人看得见」：后台循环把她的话写进 messages，
    此前对外没有任何取用通道，等于主动消息这条链路通到一半。
    """
    client.post("/api/chat", json={"message": "回来了"})
    first = client.get("/api/messages").json()
    assert [m["role"] for m in first["messages"]] == ["user", "assistant"]
    assert first["last_id"] == first["messages"][-1]["id"]

    client.post("/api/chat", json={"message": "今天很冷"})
    delta = client.get("/api/messages", params={"after_id": first["last_id"]}).json()
    assert [m["role"] for m in delta["messages"]] == ["user", "assistant"], delta
    assert delta["messages"][0]["id"] > first["last_id"]

    ids = [m["id"] for m in delta["messages"]]
    assert ids == sorted(ids), "必须旧 → 新，前端按顺序上屏"


def test_messages_endpoint_delivers_proactive_line(client):
    """她的主动消息落库之后，客户端必须能从接口取到正文。

    后台循环写库与客户端取消息是两段，任何一段断掉，用户都看不到她开口。
    """
    import app.main as m

    m.STATE.store.add_message("assistant", "降温了，你那破身子骨注意点。")
    body = client.get("/api/messages").json()
    assert body["messages"], "主动消息落库后取不到，等于这条功能不存在"
    assert body["messages"][-1]["content"] == "降温了，你那破身子骨注意点。"


def test_retry_count_comes_from_config(client, monkeypatch):
    """重生成次数必须由配置决定：配 2 次就要真的重生成两次。

    此前次数写死在代码里（只重生成一次），配置项 retry_on_repetition 没人读——
    改配置毫无作用，参数等于不存在。这里配 2，队列给「重复 / 重复 / 正常」，
    期望调用 3 次（首轮 1 次 + 重生成 2 次）。
    """
    import app.main as m

    monkeypatch.setitem(SETTINGS["filter"], "retry_on_repetition", 2)
    client.post("/api/chat", json={"message": "说句话"})          # 第 1 次调用
    m.STATE.generator.queue = [DEFAULT_REPLY, DEFAULT_REPLY, "碗放着，我来洗。"]
    body = client.post("/api/chat", json={"message": "再说一句"}).json()

    assert m.STATE.generator.calls == 4, "首轮 1 次 + 本轮 3 次（首发 + 2 次重生成）"
    assert body["reply"] == "碗放着，我来洗。", body["reply"]
    assert "复读（已重生成）" in body["filtered"], body["filtered"]

