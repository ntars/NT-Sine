"""FastAPI 入口：五层装配（方案 §3.3 的一次对话完整数据流）。

启动：python -m uvicorn app.main:app --host 127.0.0.1 --port 8420
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from app import __version__
from app.config import SETTINGS
from app.context import build_messages
from app.diary.author import write_diary
from app.emotion.engine import EmotionEngine
from app.emotion.events import detect as detect_intimacy_event
from app.generation.client import GenerationError, Generator
from app.memory.extractor import MemoryExtractor, build_profile
from app.memory.store import MemoryStore
from app.orchestrator import Orchestrator
from app.output.filters import filter_output
from app.output.rhythm import plan_delivery_now
from app.proactive.gate import ProactiveGate
from app.worldbook.engine import WorldbookEngine

logger = logging.getLogger("nt-sine")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")


class ChatRequest(BaseModel):
    """一轮用户输入。

    `min_length=1` 只保证「至少一个字符」，管不住 `"   "`、全角空格、`"\\n\\n"`
    这类纯空白——它们在 strip 之后才是空串，而校验发生在 strip 之前。若不在这里
    先 strip 再判空，空串会一路走进世界书扫描、生成与落库：messages 表里留下
    `('user', '')`，模型那一侧看到的最后一条只剩状态前缀（等于让她对着空气说话），
    而且空 user 还会占掉滑动窗口的轮次名额。所以在校验层就把这件事了结。
    """

    message: str = Field(..., min_length=1, max_length=2000)

    @field_validator("message")
    @classmethod
    def _strip_and_require_text(cls, v: str) -> str:
        """去掉首尾空白（含全角空格）；去掉后为空则直接判为非法输入。"""
        v = v.strip()
        if not v:
            raise ValueError("消息不能只有空白")
        return v


class ChatResponse(BaseModel):
    reply: str
    segments: list[dict]
    delay_s: float
    scene_mode: str
    worldbook_tokens: int
    filtered: list[str] = []


class NTState:
    """进程内共享状态（单用户单场景，不需要多进程）。"""

    def __init__(self):
        self.store = MemoryStore(SETTINGS["app"]["db_path"])
        self.emotion = EmotionEngine(SETTINGS["app"]["emotion_state_path"])
        self.worldbook = WorldbookEngine()
        self.generator = Generator()
        self.orchestrator = Orchestrator()
        self.extractor = MemoryExtractor(self.store, self.generator)
        self.proactive = ProactiveGate()
        self.turn_count = 0


STATE: NTState | None = None


async def _proactive_loop(state: NTState) -> None:
    """主动消息异步循环（方案 §11.2）：10-15 分钟检查一次，只是时间比较。"""
    interval = float(SETTINGS["proactive"]["check_interval_min"]) * 60
    while True:
        try:
            memories = [m.__dict__ for m in state.store.top_memories(10)]
            # 用 last_user_ts：她主动发的那句也落在 messages 里，若按"最后一条消息"
            # 判定，她自己的话会被当成"他回了"，冷却从此失效（实测没回也会 2 小时再发）
            decision = state.proactive.check(state.store.last_user_ts(), memories,
                                             state.store.recent_timeline(5))
            if decision.should and decision.message:
                # 推送由前端 SSE/轮询取走；这里只落一条 assistant 消息标记
                state.store.add_message("assistant", decision.message)
                logger.info("主动开口：%s", decision.message)
        except Exception:  # 主动消息失败绝不能拖垮主服务
            logger.exception("主动消息循环异常")
        await asyncio.sleep(interval)


async def _diary_loop(state: NTState) -> None:
    """夜间日记循环：23 点到凌晨 1 点之间，每天写一篇，写完即停。"""
    while True:
        try:
            hour = time.localtime().tm_hour
            if hour >= 23 or hour < 1:
                memory_text = "；".join(m.content for m in state.store.top_memories(6))
                entry = write_diary(state.generator, state.store, memory_text=memory_text)
                if entry:
                    logger.info("她写了今天的日记：%s", entry["title"])
        except Exception:  # 日记失败绝不能拖垮主服务
            logger.exception("日记循环异常")
        await asyncio.sleep(1800)  # 半小时检查一次


@asynccontextmanager
async def lifespan(app: FastAPI):
    global STATE
    STATE = NTState()
    task = asyncio.create_task(_proactive_loop(STATE))
    diary_task = asyncio.create_task(_diary_loop(STATE))
    logger.info("NT-Sine %s 已就绪（模型 %s）", __version__, SETTINGS["ollama"]["model"])
    try:
        yield
    finally:
        task.cancel()
        diary_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        with contextlib.suppress(asyncio.CancelledError):
            await diary_task
        STATE.store.close()


app = FastAPI(title="NT-Sine", version=__version__, lifespan=lifespan)


def _get_state() -> NTState:
    if STATE is None:
        raise HTTPException(503, "服务尚未初始化")
    return STATE


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """一轮完整对话：记忆/世界书 → 调度 → 生成 → 过滤 → 情感 → 节奏 → 落库。"""
    state = _get_state()
    user_text = req.message.strip()

    # 1. 记忆层：取滑动窗口（单位是"轮"）+ 世界书扫描
    recent = state.store.recent_turns(int(SETTINGS["context"]["recent_turns"]))
    history_texts = [m["content"] for m in recent if m["role"] == "user"]
    degraded = state.emotion.state.degraded_turns > 0
    assembly = state.worldbook.scan(user_text, history_texts, degraded=degraded)

    # 2. 调度层：延迟建议 + 档位透传（档位只有一个真源：扫描结果）
    turn = state.orchestrator.plan(user_text, assembly.scene_mode)

    # 3. 拼上下文并调用生成层
    messages = build_messages(user_text, assembly, state.emotion, state.store, recent)
    gen_failed = False
    try:
        reply = state.generator.chat(messages)
    except GenerationError as e:
        # 模型不可用时不在角色外报错，用一条符合人设的短句兜底
        logger.error("生成失败：%s", e)
        reply = "……我有点累，先不说了。"
        gen_failed = True

    # 4. 输出过滤（顺序不可反）；复读/过度跪舔按配置重生成
    #    重生成次数此前写死在代码里，配置里的 retry_on_repetition 没人读——
    #    改配置毫无作用，等于这个参数不存在。改为按配置循环。
    history_replies = [m["content"] for m in recent if m["role"] == "assistant"]
    result = filter_output(reply, history=history_replies)
    max_retry = max(0, int(SETTINGS["filter"]["retry_on_repetition"]))
    retried = False
    for _ in range(max_retry):
        if not result.need_retry or result.blocked:
            break
        retried = True
        try:
            retry_text = state.generator.chat(messages)
        except GenerationError:
            break   # 重生成失败就沿用这一轮已有的文本，不把回复弄丢
        retry_result = filter_output(retry_text, history=history_replies)
        # 重生成成功时也要给"判过复读"留痕：它说明模型这一轮状态不稳，
        # 情感层据此降低上下文压力，调试台也能看见（覆盖掉就什么也查不到）。
        if not retry_result.need_retry:
            retry_result.triggered = ["复读（已重生成）"] + retry_result.triggered
        result = retry_result
    # 4.5 重生成后仍然是空正文：模型这一轮确实没给出可用的话（只吐了思考块，
    #     或整段旁白被裁空）。用一条符合人设的短句兜住，否则前端会播出一段空白。
    #     这句不写进历史——与生成失败同理，写进去下一轮会被复读过滤器拦住，
    #     兜底→判复读→再兜底就成了死循环。
    if not result.text:
        logger.warning("本轮无可用正文（%s），使用兜底句", result.triggered)
        result.text = "……我有点累，先不说了。"
        gen_failed = True
        result.triggered.append("空回复兜底")

    state.emotion.mark_degraded(retried)

    # 5. 情感状态随本轮更新（用过滤后的文本，避免脏输出影响状态）
    state.emotion.update(user_text, result.text)

    # 5.5 亲密事件：关系跨段只能由真实事件推动，刷轮数无效。
    #     此前这条通路只有写入口、没有任何调用方，亲密值永远停在初始 20，
    #     「依赖」及之后三个阶段（五分之三）在真实链路上永远到不了。
    #     详见 app/emotion/events.py 顶部说明。
    event = detect_intimacy_event(user_text)
    if event and state.emotion.note_intimacy_event(event[1], kind=event[0]):
        logger.info("亲密事件：%s，亲密值 → %.0f", event[1], state.emotion.state.intimacy)

    # 6. 拟人化节奏（自带当前小时，深夜才会变速）
    hour_segments = plan_delivery_now(result.text, mood=state.emotion.state.mood)

    # 7. 落库（原始对话）+ 每 3-5 轮异步抽取记忆
    state.store.add_message("user", user_text)
    # 生成失败时的兜底句不写进历史：否则下一轮"复读"过滤器会把它当重复，
    # 兜底→判复读→再兜底，形成死循环（端到端测试实测到过）
    if not gen_failed:
        state.store.add_message("assistant", result.text)
    state.turn_count += 1
    recent2 = state.store.recent_turns(int(SETTINGS["context"]["recent_turns"]))
    try:
        state.extractor.maybe_extract(recent2)
        build_profile(state.store)
    except Exception:
        logger.exception("记忆抽取异常（不影响本轮回复）")

    return ChatResponse(
        reply=result.text,
        segments=[s.__dict__ for s in hour_segments],
        delay_s=turn.delay_s,
        scene_mode=turn.scene_mode,
        worldbook_tokens=assembly.total_tokens,
        filtered=result.triggered,
    )


@app.get("/api/state")
def get_state() -> dict:
    """情感状态 + 画像快照（前端可读；数值只用于展示，绝不进上下文）。"""
    state = _get_state()
    return {
        "emotion": state.emotion.state.to_dict(),
        # 关系阶段单独给出：五维数值只用于展示，阶段才是"关系走到哪了"的可读结论。
        # 此前接口不暴露它（1.11 记为改进项），前端与调试台只能靠推断。
        "stage": {"key": state.emotion.relation_stage["key"],
                  "name": state.emotion.relation_stage["name"]},
        "profile": state.store.all_profile(),
        "last_interaction": state.store.last_interaction_ts(),
        "turn_count": state.turn_count,
    }


@app.get("/api/debug/worldbook")
def debug_worldbook(message: str) -> dict:
    """世界书调试台：查看一次扫描激活了哪些条目、预算用了多少。"""
    state = _get_state()
    history = [m["content"] for m in
               state.store.recent_turns(int(SETTINGS["context"]["recent_turns"]))
               if m["role"] == "user"]
    assembly = state.worldbook.scan(message, history)
    return {
        "scene_mode": assembly.scene_mode,
        "total_tokens": assembly.total_tokens,
        "activated": [
            {"id": a.entry.id, "layer": a.entry.layer, "source": a.source,
             "depth": a.depth, "tokens": a.tokens}
            for a in assembly.entries
        ],
        "dropped": assembly.dropped,
    }


@app.post("/api/memory")
def add_memory(kind: str, content: str) -> dict:
    """人工补写记忆（可加 kind=fact/preference/event/promise）。"""
    if kind not in ("fact", "preference", "event", "promise"):
        raise HTTPException(400, "kind 必须是 fact/preference/event/promise")
    state = _get_state()
    mid = state.store.add_memory(kind, content, source="人工录入")
    return {"id": mid}


@app.get("/api/proactive/check")
def proactive_check() -> dict:
    """手动触发一次主动消息判定（调试用，只看不发）。

    两处口径必须和 _proactive_loop 一致，否则调试台给出的结论跟线上不是一回事：
    一是时刻取 last_user_ts（她自己的话不算"他回了"），二是 commit=False，
    否则查一次就消耗一次每日配额、还会挂上待回应标记让接下来六小时不开口。
    """
    state = _get_state()
    memories = [m.__dict__ for m in state.store.top_memories(10)]
    decision = state.proactive.check(state.store.last_user_ts(), memories,
                                     state.store.recent_timeline(5), commit=False)
    return {"should": decision.should, "reason": decision.reason, "message": decision.message}


@app.get("/api/messages")
def list_messages(after_id: int = 0, limit: int = 50) -> dict:
    """按 id 增量取消息（旧 → 新），客户端轮询用。

    补的是「她主动开口却没人看得见」这条断掉的链路：后台循环把她的话写进
    messages 表，此前对外只有「会不会开口」的只读判定与不含正文的情感状态，
    界面上永远拿不到那句话。现在客户端记住 last_id、带着它轮询即可，
    既取到新对话，也取到她主动说的那句。
    """
    state = _get_state()
    rows = state.store.messages_after(after_id, min(max(1, limit), 200))
    return {"messages": rows, "last_id": rows[-1]["id"] if rows else after_id}


@app.post("/api/diary")
def diary_now() -> dict:
    """手动触发一篇日记（调试用；每日自动在夜间循环写）。"""
    state = _get_state()
    memory_text = "；".join(m.content for m in state.store.top_memories(6))
    entry = write_diary(state.generator, state.store, memory_text=memory_text)
    if entry is None:
        return {"ok": False, "reason": "今天已写过或模型不可用"}
    return {"ok": True, "entry": entry}


@app.get("/health")
def health() -> dict:
    return {"ok": True, "version": __version__}


if __name__ == "__main__":
    # 直接 `python -m app.main` 也能起服务，主机与端口从 config/settings.yaml 读。
    # 加这一段是因为 `app.host` / `app.port` 两个配置项此前全仓无人读取——
    # 在配置里改端口不会生效，等于配置在撒谎。uvicorn 是既有依赖，不新增包。
    import uvicorn

    uvicorn.run(app, host=SETTINGS["app"]["host"], port=int(SETTINGS["app"]["port"]))

