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
from pydantic import BaseModel, Field

from app import __version__
from app.config import SETTINGS
from app.context import build_messages
from app.diary.author import write_diary
from app.emotion.engine import EmotionEngine
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
    message: str = Field(..., min_length=1, max_length=2000)


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
        app_cfg = SETTINGS["app"]
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
            decision = state.proactive.check(state.store.last_interaction_ts(), memories,
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

    # 4. 输出过滤（顺序不可反）；复读/跪舔最多重生成一次
    history_replies = [m["content"] for m in recent if m["role"] == "assistant"]
    result = filter_output(reply, history=history_replies)
    retried = bool(result.need_retry and not result.blocked)
    if retried:
        try:
            retry_text = state.generator.chat(messages)
            retry_result = filter_output(retry_text, history=history_replies)
            # 重生成成功时也要给"判过复读"留痕：它说明模型这一轮状态不稳，
            # 情感层据此降低上下文压力，调试台也能看见（覆盖掉就什么也查不到）。
            if not retry_result.need_retry:
                retry_result.triggered = ["复读（已重生成）"] + retry_result.triggered
            result = retry_result
        except GenerationError:
            pass
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
    """手动触发一次主动消息判定（调试用）。"""
    state = _get_state()
    memories = [m.__dict__ for m in state.store.top_memories(10)]
    decision = state.proactive.check(state.store.last_interaction_ts(), memories,
                                     state.store.recent_timeline(5))
    return {"should": decision.should, "reason": decision.reason, "message": decision.message}


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
