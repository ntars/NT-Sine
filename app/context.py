"""上下文拼装（方案 §4.5 + §3.3）。

插入位置（注意力：越靠后影响越强）：
  系统硬约束 → L1 角色锚点 → L6 边界规则 → L2 用户画像 → L4 记忆事件 → L5 当前场景
  → 最近 N 轮原始对话（走 messages 数组）
  → L3 情感状态 + 当前冲突（拼在最后一条 user 消息尾部 = 最响的位置）

用户侧记忆（§6.4）：滚动画像与关联召回插在 L2 位置（系统提示区），
每次对话必带，不进对话记录。
"""

from __future__ import annotations

import time
from datetime import datetime

from app.config import SETTINGS, load_prompt
from app.emotion.engine import EmotionEngine
from app.memory.retrieval import recall_block, retrieve_relevant, user_profile_block
from app.memory.store import MemoryStore
from app.worldbook.engine import Assembly, WorldbookEngine

_SEASON = {12: "冬", 1: "冬", 2: "冬", 3: "春", 4: "春", 5: "春",
           6: "夏", 7: "夏", 8: "夏", 9: "秋", 10: "秋", 11: "秋"}

_DAYPART = [(0, 5, "深夜"), (5, 9, "清晨"), (9, 12, "上午"), (12, 14, "中午"),
            (14, 18, "下午"), (18, 22, "傍晚"), (22, 24, "夜里")]


def _daypart(hour: int) -> str:
    for a, b, name in _DAYPART:
        if a <= hour < b:
            return name
    return "夜里"


def build_l5_scene(store: MemoryStore) -> str:
    """L5 当前场景：时间锚 + 距上次多久 + 上次聊到哪。动态生成，约 100-150 字。"""
    now = datetime.now()
    part = _daypart(now.hour)
    season = _SEASON[now.month]
    parts = [f"现在是{season}天的{part}。"]

    last_ts = store.last_interaction_ts()
    if last_ts:
        gap_h = (time.time() - last_ts) / 3600
        if gap_h < 1:
            parts.append("你们刚刚还在说话。")
        elif gap_h < 12:
            parts.append(f"距离上次说话约 {int(gap_h)} 小时。")
        else:
            parts.append(f"距离上次说话已经一天多了。")
    else:
        parts.append("这是你们第一次说话。")

    timeline = store.recent_timeline(2)
    if timeline:
        parts.append(f"上次聊到：{timeline[-1]['title']}。")
    return "".join(parts)


def build_messages(user_text: str, assembly: Assembly, emotion: EmotionEngine,
                   store: MemoryStore, history: list[dict]) -> list[dict]:
    """组装 OpenAI 格式 messages。L3 情感状态与冲突放在最后（最强影响位）。"""
    system_parts = [load_prompt("system")]

    # 用户侧记忆（§6.4）：滚动画像 + 关联召回，按方案 §4.5 插在 L2 位置
    recalled = retrieve_relevant(user_text, store)
    recall = recall_block(recalled)

    l1_l6 = [a.entry.content for a in assembly.entries if a.entry.layer in ("L1", "L6")]
    rest = [a.entry.content for a in assembly.entries
            if a.entry.layer not in ("L1", "L6") and a.entry.id != "L2-user-profile"]
    for c in l1_l6:
        system_parts.append(c)
    system_parts.append(user_profile_block(store))
    if recall:
        system_parts.append(recall)
    for c in rest:
        system_parts.append(c)

    # L5 当前场景（动态，拼在系统提示尾部 = 背景在前）
    system_parts.append(build_l5_scene(store))

    # 关系阶段（完整行为卡，随相处时间破冰）：放背景层最后。
    # 与"当下情绪"（L3，拼在 user 尾部）分层——关系是稳定的，情绪是波动的，
    # 两者混在一处会让模型把"现在有点烦"误当成"关系变了"。
    # 注入完整卡而非一句话：训练侧注入的是同一份卡，两侧必须同构。
    stage_card = emotion.stage_card()
    if stage_card:
        stage = emotion.relation_stage
        system_parts.append(f"【关系阶段】{stage['name']}\n{stage_card}")

    messages: list[dict] = [{"role": "system", "content": "\n\n".join(system_parts)}]
    # 最近 N 轮原始对话（滑动窗口）
    for m in history[-int(SETTINGS["context"]["recent_turns"]):]:
        messages.append({"role": m["role"], "content": m["content"]})

    # L3 情感投影 + 冲突：拼在最后一条 user 消息尾部（§4.5.2 深度 0 的"最响位置"）
    suffix_parts = []
    projection = emotion.project()
    if projection:
        suffix_parts.append(f"(你此刻的状态：{projection})")
    if emotion.state.conflict:
        suffix_parts.append(f"(未解决的事横在中间：{emotion.state.conflict})")
    final_content = user_text
    if suffix_parts:
        final_content = user_text + "\n" + "\n".join(suffix_parts)
    messages.append({"role": "user", "content": final_content})
    return messages


def build_proactive_prompt(anchor: str) -> list[dict]:
    """主动开口专用提示：一句话、≤20 字、带锚点、不用"在吗"。"""
    system = (
        "你是白玖玖。现在不是回复，是你主动先开口。"
        "一句、不超过20字、带具体锚点、不用'在吗'。说完就等，不解释。"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"开口聊聊：{anchor}"},
    ]
