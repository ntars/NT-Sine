"""上下文拼装（方案 §4.5 + §3.3）。

插入位置（注意力：越靠后影响越强）：
  系统硬约束 → L1 角色锚点 → L6 边界规则 → L2 用户画像 → L4 记忆事件 → L5 当前场景
  → 最近 N 轮原始对话（走 messages 数组）
  → 关系阶段卡 → L3 情感状态 + 当前冲突（拼在最后一条 user 消息尾部 = 最响的位置）

**关系阶段卡的位置：留在 system（做过实验，不是没试过）**。原假设是「system 里 992 字、
阶段卡只占 12.7%，被稀释到无效，挪到 user 尾部就会生效」。2026-10-24 用
`scripts/diagnose_stage.py position` 在同一份权重上实测两种位置：被夸探针 1/5 对 1/5、
加班探针 3/5 对 2/5——**前移没有收益，其中一例还更差**，假设被否掉。
所以默认值回到 system，`context.stage_signal` 只作复验开关保留。
真正的杠杆是「占比」而不是「位置」：极简 system 下阶段提示占三分之一时才分得开
（实验 4），这意味着要动的是 system 的长度，不是卡的位置。
这段状态无论如何只进请求、不上界面：用户看到的是自己的话，模型看到的是自己的话 + 关系状态。

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
                   store: MemoryStore, history: list[dict],
                   stage_signal: str | None = None) -> list[dict]:
    """组装 OpenAI 格式 messages。

    stage_signal 决定关系阶段卡注入哪儿：`user`（默认，紧贴用户消息）或 `system`
    （旧行为，系统提示尾部）。留这个参数是为了让诊断脚本能在同一份权重上做位置
    A/B，不用改两次代码——线上值取 config/settings.yaml 的 context.stage_signal。
    """
    pos = (stage_signal or SETTINGS["context"].get("stage_signal", "system")).lower()
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

    # 关系阶段（完整行为卡，随相处时间破冰）。
    # v3.4.1 起默认注入到紧贴用户消息处（见文件头说明）：原先放在 system 里，
    # 992 字的 system 中只占 12.7%，实测五阶段输出一字不差。
    # 注入完整卡而非一句话：训练侧注入的是同一份卡，两侧必须同构。
    stage_block = ""
    stage_card = emotion.stage_card()
    if stage_card:
        stage = emotion.relation_stage
        stage_block = f"【关系阶段】{stage['name']}\n{stage_card}"
        if pos != "user":
            system_parts.append(stage_block)

    messages: list[dict] = [{"role": "system", "content": "\n\n".join(system_parts)}]
    # 最近 N 轮原始对话（滑动窗口）：一轮 = 用户一句 + 她一句，所以是 2N 条。
    # 以前这里按 N 条切，等于把上游取到的 2N 条又砍成一半（实际只剩约 5 轮）。
    for m in history[-int(SETTINGS["context"]["recent_turns"]) * 2:]:
        messages.append({"role": m["role"], "content": m["content"]})

    # 拼在最后一条 user 消息尾部（§4.5.2 深度 0 的"最响位置"），顺序由稳到变：
    # 关系阶段（稳定）→ 当下情绪（波动）→ 未解决的冲突（最尖锐）。
    # 阶段放最前而不是最后：它描述的是「你们是什么关系」，紧贴他的话读起来像背景前提；
    # 情绪放后面，因为它才是「此刻」。
    suffix_parts = []
    if stage_block and pos == "user":
        suffix_parts.append(stage_block)
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
