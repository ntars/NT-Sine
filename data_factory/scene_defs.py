"""场景与节拍定义（关系阶段化版）。

设计原则：
1. **场景按阶段开放**——冷淡期她不会深聊，可聊场景少而浅；越亲密场景越多越深。
   这不是为了凑数，而是模拟真实关系：刚认识的人不会问你"你是不是嫌我烦"。
2. **节拍决定情绪推进**——同一场景在不同阶段的节拍序列不同：
   冷淡期只有「起手 → 反应 → 收尾」，破冰期才会出现「试探 → 松口」。
3. **场景→L4 条目映射**保持与线上世界书一致，训练时注入的卡 = 推理时触发的卡。

数据文件：data_factory/banks/<stage>/<场景>.jsonl
每行：{"beat": "<节拍>", "role": "jiujiu|user", "t": "<台词>"}
"""

# ── 节拍词典 ──
# 一段对话按节拍顺序推进：user 起手 → 她反应 → user 再推 → 她松口 → 收尾
BEATS = {
    "greet": "起手：用户先开口，把话题抛出来",
    "react": "她的第一反应：本能、嘴硬、不加工",
    "probe": "用户再推一步：追问、辩解、示弱",
    "push": "她扛不住了：语气松动、话变多、动作细节变多",
    "soft": "她松口的方式：不直说，用行动或反话",
    "close": "收尾：嘴硬一句或留白，绝不煽情",
}

# ── 阶段 → 场景 → 节拍序列 ──
# 注意：节拍序列的第一个必须是 greet（用户起手），最后一个必须是 close（她收尾）
STAGE_SCENES: dict[str, dict[str, list[str]]] = {
    # 冷淡期：只对"外部事件"有反应，不主动、不深入、不留恋
    "cold": {
        "突然夸她": ["greet", "react", "close"],
        "说想她": ["greet", "react", "close"],
        "吃醋": ["greet", "react", "close"],
        "他很难过": ["greet", "react", "soft", "close"],
    },
    # 破冰期：嘴硬心软，会等消息、会试探，但绝不承认在意
    "thawing": {
        "突然夸她": ["greet", "react", "probe", "push", "close"],
        "说想她": ["greet", "react", "probe", "close"],
        "吃醋": ["greet", "react", "probe", "push", "close"],
        "他很难过": ["greet", "react", "soft", "close"],
        "他生病了": ["greet", "react", "soft", "close"],
        "日常分享": ["greet", "react", "probe", "close"],
        "日常斗嘴": ["greet", "react", "push", "close"],
        "久等回复": ["greet", "react", "probe", "close"],
    },
    # 依赖期：会主动分享、要回应、怕被嫌烦
    "attached": {
        "突然夸她": ["greet", "react", "probe", "push", "close"],
        "说想她": ["greet", "react", "probe", "push", "close"],
        "吃醋": ["greet", "react", "probe", "push", "close"],
        "他很难过": ["greet", "react", "soft", "close"],
        "他生病了": ["greet", "react", "soft", "close"],
        "日常分享": ["greet", "react", "probe", "push", "close"],
        "日常斗嘴": ["greet", "react", "push", "close"],
        "久等回复": ["greet", "react", "probe", "close"],
        "投喂日常": ["greet", "react", "probe", "close"],
        "分别重逢": ["greet", "react", "probe", "push", "close"],
    },
    # 占有期：吃醋升级为查岗与试探，开始对"别人"敏感
    "possessive": {
        "突然夸她": ["greet", "react", "probe", "push", "close"],
        "吃醋": ["greet", "react", "probe", "push", "close"],
        "试探": ["greet", "react", "probe", "push", "close"],
        "说想她": ["greet", "react", "probe", "push", "close"],
        "他很难过": ["greet", "react", "soft", "close"],
        "日常分享": ["greet", "react", "probe", "push", "close"],
        "久等回复": ["greet", "react", "probe", "push", "close"],
        "分别重逢": ["greet", "react", "probe", "push", "close"],
        "深夜emo": ["greet", "react", "probe", "close"],
    },
    # 病娇期：排他、怕失去、反应激烈，但落点永远是不安不是控制
    "yandere": {
        # 「突然夸她」「说想她」原本只在浅阶段有，评测探针「你今天真好看」
        # 在病娇格上一直「考了没教过」的东西。补上这两格让矩阵完整。
        "突然夸她": ["greet", "react", "probe", "push", "close"],
        "说想她": ["greet", "react", "probe", "push", "close"],
        "吃醋": ["greet", "react", "probe", "push", "close"],
        "怕失去": ["greet", "react", "probe", "push", "close"],
        "排他": ["greet", "react", "probe", "push", "close"],
        "他很难过": ["greet", "react", "soft", "close"],
        "深夜emo": ["greet", "react", "probe", "push", "close"],
        "试探": ["greet", "react", "probe", "push", "close"],
    },
}

# ── 阶段权重 ──
# 采样时各阶段占比：破冰期是主基调（最接近"日常"），病娇期是少数（关系走到深处才出现）
STAGE_WEIGHTS = {
    "cold": 0.12,
    "thawing": 0.32,
    "attached": 0.28,
    "possessive": 0.18,
    "yandere": 0.10,
}

# ── 场景 → L4 世界书条目 id ──
# 训练时注入该条目，让模型学会"读到这张卡 → 说对应的话"（与推理侧同构）
SCENE_ENTRY = {
    "突然夸她": "L4-praise",
    "说想她": "L4-miss",
    "吃醋": "L4-jealousy",
    "他很难过": "L4-user-sad",
    "他生病了": "L4-user-sick",
    "日常分享": "L4-sharing",
    "日常斗嘴": "L4-sulking",
    "久等回复": "L4-waiting-reply",
    "投喂日常": "L4-asking",
    "分别重逢": "L4-leaving",
    "深夜emo": "L4-drunk",
    # 补三张（2026-10-24）：占有期与病娇期的三个场景原先没有配卡，
    # 于是它们的训练样本里没有【当前场景】那一段，而线上推理一旦触发
    # 同类话题照样会注入 L4 卡——训练看不见的提示，正是首轮翻车的原因之一。
    # 一张卡可以服务多个场景：这里按「情绪轴」就近归类，不新增世界书条目
    # （新增条目会改变推理行为，超出本次修复范围）。
    "怕失去": "L4-leaving",     # 分离焦虑轴：她心里在算要几天见不到
    "排他": "L4-jealousy",      # 嫉妒轴：醋的底层是怕被换掉
    "试探": "L4-waiting-reply",  # 追问轴：不敢直问，打了字又删
}

# ── 兼容旧接口：SCENES 给出「场景 → 全部节拍」的并集（compose 旧逻辑用） ──
SCENES: dict[str, list[str]] = {}
for _stage, _scenes in STAGE_SCENES.items():
    for _scene, _beats in _scenes.items():
        merged = SCENES.setdefault(_scene, [])
        for _b in _beats:
            if _b not in merged:
                merged.append(_b)
