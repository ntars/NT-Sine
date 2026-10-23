"""情感引擎：五维状态机 + 自然语言投影（方案 §7）。

原则：情绪有大惯性，关系有小惯性。
- mood 每轮向 0 回拉（情绪不会永远高涨或永远低落，但也不会立刻翻篇）
- affection 单轮涨幅硬顶 +2（防"聊两句就死心塌地"）
- trust 下降快于上升（信任崩塌比建立容易，符合真人心理）
- 冲突期间 mood 锁负值，直到和解
状态绝不以数值形式进上下文——翻译成 60-100 字的自然语言再注入。
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import yaml

from app.config import PROMPTS_DIR, SETTINGS

# ── 关系阶段（破冰曲线）：三轴共同决定，只升不降、必须逐级满足 ──
# 为什么要三条轴：
#   active_days —— 光耗时间不聊天，关系不该推进（防"放置养成病娇"）
#   intimacy    —— 只能由事件触发，刷轮数没用（防"聊一百句就死心塌地"）
#   trust       —— 吵翻了会卡住阶段，和好才能继续（关系不是单行道）
RELATION_STAGES: list[dict] = [
    {"key": "cold", "name": "冷淡", "active_days": 0, "intimacy": 0, "trust": 0},
    {"key": "thawing", "name": "破冰", "active_days": 3, "intimacy": 15, "trust": 30},
    {"key": "attached", "name": "依赖", "active_days": 14, "intimacy": 40, "trust": 50},
    {"key": "possessive", "name": "占有", "active_days": 30, "intimacy": 65, "trust": 65},
    {"key": "yandere", "name": "病娇", "active_days": 60, "intimacy": 85, "trust": 80},
]


def _load_stage_cards() -> dict:
    """读取五阶段话术卡（config/prompts/stages.yaml）。缺失则只有阶段名，不影响主链路。"""
    path = PROMPTS_DIR / "stages.yaml"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return (yaml.safe_load(f) or {}).get("stages", {})


# 情绪词典（简易信号抽取；诚实说明：仅覆盖常见表达，复杂语义要靠模型本身）
_POSITIVE = ["开心", "喜欢", "谢谢", "哈哈哈", "哈哈", "好听", "好看", "想你", "可爱",
             "对不起", "我错了", "保证", "拉钩", "陪", "辛苦", "厉害", "真棒"]
_NEGATIVE = ["滚", "讨厌", "烦", "生气", "吵架", "分手", "别理", "随便", "呵呵", "有病",
             "傻逼", "无聊", "敷衍", "走", "离开", "不爱"]
_CONFLICT = ["分手", "别来往", "你很烦", "不想理你", "讨厌你", "滚"]


@dataclass
class EmotionState:
    """五维 + 三个标签（方案 §7.1）。"""

    mood: float = 0.0            # -1.0 ~ +1.0 当下情绪
    affection: float = 40.0      # 0-100 好感（慢变量）
    trust: float = 40.0          # 0-100 信任（降得快升得慢）
    energy: float = 80.0         # 0-100 精力
    intimacy: float = 20.0       # 0-100 亲密（分 5 阶段，跨段需事件触发）
    conflict: str = ""           # 未解决的冲突描述（空 = 无）
    topic: str = ""              # 当前话题
    last_intimacy_event: str = ""  # 最近一次亲密事件
    first_meet_ts: float = 0.0   # 首次相处时间戳（0 = 尚未初始化）
    active_days: int = 0         # 累计互动天数（按自然日去重，同一天聊一百轮也只算一天）
    last_active_day: str = ""    # 最近一次互动的日期（YYYY-MM-DD），用于去重
    session_start: float = field(default_factory=time.time)
    last_turn_ts: float = field(default_factory=time.time)
    degraded_turns: int = 0      # 连续被过滤器打回的轮数（供世界书降级）

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "EmotionState":
        known = {f: d[f] for f in EmotionState.__dataclass_fields__ if f in d}
        return EmotionState(**known)


class EmotionEngine:
    """状态更新 + 自然语言投影。状态持久化为 JSON 单文件（重启不丢）。"""

    def __init__(self, state_path: str | Path):
        self.cfg = SETTINGS["emotion"]
        self.state_path = Path(state_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state = self._load()

    # ── 持久化 ──
    def _load(self) -> EmotionState:
        if self.state_path.exists():
            try:
                return EmotionState.from_dict(json.loads(self.state_path.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, TypeError):
                pass  # 损坏则重建——状态文件坏了不该拖垮服务
        return EmotionState()

    def save(self) -> None:
        self.state_path.write_text(
            json.dumps(self.state.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # ── 信号抽取 ──
    @staticmethod
    def signals(text: str) -> tuple[float, bool, bool]:
        """从一段文本粗判：情绪增量、是否冲突、是否道歉/示好。"""
        pos = sum(1 for w in _POSITIVE if w in text)
        neg = sum(1 for w in _NEGATIVE if w in text)
        conflict = any(w in text for w in _CONFLICT)
        apology = any(w in text for w in ["对不起", "我错了", "道歉", "别生气"])
        delta = (pos - neg) * 0.08
        return delta, conflict, apology

    # ── 状态更新（方案 §7.2）──
    def update(self, user_text: str, reply_text: str = "") -> EmotionState:
        s = self.state
        cfg = self.cfg
        now = time.time()
        delta, conflict, apology = self.signals(user_text)

        # 情绪：变化量受上轮同向加成（高兴时更容易被逗笑），再向 0 回拉
        inertia = 0.05 if (delta * s.mood) > 0 else 0.0
        s.mood = max(-1.0, min(1.0, s.mood + max(-cfg["mood_max_delta"],
                                                 min(cfg["mood_max_delta"], delta + inertia))))
        s.mood *= (1 - cfg["mood_decay"])
        if s.conflict:
            s.mood = min(s.mood, -0.15)   # 冲突期间锁负值，直到和解

        # 冲突标记：出现核按钮词 → 写入未解决冲突；道歉/示好并 mood 恢复 → 和解
        if conflict:
            s.conflict = user_text.strip()[:40]
        elif s.conflict and (apology or delta > 0.05):
            # 和解不是一瞬间：mood 回正才真正解除
            if s.mood > -0.1:
                s.conflict = ""

        # 好感：慢涨，单轮 +≤2；被冷落/冲突则缓慢下降
        if delta > 0:
            s.affection = min(100.0, s.affection + min(cfg["affection_max_gain"], delta * 12))
        elif delta < 0:
            s.affection = max(0.0, s.affection + max(-1.0, delta * 8))

        # 信任：下降快（-≤8）、上升慢（+≤1.5）
        if delta < 0:
            s.trust = max(0.0, s.trust + max(-cfg["trust_max_loss"], delta * 20))
        elif delta > 0 and not conflict:
            s.trust = min(100.0, s.trust + min(cfg["trust_max_gain"], delta * 6))

        # 精力：随本次会话时长衰减
        session_hours = (now - s.session_start) / 3600
        base_energy = 80.0 - session_hours * cfg["energy_decay_per_hour"]
        s.energy = max(0.0, min(100.0, base_energy))

        # 关系时间轴：按自然日去重累计——同一天聊一百轮也只算一天，
        # 防止"熬夜刷三小时就变成病娇"这种不真实的跃迁
        today = datetime.now().strftime("%Y-%m-%d")
        if s.first_meet_ts <= 0:
            s.first_meet_ts = now
        if s.last_active_day != today:
            s.active_days += 1
            s.last_active_day = today

        s.last_turn_ts = now
        self.save()
        return s

    def note_intimacy_event(self, event: str) -> None:
        """亲密跨段必须由事件触发（拉钩/赠礼/告白），不能靠刷轮数。"""
        self.state.last_intimacy_event = event
        self.state.intimacy = min(100.0, self.state.intimacy + 15.0)
        self.state.affection = min(100.0, self.state.affection + 3.0)
        self.save()

    def mark_degraded(self, degraded: bool) -> None:
        self.state.degraded_turns = self.state.degraded_turns + 1 if degraded else 0
        self.save()

    # ── 关系阶段（破冰曲线）──
    @property
    def relation_stage(self) -> dict:
        """当前关系阶段：三轴（互动天数/亲密/信任）同时满足的最高一级。

        逐级判断、满足即停——不能因为某一轴暴涨就跳级（例如亲密值刷到 90
        但只聊过两天，仍然停在破冰）。返回 RELATION_STAGES 中的一项。
        """
        s = self.state
        stage = RELATION_STAGES[0]
        for cand in RELATION_STAGES:
            if (s.active_days >= cand["active_days"]
                    and s.intimacy >= cand["intimacy"]
                    and s.trust >= cand["trust"]):
                stage = cand
            else:
                break  # 阶段必须连续满足，断了就停在这一级
        return stage

    def stage_sentence(self) -> str:
        """当前阶段的一句话关系底色（注入系统提示尾部；约 20 字，不占 L3 预算）。"""
        key = self.relation_stage["key"]
        card = _load_stage_cards().get(key) or {}
        sentence = card.get("sentence", "")
        return sentence if isinstance(sentence, str) else ""

    # ── 自然语言投影（方案 §7.3）：数值绝不直塞 ──
    def project(self) -> str:
        """把五维 + 冲突翻译成 60-100 字的"她此刻的状态"，作为 L3 注入。"""
        s = self.state
        parts: list[str] = []

        if s.conflict:
            parts.append(f"你们之间还横着一件事：{s.conflict}——你不想先低头。")
        if s.mood <= -0.4:
            parts.append("你现在有点闷，不太想说话，但也不想他察觉。")
        elif s.mood < -0.1:
            parts.append("你心里有点别扭，说话会带刺。")
        elif s.mood > 0.35:
            parts.append("你心情不错，嘴上不承认，但藏不住。")

        if s.affection >= 75:
            parts.append("你已经很依赖他了，怕被丢下，越依赖越装作不需要。")
        elif s.affection >= 55:
            parts.append("你已经很习惯他在身边，会主动关心，但还不会太黏。")
        elif s.affection >= 30:
            parts.append("你对他有点在意，但还在试探。")
        else:
            parts.append("你对他还有防备，说话客气而冷。")

        if s.trust < 30:
            parts.append("你不太敢信他的话，会忍不住想试探。")

        if s.energy <= 25:
            parts.append("你困了，打字变慢，句子变短。")
        elif s.energy <= 50:
            parts.append("你有点累，话比平时少。")

        text = "".join(parts)
        return text[:100]  # L3 预算 60-100 token（方案 §4.2）
