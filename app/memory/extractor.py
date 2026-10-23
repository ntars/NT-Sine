"""记忆抽取：每 3-5 轮用本地模型轻量抽取一次（方案 §6.3）。

四类目标：事实 / 偏好 / 事件 / 承诺。只抽结构化短句，绝不生成总结段落
（总结段落是幻觉的来源）。抽取失败静默跳过——主链路优先。
"""

from __future__ import annotations

import json
import re

from app.config import SETTINGS
from app.generation.client import Generator
from app.memory.store import MemoryStore

_SYSTEM = (
    "从对话里抽取值得记住的信息，只输出 JSON 数组，每项形如："
    '{"kind":"fact|preference|event|promise|pending|trait|topic","content":"30字内的短句","detail":{}}。'
    "pending = 他犹豫着没说出口的事；trait = 你观察到的他的性格/习惯（如\"常熬夜\"）；"
    "topic = 最近聊过的话题（滚动）。"
    # 人称约定：这些短句会被主动消息直接引用（"你上次说的X，别忘了"），
    # 用第三人称"他"引出来就穿帮了，所以统一用"你"。
    "content 里指代对方一律用\"你\"，不要用\"他\"，例如\"你周五要去面试\"。"
    "没有值得记的就输出 []。不要总结、不要感想、不要任何其他文字。"
)

# 每类记忆的容量上限（借鉴 ai-companion-kit memory.js：FIFO 淘汰，防账本无限膨胀）
KIND_CAPS = {"fact": 60, "preference": 40, "trait": 40, "topic": 20, "pending": 12,
             "event": 40, "promise": 30}

# ── 规则兜底 ──
# 为什么需要它：抽取原本完全依赖模型输出 JSON 数组，但实测微调后的模型连
# 「只输出 JSON」这条指令都做不到（回「……我去倒水。」），`chat_json` 解析失败
# 返回 None，`maybe_extract` 静默返回 0。于是五轮真实对话后 memories 表 0 条，
# 「不健忘」这条第一理念在真实链路上彻底不通，而接口上看不出任何异常。
#
# 规则法只抽高置信度的中文句式（我叫X / 我下周要去Y / 我讨厌Z），
# 不做推断、不补全、不造句——召回率低没关系，宁可漏记也不能编。
# 每条内容都统一用「你」指代对方，与模型抽取的人称约定保持一致。
_CONTENT_MAX = 30

# 人物/地点类取值，统一去掉句末语气词后再入册
_TAIL_NOISE = "了啊呀呢吧哦嘛哈"

# 动词前的插入语：「我特别喜欢吃」「我最近天天熬夜」。
# 不处理这一层，规则只能命中「我喜欢吃」这种教科书句式，真实聊天里几乎一次都撞不上。
_ADV = r"(?:很|特别|最|挺|也|一直|最近|平时|本来就|真的|其实|现在|这阵子)?"

# 疑问词排除：「我叫什么来着」「我最近在做什么」都是提问，不是陈述。
# 实测漏了这一层会把「你还记得我刚才说我叫什么吗？」抽成事实「你叫什么吗」，
# 脏数据会一直留在记忆表里被反复召回，比漏记更麻烦。
_NOT_Q = r"(?!什么|啥|谁|哪|怎么办|多久|多少)"

_RULES: list[tuple[str, re.Pattern]] = [
    # 称呼：我叫小玖 / 我的名字是小玖
    ("fact", re.compile(rf"我(?:叫|名字是){_NOT_Q}([\u4e00-\u9fa5A-Za-z0-9]{{1,10}})")),
    # 住处：我住在成都 / 我家在成都
    ("fact", re.compile(rf"我(?:住在|家在){_NOT_Q}([\u4e00-\u9fa5]{{2,10}})")),
    # 职业/在做的事：我在做模型 / 我在做一个小程序
    ("fact", re.compile(rf"我{_ADV}(?:在|正在)(?:做|弄|搞|开发|写){_NOT_Q}([\u4e00-\u9fa5A-Za-z0-9]{{2,14}})")),
    # 行程：我下周五要去杭州 / 我明天得去一趟上海
    ("event", re.compile(
        rf"我{_ADV}(?:下|这|上)?(?:周|星期|礼拜)?[一二三四五六日天]?(?:要去|得去|要飞|要回|去){_NOT_Q}([^，。！？,.!?\s]{{2,15}})")),
    # 偏好：我喜欢美式 / 我爱吃辣
    ("preference", re.compile(rf"我{_ADV}(?:喜欢|爱喝|爱吃|爱听){_NOT_Q}([^，。！？,.!?\s]{{1,12}})")),
    # 厌恶：我讨厌下雨 / 我不喝可乐
    ("preference", re.compile(rf"我{_ADV}(?:讨厌|不喜欢|不喝|不吃|不抽){_NOT_Q}([^，。！？,.!?\s]{{1,12}})")),
    # 习惯：我天天熬夜 / 我经常失眠
    ("trait", re.compile(rf"我{_ADV}(?:每天|天天|经常|总是|一直){_NOT_Q}([^，。！？,.!?\s]{{2,15}})")),
]

# 「我是…」单独处理：这个词太容易撞上「我是说」「我是来问的」，只在后面跟
# 明确的身份名词时才认，避免把口头禅记成事实。
_IDENTITY = re.compile(r"我是([\u4e00-\u9fa5]{2,6})(?:，|。|！|？|,|$)")


def extract_by_rules(dialogue: str) -> list[dict]:
    """用固定句式从对话里抽结构化记忆，抽不到返回空表。

    dialogue 形如 "user: 我叫小玖\\nassistant: ……嗯。"，
    只扫 user 那几行——她的台词是生成出来的，不能当事实来源。
    """
    user_lines = [ln.split(":", 1)[1] for ln in dialogue.splitlines()
                  if ln.lstrip().startswith("user:") and ":" in ln]
    text = " ".join(user_lines)
    if not text:
        return []

    found: list[dict] = []
    seen: set[str] = set()

    def push(kind: str, raw: str) -> None:
        content = raw.strip(_TAIL_NOISE).strip()
        if not content or len(content) > _CONTENT_MAX:
            return
        # 称呼类要拼成完整句子，其余按句式补主语，保持与模型抽取同一种话术
        if kind == "fact" and raw == content and re.search(r"我叫|名字是", text):
            phrase = f"你叫{content}" if f"我叫{content}" in text else f"你是{content}"
        elif kind == "fact" and f"我住在{content}" in text:
            phrase = f"你住在{content}"
        elif kind == "fact" and f"我家在{content}" in text:
            phrase = f"你家在{content}"
        elif kind == "event":
            phrase = f"你要去{content}"
        elif kind == "preference" and re.search(rf"我(?:讨厌|不喜欢|不喝|不吃|不抽){re.escape(content)}", text):
            phrase = f"你不喜欢{content}"
        elif kind == "preference":
            phrase = f"你喜欢{content}"
        elif kind == "trait":
            phrase = f"你{content}"
        else:
            phrase = f"你在{content}"
        if phrase in seen:
            return
        seen.add(phrase)
        found.append({"kind": kind, "content": phrase[:_CONTENT_MAX], "detail": {}})

    for kind, pattern in _RULES:
        for match in pattern.finditer(text):
            push(kind, match.group(1))

    for match in _IDENTITY.finditer(text):
        candidate = match.group(1)
        if re.fullmatch(r"[\u4e00-\u9fa5]{2,6}", candidate):
            phrase = f"你是{candidate}"
            if phrase not in seen:
                seen.add(phrase)
                found.append({"kind": "fact", "content": phrase, "detail": {}})

    return found[:8]  # 一轮最多记 8 条，防止长段自述把记忆表刷爆


class MemoryExtractor:
    def __init__(self, store: MemoryStore, generator: Generator | None = None):
        self.store = store
        self.generator = generator
        self.cfg = SETTINGS["memory"]
        self._turn_counter = 0

    def maybe_extract(self, recent_messages: list[dict]) -> int:
        """每 N 轮抽一次；返回本次写入的记忆条数。Ollama 不可用时返回 0。"""
        self._turn_counter += 1
        if self._turn_counter % int(self.cfg["extract_every_n_turns"]) != 0:
            return 0
        # 配置项写的是"轮"，一轮 = 他一句 + 她一句。原先按条数取 6 条，
        # 实际只有 3 轮，与 3.6 修过的窗口语义是同一个错：名字说轮、实现按条。
        rounds = max(1, int(self.cfg["extract_recent_rounds"]))
        window = recent_messages[-rounds * 2:]
        dialogue = "\n".join(f"{m['role']}: {m['content']}" for m in window)

        data = None
        if self.generator is not None:
            data = self.generator.chat_json(
                [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": dialogue}],
                num_ctx=2048,
            )
        # 模型抽不出来（返回 None / 非数组 / 空数组）时走规则兜底。
        # 这一步是必需的：实测微调模型连「只输出 JSON」都做不到，
        # 没有兜底就等于记忆功能不存在，而失败完全不可见。
        if not isinstance(data, list) or not data:
            data = extract_by_rules(dialogue)

        if not isinstance(data, list):
            return 0
        count = 0
        for item in data:
            if not isinstance(item, dict):
                continue
            kind = item.get("kind", "")
            content = str(item.get("content", "")).strip()
            if kind not in KIND_CAPS or not content:
                continue
            detail = item.get("detail") if isinstance(item.get("detail"), dict) else {}
            self.store.add_memory(kind, content, source=dialogue[-200:], detail=detail)
            if kind == "event":
                self.store.add_timeline(content, mood_tag="event")
            count += 1
        self.prune()
        return count

    def prune(self) -> None:
        """按 KIND_CAPS 淘汰最旧的低价值记忆（FIFO，mention 多的不动）。"""
        for kind, cap in KIND_CAPS.items():
            kept = self.store.memories_by_kind(kind, limit=10_000)
            if len(kept) > cap:
                # memories_by_kind 按 last_ts 降序，砍掉尾部最旧的
                for m in kept[cap:]:
                    self.store.update_memory_status(m.id, "dropped")


def build_profile(store: MemoryStore) -> dict[str, str]:
    """用户画像滚动更新：SQLite 聚合即可，不调模型（方案 §6.4 改动一）。"""
    facts = [m.content for m in store.memories_by_kind("fact", limit=10)]
    prefs = [m.content for m in store.memories_by_kind("preference", limit=10)]
    if facts:
        store.set_profile("facts", "；".join(facts))
    if prefs:
        store.set_profile("preferences", "；".join(prefs))
    return store.all_profile()
