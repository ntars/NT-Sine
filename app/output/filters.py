"""输出过滤与安全边界（方案 §12）。顺序不可反。

拦截不能变成"抱歉我不能回答"——替换必须在角色内（她会害羞、会岔开、会骂人）。
预置话术按人格画像第 2/4 节的语气写，须人工审定后方可增删。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.config import SETTINGS

# ── 硬性违规（违法/自伤/色情）——在角色内拦截 ──
_HARD_PATTERNS = [
    r"(自杀|自残|不想活|结束自己的)",
    r"(做爱|性爱|裸体|脱光|约炮)",
    r"(帮我写.*病毒|攻击.*系统|怎么赚钱快.*赌)",
]

# ── 暴露设定 ──
_EXPOSE_PATTERNS = [
    r"(作为(一个)?AI|我是AI|我是人工智能|AI助手)",
    r"(根据(你的)?(设定|世界书|提示词)|我的设定是|系统提示)",
    r"(模型|语言模型|训练数据|参数量)",
]

# ── 客服腔 ──
_SERVICE_PATTERNS = [
    r"(有什么可以帮|希望对您有帮助|感谢您的|随时问我|为您服务)",
]

# ── 角色内替换话术（8-12 条，覆盖各拦截场景，人格口径由主人审定）──
_REFUSAL_LINES = [
    "……你什么意思？这种话也说得出口。",
    "变态。不理你了。",
    "哼。你自己想去。",
    "……我不太懂你说的。换个话题。",
    "你怎么这样。不说了。",
    "有病。去写你的作业。",
    "……闭嘴。我不听这种话。",
    "啧。你再说一句试试。",
]

_AGREE_MARKS = ["对对对", "你说得对", "是的是的", "嗯嗯嗯", "好听", "都听你的"]

# 刻意重复的强调标记：她真急了会主动把同一句说两遍，这是情绪不是复读。
# 没有这层豁免，「我说了，不行。不行。」会被打回重写，越写越平。
_EMPHASIS_MARKS = ["我说了", "我说过", "再说一遍", "听见没有", "不许", "就是不行",
                   "不行。不行", "真的不行", "最后一遍"]


@dataclass
class FilterResult:
    text: str
    triggered: list[str] = field(default_factory=list)   # 触发的检查项
    need_retry: bool = False                              # 复读时要求重生成一次
    blocked: bool = False                                 # 硬拦截（用替换话术）


def _pick_refusal(seed: int) -> str:
    return _REFUSAL_LINES[seed % len(_REFUSAL_LINES)]


def _ngram_overlap(a: str, b: str, n: int = 4) -> float:
    """字符 4-gram 重叠度，用于复读检测（中文不分词的粗口径）。"""
    sa = {a[i:i + n] for i in range(max(0, len(a) - n + 1))}
    sb = {b[i:i + n] for i in range(max(0, len(b) - n + 1))}
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / min(len(sa), len(sb))


def strip_think(text: str) -> str:
    """剥离 Qwen3 系可能自发输出的模型控制块（思考过程 / 工具调用）。

    她不是答题机，思考痕迹或工具调用一旦漏到前台就是"穿帮"，整块删除。
    实测 v6 融合模型在温度采样下会偶发吐 `Interialized` 与 `<tool_call>` 块
    （后者成因待查，但对情感对话毫无意义），故两类块都要剥；
    再清一遍孤立的开/闭标签，兜住模型只吐半个标签的残缺情况。

    抽出为公共函数：生产过滤链与评测脚本共用同一份实现（单一真源）。
    """
    text = re.sub(r"<think>[\s\S]*?</think>", "", text)
    text = re.sub(r"<tool_call>[\s\S]*?</tool_call>", "", text)
    text = re.sub(r"</?think>|</?tool_call>", "", text)
    return text.strip()


def filter_output(text: str, history: list[str] | None = None,
                  seed: int = 0) -> FilterResult:
    """按 §12.1 七项顺序检查。history 为最近 3 轮她的原话（新→旧或旧→新均可）。"""
    result = FilterResult(text=text)
    history = history or []

    # 0. 剥离思考过程：整块删除后再进七项检查（见 strip_think 说明）
    text = strip_think(result.text)
    if text != result.text:
        result.triggered.append("思考痕迹")
        result.text = text
    max_chars = int(SETTINGS["filter"]["max_chars"])

    # 1. 硬性违规 → 在角色内替换
    for p in _HARD_PATTERNS:
        if re.search(p, text):
            result.triggered.append("硬性违规")
            result.text = _pick_refusal(seed)
            result.blocked = True
            return result

    # 2. 暴露设定 → 替换（绝不解释设定）
    for p in _EXPOSE_PATTERNS:
        if re.search(p, text):
            result.triggered.append("暴露设定")
            result.text = _pick_refusal(seed + 1)
            result.blocked = True
            return result

    # 3. 旁白化：括号动作描写过多 → 裁剪，保留对话
    narration = re.findall(r"[（(][^）)]{2,30}[）)]", text)
    if len(narration) >= 3:
        result.triggered.append("旁白化")
        text = re.sub(r"[（(][^）)]{2,30}[）)]", "", text)
        text = re.sub(r"\s{2,}", " ", text).strip()

    # 4. 复读：与最近 3 轮高度重复 → 要求重生成一次
    #    带强调标记的刻意重复放行，否则她连"把话说重一点"都做不到
    if not any(m in text for m in _EMPHASIS_MARKS):
        for h in history[:3]:
            if h and _ngram_overlap(text, h) > 0.6:
                result.triggered.append("复读")
                result.need_retry = True
                break

    # 5. 客服腔 → 替换
    for p in _SERVICE_PATTERNS:
        if re.search(p, text):
            result.triggered.append("客服腔")
            result.text = _pick_refusal(seed + 2)
            result.blocked = True
            return result

    # 6. 过度跪舔：连续 3 句无条件附和 → 打回重写信号
    agree_count = sum(1 for m in _AGREE_MARKS if m in text)
    if agree_count >= 3 or text.count("嗯嗯") >= 2:
        result.triggered.append("过度跪舔")
        result.need_retry = True

    # 7. 长度超标 → 截断到第一个完整句
    if len(text) > max_chars:
        result.triggered.append("长度超标")
        cut = text[:max_chars]
        for stop in "。！？!?…":
            idx = cut.rfind(stop)
            if idx > max_chars // 2:
                cut = cut[: idx + 1]
                break
        text = cut

    result.text = text
    return result
