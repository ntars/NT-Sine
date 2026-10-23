"""世界书引擎：六层条目、中文触发、仅 L4 递归、预算裁剪、上下文拼装。

设计依据：《Sine-2.1-完整方案》§4。核心立场——
条目是"事实卡"不是"小作文"；触发词是"回忆的钩子"；
递归是"联想"不是"展开"，只允许 L4 记忆事件层递归，最大递归步骤 = 3。
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from app.config import SETTINGS, WORLDBOOK_DIR

# ── token 估算 ──────────────────────────────────────────────
_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]")


def estimate_tokens(text: str) -> int:
    """保守估算：1 汉字/全角符号 ≈ 1 token，西文按 4 字符 ≈ 1 token。

    与方案 §4.6 的验算口径一致（宁可高估，避免预算超限被截断）。
    精确值以 Qwen tokenizer 为准，此处只用于预算裁剪。
    """
    if not text:
        return 0
    cjk = len(_CJK_RE.findall(text))
    other = len(text) - cjk
    return cjk + math.ceil(other / 4)


# ── 数据结构 ────────────────────────────────────────────────
@dataclass
class Entry:
    """一条世界书条目（六层通用）。"""

    id: str
    layer: str                 # L1..L6
    content: str               # 事实短句，30-80 字
    order: int                 # 同层内排序，越小越靠前（常驻层 Order：L1<6<L2<5<3）
    priority: int = 0          # 激活冲突时的优先级
    triggers: list[str] = field(default_factory=list)   # 关键词触发
    regex_triggers: list[str] = field(default_factory=list)  # 正则兜底（≤20%）
    recursive: bool = False    # 仅 L4 为 True（可被递归拉起）
    stop_recursion: bool = False  # 命中后是否阻止继续传染（L1/L2/L6 为 True）
    always: bool = False       # 常驻注入（L1/L2/L6）

    @staticmethod
    def from_dict(d: dict) -> "Entry":
        return Entry(
            id=d["id"],
            layer=d["layer"],
            content=d["content"],
            order=int(d.get("order", 100)),
            priority=int(d.get("priority", 0)),
            triggers=list(d.get("triggers", [])),
            regex_triggers=list(d.get("regex_triggers", [])),
            recursive=bool(d.get("recursive", False)),
            stop_recursion=bool(d.get("stop_recursion", False)),
            always=bool(d.get("always", False)),
        )


@dataclass
class ActivatedEntry:
    """激活记录：条目 + 激活方式 + 深度（0=直接命中，1/2=递归层）。"""

    entry: Entry
    source: str   # "direct" / "recursive"
    depth: int    # 0 直接；1、2 递归
    tokens: int


@dataclass
class Assembly:
    """一次扫描的产物：按插入顺序排好的条目 + 预算信息。"""

    entries: list[ActivatedEntry] = field(default_factory=list)
    total_tokens: int = 0
    dropped: list[str] = field(default_factory=list)  # 因预算被淘汰的条目 id
    suppressed: list[str] = field(default_factory=list)  # 因 L4 互斥被压制的条目 id
    scene_mode: str = "normal"
    degraded: bool = False


# ── 引擎 ────────────────────────────────────────────────────
class WorldbookEngine:
    """世界书引擎。数据来自 data/worldbook/*.json；参数来自 settings.yaml。"""

    # 最终插入顺序的层前缀：L1(30) → L6(40) → L2(50) → L4(55) → L5(60)。
    # L4 必须显式写出来：原先缺这一项，回落成 80，被排到 L5 之后——
    # 与方案 §4.5 的"记忆事件在前、当前场景在后"正好相反。现在世界书里
    # 没有 L5 条目（场景由 context.build_l5_scene 动态生成），所以顺序错位
    # 还没显形；一旦补上 L5 条目就会静默反序，必须先把契约钉住。
    LAYER_PREFIX = {"L1": 30, "L6": 40, "L2": 50, "L4": 55, "L5": 60, "L3": 70}

    def __init__(self, data_dir: Path | None = None):
        self.cfg = SETTINGS["worldbook"]
        if self.cfg.get("whole_word_match", False):
            # 方案 §4.3.1：中文必须关闭"匹配完整单词"，引擎层强制，不可配置打开
            raise ValueError("whole_word_match 必须为 false：中文分词无空格，开启会导致触发器全部失效")
        self.stopwords = set(self.cfg.get("stopwords", []))
        self.short_trigger_len = int(self.cfg.get("short_trigger_len", 2))
        self.regex_ratio_max = float(self.cfg.get("regex_ratio_max", 0.2))
        self.entries: list[Entry] = []
        self.load(data_dir or WORLDBOOK_DIR)

    # ── 加载 ──
    def load(self, data_dir: Path) -> None:
        """按需加载全部层文件（data/worldbook/*.json）。"""
        self.entries = []
        for path in sorted(Path(data_dir).glob("*.json")):
            raw = json.loads(path.read_text(encoding="utf-8"))
            items = raw if isinstance(raw, list) else raw.get("entries", [])
            for item in items:
                self.entries.append(Entry.from_dict(item))
        # 正则条目占比校验（方案 §4.3.3：≤20%，太多会"什么都命中"）
        with_regex = [e for e in self.entries if e.regex_triggers]
        if self.entries and len(with_regex) / len(self.entries) > self.regex_ratio_max:
            raise ValueError(
                f"正则条目占比 {len(with_regex)}/{len(self.entries)} 超过上限 {self.regex_ratio_max}"
            )
        # 递归属性校验：只有 L4 可递归（方案 §4.4.2）
        for e in self.entries:
            if e.recursive and e.layer != "L4":
                raise ValueError(f"条目 {e.id}: 只有 L4 记忆事件层允许递归")

    # ── 触发匹配 ──
    def _trigger_hit(self, trigger: str, text: str, recent_text: str) -> bool:
        """中文子串匹配（关闭整词边界）。短触发词需出现在最近 2 条消息内。"""
        trigger = trigger.strip()
        if not trigger or trigger in self.stopwords:
            return False
        if len(trigger) <= self.short_trigger_len:
            return trigger in recent_text
        return trigger in text

    def _match_entry(self, entry: Entry, text: str, recent_text: str) -> bool:
        for t in entry.triggers:
            if self._trigger_hit(t, text, recent_text):
                return True
        for pattern in entry.regex_triggers:
            if re.search(pattern, text):
                return True
        return False

    # ── 场景档位 ──
    def detect_scene(self, text: str) -> str:
        """按用户话里的回忆性说法选档位（方案 §4.6.4）。"""
        deep_hits = sum(1 for t in SETTINGS["scene_profiles"]["deep_triggers"] if t in text)
        if deep_hits >= 1 and len(text) <= 30:
            return "deep"
        if deep_hits >= 1:
            return "longrun"
        if len(text) >= 40:
            return "normal"
        return "daily"

    # ── 核心：扫描 + 递归 + 预算 ──
    def scan(self, text: str, history: list[str], scene_mode: str | None = None,
             degraded: bool = False) -> Assembly:
        """扫描输入文本，返回预算内的激活条目。

        history: 最近若干条原始消息（旧 → 新），用于扫描深度与短触发词判定。
        degraded: 上一轮检测到"背设定/旁白化/复读"时传 True，深度降一档、L4 上限减 1。
        """
        profiles = SETTINGS["scene_profiles"]
        mode = scene_mode or self.detect_scene(text)
        profile = profiles.get(mode, profiles[profiles["default"]])
        depth = int(profile["scan_depth"])
        max_steps = int(profile["max_recursion_steps"])
        l4_cap = int(profile["l4_cap"])
        if degraded:
            # 运行时自愈（方案 §4.6.4 降级机制）
            depth = max(3, depth - 2)
            l4_cap = max(2, l4_cap - 1)
            mode = f"{mode}(降级)"

        scan_messages = history[-depth:] + [text]
        scan_text = "\n".join(scan_messages)
        # 短触发词上下文窗口：最近 2 条消息
        recent_text = "\n".join(history[-2:] + [text])

        direct_cap = int(self.cfg["direct_cap"])
        always = [e for e in self.entries if e.always]
        pool = [e for e in self.entries if not e.always]

        # 第 0 轮：直接关键词命中
        activated: list[ActivatedEntry] = []
        activated_ids: set[str] = set()
        direct_hits: list[Entry] = []
        for e in pool:
            if self._match_entry(e, scan_text, recent_text):
                direct_hits.append(e)
        direct_hits.sort(key=lambda e: (-e.priority, e.order))
        # L4 单独限流，其他层（如 L5 的条件条目）按优先级取
        l4_direct = [e for e in direct_hits if e.layer == "L4"]
        other_direct = [e for e in direct_hits if e.layer != "L4"]
        # L4 互斥：同一时刻只保留优先级最高的一张场景卡。
        # 原因：多卡并行会让模型串味（v2 实测 sad+leaving 同时命中 → 用了重逢语气）。
        # 优先级高的胜出（如 user-sad 65 > leaving 55），其余压制并记入 suppressed。
        suppressed: list[str] = []
        if self.cfg.get("l4_exclusive", True) and l4_direct:
            top_pri = max(e.priority for e in l4_direct)
            for e in l4_direct:
                if e.priority < top_pri:
                    suppressed.append(e.id)
            l4_direct = [e for e in l4_direct if e.priority >= top_pri]
        for e in other_direct:
            activated.append(ActivatedEntry(e, "direct", 0, estimate_tokens(e.content)))
            activated_ids.add(e.id)
        for e in l4_direct[:l4_cap]:
            activated.append(ActivatedEntry(e, "direct", 0, estimate_tokens(e.content)))
            activated_ids.add(e.id)

        # 递归：只有 L4 可递归；最大递归步骤 = 3 → 允许两轮递归
        direct_l4_count = sum(1 for a in activated if a.entry.layer == "L4")
        recursive_added = 0
        for round_no in range(1, max_steps):
            # 关联前沿只取 L4 场景卡的内容：L1/L6 的常驻文本含大量边界词，
            # 拿来当关联源会误召无关场景卡（实测 L6 里的"生气/不理你"误召 L4-sulking）
            frontier_text = "\n".join(a.entry.content for a in activated
                                       if a.entry.recursive and a.entry.layer == "L4")
            if not frontier_text:
                break
            added_this_round = 0
            for e in pool:
                if e.id in activated_ids or e.layer != "L4" or not e.recursive:
                    continue
                if added_this_round >= int(self.cfg["recursion_cap_per_round"]):
                    break
                if direct_l4_count + recursive_added >= l4_cap:
                    break
                # 先判有没有命中，再判要不要压制。顺序反了会把同层所有低优先级卡
                # 都记成"被压制"——实测「下雨等伞」的压制名单里塞进了 drunk/money/cat，
                # 那些卡压根没被触发过。调试台的数字必须对应真事。
                if not self._match_entry(e, frontier_text, frontier_text):
                    continue
                if self.cfg.get("l4_exclusive", True):
                    # 互斥：递归也不得引入比当前场景卡优先级低的卡（否则同样串味）
                    cur_pri = max([a.entry.priority for a in activated
                                   if a.entry.layer == "L4"] or [0])
                    if e.priority < cur_pri:
                        suppressed.append(e.id)
                        continue
                activated.append(ActivatedEntry(e, "recursive", round_no,
                                                estimate_tokens(e.content)))
                activated_ids.add(e.id)
                added_this_round += 1
                recursive_added += 1
            if added_this_round == 0:
                break  # 无新联想，链到头

        # 互斥兜底：直接命中与递归加入的 L4 合起来，最终只留优先级最高的一张。
        # 理由：训练样本里每段对话只带一张场景卡，多卡注入属于分布外 → 模型串味。
        if self.cfg.get("l4_exclusive", True):
            l4_acts = [a for a in activated if a.entry.layer == "L4"]
            if len(l4_acts) > 1:
                # 规则：直接命中之间互斥（只留优先级最高的一张，串味主因）；
                # 递归关联最多再留 1 张——它由主卡内容联想而来，是原设计的一部分
                #（例：下雨等伞 → 内容含"冷战" → 闹别扭）。
                directs = [a for a in l4_acts if a.source == "direct"]
                recs = [a for a in l4_acts if a.source == "recursive"]
                if directs:
                    main_card = max(directs, key=lambda a: a.entry.priority)
                    keep_l4 = [main_card] + sorted(recs, key=lambda a: -a.entry.priority)[:1]
                else:
                    keep_l4 = [max(l4_acts, key=lambda a: a.entry.priority)]
                for a in l4_acts:
                    if a not in keep_l4:
                        suppressed.append(a.entry.id)
                activated = [a for a in activated if a.entry.layer != "L4"] + keep_l4

        # 常驻层不参与触发，直接进入预算（L3 情感状态由外层动态生成，不在此处）
        always_acts = [ActivatedEntry(e, "direct", 0, estimate_tokens(e.content)) for e in always]

        # 预算裁剪（方案 §4.7 保留顺序）：
        # L1 → L6 → L3(外层) → 冲突(外层) → L2 → L4 直接 → L5 → L4 递归
        def keep_priority(a: ActivatedEntry) -> int:
            if a.entry.layer == "L1":
                return 0
            if a.entry.layer == "L6":
                return 1
            if a.entry.layer == "L2":
                return 4
            if a.entry.layer == "L4" and a.source == "direct":
                return 5
            if a.entry.layer == "L5":
                return 6
            return 7  # L4 递归最易变噪声，最先被淘汰

        all_acts = always_acts + activated
        all_acts.sort(key=lambda a: (keep_priority(a), a.entry.order))

        layer_budget = {k: int(v) for k, v in self.cfg["layer_budget"].items()}
        layer_used: dict[str, int] = {k: 0 for k in layer_budget}
        kept: list[ActivatedEntry] = []
        dropped: list[str] = []
        total = 0
        for a in all_acts:
            lb = layer_budget.get(a.entry.layer, 0)
            if layer_used.get(a.entry.layer, 0) + a.tokens > lb or total + a.tokens > int(self.cfg["total_budget"]):
                dropped.append(a.entry.id)
                continue
            kept.append(a)
            layer_used[a.entry.layer] = layer_used.get(a.entry.layer, 0) + a.tokens
            total += a.tokens

        # 按最终插入顺序排列：L1(30) → L6(40) → L2(50) → L4 → L5(60)
        kept.sort(key=lambda a: (self.LAYER_PREFIX.get(a.entry.layer, 80),
                                 0 if a.entry.layer == "L4" and a.source == "direct" else 1,
                                 a.entry.order))
        return Assembly(entries=kept, total_tokens=total, dropped=dropped,
                        suppressed=suppressed, scene_mode=mode, degraded=degraded)

    # ── 常驻层快捷读取（供拼装与调试） ──
    def layer(self, name: str) -> list[Entry]:
        return sorted([e for e in self.entries if e.layer == name], key=lambda e: e.order)

    # ── 角色卡 lint（借鉴 ai-companion-kit 的 role.js lint：防人设写崩）──
    def lint(self) -> list[str]:
        """对世界书条目做写法体检，返回警告列表（空 = 健康）。

        检查项：常驻锚点没写"怎么说话"、出现助手/客服字样、
        条目过长、边界层没写拒绝方式。
        """
        warnings: list[str] = []
        l1 = self.layer("L1")
        l6 = self.layer("L6")
        if not l1:
            warnings.append("缺少 L1 角色锚点：没人设的世界书等于没人格")
        else:
            joined = "".join(e.content for e in l1)
            if not any(k in joined for k in ("说话", "语气", "口吻", "句子", "短句")):
                warnings.append("L1 没写「怎么说话」——这是最影响像不像人的部分")
        if not l6:
            warnings.append("缺少 L6 边界规则")
        elif not any(k in e.content for e in l6 for k in ("拒绝", "角色内", "岔开")):
            warnings.append("L6 没写「拒绝怎么做」——边界缺失会导致出戏或越界")
        for e in self.entries:
            if len(e.content) > 400:
                warnings.append(f"{e.id}: 单条 {len(e.content)} 字，超过 400——"
                                f"世界书条目是事实卡不是小作文")
            if any(k in e.content for k in ("助手", "客服", "助理")) and e.layer in ("L1", "L6"):
                warnings.append(f"{e.id}: 出现「助手/客服」字样，容易带出客服腔")
            if e.layer in ("L1", "L2", "L6") and not e.stop_recursion:
                warnings.append(f"{e.id}: 常驻层应设 stop_recursion（人格不是钩子）")
        return warnings
