#!/usr/bin/env python3
"""冒烟测试：不依赖 Ollama，验证世界书触发/递归/预算、过滤器、节奏层、记忆库。

运行：python scripts/smoke_test.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.output.filters import filter_output          # noqa: E402
from app.output.rhythm import plan_delivery           # noqa: E402
from app.worldbook.engine import WorldbookEngine      # noqa: E402

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✓ {name}")
    else:
        FAIL += 1
        print(f"  ✗ {name} {detail}")


def main() -> int:
    print("== 世界书引擎 ==")
    wb = WorldbookEngine()
    check("条目加载 ≥ 20", len(wb.entries) >= 20, f"实际 {len(wb.entries)}")

    # 角色卡 lint（借鉴 ai-companion-kit：防人设写崩）
    warnings = wb.lint()
    check("角色卡 lint 无警告", not warnings, f"{warnings}")

    # 直接触发：夸她
    a = wb.scan("你今天真好看", [])
    ids = [x.entry.id for x in a.entries]
    check("夸她触发 L4-praise", "L4-praise" in ids, f"激活 {ids}")
    check("常驻层 L1/L2/L6 在场",
          any(x.entry.layer == "L1" for x in a.entries)
          and any(x.entry.layer == "L6" for x in a.entries))

    # 用户难过
    a2 = wb.scan("今天工作好累啊，压力好大", [])
    ids2 = [x.entry.id for x in a2.entries]
    check("难过触发 L4-user-sad", "L4-user-sad" in ids2, f"激活 {ids2}")

    # 递归链：下雨等伞（直接命中）→ 内容含"冷战" → 闹别扭被递归拉起
    a3 = wb.scan("昨天下雨，你在哪，不是说好来接我吗", [])
    ids3 = [(x.entry.id, x.source, x.depth) for x in a3.entries]
    rain = [t for t in ids3 if t[0] == "L4-rain-wait"]
    sulk = [t for t in ids3 if t[0] == "L4-sulking"]
    check("下雨直接命中 L4-rain-wait", bool(rain) and rain[0][1] == "direct", f"{ids3}")
    check("递归拉起 L4-sulking", bool(sulk) and sulk[0][1] == "recursive", f"{ids3}")

    # 预算硬顶
    check("总预算 ≤ 1600 token", a.total_tokens <= 1600, f"用了 {a.total_tokens}")
    check("预算裁剪记录", isinstance(a.dropped, list))

    # 深度回忆档位
    a4 = wb.scan("还记得我们之前拉过钩的事吗", [])
    check("深度回忆档位识别", a4.scene_mode in ("deep", "longrun"), a4.scene_mode)

    # 单字/高频词不应误触大量条目
    a5 = wb.scan("嗯", ["今天", "还好"])
    check("高频泛词不乱触发", len([x for x in a5.entries if x.entry.layer == "L4"]) <= 1,
          f"激活过多：{[x.entry.id for x in a5.entries]}")

    print("== 输出过滤 ==")
    r = filter_output("作为一个AI，我无法回答这个问题。")
    check("暴露设定被拦截", r.blocked and "有病" in r.text or "什么意思" in r.text or "变态" in r.text
          or "不太懂" in r.text or "这样" in r.text or "闭嘴" in r.text or "试试" in r.text or "作业" in r.text,
          r.text)

    r2 = filter_output("希望对您有帮助，感谢您的咨询。")
    check("客服腔被拦截", r2.blocked, r2.text)

    long_text = "这句话特别长" * 40
    r3 = filter_output(long_text)
    check("超长被截断", len(r3.text) <= 190, f"长度 {len(r3.text)}")

    r4 = filter_output("（低头）（踢石子）（小声）嗯。（别过脸）（耳根红了）好。")
    check("旁白过多被裁剪", "旁白化" in r4.triggered, r4.triggered)

    r5 = filter_output("就是你啊就是你啊", history=["就是你啊就是你啊，哼"])
    check("复读触发重生成信号", r5.need_retry, r5.triggered)

    print("== 拟人化节奏 ==")
    segs = plan_delivery("笨死了，也就我管你。吃完去睡，现在。别熬了，听到没有。", mood=-0.2)
    check("分条数 1-4", 1 <= len(segs) <= 4, f"{len(segs)} 条")
    check("每条时长为正", all(s.typing_s > 0 for s in segs))
    check("末条停顿为 0", segs[-1].pause_s == 0.0)

    segs_night = plan_delivery("嗯。", mood=0.0, hour=3)
    check("深夜模式可运行", len(segs_night) >= 1)

    print("== 记忆库 ==")
    from app.memory.store import MemoryStore
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        store = MemoryStore(Path(td) / "t.db")
        store.add_message("user", "我妈叫王秀兰")
        store.add_memory("fact", "妈妈叫王秀兰", source="测试")
        store.add_memory("fact", "妈妈叫王秀兰", source="测试")  # 去重
        found = store.search_memories("王秀兰")
        check("记忆写入与去重", len(found) == 1 and found[0].mentions == 2, f"{found}")
        check("滑动窗口读取", store.recent_messages(5)[0]["content"] == "我妈叫王秀兰")
        store.close()

    print(f"\n结果：{PASS} 通过，{FAIL} 失败")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
