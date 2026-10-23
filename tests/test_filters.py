"""输出过滤层专项测试。

这里的用例来自 2026-10-24 的代码审计。共同点是「旧测试全绿、线上却不生效」：
原有的复读用例各只喂一条历史，`history[:3]` 与 `history[-3:]` 结果恰好相同，
顺序缺陷与空文本缺陷都被绕过去了。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.output.filters import filter_output


# ── 复读检测的比对口径 ──
def test_repeat_checked_against_recent_turns():
    """复读要比她最近说过的话，不是比窗口里最老的三条。

    main.py 把滑动窗口按「旧 → 新」传进来，越新的越靠后。早前实现用
    `history[:3]` 取到的是最老三条，她刚说过的同一句根本不在比对范围里，
    等于这道防线在真实链路上从没生效。
    """
    history = ["水在桌上。", "面下好了，先吃。", "碗放着，我来洗。",
               "三十七度八。", "药在抽屉第二格。", "……你别把这事说出去。"]
    result = filter_output("……你别把这事说出去。", history=history)
    assert "复读" in result.triggered, f"最近一句原样复读没被拦下：{result.triggered}"
    assert result.need_retry, "判出复读就应要求重生成一次"


def test_repeat_ignores_distant_history():
    """窗口最老的那几条不该再参与比对——它们已经隔了十几句。"""
    history = ["……你别把这事说出去。", "水在桌上。", "面下好了，先吃。",
               "三十七度八。", "药在抽屉第二格。", "碗放着，我来洗。"]
    result = filter_output("……你别把这事说出去。", history=history)
    assert "复读" not in result.triggered, f"最老的三条不该参与比对：{result.triggered}"


# ── 空正文 ──
def test_think_only_is_marked_empty():
    """只吐思考块、没有正文时，要标记重来一次，而不是把空串交给前端。

    播放层拿到空串会播出一段「什么都没说」的空白；这也是 README 里
    「约 8% 空返回」的一部分来源——不是模型没输出，是被剥成了空。
    """
    result = filter_output("<think>\n\n</think>\n\n")
    assert result.text == "", result.text
    assert "空回复" in result.triggered, result.triggered
    assert result.need_retry, "空正文应要求重生成"


def test_narration_only_is_marked_empty():
    """整段都是括号旁白时，裁完什么都不剩，同样算空回复。"""
    result = filter_output("（她低头）（把杯子推过来）（没有再看他）")
    assert result.text == "", result.text
    assert "空回复" in result.triggered, result.triggered
