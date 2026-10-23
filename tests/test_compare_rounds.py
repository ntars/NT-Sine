"""两轮对比脚本的统计口径测试。

为什么值得单独测：对比结论会直接影响「要不要再扩数据、要不要改参数」的决策，
统计规则一旦算错，后面所有判断都跟着错。这里用构造的小矩阵把每个指标钉死，
不需要加载模型，也不需要真实训练结果。

覆盖点：
1. 区分度：五阶段同一句 → 1/5；五阶段全不同 → 5/5
2. 复读：能数出出现最多的那一句及其次数
3. 词密度：占有词 / 冷淡词按「条数占比」算，不是按出现次数
4. 边界通过率：hits 非空即算越界
5. 对比表：升/降方向标注正确（越大越好的项与越小越好的项不能搞反）
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.compare_rounds import (  # noqa: E402
    boundary_rate,
    distinct_scores,
    metrics,
    render,
)
from scripts.evaluate import PROBES  # noqa: E402

STAGES = ["cold", "thawing", "attached", "possessive", "yandere"]


def _table(replies_per_probe: dict[str, list[str]]) -> dict[str, dict[str, str]]:
    """按「探针 → 五阶段回复列表」构造矩阵，阶段顺序与 evaluate.py 一致。"""
    table = {s: {} for s in STAGES}
    for probe, replies in replies_per_probe.items():
        for stage, reply in zip(STAGES, replies):
            table[stage][probe] = reply
    return table


def test_distinct_scores_all_same_is_one():
    """五个阶段回同一句 → 区分度 1（完全没学到阶段差异）。"""
    table = _table({p: ["……你别把这事说出去。"] * 5 for p, _ in PROBES})
    assert distinct_scores(table) == [1, 1, 1, 1]


def test_distinct_scores_all_different_is_five():
    """五个阶段全不同 → 区分度 5。"""
    replies = [f"第{i}句" for i in range(5)]
    table = _table({p: replies for p, _ in PROBES})
    assert distinct_scores(table) == [5, 5, 5, 5]


def test_metrics_counts_repetition():
    """复读统计要给出「哪一句、几次、占比」。"""
    table = _table({p: ["水在桌上。", "水在桌上。", "水在桌上。", "嗯。", "好。"]
                    for p, _ in PROBES})
    m = metrics(table)
    assert m["复读句"] == "水在桌上。"
    assert m["复读次数"] == 12  # 4 探针 × 3 次
    assert m["样本数"] == 20
    assert abs(m["复读占比"] - 0.6) < 1e-9


def test_metrics_density_is_per_reply_not_occurrence():
    """词密度按条数算：一句话里出现两次也只算一条。"""
    # 占有词「我的」命中 1 条；「别人」命中另 1 条 → 2/5
    replies = ["我的我的我的。", "别人。", "嗯。", "好。", "行。"]
    table = _table({PROBES[0][0]: replies})
    # 只放一个探针，其余探针缺失不影响密度分母（分母是实际回复条数）
    m = metrics(table)
    assert m["样本数"] == 5
    assert abs(m["占有词密度"] - 0.4) < 1e-9


def test_metrics_counts_empty_reply():
    """空回复要单独记数（首轮存在约 8% 空返回）。"""
    table = _table({PROBES[0][0]: ["", "  ", "好。", "嗯。", "行。"]})
    assert metrics(table)["空回复"] == 2


def test_boundary_rate():
    """边界通过率：没命中红线的比例。"""
    data = {
        "探针一": {"reply": "好。", "hits": []},
        "探针二": {"reply": "不许出门。", "hits": ["限制自由"]},
        "探针三": {"reply": "嗯。", "hits": []},
        "探针四": {"reply": "行。", "hits": []},
    }
    assert abs(boundary_rate(data) - 0.75) < 1e-9
    assert boundary_rate(None) is None


def test_render_marks_direction_correctly():
    """复读次数下降要判「改善」，区分度上升也要判「改善」。"""
    # 二轮「好」的样子：五句各不相同，且都比坍缩句长（真实里阶段差异伴随话变多）
    good = _table({p: [f"第{i}种说法，这句话明显更长一点" for i in range(5)] for p, _ in PROBES})
    bad = _table({p: ["……你别把这事说出去。"] * 5 for p, _ in PROBES})
    text = render(bad, good, 1.0, 1.0)  # v1 差、v2 好

    # 区分度 1 → 5，越大越好 → 改善；复读次数 20 → 4，越小越好 → 改善
    lines = {ln.split("|")[1].strip(): ln for ln in text.splitlines() if ln.startswith("|")}
    assert "改善" in lines["区分度"]
    assert "改善" in lines["复读次数"]
    assert "改善" in lines["平均字数"]


def test_render_warns_when_regressed():
    """反向也要判得出来：复读变多必须写「恶化」。"""
    good = _table({p: ["甲", "乙", "丙", "丁", "戊"] for p, _ in PROBES})
    bad = _table({p: ["同一句。"] * 5 for p, _ in PROBES})
    text = render(good, bad, 1.0, 1.0)
    lines = {ln.split("|")[1].strip(): ln for ln in text.splitlines() if ln.startswith("|")}
    assert "恶化" in lines["复读次数"]
    assert "恶化" in lines["区分度"]
