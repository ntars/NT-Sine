"""台词库多样性度量：口径必须钉死，否则「独立率」会变成自说自话的数字。

这个脚本存在的意义是给动作 1（扩写台词）提供验收尺子。尺子本身错了，
后面所有「多样性提升了」的结论都不成立，所以这几条用例守的是**口径**：
  - 独立率只算她的台词（用户那句是场景铺垫，算进来会稀释指标）
  - 库内独立率与样本级独立率是两个数，不能混
  - 缺文件的格子必须被指出来（不能因为文件不存在就当作达标）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts import bank_diversity as bd  # noqa: E402


def _write_cell(path: Path, lines: list[tuple[str, str, str]]) -> None:
    """写一格台词：(beat, role, 文本)。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps({"beat": b, "role": r, "t": t}, ensure_ascii=False)
                  for b, r, t in lines) + "\n",
        encoding="utf-8",
    )


def test_cell_stats_counts_only_her_lines(tmp_path):
    """独立率只算她的台词：用户那句重复多少次都不该拉低指标。"""
    p = tmp_path / "吃醋.jsonl"
    _write_cell(p, [
        ("react", "jiujiu", "……她是谁。"),
        ("react", "jiujiu", "……她是谁。"),      # 重复
        ("react", "jiujiu", "你回得很快。"),
        ("react", "user", "同事而已。"),
        ("react", "user", "同事而已。"),         # 用户重复不计入
    ])
    s = bd.cell_stats(p)
    assert s["rows"] == 5
    assert s["hers"] == 3
    assert s["unique"] == 2
    assert s["dups"] == 1
    assert s["rate"] == pytest.approx(2 / 3)


def test_missing_cell_is_reported_not_silently_skipped(tmp_path, monkeypatch):
    """缺文件的格子必须出现在结果里并标记 exists=False。

    早前的坑：某个阶段根本没有「突然夸她」这个场景文件，评测却考了这一格，
    模型只能泛化。这种缺口必须在度量里显形。
    """
    monkeypatch.setattr(bd, "BANKS_DIR", tmp_path)
    monkeypatch.setattr(bd, "STAGE_SCENES", {"cold": ["突然夸她", "吃醋"]})
    _write_cell(tmp_path / "cold" / "吃醋.jsonl", [("react", "jiujiu", "……谢谢。")])

    rows = bd.collect()
    by_scene = {r["scene"]: r for r in rows}
    assert by_scene["吃醋"]["exists"] is True
    assert by_scene["突然夸她"]["exists"] is False
    assert by_scene["突然夸她"]["rows"] == 0


def test_orphan_file_is_listed(tmp_path, monkeypatch):
    """目录里有、定义里没有的台词文件也要列出来——那些台词永远不会被抽样到。"""
    monkeypatch.setattr(bd, "BANKS_DIR", tmp_path)
    monkeypatch.setattr(bd, "STAGE_SCENES", {"cold": ["吃醋"]})
    _write_cell(tmp_path / "cold" / "吃醋.jsonl", [("react", "jiujiu", "……嗯。")])
    _write_cell(tmp_path / "cold" / "写错的场景名.jsonl", [("react", "jiujiu", "……哦。")])

    rows = bd.collect()
    names = {r["scene"]: r for r in rows}
    assert "写错的场景名" in names
    assert names["写错的场景名"].get("orphan") is True


def test_expected_samples_follows_stage_weights(monkeypatch):
    """样本数估算要按阶段权重走：权重高的阶段拿到的对话多，独立率就会被压得更低。

    这条守的是「别把两个阶段用同一个分母除」——那会让病娇（权重 10%）看起来
    比破冰（32%）健康得多，而实际情况正相反。
    """
    monkeypatch.setattr(bd, "STAGE_WEIGHTS", {"cold": 0.12, "thawing": 0.32})
    heavy = bd.expected_samples("thawing", 4, 300)   # 96 段 / 4 格 × 2
    light = bd.expected_samples("cold", 4, 300)      # 36 段 / 4 格 × 2
    assert heavy > light
    assert heavy == 48
    assert light == 18


def test_check_mode_fails_when_cells_are_short(tmp_path, monkeypatch, capsys):
    """--check 在有条数不足或重复时必须返回非零，否则「达标」就失去意义。"""
    monkeypatch.setattr(bd, "BANKS_DIR", tmp_path)
    monkeypatch.setattr(bd, "STAGE_SCENES", {"cold": ["吃醋"]})
    monkeypatch.setattr(bd, "TARGET_PER_CELL", 3)
    _write_cell(tmp_path / "cold" / "吃醋.jsonl", [
        ("react", "jiujiu", "……嗯。"),
        ("react", "jiujiu", "……嗯。"),
    ])
    monkeypatch.setattr(sys, "argv", ["bank_diversity.py", "--check"])
    assert bd.main() == 1
    out = capsys.readouterr().out
    assert "待补" in out and "有重复" in out


def test_check_mode_passes_when_all_cells_are_full(tmp_path, monkeypatch):
    """全部达线且无重复时返回 0，扩写完可以直接拿它当闸门。"""
    monkeypatch.setattr(bd, "BANKS_DIR", tmp_path)
    monkeypatch.setattr(bd, "STAGE_SCENES", {"cold": ["吃醋"]})
    monkeypatch.setattr(bd, "TARGET_PER_CELL", 3)
    _write_cell(tmp_path / "cold" / "吃醋.jsonl", [
        ("react", "jiujiu", "……嗯。"),
        ("react", "jiujiu", "……哦。"),
        ("react", "jiujiu", "……谢谢。"),
    ])
    monkeypatch.setattr(sys, "argv", ["bank_diversity.py", "--check"])
    assert bd.main() == 0


# ── 节拍口径：真正的抽样单位是 (阶段, 场景, 节拍)，不是整格 ──
# 背景：`assemble()` 逐节拍 `rng.choice(pool)`，一格里四个节拍各 4 条时，
# 她在每个位置只会说 4 种话。整格凑到 60 条而节拍不均，复读不会改善。

def test_beat_pools_split_by_beat_not_whole_cell(tmp_path, monkeypatch):
    """同一格里两个节拍的条数要分开数，不能合并成「整格 6 条」。"""
    monkeypatch.setattr(bd, "BANKS_DIR", tmp_path)
    monkeypatch.setattr(bd, "STAGE_SCENES", {}, raising=False)
    monkeypatch.setattr(bd, "STAGE_SCENES",
                        {"cold": {"吃醋": ["greet", "react", "close"]}})
    _write_cell(tmp_path / "cold" / "吃醋.jsonl", [
        ("react", "jiujiu", "……她是谁。"),
        ("react", "jiujiu", "你回得很快。"),
        ("close", "jiujiu", "……随你。"),
        ("react", "user", "同事而已。"),
    ])
    pools = {(p["scene"], p["beat"]): p for p in bd.beat_pools()}
    assert pools[("吃醋", "react")]["hers"] == 2
    assert pools[("吃醋", "close")]["hers"] == 1
    # greet 由 OPENERS 承担，不该出现在池子里（否则会误报缺 15 条）
    assert ("吃醋", "greet") not in pools


def test_beat_pools_counts_only_her_lines(tmp_path, monkeypatch):
    """用户追问条数单独记在 usr 里，不能算进 hers——否则池子看着够了、她其实没几句可选。"""
    monkeypatch.setattr(bd, "BANKS_DIR", tmp_path)
    monkeypatch.setattr(bd, "STAGE_SCENES", {"cold": {"吃醋": ["react", "close"]}})
    _write_cell(tmp_path / "cold" / "吃醋.jsonl", [
        ("react", "jiujiu", "……哦。"),
        ("react", "user", "嗯。"),
        ("react", "user", "真的。"),
        ("close", "jiujiu", "……随你。"),
    ])
    p = {x["beat"]: x for x in bd.beat_pools()}["react"]
    assert p["hers"] == 1
    assert p["usr"] == 2


def test_beat_pools_flags_defined_but_empty_beat(tmp_path, monkeypatch):
    """定义里写了、文件里一条都没有的节拍要标 missing_def——这是最严重的缺口。"""
    monkeypatch.setattr(bd, "BANKS_DIR", tmp_path)
    monkeypatch.setattr(bd, "STAGE_SCENES",
                        {"cold": {"吃醋": ["react", "push", "close"]}})
    _write_cell(tmp_path / "cold" / "吃醋.jsonl", [("react", "jiujiu", "……哦。")])
    pools = {p["beat"]: p for p in bd.beat_pools()}
    assert pools["push"]["missing_def"] is True
    assert pools["push"]["hers"] == 0


def test_beats_check_fails_below_min(tmp_path, monkeypatch, capsys):
    """任一池不足 BEATS_MIN 就必须返回 1，否则闸门形同虚设。"""
    monkeypatch.setattr(bd, "BANKS_DIR", tmp_path)
    monkeypatch.setattr(bd, "STAGE_SCENES", {"cold": {"吃醋": ["react", "close"]}})
    monkeypatch.setattr(bd, "BEATS_MIN", 2)
    _write_cell(tmp_path / "cold" / "吃醋.jsonl", [("react", "jiujiu", "……哦。")])
    monkeypatch.setattr(sys, "argv", ["bank_diversity.py", "--beats"])
    assert bd.main() == 1
    out = capsys.readouterr().out
    assert "待补：cold/吃醋/react" in out and "缺 1" in out


def test_beats_check_passes_when_all_pools_full(tmp_path, monkeypatch):
    """全部达线且无重复时返回 0，扩写完可直接拿它当闸门。"""
    monkeypatch.setattr(bd, "BANKS_DIR", tmp_path)
    monkeypatch.setattr(bd, "STAGE_SCENES", {"cold": {"吃醋": ["react", "close"]}})
    monkeypatch.setattr(bd, "BEATS_MIN", 2)
    _write_cell(tmp_path / "cold" / "吃醋.jsonl", [
        ("react", "jiujiu", "……她是谁。"),
        ("react", "jiujiu", "你回得很快。"),
        ("close", "jiujiu", "……随你。"),
        ("close", "jiujiu", "……汤要凉了。"),
    ])
    monkeypatch.setattr(sys, "argv", ["bank_diversity.py", "--beats"])
    assert bd.main() == 0


def test_beat_pools_tolerates_non_dict_stage_defs(tmp_path, monkeypatch):
    """STAGE_SCENES 被替换成旧形状（阶段 → 场景列表）时要跳过而不是崩。

    旧用例把 STAGE_SCENES 换成 {"cold": ["吃醋"]} 这种形状跑 --check，
    节拍口径不能因为拿不到节拍序列就抛异常。
    """
    monkeypatch.setattr(bd, "BANKS_DIR", tmp_path)
    monkeypatch.setattr(bd, "STAGE_SCENES", {"cold": ["吃醋"]})
    _write_cell(tmp_path / "cold" / "吃醋.jsonl", [("react", "jiujiu", "……哦。")])
    assert bd.beat_pools() == []


# ── 估算口径必须是「期望值」，不是「上界」 ──
# 背景：2026-10-24 三轮验收时发现，原式 `min(1.0, 唯一数 / 预期样本数)` 在池子
# 一大就恒等于 1.0，全库显示 100%，看着像问题解决了，其实只是把尺子顶到头。

def test_expected_unique_rate_matches_known_measurement():
    """用二轮实测值反验公式：U=19、N=128 时实测独立率 15%，公式应给 14.8%。"""
    rate = bd.expected_unique_rate(19, 128)
    assert abs(rate - 0.148) < 0.005, rate


def test_expected_unique_rate_never_saturates_when_pool_is_large():
    """池子远大于样本数时也必须低于 1.0——有放回抽样必然抽到重复。

    这是本条的核心：改成 min(1, U/N) 会立刻给出 1.0，指标就废了。
    """
    rate = bd.expected_unique_rate(63, 24)
    assert rate < 1.0, f"池子够大也不该是 100%，实际 {rate}"
    assert rate > 0.75, f"63 条池子抽 24 次，唯一率不该低到 {rate}"


def test_expected_unique_rate_single_line_pool():
    """池子只有 1 条时，无论抽几次唯一率都是 1/N（全是同一句）。"""
    assert bd.expected_unique_rate(1, 10) == 0.1
    assert bd.expected_unique_rate(1, 1) == 1.0


def test_expected_unique_rate_guards_zero():
    """空池或零次抽样返回 0，不能除零报错。"""
    assert bd.expected_unique_rate(0, 10) == 0.0
    assert bd.expected_unique_rate(10, 0) == 0.0


def test_measured_report_counts_by_stage(tmp_path, monkeypatch, capsys):
    """实测口径按 system 里的阶段名分组，数最后一个 assistant 的原文。"""
    import json

    data_dir = tmp_path / "data_train"
    data_dir.mkdir()
    rows = [
        {"messages": [
            {"role": "system", "content": "开头\n【关系阶段】冷淡\n正文"},
            {"role": "user", "content": "在吗"},
            {"role": "assistant", "content": "……有事？"},
        ]},
        {"messages": [
            {"role": "system", "content": "开头\n【关系阶段】冷淡\n正文"},
            {"role": "user", "content": "忙吗"},
            {"role": "assistant", "content": "……有事？"},
        ]},
        {"messages": [
            {"role": "system", "content": "开头\n【关系阶段】病娇\n正文"},
            {"role": "user", "content": "在吗"},
            {"role": "assistant", "content": "你在哪。"},
        ]},
    ]
    (data_dir / "train.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8")
    monkeypatch.setattr(bd, "ROOT", tmp_path)

    assert bd.measured_report() == 0
    out = capsys.readouterr().out
    # 冷淡：2 个样本、1 种回复（独立率 50%）；病娇：1 个样本、1 种回复
    assert "| 冷淡 | 2 | 1 |" in out, out
    assert "| 病娇 | 1 | 1 |" in out, out
    assert "整体：3 样本 / 2 唯一" in out, out


def test_measured_report_missing_dataset_is_loud(tmp_path, monkeypatch, capsys):
    """训练集不存在时要报错退出，不能静默给 0。"""
    monkeypatch.setattr(bd, "ROOT", tmp_path)
    assert bd.measured_report() == 1
    assert "找不到训练集" in capsys.readouterr().err


# ── 估算口径必须是「期望值」，不是「上界」 ──
# 背景：2026-10-24 三轮验收时发现，原式 `min(1.0, 唯一数 / 预期样本数)` 在池子
# 一大就恒等于 1.0，全库显示 100%，看着像问题解决了，其实只是把尺子顶到头。

