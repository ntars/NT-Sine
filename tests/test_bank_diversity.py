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
