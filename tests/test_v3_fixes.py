"""世界书 L4 互斥与日记落库的回归测试。

注：原文件中的「复读检测」测试依赖旧数据工厂脚本（filter_dataset / compose_diary），
旧链路已整体下线（Sine 3.4 起数据从零重建），相关断言随之下线；
新链路的复读控制由「台词人工把关 + 单句复用上限」承担，见 scripts/build_dataset.py。
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.worldbook.engine import WorldbookEngine


def _l4_ids(assembly):
    return [a.entry.id for a in assembly.entries if a.entry.layer == "L4"]


def test_l4_exclusive_single_card():
    """多卡同时命中时只保留优先级最高的一张（防串味）。"""
    wb = WorldbookEngine()
    a = wb.scan("我今天好累，加班到十点才下班", [])
    ids = _l4_ids(a)
    assert ids == ["L4-user-sad"], ids  # user-sad(65) > leaving(55)
    assert "L4-leaving" in a.suppressed


def test_l4_exclusive_keeps_single_hit():
    wb = WorldbookEngine()
    a = wb.scan("拉钩，一百年不许变", [])
    assert _l4_ids(a) == ["L4-promise"], _l4_ids(a)


def test_l4_exclusive_nickname():
    wb = WorldbookEngine()
    a = wb.scan("宝宝~我今天超想你", [])
    assert _l4_ids(a) == ["L4-nickname"], _l4_ids(a)


def test_diary_training_file_exists():
    """若生成了日记训练文件，结构必须是 system/user/assistant。"""
    p = Path("data_train/diary.jsonl")
    if not p.exists():
        return  # 日记语料尚未重建，跳过
    first = json.loads(p.read_text(encoding="utf-8").splitlines()[0])
    roles = [m["role"] for m in first["messages"]]
    assert roles == ["system", "user", "assistant"], roles
