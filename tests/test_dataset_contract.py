"""训练数据的装配契约：场景与场景卡必须一一对得上。

为什么值得单独守：`build_system()` 里「场景卡查不到就跳过」是静默的——
少配一张卡，训练样本只是少一段【当前场景】，脚本不报错、损失曲线也照常下降，
只有等评测发现"某些阶段学不会"才回头找。而这个项目已经因为「训练看不见的
提示」翻过一次车（模板注入空思考块），同类风险不该再靠肉眼。

另一条：训练集与推理必须用同一份卡。测试直接读 `data/worldbook/*.json`，
映射指向不存在的 id 会立刻失败。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from data_factory.scene_defs import SCENE_ENTRY, STAGE_SCENES

ROOT = Path(__file__).resolve().parent.parent


def _worldbook_ids() -> set[str]:
    ids: set[str] = set()
    for path in (ROOT / "data" / "worldbook").glob("*.json"):
        raw = json.loads(path.read_text(encoding="utf-8"))
        for entry in raw.get("entries", []):
            ids.add(entry["id"])
    return ids


def _all_scenes() -> set[str]:
    return {scene for scenes in STAGE_SCENES.values() for scene in scenes}


def test_every_scene_has_a_card_mapping():
    """每个场景都要配一张场景卡，缺一张就是一段训练时看不见的提示。"""
    missing = sorted(_all_scenes() - set(SCENE_ENTRY))
    assert not missing, f"这些场景没配场景卡：{missing}"


def test_mapping_points_to_existing_entries():
    """映射必须指向世界书里真实存在的条目 id。"""
    known = _worldbook_ids()
    broken = {scene: eid for scene, eid in SCENE_ENTRY.items() if eid not in known}
    assert not broken, f"映射指向不存在的条目：{broken}"


def test_mapping_only_covers_l4_layer():
    """场景卡属 L4 记忆事件层；映射到常驻层等于把人格当场景注入。"""
    layers: dict[str, str] = {}
    for path in (ROOT / "data" / "worldbook").glob("*.json"):
        raw = json.loads(path.read_text(encoding="utf-8"))
        for entry in raw.get("entries", []):
            layers[entry["id"]] = entry["layer"]

    wrong = {s: layers.get(e, "缺") for s, e in SCENE_ENTRY.items() if layers.get(e) != "L4"}
    assert not wrong, f"这些场景映射到的不是 L4 卡：{wrong}"


def test_extra_mappings_are_not_orphaned():
    """映射表里不该留下已经不用的场景名，否则读表的人会以为它还在生效。"""
    extra = sorted(set(SCENE_ENTRY) - _all_scenes())
    assert not extra, f"这些映射对应的场景已不在台词库里：{extra}"


@pytest.mark.parametrize("scene,entry", [
    ("怕失去", "L4-leaving"),
    ("排他", "L4-jealousy"),
    ("试探", "L4-waiting-reply"),
])
def test_backfilled_scenes_map_to_expected_axis(scene, entry):
    """补配的三张按情绪轴就近归类，钉住结论免得日后被随手改散。"""
    assert SCENE_ENTRY[scene] == entry
