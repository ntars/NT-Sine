"""配置加载：读取 config/settings.yaml 与 config/prompts/，全项目唯一入口。"""

from __future__ import annotations

from pathlib import Path

import yaml

# 项目根目录（app/ 的上一级）
ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
PROMPTS_DIR = CONFIG_DIR / "prompts"
DATA_DIR = ROOT / "data"
WORLDBOOK_DIR = DATA_DIR / "worldbook"


def load_settings() -> dict:
    """读取 settings.yaml，缺失文件直接报错（配置是硬约束，不容静默降级）。"""
    path = CONFIG_DIR / "settings.yaml"
    if not path.exists():
        raise FileNotFoundError(f"缺少配置文件：{path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_prompt(name: str) -> str:
    """读取提示词模板文件（按名取用，按需加载）。"""
    path = PROMPTS_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"缺少提示词文件：{path}")
    return path.read_text(encoding="utf-8").strip()


SETTINGS = load_settings()
