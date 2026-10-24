"""测试套件自身的体检：用例不许重名。

为什么需要这条
--------------
2026-10-24 实测：多个执行体并发写同一个工作区时，同一个测试文件被重复追加，
`tests/test_bank_diversity.py` 里出现 18 个 `def` 但只收集到 12 项；
`test_dataset_contract.py`、`test_diary.py`、`test_worldbook.py` 各有 1–2 组同名用例。
Python 里后定义的函数**静默覆盖**前一个，pytest 不报错、测试照旧全绿，
于是「用例数」这个交付物上天天引用的数字是虚的，而没人看得出来。

这条用例把「同名即覆盖」变成红灯，并发导致重复时第一时间暴露。
"""

from __future__ import annotations

import ast
import collections
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent


def _test_names(path: Path) -> list[str]:
    """取一个测试文件里所有「会被 pytest 收集」的用例名。

    含顶层 `def test_*` 与类内 `def test_*`（类内按 `类名::方法名` 归一，
    因为类内同名方法同样会覆盖）。
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test"):
            names.append(node.name)
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and sub.name.startswith("test"):
                    names.append(f"{node.name}::{sub.name}")
    return names


def test_no_duplicate_test_function_names():
    """同一文件内不许出现同名用例——重名只会静默覆盖，不会报错。"""
    dup: list[str] = []
    for path in sorted(TESTS_DIR.glob("test_*.py")):
        counter = collections.Counter(_test_names(path))
        for name, times in counter.items():
            if times > 1:
                dup.append(f"{path.name}::{name}（出现 {times} 次）")
    assert not dup, "以下用例重名，后者会覆盖前者、测试数虚高：\n" + "\n".join(dup)


def test_every_test_file_has_cases():
    """测试文件里至少要有一个用例。

    防止「文件建好了、内容写空了」这种半成品被当成已覆盖——
    空文件在 pytest 眼里和不存在没区别，但会让人以为这块有测试。
    """
    empty = [p.name for p in sorted(TESTS_DIR.glob("test_*.py")) if not _test_names(p)]
    assert not empty, f"以下测试文件没有任何用例：{empty}"
