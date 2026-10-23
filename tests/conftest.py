"""pytest 全局配置。

只做一件事：把临时目录固定到项目内的 `.pytest_tmp/`。

为什么需要：pytest 默认会在系统临时目录里自动新建
`pytest-of-<用户名>/pytest-<序号>` 作为基线目录。在受限沙箱里，
这次新建目录的调用会被拦截，凡是使用 `tmp_path` 的用例都会在 setup
阶段直接抛 FileExistsError / FileNotFoundError，测试根本没跑起来就被判失败。
指定一个已经存在的固定目录后，同一批用例可稳定通过。
"""

from pathlib import Path

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """未显式指定 --basetemp 时，改用项目内的固定目录。"""
    if config.option.basetemp:
        return
    base = Path(__file__).resolve().parent.parent / ".pytest_tmp"
    base.mkdir(parents=True, exist_ok=True)
    config.option.basetemp = str(base)
