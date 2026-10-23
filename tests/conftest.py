"""pytest 全局配置。

三件事：

1. 把临时目录固定到项目内的 `.pytest_tmp/`。pytest 默认会在系统临时目录里
   自动新建 `pytest-of-<用户名>/pytest-<序号>` 作为基线目录；在受限沙箱里，
   这次新建目录的调用会被拦截，凡是使用 `tmp_path` 的用例都会在 setup 阶段
   直接抛 FileExistsError，测试根本没跑起来就被判失败。指定一个固定目录后
   可稳定通过。

2. 提供 `freeze(模块路径, 时刻)`：把某个模块眼里的 `time.time()` 钉死。
   消息表写入时刻、主动闸门的"现在几点"都读 time.time()，要摆出"几小时前
   聊过""现在是不是深夜"就必须能指定时刻——否则同一批用例下午跑通过、
   凌晨跑失败（深夜禁主动会先拦住）。

3. 提供 `insert_at(store, role, content, ts)`：在指定时刻往消息表写一条。

替身只顶掉目标模块里的 `time` 名字，不动全局 time 模块（否则会影响
pytest 自身的计时）；其余属性（localtime 等）转交真实模块，不至于把
依赖时间格式化的代码打断。
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


class FrozenTime:
    """只把 time() 钉死的替身，其余属性转交真实 time 模块。"""

    def __init__(self, ts: float) -> None:
        import time as _real

        self.ts = ts
        self._real = _real

    def time(self) -> float:
        return self.ts

    def __getattr__(self, name: str):
        return getattr(self._real, name)


@pytest.fixture()
def freeze(monkeypatch):
    """返回 freeze(模块路径, 时刻)，例如 freeze("app.proactive.gate.time", ts)。"""

    def _freeze(module_path: str, ts: float) -> None:
        monkeypatch.setattr(module_path, FrozenTime(ts))

    return _freeze


@pytest.fixture()
def insert_at(freeze):
    """返回 insert(store, role, content, ts)：在 ts 这一刻插一条消息。"""

    def _insert(store, role: str, content: str, ts: float) -> None:
        freeze("app.memory.store.time", ts)
        store.add_message(role, content)

    return _insert
