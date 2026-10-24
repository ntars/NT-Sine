#!/usr/bin/env python3
"""训练模板去思考块的回归测试。

背景见 `scripts/prepare_base_nothink.py` 开头：Qwen3-4B-Instruct-2507 的模板
会把 `<think>` 空块写进每条样本的监督目标，白烧约四成梯度预算，还造成训练
与推理的分布不一致。这里守住两件事：

1. 剥离逻辑本身不能坏（匹配串失效时必须报错，不能静默放过）
2. 真实基座模板能被正确剥离（基座不在本机时跳过，不误报失败）
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_base_nothink import (  # noqa: E402
    _NEW_ASSISTANT,
    _ORIG_ASSISTANT,
    DEFAULT_BASE,
    strip_think,
)


def test_strip_think_removes_injection() -> None:
    """含思考块注入的模板，剥离后不应再出现 think 标签。"""
    template = "前缀\n" + _ORIG_ASSISTANT + "\n后缀"
    out = strip_think(template)
    assert "<think>" not in out
    assert "</think>" not in out
    # 模板其余部分必须原样保留，不能顺手改坏别的分支
    assert out.startswith("前缀\n")
    assert out.endswith("\n后缀")
    assert _NEW_ASSISTANT in out


def test_strip_think_refuses_unknown_template() -> None:
    """模板结构变了要报错停下，不能悄悄生成一份错的训练模板。"""
    with pytest.raises(SystemExit):
        strip_think("这个模板里没有预期的 assistant 分支")


@pytest.mark.skipif(not (DEFAULT_BASE / "chat_template.jinja").exists(),
                    reason="本机没有基座模型，跳过真实模板校验")
def test_real_base_template_strips() -> None:
    """真实基座模板：剥离后 assistant 尾巴只剩她的原话。"""
    from transformers import AutoTokenizer

    src = (DEFAULT_BASE / "chat_template.jinja").read_text(encoding="utf-8")
    out = strip_think(src)
    assert "loop.last" not in out.split("<|im_start|>assistant")[-1], "think 注入分支仍然存在"
    assert "<think>" not in out.split("message.role == \"assistant\"")[-1]

    # 用剥离后的模板实测一次分词结果
    from transformers import PreTrainedTokenizerFast

    tk = PreTrainedTokenizerFast(
        tokenizer_file=str(DEFAULT_BASE / "tokenizer.json"),
        chat_template=out,
    )
    msgs = [
        {"role": "user", "content": "在吗"},
        {"role": "assistant", "content": "在。"},
    ]
    text = tk.apply_chat_template(msgs, tokenize=False)
    assert "<think>" not in text
    assert text.rstrip().endswith("在。<|im_end|>")


@pytest.mark.skipif(
    not (DEFAULT_BASE / "chat_template.jinja").exists(),
    reason="本机没有基座模型，跳过",
)
def test_inference_path_identical_between_base_and_nothink() -> None:
    """推理路径上，原基座模板与去 think 版必须渲染出**逐字相同**的 prompt。

    为什么这条重要：`evaluate.py` / `boundary_check.py` 加载的是**原基座目录**
    （`load(base, adapter_path=...)`），而训练用的是 `runs/base-nothink`。
    两者模板文件不同（4040 字节 vs 3130 字节、think 出现 6 次 vs 0 次），所以
    「评测用原基座安不安全」不能靠假设——只有「add_generation_prompt=True 这条
    路径上两者一字不差」被验证过，评测结果才与线上等价。

    这条一旦红，说明 Qwen 模板的 generation 分支被动过，评测口径立刻失真。
    """
    from transformers import AutoTokenizer

    nothink = ROOT / "runs" / "base-nothink"
    if not (nothink / "chat_template.jinja").exists():
        pytest.skip("本机没有 runs/base-nothink，跳过")

    msgs = [
        {"role": "system", "content": "你是白玖玖。"},
        {"role": "user", "content": "我今天好累"},
    ]
    a = AutoTokenizer.from_pretrained(str(DEFAULT_BASE))
    b = AutoTokenizer.from_pretrained(str(nothink))
    ta = a.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    tb = b.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    assert ta == tb, f"推理路径不一致：\n原基座 {ta!r}\n去think {tb!r}"
    assert ta.rstrip().endswith("<|im_start|>assistant"), ta
    assert "<think>" not in ta
