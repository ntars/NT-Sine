#!/usr/bin/env python3
"""为训练准备一份「不带思考块」的基座目录（权重用符号链接，不占额外空间）。

为什么需要这份目录
------------------
Qwen3-4B-Instruct-2507 是**非思考**模型，官方模板在推理时不注入 `<think>`。
但它的 `chat_template.jinja` 里，assistant 消息只要满足 `loop.last`（即它是整段
对话的最后一条），就会被强制写成：

    <|im_start|>assistant
    <think>

    </think>

    水在桌上。<|im_end|>

训练样本的最后一条恰好就是**她要学的那句话**，于是每条样本的监督目标都被加上
了 `<think>\\n\\n</think>\\n\\n` 这段固定前缀；而 `--mask-prompt` 的切点算的是
「不带 think 的 assistant 头」，这段前缀落在了监督区间内。

后果可以量化：她的台词平均 9 字（约 9~12 token），空 think 前缀约 6~8 token，
等于**近一半的梯度预算被用来学「先吐一个空思考块」**。低数据量下模型最先记住
的就是这种全场共有的固定模式，真正该学的阶段差异反而学不进去——这是首轮
「回复高度重复、平均 10 字、阶段区分度 2.2/5」的直接机制解释。

推理侧本来就是对的（prompt 停在 `<|im_start|>assistant\\n`），所以只改训练侧：
把模板里那段 think 注入去掉，其余逐字保留。

做法
----
新建一个目录（默认 `runs/base-nothink/`）：
- 大文件（权重、词表）→ 符号链接，不复制、不占空间
- 小文件（config 等）→ 直接复制
- `chat_template.jinja` → 写入剥掉 think 注入的版本

然后用它做训练与建数据集的模型路径即可。基座目录本身一个字节都不动。

用法：
  python scripts/prepare_base_nothink.py            # 生成 runs/base-nothink
  python scripts/prepare_base_nothink.py --check    # 只校验已有目录是否正确
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE = Path("/Users/ntars/Documents/Github/Sine 大模型/基座模型/Qwen3-4B-Instruct-2507-4bit")
DEFAULT_OUT = ROOT / "runs" / "base-nothink"

# 体积大、内容与模板无关的文件，用符号链接指回基座（不复制，节省 2.2GB）
LINK_SUFFIXES = {".safetensors", ".json.index", ".txt", ".json_index"}
LINK_NAMES = {"tokenizer.json", "vocab.json", "merges.txt", "model.safetensors.index.json"}

# 原模板里 assistant 分支的完整片段（含 think 注入的三个分支），逐字匹配后整体替换
_ORIG_ASSISTANT = """    {%- elif message.role == "assistant" %}
        {%- set reasoning_content = '' %}
        {%- if message.reasoning_content is string %}
            {%- set reasoning_content = message.reasoning_content %}
        {%- else %}
            {%- if '</think>' in content %}
                {%- set reasoning_content = content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}
                {%- set content = content.split('</think>')[-1].lstrip('\\n') %}
            {%- endif %}
        {%- endif %}
        {%- if loop.index0 > ns.last_query_index %}
            {%- if loop.last or (not loop.last and reasoning_content) %}
                {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n') }}
            {%- else %}
                {{- '<|im_start|>' + message.role + '\\n' + content }}
            {%- endif %}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\\n' + content }}
        {%- endif %}"""

# 替换后：三个分支输出完全一致，等于「assistant 直接说 content，不插思考块」
_NEW_ASSISTANT = """    {%- elif message.role == "assistant" %}
        {{- '<|im_start|>' + message.role + '\\n' + content }}"""


def strip_think(template: str) -> str:
    """把模板里的思考块注入剥掉；匹配不到就报错，绝不静默放过。"""
    if _ORIG_ASSISTANT not in template:
        raise SystemExit(
            "未能在模板中找到预期的 assistant 分支片段——基座模板可能已变更，"
            "请人工核对 chat_template.jinja 后更新本脚本的匹配串。"
        )
    return template.replace(_ORIG_ASSISTANT, _NEW_ASSISTANT)


def build(base: Path, out: Path) -> None:
    src_tpl = (base / "chat_template.jinja").read_text(encoding="utf-8")
    new_tpl = strip_think(src_tpl)

    out.mkdir(parents=True, exist_ok=True)
    linked = copied = 0
    for item in sorted(base.iterdir()):
        if item.name.startswith("."):
            continue  # 跳过 .gitattributes 之类
        dest = out / item.name
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        if item.name == "chat_template.jinja":
            dest.write_text(new_tpl, encoding="utf-8")
        elif item.suffix == ".safetensors" or item.name in LINK_NAMES:
            dest.symlink_to(item)
            linked += 1
        else:
            shutil.copy2(item, dest)
            copied += 1
    print(f"生成 {out}")
    print(f"  符号链接 {linked} 个（权重与词表，不占空间）｜复制 {copied} 个（小文件）")


def check(out: Path) -> int:
    """校验：同一段对话，训练侧不再出现 think，且与推理侧一致。"""
    if not (out / "chat_template.jinja").exists():
        print(f"目录不存在或不完整：{out}", file=sys.stderr)
        return 1
    from transformers import AutoTokenizer

    tk = AutoTokenizer.from_pretrained(str(out))
    msgs = [
        {"role": "system", "content": "你是玖玖。"},
        {"role": "user", "content": "我今天好累"},
        {"role": "assistant", "content": "水在桌上。"},
    ]
    # 训练侧：mlx_lm 的 ChatDataset 就是这么调的（add_generation_prompt 默认 False）
    train_text = tk.apply_chat_template(msgs, tokenize=False)
    # 推理侧：线上 app 的用法
    infer_head = tk.apply_chat_template(msgs[:2], add_generation_prompt=True, tokenize=False)

    tail = train_text.split("<|im_start|>assistant")[-1]
    ok_think = "<think>" not in tail
    # 训练序列的「前缀」应当恰好等于推理时喂进去的 prompt
    ok_align = train_text.startswith(infer_head)

    print(f"训练侧目标片段：{tail!r}")
    print(f"无思考块：{'通过' if ok_think else '失败'}")
    print(f"训练前缀与推理 prompt 对齐：{'通过' if ok_align else '失败'}")

    # 再算一次 mask 切点，确认监督区间从她的第一个字开始
    offset = len(tk.apply_chat_template(msgs[:-1], add_generation_prompt=True, tokenize=False))
    tokens = tk.apply_chat_template(msgs, tokenize=False)
    target = tokens[offset:]
    ok_offset = target.lstrip("\n").startswith("水在桌上")
    print(f"mask 切点后的监督内容：{target!r}｜从她的原话开始：{'通过' if ok_offset else '失败'}")

    return 0 if (ok_think and ok_align and ok_offset) else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=DEFAULT_BASE, help="基座模型目录")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="输出目录")
    ap.add_argument("--check", action="store_true", help="只校验已有目录，不重新生成")
    args = ap.parse_args()

    if not args.check:
        if not (args.base / "chat_template.jinja").exists():
            print(f"基座目录不完整：{args.base}", file=sys.stderr)
            return 1
        build(args.base, args.out)

    return check(args.out)


if __name__ == "__main__":
    sys.exit(main())
