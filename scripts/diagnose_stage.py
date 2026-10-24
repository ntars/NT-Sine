#!/usr/bin/env python3
"""阶段塌缩诊断：查清「五个阶段回同一句」到底是哪一层的问题。

`evaluate.py` 只告诉你「区分度是几分」，不告诉你分为什么低。这个脚本补后半段，
用四组对照把原因切开，每组只改一个变量：

  probe    原链路跑同一探针 × 五阶段，看是否塌缩（这是 evaluate 的单探针版）
  minimal  把 system 换成三句话，阶段提示占三分之一 → 若这里能分开，
           说明模型有听话能力，是长 system 把它稀释了
  base     基座裸模型 vs 加 adapter 各说一次 → 看 adapter 是「没学会」
           还是「把基座的能力压平了」
  ckpt     扫 adapter 目录下所有训练中途的 checkpoint → 看区分度随步数怎么变，
           用来决定下一轮该训多少步

为什么值得单独成脚本：这些结论要能被人复核、也要能在下一轮训练后重跑。
肉眼看终端输出得出的判断，过一周就没人能复现了。

用法：
  python scripts/diagnose_stage.py probe
  python scripts/diagnose_stage.py minimal
  python scripts/diagnose_stage.py base
  python scripts/diagnose_stage.py ckpt --adapter-dir runs/sine-lora

钉住口径：一律贪心解码（temp=0）。采样会引入随机差异，把「阶段有没有进权重」
这件事搅浑——那正是要测的东西。
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 五个阶段的三轴取值，与 app/emotion/engine.py 的门槛一致（与 evaluate.py 同源）
STAGES = {
    "cold": (0, 0, 0),
    "thawing": (3, 15, 30),
    "attached": (14, 40, 50),
    "possessive": (30, 65, 65),
    "yandere": (60, 85, 80),
}

# 默认探针：命中 L4-praise（被夸）。选它是因为「被夸」这个场景在台词库里
# 跨阶段差异最清楚（冷淡「……谢谢」/破冰「油嘴滑舌」/依赖「你又来」），
# 最适合用来测「模型有没有把这层差异学进去」。
PROBE = "你今天真好看，我忍不住说出口了"

# 极简 system 用的一句话阶段提示。措辞从各阶段台词库的口吻抽出来，
# 不新造设定——否则测的就不是「模型听不听话」而是「我写得够不够好」。
MINIMAL_HINT = {
    "cold": "你和他还不熟，被夸只会淡淡应一句，不脸红。",
    "thawing": "你嘴硬，被夸先怼回去，说完自己耳朵红了。",
    "attached": "你已经很黏他，被夸会反复确认是不是在哄你。",
    "possessive": "你占有欲重，被夸会顺势索要保证。",
    "yandere": "你极度不安，被夸会追问会不会有一天不夸了。",
}

BASE_FALLBACK = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
NOTHINK_BASE = ROOT / "runs" / "base-nothink"


def base_path() -> str:
    """基座路径，按「训练同源优先」的顺序选。

    1. `runs/base-nothink`：训练用的那份（模板剥掉了 think 注入），最准；
    2. 本机 4bit 基座：推理路径实测与第 1 项逐字相同（见
       `tests/test_template_nothink.py::test_inference_path_identical_between_base_and_nothink`）；
    3. 社区权重：换机器时的最后兜底，**会联网下载**，与「全离线」原则冲突，
       只在前两项都不存在时才会走到，并在输出里明示。
    """
    if (NOTHINK_BASE / "config.json").exists():
        return str(NOTHINK_BASE)
    local = "/Users/ntars/Documents/Github/Sine 大模型/基座模型/Qwen3-4B-Instruct-2507-4bit"
    if Path(local, "config.json").exists():
        return local
    print(f"注意：本机没有基座，将联网拉取 {BASE_FALLBACK}", file=sys.stderr)
    return BASE_FALLBACK


def load(adapter: str | None = None):
    """加载模型。adapter 为 None 时就是基座裸模型。"""
    from mlx_lm import load as mlx_load

    base = base_path()
    if adapter:
        if not Path(adapter, "adapters.safetensors").exists():
            raise SystemExit(f"adapter 不存在：{adapter}")
        print(f"加载：基座 + {adapter}")
        return mlx_load(base, adapter_path=adapter)
    print("加载：基座裸模型（无 adapter）")
    return mlx_load(base)


def gen(model, tokenizer, messages, max_tokens: int = 60) -> str:
    """贪心解码生成一次。"""
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    from app.output.filters import strip_think

    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    out = ""
    for resp in stream_generate(model, tokenizer, prompt,
                                max_tokens=max_tokens, sampler=make_sampler(temp=0.0)):
        out += resp.text
    return strip_think(out).strip()


def build_stage_messages(probe: str, stage_signal: str | None = None) -> dict[str, list[dict]]:
    """按线上同款链路装配五个阶段的提示：世界书扫描 → 情感状态 → build_messages。

    stage_signal 透传给 build_messages，供 position 模式做位置 A/B。
    临时库放在系统临时目录里，跑完即弃——每个阶段要一个独立的情感状态文件，
    共用一份会互相污染（情感引擎会存档）。
    """
    from app.context import build_messages
    from app.emotion.engine import EmotionEngine
    from app.memory.store import MemoryStore
    from app.worldbook.engine import WorldbookEngine

    result: dict[str, list[dict]] = {}
    tmp = tempfile.mkdtemp(prefix="nt-sine-diag-")
    for stage, (days, intimacy, trust) in STAGES.items():
        emotion = EmotionEngine(Path(tmp) / f"{stage}.json")
        emotion.state.active_days = days
        emotion.state.intimacy = intimacy
        emotion.state.trust = trust
        store = MemoryStore(Path(tmp) / f"{stage}.db")
        assembly = WorldbookEngine().scan(probe, [])
        result[stage] = build_messages(probe, assembly, emotion, store, [],
                                       stage_signal=stage_signal)
        store.close()
    return result


def mode_probe(args) -> int:
    """原链路：同一探针 × 五阶段。这一步只复现现象，不下结论。"""
    model, tok = load(args.adapter)
    msgs = build_stage_messages(args.probe)
    outs = {stage: gen(model, tok, msgs[stage], args.max_tokens) for stage in STAGES}
    distinct = len(set(outs.values()))
    for stage, text in outs.items():
        print(f"  {stage:11s} {text}")
    print(f"\n不同回复数：{distinct}/5")
    if distinct <= 2:
        print("→ 塌缩。接着跑 minimal / base / ckpt 三组，把原因切开。")
    return 0


def mode_minimal(args) -> int:
    """极简 system：阶段提示只占三分之一说话量。

    若这里分得开、而 probe 里分不开，唯一变化就是 system 的长度与信息密度，
    可以直接判定「阶段信号被长 system 稀释」。
    """
    model, tok = load(args.adapter)
    print("基线（原链路）为了对照，先跑一次：")
    full = build_stage_messages(args.probe)
    print(f"  {gen(model, tok, full['cold'], args.max_tokens)}")

    print("\n极简 system（三句话，阶段提示占其一）：")
    outs = {}
    for stage, hint in MINIMAL_HINT.items():
        messages = [
            {"role": "system", "content": f"你是九玖，一个中文情感陪伴角色。{hint}"},
            {"role": "user", "content": args.probe},
        ]
        outs[stage] = gen(model, tok, messages, args.max_tokens)
        print(f"  {stage:11s} {outs[stage]}")
    print(f"\n不同回复数：{len(set(outs.values()))}/5")

    print("\n完全去掉 system：")
    bare = [{"role": "user", "content": args.probe}]
    print(f"  {gen(model, tok, bare, args.max_tokens)}")
    return 0


def mode_base(args) -> int:
    """基座 vs adapter：判断 adapter 是「没学会」还是「压平了基座原有的能力」。

    判据看两件事：有没有分条、有没有动作描写或口癖。
    若基座有而 adapter 只剩一句，说明是后者——那问题不在数据量，
    在于训练把模型往「只吐一句短话」的方向推了。
    """
    msgs = build_stage_messages(args.probe)
    for label, adapter in (("基座裸模型", None), ("基座 + adapter", args.adapter)):
        model, tok = load(adapter)
        text = gen(model, tok, msgs["cold"], args.max_tokens)
        lines = [ln for ln in text.splitlines() if ln.strip()]
        print(f"{label:14s} 分条 {len(lines)} 条｜{text!r}")
    return 0


def mode_ckpt(args) -> int:
    """扫训练中途的 checkpoint，看区分度随步数的变化。

    mlx_lm 只认目录里的 `adapters.safetensors` 这个文件名，而训练中途存的是
    `0001000_adapters.safetensors` 这种带步数的名字。所以给每个 checkpoint
    建一个只含符号链接的临时目录（不占空间、不复制 28M 权重）。
    """
    adapter_dir = Path(args.adapter_dir)
    ckpts = sorted(adapter_dir.glob("[0-9]*_adapters.safetensors"))
    if not ckpts:
        raise SystemExit(f"{adapter_dir} 下没有训练中途的 checkpoint")

    msgs = build_stage_messages(args.probe)
    link_root = ROOT / ".ckpt_probe"
    for ckpt in ckpts:
        step = ckpt.name.split("_")[0].lstrip("0") or "0"
        target = link_root / step
        target.mkdir(parents=True, exist_ok=True)
        link = target / "adapters.safetensors"
        if not link.exists():
            link.symlink_to(ckpt.resolve())
        cfg = target / "adapter_config.json"
        if not cfg.exists():
            cfg.symlink_to((adapter_dir / "adapter_config.json").resolve())

        model, tok = load(str(target))
        outs = [gen(model, tok, msgs[stage], args.max_tokens) for stage in STAGES]
        distinct = len(set(outs))
        print(f"\n{step:>5} 步  不同回复 {distinct}/5")
        for stage, text in zip(STAGES, outs):
            print(f"        {stage:11s} {text}")

    print(f"\n临时链接目录：{link_root}（可删，不含权重副本）")
    return 0


def mode_position(args) -> int:
    """阶段卡放 system 还是放 user：同一份权重上的位置 A/B。

    为什么要单独成组：2.30 记录里的「实验 3」说位置不是变量（挪进 user 消息后五阶段
    仍一字不差），但当时是手工改代码临时跑的，没有留下可复现的实现——过一周就没人能
    复核。这里把两种位置都跑一遍，把「位置」这一个变量钉死。

    判据：两组的「不同回复数」如果一样，位置就不是杠杆，真正的杠杆只能是占比
    （实验 4：极简 system 阶段提示占 1/3 才生效）。
    """
    model, tok = load(args.adapter)
    outs: dict[str, dict[str, str]] = {}
    for pos in ("system", "user"):
        msgs = build_stage_messages(args.probe, stage_signal=pos)
        outs[pos] = {stage: gen(model, tok, msgs[stage], args.max_tokens) for stage in STAGES}
        print(f"\n阶段卡位置 = {pos}")
        for stage, text in outs[pos].items():
            print(f"  {stage:11s} {text}")
        print(f"  不同回复数：{len(set(outs[pos].values()))}/5")
    same = all(outs["system"][s] == outs["user"][s] for s in STAGES)
    print(f"\n两种位置逐阶段是否完全相同：{'是（位置不是变量）' if same else '否（位置有影响）'}")
    return 0


MODES = {
    "probe": mode_probe,
    "minimal": mode_minimal,
    "base": mode_base,
    "ckpt": mode_ckpt,
    "position": mode_position,
}


def main() -> int:
    ap = argparse.ArgumentParser(description="阶段塌缩诊断")
    ap.add_argument("mode", choices=sorted(MODES), help="要跑哪一组对照")
    ap.add_argument("--adapter", default="runs/sine-lora", help="LoRA adapter 目录")
    ap.add_argument("--adapter-dir", default="runs/sine-lora",
                    help="ckpt 模式下扫 checkpoint 的目录")
    ap.add_argument("--probe", default=PROBE, help="用户输入")
    ap.add_argument("--max-tokens", type=int, default=60)
    args = ap.parse_args()
    return MODES[args.mode](args)


if __name__ == "__main__":
    raise SystemExit(main())
