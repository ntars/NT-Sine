# NOTICE — 第三方版权与修改声明

## 基座模型

本项目产出的模型 **Sine 玄玖大模型 3.4** 基于两个上游（均为 Apache-2.0）：

1. **上游官方仓库**：`Qwen/Qwen3-4B-Instruct-2507`（阿里巴巴通义千问团队）
2. **MLX 转换仓库**：`mlx-community/Qwen3-4B-Instruct-2507-4bit`
   （MLX 4bit 量化版，本项目训练与推理实际加载的对象；
   本地实体副本在 `Sine 大模型/基座模型/Qwen3-4B-Instruct-2507-4bit/`）

- 许可证：**Apache License 2.0**（全文见 `LICENSE-Apache-2.0.txt`）
- 原始版权：阿里巴巴集团（通义千问团队）
- 修改内容：针对中文情感陪伴场景做风格对齐（LoRA 微调），
  具体为人称、语气、句长分布、边界表达方式，不改变模型结构与能力定位。
- 原始 NOTICE：随 Qwen 官方发布物与 MLX 社区转换仓库的模型卡一并保留。

**不得使用 Qwen 的名称或商标暗示本项目或其衍生模型获得官方背书。**

## 许可证兼容性说明

- 基座（Apache-2.0）与 MLX 转换版（Apache-2.0）
  允许商用与再分发，前提是保留上述版权声明与修改标注。
- 注意：Qwen 系列中部分型号（如 Qwen2.5-3B）为 Qwen-Research License（不可商用），本项目未使用。

## 借鉴来源（只借鉴设计，未复制代码）

| 来源 | 许可证 | 借鉴内容 | 是否复制代码 |
|---|---|---|---|
| SillyTavern | AGPL-3.0 | 世界书机制设计（触发/递归/插入位置/预算） | **否** |
| ai-companion-kit | MIT | 记忆分层思路、拟人化打字节奏、唤醒策略 | 否（MIT 亦允许） |

SillyTavern 为 AGPL-3.0 项目：复制其源码会使本项目前端被"传染"。
本项目仅阅读其官方文档理解机制后自行实现（见 `app/worldbook/`），
**仓库内不含任何 SillyTavern 源码**。

---

## ai-companion-kit（引用声明）

本项目以下四个模块的**机制与参数设计**借鉴自
[pinganbbOvO/ai-companion-kit](https://github.com/pinganbbOvO/ai-companion-kit)：

| # | 原项目模块 | 借鉴内容 | 本项目的独立实现 |
|---|---|---|---|
| 1 | `role.js`（角色卡） | 角色卡三段式（是谁/怎么说话/边界）+ **lint 自检**（没写说话方式、出现"助手/客服"字样、超长段落均报警） | `app/worldbook/engine.py` → `WorldbookEngine.lint()` |
| 2 | `memory.js`（记忆账本） | 按人存账（traits/facts/topics/pending）+ **每类容量上限 FIFO 淘汰**（facts 60 / traits 40 / topics 20 / pending 12） | `app/memory/store.py` + `app/memory/extractor.py` → `KIND_CAPS` 淘汰 |
| 3 | `humanize.js`（拟人化发送） | 打字时长按字数+抖动、深夜 ×1.6、分条 ≤4、**条间长停顿**（25% 概率 3–9 秒，模拟走神/被打断） | `app/output/rhythm.py`（参数：`long_pause_chance` / `long_pause_range_s`） |
| 4 | `diary.js` + `diary-author.js`（她的日记） | 日记是**独白体不是汇报体**；写作角度轮换防模板化；宽松 JSON 解析 + 失败兜底；幂等（同日只写一篇） | `app/diary/author.py`（ANGLES 轮换、`parse_loose`、夜间循环、`POST /api/diary`） |

**原项目信息**：
- 仓库：`pinganbbOvO/ai-companion-kit`（https://github.com/pinganbbOvO/ai-companion-kit）
- 许可证：**MIT**
- 作者：pinganbbOvO

**代码关系**：上述模块为本项目**独立重写**（原作为 JavaScript/Node，本项目为 Python/FastAPI），
**未复制其任何源码**；借鉴限于机制理解与参数选择。按 MIT 许可证要求，
保留原项目署名与本声明。

**未采纳部分**：原项目的 `wake.js`（群聊唤醒：@/拍一拍/概率插话/潜水）与
`phone-mcp`（安卓手机控制）未采纳——本项目为单用户本地场景，两者均不适用。

---

## DeepInfer（技术参考声明，未使用其代码）

**原项目信息**
- 仓库：[deepseek-ai/DeepInfer](https://github.com/deepseek-ai/DeepInfer)
- 许可证：**Apache-2.0**（2026-10-16 创建，2026-10-20 开源，公开时约 1.87 万星）
- 作者：DeepSeek（深度求索）
- 性质：README 原文 —— "high-performance, large-scale **inference** framework"
  （**推理框架，非训练框架**；仓库 342 个文件中无任何 train/finetune/sft/lora 相关代码）
- 支持范围：仅 DeepSeek-V4.1-Flash；硬件仅 NVIDIA SM100 与华为昇腾 950；Linux + GCC ≥13
- 配套项目：`deepseek-ai/deepseek-recipe`（Token 编解码，与官方 API 行为对齐）

**本项目借鉴了什么（仅设计思想，代码零引用）**

| # | 借鉴点 | 本项目落地 |
|---|---|---|
| 1 | **确定性采样 / 批量不变性**：同输入必得同输出，便于复现与排查 | `scripts/evaluate.py --deterministic`（贪心解码 + 固定种子）。落地前实测：同一探针两次运行输出逐字一致 |
| 2 | **KV Cache 长保留**的取舍思维：能复用就不重算，但要算清容量与管理成本 | 未实现。本项目上下文短、单用户，重算成本可接受；记录为"当对话轮次上千时的优化方向" |
| 3 | **API 协议 ↔ Token 序列严格对齐**（deepseek-recipe 的动机） | `scripts/check_tokens.py` 增加训练/服务提示一致性检查，防"训练看不见的提示"漂移 |

**为什么不能直接用它**
1. 它是**推理**框架，本项目需要的是训练/微调链路（用 mlx-lm）；
2. 硬件门槛：仅 NVIDIA SM100 / 昇腾 950，**Apple Silicon 无法运行**；
3. 模型绑定：仅支持 DeepSeek-V4.1-Flash，不支持 Qwen 系基座。

因此本项目**未引入其任何代码或依赖**，声明仅为如实标注参考来源，符合 Apache-2.0 的署名要求。
