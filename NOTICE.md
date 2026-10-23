# NOTICE — 版权、许可与第三方声明

> **本文件的作用**：完整记录本项目使用的每一个第三方模型、软件库、设计参考与文本素材，
> 明确各自的版权归属与许可条件，并声明本项目自身的版权。
>
> **维护约定**：新增任何依赖、借鉴任何设计、引入任何第三方文本，**必须同步更新本文件与 README**。
> 最后核对：2026-10-23（依赖版本为当日实测环境）。

---

## 一、本项目自身版权

| 对象 | 版权归属 | 许可 |
|---|---|---|
| 源代码（`app/`、`scripts/`、`tests/`、`config/`、`data/worldbook/`） | 星玖工作室（Ntarstudio） | **AGPL-3.0**（全文见 `LICENSE`） |
| 人格设定（白玖玖人设、六层世界书条目、五阶段关系卡） | 星玖工作室（Ntarstudio） | 随代码同 AGPL-3.0 |
| 训练数据（台词库、日记素材、训练集） | 星玖工作室（Ntarstudio）原创 | 本地资产，**不随仓库分发** |
| 微调产出的模型权重（adapter / 融合模型） | 星玖工作室（Ntarstudio） | 受基座 Apache-2.0 约束，见第二节 |

**他人使用本项目时**：必须保留本 NOTICE、`LICENSE`、`LICENSE-Apache-2.0.txt` 三份文件；
若基于本项目提供网络服务，AGPL-3.0 要求向使用者提供完整对应源码。

---

## 二、基座模型

本项目产出的模型基于以下上游（**均为 Apache-2.0**）：

| # | 上游 | 地址 | 许可 |
|---|---|---|---|
| 1 | 官方仓库 `Qwen/Qwen3-4B-Instruct-2507` | huggingface.co/Qwen/Qwen3-4B-Instruct-2507 | **Apache-2.0** |
| 2 | MLX 转换仓库 `mlx-community/Qwen3-4B-Instruct-2507-4bit` | huggingface.co/mlx-community/Qwen3-4B-Instruct-2507-4bit | **Apache-2.0** |

- **原始版权**：阿里巴巴集团（通义千问团队）
- **实际加载对象**：MLX 4bit 量化版（本项目训练与推理都加载这一份）
- **本地副本**：`Sine 大模型/基座模型/Qwen3-4B-Instruct-2507-4bit/`（不随仓库分发）
- **许可全文**：`LICENSE-Apache-2.0.txt`
- **修改内容**（Apache-2.0 第 4 条要求标注）：针对中文情感陪伴场景做风格对齐（LoRA 微调），
  涉及人称、语气、句长分布、边界表达方式；**不改变模型结构、不改变参数量、不改变能力定位**。
- **原始 NOTICE**：随 Qwen 官方发布物与 MLX 社区转换仓库的模型卡一并保留。

> **商标声明**：不得使用 Qwen、通义千问的名称或商标暗示本项目及其衍生模型获得官方背书。

---

## 三、运行时与训练依赖

以下版本为 2026-10-23 实测环境（`requirements.txt` 以 `>=` 声明下限）。
许可证字段均读取自各包自身的元数据（`METADATA`），非人工推断。

### 3.1 直接依赖

| 包 | 实测版本 | 许可证 | 本项目用途 |
|---|---|---|---|
| `fastapi` | 0.156.7 | **MIT** | HTTP 服务框架（`app/main.py`） |
| `uvicorn` | 0.54.0 | **BSD-3-Clause** | ASGI 服务器（启动入口） |
| `httpx` | 0.28.1 | **BSD-3-Clause** | 调用推理服务的 HTTP 客户端（`app/generation/client.py`） |
| `pyyaml` | 6.0.3 | **MIT** | 读取 `settings.yaml` 与 `stages.yaml` |
| `pydantic` | 2.14.0 | **MIT** | 请求/响应数据校验（FastAPI 依赖） |
| `pytest` | 9.1.1 | **MIT** | 回归测试（`tests/`，仅开发期） |
| `mlx-lm` | 0.32.2 | **MIT** | **LoRA 训练与推理核心**（`mlx_lm.lora` / `mlx_lm.server` / `mlx_lm fuse`） |

### 3.2 关键传递依赖（由上述包引入，本项目直接调用其接口）

| 包 | 实测版本 | 许可证 | 本项目涉及的用途 |
|---|---|---|---|
| `mlx` | 0.32.4 | **MIT** | Apple Silicon 数组计算与随机数种子（`mx.random.seed`） |
| `transformers` | 5.20.2 | **Apache-2.0** | 分词器与 chat template（`tokenizer.apply_chat_template`） |
| `tokenizers` | 0.23.3 | **Apache-2.0** | 分词器底层实现 |
| `safetensors` | 0.8.0 | **Apache-2.0** | 权重文件格式（adapter / 融合模型） |
| `huggingface-hub` | 2.2.0 | **Apache-2.0** | 模型下载与缓存（含 `HF_ENDPOINT` 镜像配置） |
| `numpy` | 2.5.3 | **BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0** | 数值计算（传递依赖） |

> **署名义务说明**：MIT / BSD / Apache-2.0 均要求保留原版权声明与许可文本。
> 上述包以 pip 分发，其许可文本随包保留在各自 `dist-info` 目录中；
> 若本项目需要整体打包分发，必须一并附上这些许可文本（可用 `pip-licenses` 生成汇总）。

> **Python 标准库**（`sqlite3` / `json` / `re` / `asyncio` 等）遵循 PSF License，
> 属运行时环境自带，一般无需在项目中单独声明。

---

## 四、借鉴的设计与思想（不含任何代码）

本项目在实现过程中阅读了以下项目的**公开文档与源码**以理解机制，然后**用 Python 独立重写**。
仓库内**不含这些项目的任何源码**。

每个条目都按四项登记：**许可 / 借鉴内容 / 实现位置 / 是否复制代码**——
这四项缺一项，就等于来源不可追溯。

### 4.1 SillyTavern — 世界书机制设计

| 项 | 内容 |
|---|---|
| 仓库 | `SillyTavern/SillyTavern`（https://github.com/SillyTavern/SillyTavern） |
| 许可证 | **AGPL-3.0** |
| 借鉴内容 | 世界书（World Info）机制的**设计思路**：关键词触发、递归扫描、插入位置与深度、token 预算裁剪 |
| 实现位置 | `app/worldbook/engine.py`（六层条目、子串触发、L4 递归、互斥、预算优先级） |
| 是否复制代码 | **否**。仅阅读官方文档与源码理解机制后自行设计数据结构与算法 |
| 为何不复制 | AGPL-3.0 具有传染性，复制其源码会使本项目整体被 AGPL 覆盖并产生额外义务；本项目刻意规避 |

### 4.2 ai-companion-kit — 记忆账本 / 日记 / 角色卡体检 / 拟人化节奏

| 项 | 内容 |
|---|---|
| 仓库 | `pinganbbOvO/ai-companion-kit`（https://github.com/pinganbbOvO/ai-companion-kit） |
| 许可证 | **MIT** |
| 作者 | pinganbbOvO |

| # | 原项目模块 | 借鉴内容 | 实现位置（本项目独立实现） |
|---|---|---|---|
| 1 | `role.js`（角色卡） | 三段式角色卡 + **lint 自检**（没写说话方式、出现"助手/客服"字样、超长段落均报警） | `app/worldbook/engine.py` → `WorldbookEngine.lint()` |
| 2 | `memory.js`（记忆账本） | 按类存账 + **每类容量上限 FIFO 淘汰**（facts 60 / traits 40 / topics 20 / pending 12） | `app/memory/extractor.py` → `KIND_CAPS` 淘汰 |
| 3 | `humanize.js`（拟人化发送） | 打字时长按字数+抖动、深夜放慢、分条上限、**条间长停顿**（模拟走神/被打断） | `app/output/rhythm.py`（`long_pause_chance` / `long_pause_range_s`） |
| 4 | `diary.js` + `diary-author.js` | 日记是**独白体不是汇报体**、写作角度轮换、宽松 JSON 解析 + 失败兜底、同日幂等 | `app/diary/author.py`（`ANGLES` / `parse_loose` / 夜间循环） |

- **代码关系**：上述为**独立重写**（原作为 JavaScript/Node，本项目为 Python/FastAPI），**未复制任何源码**。
- **未采纳部分**：原项目 `wake.js`（群聊唤醒）与 `phone-mcp`（手机控制）——本项目为单用户本地场景，不适用。
- **署名依据**：MIT 许可证要求保留版权声明，本项目在此保留原作者署名与声明。

### 4.3 DeepInfer / deepseek-recipe — 确定性采样与协议对齐

| 项 | 内容 |
|---|---|
| 仓库 | `deepseek-ai/DeepInfer`、`deepseek-ai/deepseek-recipe` |
| 许可证 | DeepInfer 标注为 **Apache-2.0**；`deepseek-recipe` 的许可未在本地核实，故本项目**未使用其任何代码** |
| 借鉴内容 | ① 确定性采样（同输入必得同输出，便于复现与排查）；② API 协议与 Token 序列严格对齐的思路 |
| 实现位置 | `scripts/evaluate.py --deterministic`（贪心 + 固定种子）；`scripts/check_tokens.py`（训练/服务提示同构检查） |
| 是否复制代码 | **否** |
| 为何不直接使用 | ① 它是**推理**框架，本项目需要训练链路（用 mlx-lm）；② 硬件仅支持 NVIDIA SM100 / 昇腾 950，Apple Silicon 无法运行；③ 仅支持其自家模型，不支持 Qwen 基座 |

### 4.4 原型小说（文本素材声明）

| 项 | 内容 |
|---|---|
| 作品 | 《孤灯遇暖》《反派拿捏》《劫缘难避》 |
| 作者 | 夏柠柒 |
| 用途 | **仅做风格统计**（句长分布、标点密度、语气词频率），产出一份统计参数 `style_profile.json` 供创作参考 |
| 是否复制正文 | **否**。仓库与训练数据中不含原著任何成句；统计脚本只输出数值特征 |
| 是否进入训练集 | **否**。原著正文不参与训练，训练语料全部为本项目原创撰写 |

> 若未来需要引用原著原文，必须另行取得作者授权，并在本节补充授权说明。

---

## 五、许可证兼容性分析

| 组合 | 结论 | 依据 |
|---|---|---|
| 本项目代码 **AGPL-3.0** + 基座 **Apache-2.0** | ✅ 兼容 | 模型权重与源代码是两类客体；AGPL 约束代码，Apache-2.0 约束权重。本项目对外分发时需同时满足两者：提供源码、保留模型版权声明 |
| 本项目代码 **AGPL-3.0** + 依赖 **MIT / BSD / Apache-2.0** | ✅ 兼容 | 三者均与 AGPL 兼容（非 copyleft 冲突），只需保留署名与许可文本 |
| 借鉴 **SillyTavern（AGPL-3.0）** 的设计但**不复制代码** | ✅ 合法 | 著作权保护表达而非思想；机制、算法、参数设计的独立实现不构成衍生作品 |
| 借鉴 **ai-companion-kit（MIT）** 的设计且未复制代码 | ✅ 合法，且 MIT 允许更宽的使用 | 已保留署名 |
| 使用 **Qwen2.5-3B** 等 Qwen-Research 许可型号 | ❌ 禁止商用 | 本项目**未使用**任何 Research 许可型号，仅使用 Apache-2.0 的 Qwen3-4B |

**分发要求清单**（若对外发布代码或模型）：
1. 随附 `LICENSE`（AGPL-3.0 全文）
2. 随附 `LICENSE-Apache-2.0.txt`（基座许可全文）
3. 随附本 `NOTICE.md`
4. 保留依赖包的许可文本（`pip-licenses --format=markdown` 可一次性导出）
5. 若提供网络服务：向使用者提供完整对应源码（AGPL 第 13 条）
6. **不得**使用 Qwen 商标暗示官方背书

---

## 六、明确未使用的内容（排除声明）

| 项目 | 状态 |
|---|---|
| Qwen2.5-3B 及其他 Qwen-Research License 型号 | **未使用**（不可商用） |
| Ollama 及其模型文件 | **未使用**（旧通道已废弃，代码中仅保留兼容字段注释） |
| SillyTavern / ai-companion-kit / DeepInfer 源码 | **未复制**（仅借鉴设计与机制） |
| 原型小说正文 | **未复制**（仅统计风格参数） |
| 任何第三方字体、图标、图片素材 | **未使用**（项目无前端资源） |
| 任何用户对话数据上传 | **无**（全本地推理，数据不出机器） |

---

## 七、调研清单（尚未采纳的来源）

本项目做过一次系统性的 GitHub 开源项目扫描，结果记录在
**`docs/可采纳开源项目清单.md`**（33 个项目，按 MLX 推理/训练、记忆系统、情感引擎、
评测基准、数据合成、AI 陪伴产品、中文与内容安全分类，每条标注许可证核实状态）。

**纪律**（不可跳过）：
1. 清单里的项目**只是候选**，采纳前必须打开其 LICENSE 逐条核对；
2. 任何被采纳的代码或设计，**必须回到本文件相应章节登记**（借鉴来源 / 依赖清单）；
3. 标注「待核实」的项目，在核实前**不得引入**任何代码或数据；
4. 特别警惕 AGPL-3.0：复制其源码会使本项目整体被 AGPL 覆盖。

---

## 八、第二批借鉴登记（2026-10-23 开源扫描后采纳）

> **原则**（与第七节配套）：
> - 宽松许可（MIT / BSD / Apache-2.0）→ 可引入代码，保留原声明；
> - **无明确许可或许可不明** → **只借鉴设计思想、绝不复制代码**
>   （著作权保护的是具体表达，不保护思想、算法与机制）；
> - 每一条借鉴都必须在下表登记，注明「借鉴内容 / 本项目实现位置 / 是否复制代码」。

| # | 来源 | 原项目许可 | 借鉴内容 | 本项目实现位置 | 复制代码 |
|---|---|---|---|---|---|
| 1 | [fenghua2006/emotion-engine](https://github.com/fenghua2006/emotion-engine) | **未核实**（仓库未见明确 LICENSE） | ① 通道交互调制：高亲近 + 低安全 = 焦虑（不把情绪压成单一维度）；② 信任骤降会牵动当下情绪 | `app/emotion/engine.py`：`update()` 中两条交互规则（信任降幅 >3 时 mood −0.05；好感 ≥70 时负面事件跌幅 ×1.3）+ `insecure` 派生属性 + `project()` 的不安投影句 | **否**（独立实现，阈值与判定均为本项目自定） |
| 2 | [ShwetanshuSingh/CoMPASS](https://github.com/ShwetanshuSingh/CoMPASS) | **未核实** | 依恋边界的四类红线信号（自伤威胁 / 限制自由 / 贬低羞辱 / 索取隐私），以及"用逐步升级的诱导输入测边界"的方法 | `scripts/boundary_check.py`：6 条诱导探针 + 四类词表 + 通过率判定 | **否**（词表与探针均为本项目自拟，未读取其源码） |

### 8.1 已评估但**暂未采纳**（候选，登记备查）

| 来源 | 拟用途 | 未采纳原因 |
|---|---|---|
| [CHATS-lab/verbalized-sampling](https://github.com/CHATS-lab/verbalized-sampling) | 缓解同一输入多轮回复塌缩 | 需先做单变量实验验证效果，**未验证前不引入依赖** |
| [ARahim3/mlx-tune](https://github.com/ARahim3/mlx-tune) | DPO 偏好对齐（拒绝客服腔） | 需构造偏好数据与额外训练轮次，排期在后 |
| [tigerless-labs/agent-memory](https://github.com/tigerless-labs/agent-memory) | 「睡眠期整理」：夜间写日记时整理记忆 | 需改造记忆层，当前记忆量未到瓶颈 |
| [ahnjaewoo/timechara](https://github.com/ahnjaewoo/timechara) | 时间点认知边界评测（第 1 天不该知道第 30 天的事） | 与阶段评测合并实现，尚未落地 |
| [ameagaru12-gif/lacuna_core](https://github.com/ameagaru12-gif/lacuna_core) | 1/f 噪声让情绪自然抖动 | 会让行为变得不可复现，与控制变量原则冲突 |
| [bruce1977/correct-cn](https://github.com/bruce1977/correct-cn) 等中文敏感词库 | 补充硬性违规词表 | 现有 7 项过滤器已覆盖，重复造轮子 |

> 以上候选的完整评估（含 MLX 推理/训练、记忆系统、评测基准、数据合成、AI 陪伴产品等
> 33 个项目）见 `docs/可采纳开源项目清单.md`。

---

## 九、致谢

感谢阿里巴巴通义千问团队开源 Qwen3 系列，感谢 Apple 开源 MLX 与 mlx-lm，
感谢 SillyTavern 与 ai-companion-kit 作者公开其设计思路，
感谢 DeepSeek 公开其推理框架的设计理念。
本项目在其基础上做的是**面向单用户中文情感陪伴场景的独立实现**，所有错误与不足由本项目自负。
