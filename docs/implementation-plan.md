# Tritfold 实施规划（PrismML Bonsai 方法的等效复现）

- 日期：2026-09-22
- 版本：v1.3（经三轮 review→fix 迭代，修订记录见文末）
- 依据文档：[forensics-notes.md](forensics-notes.md)（取证与环境）、[method-reconstruction.md](method-reconstruction.md)（方法重建，下称"重建文档"）
- 性质：执行层规划——把重建文档的 Phase 0–4 管线与笔记 §10.4 的分级原型计划落成可执行、可验收的任务序列

---

## 1. 目标与总体策略

**总目标**：用公开数学重建与 PrismML Bonsai 2 等效的三值化管线（本项目：Tritfold），产出满足 artifact 契约（§2 的 C1–C6）的三值模型，并经 PrismML fork 运行时端到端验证；质量对标已发布的 Ternary-Bonsai 系列。

**策略要点**：

1. **分级推进、每级设验收门与止损点**：M0 环境 → M1 真值挖掘（L0′）→ M2 机制验证（L0）→ M3 端到端 micro（L1-micro）→ M4 完整对拍（L1）→ M5 消融（L2）→ M6 27B 外推（L3，决策门）。任何一级未过验收，不进入下一级。
2. **工程派引擎（E2/PV-Tuning 式）先行**，因为公开文献已验证其稳定性；理论派引擎（E1/矩阵 SMD）作为 M5 消融项实现，回答"PrismML 到底用哪个"并非跑通管线的先决条件。
3. **真值驱动**：M1 对官方 PQ2_0 artifact 的统计挖掘结论优先于一切文献推断——若实测与假设冲突（如 scale ≠ amax、零占比远离 12%），回修管线假设再继续。
4. **不做的事**：不尝试逆向 Caltech 专有训练 IP 本身；目标是以公开方法达到等效出货物。不碰 CUDA 13.3 预编译二进制（本机 Pascal 不可用），运行时验证一律走 CPU 编译的 fork。

---

## 2. 硬约束（每级的共同验收门）

来自重建文档 §1 的代码取证结论，任何阶段的出货物必须满足：

| # | 约束 | 验证手段 |
|---|---|---|
| C1 | 固定块对角 Hadamard 基 R = H_n·S/√n，n=1024，S 为固定 ±1 向量随模型发布 | hadamard_packing.json 字段 + GGUF `prism.hadamard.block_size/sign_mode` |
| C2 | 每组 128 权重 ∈ {-s, 0, +s}，单一 FP16 scale，**无 shift/bias** | 打包后反量化逐组检查码值集合 |
| C3 | 成品权重对 RTN（amax+round）无损 | 全张量 ‖s_g·T − Z‖∞/amax < 1e-3 |
| C4 | 折叠只沿输入特征轴（axis=-1）；embedding 存旋转域、查表后逆变换（0.6B/1.7B 均 tie_word_embeddings=true，必经此路径） | GGUF `prism.hadamard.axis` + inverse_weight_names 含 token_embd |
| C5 | 敏感小岛全精度：归一化层、线性注意力循环状态路径（in_proj_a/b、conv1d、A_log/dt_bias） | 转换后 GGUF 中对应张量仍为 F16/F32 |

注：纯 Qwen3 架构（0.6B/1.7B）的小岛 = 全部归一化层（含 q_norm/k_norm）；in_proj_a/b、conv1d、A_log/dt_bias 为 qwen35 GDN 特有，M6 才涉及。
| C6 | 后训练路线（从现成预训练模型出发）、架构无关 | 管线输入为 Qwen3 原始 safetensors，不改预训练 |

---

## 3. 环境与资产基线（2026-09-22 实测复核）

### 3.1 硬件与系统

- GPU：GTX 1060 6GB（Pascal sm_61）。torch 2.7.1+cu126 实测可用；**训练/推理一律 fp16**（bf16 仅 math SDPA 后端，fp16 可用 mem-efficient）。
- 12 核 CPU / 15GB RAM / 空闲盘 47GB。
- WSL2，gcc/g++/make 在位；**cmake、ninja 缺失**（M0 安装）。

### 3.2 软件栈

| 项 | 状态 | 路径/备注 |
|---|---|---|
| Python venv | ✅ torch 2.7.1+cu126、transformers 4.57.1、accelerate 1.14.0、datasets 5.0.1、numpy 2.2.6、safetensors 0.8.0 | `<VENV>/` |
| fork 源码 | ✅ prism 分支 @ 9a9394a89（已核对） | `<FORK_DIR>/` |
| fork CPU 构建 | ❌ 未构建（无 build 目录，且缺 cmake） | `build_cpu_linux.sh <FORK_DIR>`（须传仓库路径；产物在 `Bonsai-demo/bin/cpu/`） |
| gguf-py | ❌ venv 未安装 | `pip install -e <FORK_DIR>/gguf-py` |
| 论文全文缓存 | ✅ 持久副本存在 | `<DEMO_DIR>/papers/*.txt`（/tmp 副本易失，勿依赖） |
| 网络 | HF 直连不通 | `export https_proxy=<proxy>`；ModelScope 直连备份 |

### 3.3 模型资产（<MODELS_DIR>/）

| 资产 | 关键规格 | 用途 |
|---|---|---|
| `Qwen3-0.6B` | hidden **1024**（恰等于 block_size）、intermediate 3072、head_dim 128、28 层、绑定嵌入 | M2/M3 基座；绑定嵌入 → 天然覆盖 C4 的 inverse-after-lookup 路径 |
| `Qwen3-1.7B` | hidden 2048、intermediate 6144、28 层、绑定嵌入 | M4 基座（与上一代 Ternary-Bonsai-1.7B 同基座，可对拍） |
| `Ternary-Bonsai-2-27B/Ternary-Bonsai-2-27B-PQ2_0.gguf`（7.2GB） | qwen35 架构、PQ2_0、52 条 KV 元数据 | **地面真值**：M1 挖掘对象 |
| `MiniCPM5-2B`（Llama 架构）、`Spark-X2.5-1.7B`（绑定嵌入） | — | M5/M6 架构变体备用 |

注：0.6B/1.7B 的所有矩阵输入维（1024/2048/3072/6144）均为 1024 的整数倍，块对角 Hadamard 无需补齐。

### 3.4 目录约定

- 文档与结果：`./`（本计划、各级 findings、results-log.md）
- 原型代码：`proto/`（新建；`common/` FWHT 与打包共用库、`l0prime/`、`l0/`、`l1micro/`）
- 大产物（checkpoint、GGUF）：`<WORK_DIR>/`（新建，避免污染原始模型目录）
- 磁盘预算（47GB 内）：CPU 构建 ~2GB、WikiText-2 <0.1GB、0.6B 各阶段产物 ~4GB、1.7B fp16 副本+产物 ~8GB、checkpoint 周转 ~10GB，余量充足。

注：上游私有工作档案与本项目发布版分离维护。

---

## 4. 里程碑总览

| # | 名称 | 对应分级 | 估时 | 前置 | 验收摘要 | 状态 |
|---|---|---|---|---|---|---|
| M0 | 环境与工具链就绪 | — | 0.5 天 | — | CPU 版 llama-cli 可跑；import gguf 通过；WikiText-2 本地可用 | ✅ 2026-09-22 |
| M1 | 地面真值挖掘 | L0′ | 1–2 天（纯 CPU） | M0 | PQ2_0 元数据全解码 + 五项统计检验全部有结论 | ✅ 2026-09-22（[findings](findings-1-truth-mining.md)） |
| M2 | 旋转机制验证 | L0 | 1–2 天 | M0（建议 M1 后） | 修订后：机制恒等（rel 0.24%）+ RTN 统计符合理论；原"QuaRot 效应"预期被实测证伪 | ✅ 2026-09-22（[findings](findings-2-rotation-study.md)） |
| M3 | 端到端 micro 管线 | L1-micro | ~1 周 | M2 | ①③④ 过；② 经 Colab 端到端 QAT 推进到 **48.08（1.77×FP）**，本机天花板 91.6 已真值突破（80.75），41 门槛差 15%（终点仍创新低） | ✅ 管线；🔶②（[e2e 战报](findings-5-e2e-qat.md)） |
| M4 | 完整规模对拍 | L1 | 数周（默认本机降级） | M3 | 原门 ±1 分未按原义执行（官方分数口径不可及）；**代之以 ppl 口径：1.41×FP（28.77），≤1.5× 质量门达成** | ✅ 2026-09-26（[findings](findings-6-1p7b.md)） |
| M5 | 消融 | L2 | 与 M4 并行（小规模） | M3 | **A1 已结**（V-step 三变体 + E1 玩具 + 逐层 STE 三变体共七项实验，微预算离散移动系统性证伪，见 [m5a1-findings](findings-4-discrete-movement.md) §5-6）；A2–A4 待做 | 🔶 A1 完成 2026-09-24 |
| M6 | 27B 外推 | L3 | 决策门（200–800 A100·时） | M4+M5 | 对 FP 保真 ≥98%（20 基准）且 AIME/LiveCodeBench 不塌 | 未开始（需多卡） |
| **M5′** | **能力路线**（wiki 蒸馏 → 指令模型） | 增设 | 2026-09-26 立项 | M4 checkpoint | A1 指令数据蒸馏（ultrachat 60% + wiki 40% 混合，~3000 步）+ A2 评估闭环（ARC-c 迷你 harness + 生成抽检 + FP 基线对照）。**验收**：wiki ppl 回退 ≤0.15×FP（回归护栏）；ARC-c 达到 FP 基线的 ≥70%；生成显著跟随指令 | 未开始；notebook `notebooks/tritfold-instruct-1p7b.ipynb` |

**当前起点**：M0 → M1（笔记 §10.4 指定的"明日起点"即 M1，但 M0 的 cmake/CPU 构建是其硬前置，合并执行）。

---

## 5. 里程碑详述

### M0 环境与工具链（0.5 天）

| 任务 | 内容 |
|---|---|
| T0.1 | 安装 cmake（`pip install cmake` 进 venv，或 apt）；可选 ninja |
| T0.2 | 构建 fork CPU 版：`bash <DEMO_DIR>/scripts/build_cpu_linux.sh <FORK_DIR>`（**必须传仓库路径参数**，否则脚本会在 demo 目录下重新 clone；全量构建约 30–60 分钟/12 核，RAM 紧张时将脚本内 `-j$(nproc)` 改为 `-j8`；产物安装在 `<RUNTIME_BIN>/`）；冒烟 `llama-cli --version` |
| T0.3 | venv 安装 gguf-py：`pip install -e <FORK_DIR>/gguf-py`；冒烟 `python -c "import gguf"` |
| T0.4 | 设代理下载 WikiText-2（datasets：`Salesforce/wikitext`, `wikitext-2-raw-v1`） |
| T0.5 | 建 `proto/` 与 `<WORK_DIR>/` 目录骨架；建 `results-log.md` 表头（日期/级别/配置哈希/指标/备注） |
| T0.6 | `git init` 于 `proto/`（实验代码版本化，results-log 的配置哈希对应 git commit） |

**验收**：三项冒烟全过；`results-log.md` 就位。
**交付**：`<RUNTIME_BIN>/` 二进制套件、就绪的 venv、`proto/` git 仓库骨架。

### M1 地面真值挖掘（L0′，1–2 天，纯 CPU）

对象：`Ternary-Bonsai-2-27B-PQ2_0.gguf`。已知 52 条 KV 命中契约（笔记 §10.3），本级补齐未完成项并做权重统计。

| 任务 | 内容 | 预期（据笔记/重建推断，实测为准） |
|---|---|---|
| T1.1 | 用 gguf-py 正规 API 完整解码 `prism.hadamard.weight_names` / `inverse_weight_names` 字符串数组（此前快速脚本偏移读取失败项） | 折叠清单与 `_HADAMARD_KINDS` 白名单（base.py:700-711，清单见 M3 已知风险行）一致 |
| T1.1b | 核对 `sign_widths=[17408]` 与各折叠张量输入维的对应关系：27B 为何只有一条符号宽度（按输入维去重？仅部分张量携带符号？） | 结论直接决定 M3 manifest 符号表的写法 |
| T1.2 | 实现 PQ2_0 反量化（`block_pq2_0` = FP16 scale `d` **在前** + 32 B 2-bit 码 `qs` 在后，共 34 B/128，`ggml-common.h:199-207`；2-bit 解码规则与官方 Q2_0 相同，仅组宽为 128），导出若干代表性张量（注意力投影、MLP、ssm_out、embedding） | 反量化值与 scale 关系可复算 |
| T1.3 | 检验 (a)：零值占比 | ~12%（RTN 高斯阈值 ±0.5·amax 的理论值）；显著偏离 ⇒ 训练端零态比例被主动优化过 |
| T1.4 | 检验 (b)：scale == 组 amax？ | 是（C3 的直接推论）；若否 ⇒ 打包尺度另有规则 |
| T1.5 | 检验 (c)：旋转域权重分布直方图/峰度 vs 高斯 | 近高斯、薄尾（incoherence 成立证据） |
| T1.6 | 检验 (d)：逐张量折叠清单与角色（fold-before-matmul / inverse-after-lookup） | 与 conversion/base.py 白名单一致 |
| T1.7 | 检验 (e)：embedding 是否真为旋转域存储（对 token_embd 做逆变换后应恢复常规嵌入统计） | 是 |

**交付**：`findings-1-truth-mining.md` + `proto/l0prime/` 脚本（可重跑）。
**验收门**：五项检验全部有结论。**止损/回修**：(b) 或 (a) 不符时，先更新对 C2/C3 的解释并修订 M3 的 Phase 3 吸附判据，再继续。

### M2 旋转机制验证（L0，1–2 天）

基座 Qwen3-0.6B。目标：用最小代价复现"旋转使 RTN 三值化变得可用"的 QuaRot 效应。

| 任务 | 内容 |
|---|---|
| T2.1 | `proto/common/`：PyTorch 实现 **fork 契约式旋转**——残差流保持原基，**不做** QuaRot R1 式离线并入：每个折叠权重存 W′=W·R⁻¹（R=H_n·S/√n 沿输入轴块对角，n=1024，S 为固定种子 ±1），推理时在该 matmul 的输入激活上在线施加 sign+FWHT（同一激活被多个折叠权重复用时去重）；embedding 折叠存储、查表后逆变换还原（0.6B 绑定嵌入必经此路径）；全部归一化层（含 Qwen3 的 q_norm/k_norm）保持 FP 不动。**依据**：fork 契约要求全部折叠权重 axis=-1 fold-before-matmul、inverse-after-lookup 仅允许 token_embd（base.py:700-735），且真值 artifact 的 embedding 需要查表后逆变换——二者共同证明运行时逐 matmul 在线旋转、残差流在原基。QuaRot 文献仅作"旋转为何有效"的机制参照，不作实现模板 |
| T2.2 | RTN 三值化：组 128、amax scale、round 到 {-1,0,+1}；数学等价推理（不求速度） |
| T2.3 | WikiText-2 test ppl 三档对比：FP / 无旋转 RTN / 旋转 RTN。fp16、mem-efficient SDPA、固定 stride/种子，结果登记 results-log.md |

**预期（原版，已被实测否定）**：旋转 RTN ≫ 无旋转 RTN，两者均明显差于 FP（缺 Phase 2 训练）。
**实测结论（2026-09-22）**：FP 19.29 / naive RTN 1.12×10⁹ / 旋转 RTN 1.24×10⁸——旋转仅 9× 改善且双方均崩溃；逐层探针显示旋转对权重-only RTN 误差比值 ≈1（权重本就近高斯，正交变换不改变量化难度；84% 置零才是崩溃主因）。旋转的作用在**训练侧**（制造理论成立的非相干 regime + 对称三值无损表示），不是 RTN 修复术。详见 [findings-2-rotation-study.md](findings-2-rotation-study.md)。
**验收门（修订后，已通过）**：① 机制数学恒等（W′(Rx)≈Wx，rel 0.24%，argmax 99.61%）；② RTN 统计与 Gaussian+amax 理论一致（零占比 ~84%）；③ 实现与 artifact 契约逐项一致。**止损**（保留语义）：恒等性若不过，排查 W′(R x) 对齐与 embedding 逆变换，而非调实现细节。
**交付**：`findings-2-rotation-study.md`（三档 ppl + 恒等性测试记录）+ `proto/common/`、`proto/l0/` 可重跑代码。

### M3 端到端 micro 管线（L1-micro，约 1 周）

基座 Qwen3-0.6B，目标跑通 Phase 0–4 全链路。

| 阶段 | 任务 | 要点 |
|---|---|---|
| Phase 0 | T3.1 基底插入落地为可转换产物 | 复用 T2.1 的契约式旋转（无 QuaRot 离线并入）；产出旋转后模型 W′₀（safetensors，折叠张量保持训练序）+ 折叠清单记录 |
| Phase 1 | T3.2 初始化 | Z₀ ← W′₀；s_g ← 组 amax；T₀ ← round(Z₀/s_g)；从 Z₀ 出发（不用已收敛点，2603.10485 Remark 2） |
| Phase 2 | T3.3 E2 引擎训练（冻结主体） | **自研 PyTorch 训练循环**（V 步吸附逻辑与 HF Trainer 不兼容）；可训练集：全部 s_g + 全精度小岛 + embedding（C4/C5），主体 Z 冻结；P 步 Adam：**s_g（即码本）lr 3e-3 恒定，小岛/embedding lr 3e-4**，β=(0.9,0.95)；V 步每步对 \|Adam 更新\| top-1% 权重吸附到最近三值点（信赖比 ≤0.01）；目标 KL(student‖teacher logits)。**教师 logits 预计算**：teacher 单独上卡对训练语料前向一次，top-50 logits 缓存落盘（2K 条×1024 tok ≈ 0.4GB），训练时仅 student 在卡；fp32 主权重 + fp16 计算，梯度检查点按需。0.6B 学生 fp16 ≈1.2GB，6GB 显存充裕；数据先用 BitDistiller 式 ~2K 教师样本，数百步 |
| Phase 3 | T3.4 吸附与清单 | T ← round(Z/s_g)，s_g ← amax；全张量无损校验 <1e-3（C3）；输出 `hadamard_packing.json`：schema 1/2、kind=`hadamard-weight-fold`、transform.name=`normalized-signed-sylvester-walsh-hadamard`（校验硬要求，base.py:656）、block_size=1024、sign_mode=explicit + 符号表（写法以 M1-T1.1b 结论为准）、逐张量记录 {name, axis=-1, role}（0.6B 折叠集 = attn_q/k/v、attn_output、ffn_gate/up/down、output.weight(lm_head)；token_embd 标 inverse-after-lookup）；**先用 base.py:635-760 的校验函数空跑单测，通过后再进 Phase 4** |
| Phase 4 | T3.5 打包与运行时验证 | 将 `hadamard_packing.json` **放入旋转后模型目录**（base.py:626/637 从 `dir_model` 读取，无 CLI 参数）→ fork `convert_hf_to_gguf.py` 转 F16 GGUF（add_hadamard_metadata 校验并写入 `prism.hadamard.*`）→ `llama-quantize` 转 PTQ1_0 → CPU 版 `llama-cli` 加载生成 + **`llama-perplexity` 跑 WikiText-2 做运行时侧独立 ppl 核验**；核对加载日志 `loaded N Hadamard-folded weight(s)` 且无 "consumed without its activation transform" 报错 |

**实测结论（2026-09-23 更新，findings-3-pipeline.md）**：管线端到端打通（PTQ1_0 157MB、C1–C5 全 PASS、打包值级无损、运行时 ppl 一致）。三处偏离记录：embedding 码冻结只训尺度（显存）、尺度 lr 1e-3+裁剪（3e-3 发散）、V-step 隔步+抽样阈值（性能）。**关键消融：V-step 直译版是振荡源；纯 P-step 尺度训练 600 步平滑降到 167。续训更新（2M token 缓存、总 1600 步）：167→91.6 后打平——纯尺度训练在 ~91 饱和，闭合差距需离散码迁移（M5-A1 提前为瓶颈项）**。初始化消融：目标零占比 1/3 初始化起点优 93 倍，选为默认。v3 artifact 生成连贯英文句。

**验收门**：① 端到端跑通（生成文本连贯）；② **训练后三值 ppl 逼近 FP（0.6B 目标 ≤1.5×FP≈29）**——M2 实测裸 RTN 基线 ~1×10⁸，"优于裸 RTN"无区分度，已弃用该判据；③ C1–C5 逐项检查通过（§2 表）；④ `llama-perplexity`（运行时侧）与 PyTorch 侧三值 ppl 偏差在量化噪声量级内——交叉证明打包链路无损。
**训练健康指标**（M1 实测指纹）：收敛期零占比应趋向 ~1/3（近最大熵三值），±1 平衡 ≈1；若零占比停在 ~80%（RTN 初值附近）说明吸附未生效。
**初始化提示**（M2 实测）：裸 RTN 初始化零占比 ~84%，与成品 ~33% 差距大；T3.3 开始前先做小时级初始化小实验（amax RTN 初始化 vs 按目标零占比 1/3 设尺度的初始化）再定默认。
**已知风险**：0.6B 张量映射必须落在 `_HADAMARD_KINDS` 白名单内（base.py:700-711：`output.weight`、`attn_q/k/v/qkv/gate/output`、`ffn_gate/up/down` 及专家变体、`ssm_out`；Qwen3 dense 命中 `attn_q/k/v`、`attn_output`、`ffn_gate/up/down`、`output.weight`）；inverse-after-lookup 只允许映射到 `token_embd.weight`（base.py:724-732，0.6B 绑定嵌入恰好合规）；GDN 字段（`gdn_v_grouped`）对纯 Qwen3 架构不触发。
**交付**：0.6B 三值 PTQ1_0 GGUF + `hadamard_packing.json` + `findings-3-pipeline.md`（端到端验证 + 双侧 ppl 对照）。

### M4 完整规模对拍（L1，数周）

基座 Qwen3-1.7B，跑充分训练的全管线，对照已发布 Ternary-Bonsai-1.7B 的基准分（MMLU 等，白皮书附录 C 有数，无需下载该模型）。

- **算力账与默认方案**：沿用 M3 的"教师 logits 预计算 + student 单独在卡"设计——1.7B 学生 fp16 ≈3.4GB + 激活在 6GB 内可行（teacher 与 student 同时在卡 ≈6.8GB 会爆显存，故必须串行）；优化器状态只在 s_g/小岛/embedding 上（MB 量级，无需 CPU offload）。瓶颈是墙钟而非显存：本机 1060 跑充分 E2 约天数级。**决策点（W3 末）**：默认走本机方案（缩小 token 批、梯度检查点、接受天数级墙钟）；仅当需要 1M token 批全量配置或更大端到端成分时，才评估云短租（PV-Tuning 70B 数据点外推）。教师 logits 缓存随语料线性增长（1M tok × top-50 ≈ 0.2GB），磁盘可忽略。
- **验收门**：对拍分数差 ±1 以内 = 管线等效；差距 >2 分 → 进入 M5 消融定位短板（优先怀疑数据构成与 s_g 处理，重建文档 §6-3/§6-4）。
- **交付**：1.7B 三值 GGUF + `docs/findings-*.md`（对拍报告）。

### M5 消融（L2，与 M4 并行，小规模 0.6B）

| 实验 | 内容 |
|---|---|
| A1 | **E1 vs E2**（2026-09-23 已结，[findings-4-discrete-movement.md](findings-4-discrete-movement.md)）：E2 的 P 步（尺度+小岛）有效但饱和于 3.3×FP；**三种 V-step 变体（末样本/EMA 排序/带符号 EMA+接受回退）全部无法移动码**（盲翻恶化 91.6→308/3054；闸门版全拒 flips 0 rej 19.7K/步）；E1 玩具显示浅井 ψ 退化为 GD、深井 ψ 卡死——朴素 mirror descent 不是 2602.18997 的约束对偶算法。**结论：闭合质量缺口需 Z 潜变量 + STE 的 QAT 式训练（TernaryLLM 路线）或按原论文实现 SMD 对偶形式**；M4 上量前应先建 Z+STE 训练器 |
| A2 | 特征 KD 开关：前 ~18 层余弦特征 KD（δ=5，ε=0.001）增删对比 |
| A3 | s_g 初始化：amax vs MSE 最优尺度 |
| A4 | S 符号向量候选数：固定种子 1 个 vs 离散搜索少数候选取校准 MSE 最优（SpinQuant 报告的随机基方差压缩） |

每项实验 0.6B、固定步数与数据，ppl + 小基准（如 ARC-easy 子集）双指标，登记 results-log.md。
**交付**：`docs/findings-4-discrete-movement.md`（消融结论 + E1 玩具验证记录）。

### M6 27B 外推（L3，决策门）

- 目标：混合注意力（qwen35：~75% 线性 + ~25% 全注意力）+ 262K 上下文，对 FP 保真 ≥98%（20 基准套件）且 AIME/LiveCodeBench 不塌（白皮书指出这两类最先退化）。
- 成本：200–800 A100·时（重建文档 §4 外推），本机不可行 → **云预算决策门**。
- 新增工作：GDN/线性注意力路径（softmax 层沿用 M2 同款逐 matmul 折叠；`ssm_out` fold-before-matmul；激活侧 [hd,nk,rep]→[hd,rep,nk] 置换与 `gdn_v_grouped` 契约字段）、视觉塔与 emb/head 保持原契约；KV cache 旋转域存储与 RoPE 旋回（llama-kv-cache.cpp:2067）为运行时内部行为，打包侧无需处理。
- **降级目标**（若云预算不批）：用 MiniCPM5-2B（Llama 架构）与 Spark-X2.5-1.7B 做架构变体验证，坐实 C6 的架构无关性主张。
- **交付**：`docs/findings-*.md`（云预算评估或降级方案结论）。

---

## 6. 验证与度量规范

1. **ppl 协议**：WikiText-2 raw test，全文滑窗 stride=512、ctx=1024，fp16，固定种子；每次实验的完整配置记入 results-log.md（一行一实验）。
2. **契约自检**：`proto/common/check_contract.py`（M3 起用）——读 GGUF 元数据与打包权重，逐项输出 C1–C5 通过/失败。
3. **无损判据**：C3 的全张量 ‖s_g·T − Z‖∞/amax < 1e-3 作为 Phase 3 固定闸门，不通过不进 Phase 4。
4. **结果登记**：`docs/results-log.md` 从 M0 起维护；每级 findings 单独成文（命名 `findings-<序号>-<主题>.md`），在 `docs/` 目录。
5. **活文档**：每级验收后在 §4 表更新状态列，并将与计划的偏差、决策点结论回写对应里程碑小节；重大修订追加文末修订记录。

---

## 7. 风险登记

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| WSL 上 fork CPU 构建失败（cmake 缺失之外的依赖问题） | 中 | 阻塞 M3 起所有运行时验证 | M0 预留半天排错；构建日志归档；必要时对照 build_cpu_linux.sh 手工 cmake |
| M1 实测推翻假设（scale≠amax、零占比远离 12%） | 低-中 | 重建理论解释需修订 | 验收门设计即为此；修订后重估 M3 Phase 3 |
| Pascal fp16 训练数值不稳 | 中 | M3/M4 训练发散 | fp32 主权重 + fp16 计算；梯度范数监控；lr 降档预案 |
| 15GB RAM 限制 GGUF 转换/反量化大文件 | 低 | M1/M3 操作变慢 | 分张量流式处理；27B GGUF 只读元数据与抽样张量，不全量载入 |
| E1 引擎实现不确定（ψ 设计自由度大） | 高 | M5-A1 延期 | E2 先行保证主线；E1 仅限 M5，不阻塞 M3/M4 |
| 1.7B 训练算力缺口 | 高 | M4 墙钟失控 | M4 前置决策点：云短租 vs 本机降级，二选一写进 results-log |
| 27B 云预算不批 | 中 | M6 无法执行 | 降级目标（架构变体验证）已内置 |
| 磁盘耗尽（47GB） | 低 | 全线阻塞 | §3.4 预算表；每级结束清理中间 checkpoint |
| 网络（代理失效） | 中 | 数据集/依赖下载失败 | ModelScope 备份通道；WikiText-2 一次下载后本地缓存 |
| hadamard_packing.json schema 校验反复被拒（transform.name、sign_values 打包格式等细节多） | 中 | M3 Phase 4 阻塞 | T3.4 先用 base.py 校验函数空跑单测；符号表写法由 M1-T1.1b 先行确定 |

---

## 8. 未决问题 → 决策点映射（重建文档 §6）

| 未决问题 | 在哪个里程碑回答 | 方式 |
|---|---|---|
| E1 还是 E2（或混合） | M5-A1 | 同配置对拍 |
| 逐层还是端到端 | M3 先端到端；若 M4 分数差 >2 分，回试逐层初始化再端到端 | 对照实验 |
| s_g 联合训练 vs 闭式交替 | M5-A3 | 初始化消融 + 训练期 MSE 尺度试验 |
| 数据构成与算力 | M3（2K 起步）→ M4（1M token 批对比） | 数据量阶梯实验 |
| 98.2% 的最后一里路 | M6 | 仅外推后可判 |
| 1-bit 支线（ℓ∞ 单独 → {±s}） | 不在本规划范围；M5 后视兴趣单独立项 | — |

---

## 9. 日程建议（相对周）

| 周 | 内容 |
|---|---|
| W0 | M0 + M1（当前起点） |
| W1 | M2 |
| W2–W3 | M3（Phase 0–4 端到端） |
| W4 起 | M4 与 M5 并行（M4 算力决策在 W3 末做出） |
| 门控 | M4+M5 通过后再评估 M6 云预算 |

---

## 附录 A：关键路径索引

| 内容 | 路径 |
|---|---|
| 本规划与姊妹文档 | `./` |
| fork 源码（prism @ 9a9394a89） | `<FORK_DIR>/` |
| CPU 构建脚本 | `<DEMO_DIR>/scripts/build_cpu_linux.sh` |
| 格式契约文档 | `<DEMO_DIR>/MODEL-FORMATS.md` |
| 白皮书（4 份 PDF） | `<DEMO_DIR>/*.pdf` |
| 论文全文缓存（持久） | `<DEMO_DIR>/papers/*.txt` |
| venv | `<VENV>/` |
| 基座模型 | `<MODELS_DIR>/Qwen3-0.6B`、`Qwen3-1.7B` |
| 地面真值 GGUF | `<MODELS_DIR>/Ternary-Bonsai-2-27B/Ternary-Bonsai-2-27B-PQ2_0.gguf` |
| 环境坑位手册 | `ENVIRONMENT.md (local-only)` |
| 原型代码（新建） | `proto/` |
| Colab 端到端 QAT notebook（≥16GB GPU） | `./notebooks/tritfold-train-0p6b.ipynb`（生成器 make_notebook.py；核心 cell 已在本机 GPU 冒烟：恒等性/梯度流/训练步/内存路径） |
| Colab A100 serving notebook | `./notebooks/tritfold-serving-a100.ipynb`（生成器 make_serving_notebook.py；CUDA fork 构建 + llama-server + cloudflared 隧道） |
| 三值模型运行指南（三种方式归档） | `./serving-guide.md` |
| 大产物（新建） | `<WORK_DIR>/` |

## 附录 B：命令速查（从零到验证的完整链）

### 0) 依赖（一次性）

```bash
git clone -b prism https://github.com/PrismML-Eng/llama.cpp.git <FORK_DIR>
python -m venv .venv && source .venv/bin/activate
pip install torch "transformers==4.57.1" "tokenizers==0.22.2" "huggingface-hub==0.36.2" \
     "datasets==5.0.1" sentencepiece        # 版本钉死是实测结论，勿随意升级
pip install cmake && pip install -e <FORK_DIR>/gguf-py
bash <DEMO_DIR>/scripts/build_cpu_linux.sh <FORK_DIR>    # 产物在 <RUNTIME_BIN>/
```

### 1) 取模型（或用训练 notebook 自己产出）

```bash
huggingface-cli download benzeng/tritfold-1.7b-ptq1_0
```

### 2) 打包（训练 notebook 的导出 cell 产出的目录已内置 hadamard_packing.json）

```bash
python <FORK_DIR>/convert_hf_to_gguf.py <export_dir> --outfile out.f16.gguf
<RUNTIME_BIN>/llama-quantize \
    --token-embedding-type PTQ1_0 --output-tensor-type PTQ1_0 \
    out.f16.gguf out.ptq1_0.gguf PTQ1_0
# 两个类型标志是硬性要求：默认规则会把 token_embd/output 降级 Q4_K（llama-quant.cpp），
# 嵌入表不再是三值、契约即被破坏
```

### 3) 契约校验（C1–C5：元数据/符号表逐字节/折叠清单/无损/小岛全精度）

```bash
python proto/common/check_contract.py out.ptq1_0.gguf <export_dir>
```

### 4) 运行时 ppl 与生成（采样配方必带——三值分布尾部平坦，默认采样会 token 循环）

```bash
# 评估切片：任意 wiki 风格文本即可（口径需注明；与训练评估协议的差异见 findings-6）
<RUNTIME_BIN>/llama-perplexity -m out.ptq1_0.gguf -f <wiki_text>.txt -t 12 -c 512
<RUNTIME_BIN>/llama-cli -m out.ptq1_0.gguf \
    -p "The Great Wall of China was originally built to" -n 96 -st \
    --temp 0.5 --top-p 0.85 --top-k 20 --repeat-penalty 1.1
```

### 5) 真值挖掘（可选：对任何 PTQ1_0/PQ2_0 公开 artifact 做契约与统计分析）

```bash
python proto/l0prime/dump_metadata.py <gguf> [out.json]
python proto/l0prime/dequant_stats.py <PQ2_0_gguf> [out.json]
```

> 网络：受限环境可设 `HF_ENDPOINT=https://hf-mirror.com` 等镜像；本仓库不依赖任何内网资源。

---

## 修订记录

- **v1.0**（2026-09-22）：初版，由 notes + reconstruction 两份文档整理而成。
- **v1.1**（第 1 轮，技术准确性，对照 fork 源码）：① **旋转实现修正为 fork 契约式**（残差流保持原基 + 逐 matmul 在线 FWHT + embedding 逆变换，替换原 QuaRot R1/R2/R4 离线并入写法——依据 base.py:700-735 的 axis=-1/inverse 契约与真值 artifact 的 embedding 逆变换证据）；② PQ2_0 块布局更正为 scale 在前（ggml-common.h:199-207）；③ manifest 须放模型目录、transform.name 固定串、白名单与 inverse 限制落实行号；④ 构建脚本必须传仓库路径、产物位置与构建时长；⑤ M3 验收增加 llama-perplexity 运行时侧核验；⑥ 新增 T1.1b（sign_widths 对应关系）。
- **v1.2**（第 2 轮，可行性）：① M3/M4 显存账修正为"教师 top-50 logits 预计算落盘 + student 单独在卡"（1.7B 双模型同时在卡 ~6.8GB 会爆）；② 明确自研 PyTorch 训练循环（V 步吸附与 HF Trainer 不兼容）；③ lr 分工落实（s_g 码 3e-3 / 小岛与 embedding 3e-4）；④ M4 默认本机降级方案、云为可选加速；⑤ E1 先玩具问题验证隐偏置再上模型；⑥ proto/ git 化；⑦ 风险表补 manifest schema 被拒一行。
- **v1.3**（第 3 轮，编辑一致性）：§4 总览加状态列；M0/M2/M3/M4/M5/M6 补齐交付物行；§6 增活文档条款；§3.4 注明 `<BONSAI_DIR>/` 镜像同步；§2 注明纯 Qwen3 架构的小岛范围。
