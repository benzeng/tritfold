# Bonsai 三值化：数学与技术方案探索笔记

- 日期：2026-09-20
- 对象：PrismML Ternary Bonsai 2 27B 及其运行时（PrismML-Eng/llama.cpp fork）
- 源码版本：`prism` 分支，提交 `9a9394a89`（与预编译包 `llama-prism-b10709-9a9394a` 对应）
- 关键材料：`Bonsai-demo/bonsai-2-27b-whitepaper.pdf`（2026-09）、`MODEL-FORMATS.md`、fork 源码

---

## 0. 结论速览

三值化方案 = **固定的随机 Hadamard 旋转基 + 组 128 的 amax/RTN 三值量化 + base-3 trit 打包 + 激活侧 FWHT 运行时**。

数学洞察一句话：**用固定的正交旋转把三值化的难度从量化器转移到训练过程**——运行时看到的永远是分布良好、可以无脑 round 的旋转后权重；所有保质量的计算都预付在上游（专有，未公开）。

公开程度：

| 层 | 内容 | 公开程度 |
|---|---|---|
| 表示 | w = s_g·t_i，t∈{-1,0,+1}，g128 FP16 scale | 完全公开 |
| 编码 | base-3 打包（5 trit/字节）或 2 bit/trit | 完全公开 |
| 基变换 | 固定 RHT：R = H_n·S/√n | 结构公开，符号向量随模型发布 |
| 量化函数 | amax + round-to-nearest | 完全公开（对三值检查点无损） |
| **三值权重怎么训出来** | 训练/优化过程 | **未公开**（Caltech 专有 IP） |
| 执行 | FWHT 蝶形（仅加减）+ 融合整数 GEMM | 完全公开（CUDA/Metal/CPU 全后端） |

---

## 1. 表示数学

- 每个权重 `w_i = s_g · t_i`，`t_i ∈ {-1, 0, +1}`，每 128 个权重共享一个 FP16 scale。
- trit 信息量 log₂3 ≈ 1.585 bit；scale 摊销 16/128 = 0.125 bit → 理论下限 **1.71 bpw**。
- 零值带来非结构化稀疏：RTN 阈值 ±0.5·amax，对近高斯权重约 12% 置零（不存位置，只体现在数值）。
- Bonsai 2 保留全精度的张量（白皮书 Table 2）：线性注意力循环状态路径（in_proj_a/b、conv1d、A_log/dt_bias）+ 所有归一化层，共 26.2M 参数（0.0976%，52 MB），把 1.71 推到 1.72 bpw。

## 2. 编码数学（打包格式）

源码：`ggml/src/ggml-common.h:204-220`，`ggml/src/ggml-quants.c`

### PTQ1_0 — 1.75 bpw，28 字节/128 权重

- base-3 算术编码：5 个 trit 作一个三进制数存 1 字节（3⁵=243 ≤ 256，字节利用率 log₂243/8 = 99.06%）。
- 布局：24 B 主体（120 trit）+ 2 B 尾部（每字节 4 trit，共 8）+ 2 B FP16 scale。
- 定点编解码技巧（`ggml-quants.c:2231`）：
  - 编码：码 c∈[0,242] → 字节 q = ⌈c·256/243⌉；
  - 解码：x_n = ((q·3ⁿ mod 256)·3) >> 8——uint8 溢出回绕恰好是"丢弃已消费高位 trit"的模运算，一次乘加一次移位出一个 trit，无除法。
  - 验证：`tests/test-ptq1_0-element-map.cpp` 逐元素测试该映射。
- 内存布局分三级交错 `ptq1_0_stages = {32, 16, 8}`（`ggml-quants.c:2203`），便于 kernel 按 8/16/32 向量化取 trit。
- 注释确认：**对已是三值的检查点是无损重打包**；上游 TQ1_0（g256）无法无损表示 g128 三值（一个 256 尺度要跨两个组尺度，必须丢一个）。

### PQ2_0 — 2.125 bpw，34 字节/128 权重

- 每 trit 直接 2 bit（ggml type id 142；group 128）。浪费 1/4 码空间，但解包只需移位+掩码。
- 与 PTQ1_0 是"字节流量 vs 解包指令数"的权衡：
  - PTQ1_0 在带宽受限的 Ada/L4 上 decode 更快；
  - PQ2_0 在 Hopper/Blackwell/Ampere 大卡和 Apple silicon 上更快，prompt 处理明显更快（白皮书 Table 5）。

### 量化函数（RTN，无魔法）

`quantize_row_pq2_0_ref` / `quantize_row_ptq1_0_ref`：组内 amax 作 scale → `lroundf(x·id)` 归到 {-1,0,+1} → 打包。`quantize_ptq1_0` 注释："ternary codes come from the weights themselves; an imatrix has no role"。

## 3. 旋转基（方案真正的数学核心）

### 恒等结构

原层 y = Wx。取正交阵 R = (1/√n)·H_n·S（n=1024，H_n 为 Sylvester Walsh–Hadamard 阵，S 为固定 ±1 对角阵），则 y = (W·R⁻¹)(R·x) 恒等。方案：**令 W′ = W·R⁻¹ 为三值矩阵存起来**，推理算 f(x) = W′(Rx)（白皮书 2.4 节公式）。

### 为什么旋转后 RTN 就够用

- 正交变换保持内积/范数，但改变坐标分布形状：Hadamard 元素全是 ±1/√n，每个输出坐标是整行等权加减混合 → 把 LLM 权重的 outlier **摊平**到整个 1024 维块，旋转后分布近高斯、尾部薄。
- 效果：(1) amax 不再被个别 outlier 撑大；(2) round 误差均匀化，不集中于少数敏感方向。
- 思想家族：QuIP# 的 incoherence processing（随机 Hadamard）、SpinQuant（学习旋转，白皮书参考文献 [28]）。Bonsai 选**固定** Hadamard + 固定符号（即 RHT）；S 打破 Walsh 函数周期结构、去相关。
- 代价：免学习、硬件友好，但"怎样旋转才能三值化"的能力转移到上游训练侧。

### 契约式对接（hadamard_packing.json）

- 上游打包管线随检查点输出 `hadamard_packing.json`：schema 1/2，kind=`hadamard-weight-fold`，变换名 `normalized-signed-sylvester-walsh-hadamard`，block_size（2 的幂），sign_mode（identity/explicit + 符号表），被折叠张量清单，status=`requires-matching-runtime`。
- `conversion/base.py` `add_hadamard_metadata()` 校验后写入 GGUF 元数据。
- `conversion/qwen.py`：GDN/线性注意力头重排；**折叠张量保持训练序**，激活侧运行时置换（`perm_hd/perm_nk/perm_rep`，[hd,nk,rep]→[hd,rep,nk]）。

## 4. 运行时执行

源码：`src/llama-graph.{h,cpp}`、`src/llama-impl.h:57`、`ggml/src/ggml-cuda/fwht.cu`

1. `llama_hadamard_transform`（`llama-graph.h:23`）：每个折叠权重对应 rot（块对角 ±1 矩阵）、signs（可空）、可选激活置换参数。
2. 图构建：先 `ggml_mul` 乘符号（融合），再 `llama_mul_mat_hadamard`：激活 reshape 成 n 列 → matmul(rot) → 打 `GGML_HINT_SRC0_IS_HADAMARD` 标记 → reshape 回去。同一激活被多个折叠权重复用时 `hadamard_memo` 去重。
3. 后端识别标记后**不走稠密 GEMM，改派 FWHT kernel**（`ggml-cuda.cu:1822`；CUDA 还做图模式匹配把 sign-mul 融进变换）。
4. FWHT 实现（`fwht.cu`）：教科书蝶形，仅加减法，O(n log n)。
   - N ≤ 2048：寄存器路径，warp 内 `__shfl_xor_sync` 做前 log₂32 级，寄存器内做剩余级；
   - 更大 N：shared-memory 路径（一行一个 block，256 线程）；
   - 符号翻转融合在加载路径（白皮书附录 A.2 一致）。
5. 反向路径：embedding/latent 表有 `hadamard_inverses` 逆变换；KV cache 量化后做 RoPE shift 时需"旋回 → RoPE → 旋入"（`llama-kv-cache.cpp:2067`）。
6. 加载日志：`loaded %zu Hadamard-folded weight(s) ... using %zu rotation(s) and %zu sign vector(s)`；安全检查：`Hadamard-folded weight ... consumed without its activation transform`（不支持折叠的 matmul 路径直接报错，不静默算错）。

## 5. 计算数学（kernel）

- 权重从不反量化成稠密 FP16。CUDA MMQ kernel（`mmq-instance-pq2_0.cu` / `mmq-instance-ptq1_0.cu`，tile 加载见 `mmq-load-tiles.cuh`）把码解成整数、与量化激活整数点积（dp4a/Tensor Core），**在 GEMM 内部**乘组 scale。
- Metal：`ggml-metal/kernels/{mul_mm,mul_mv,quantize}.metal`、`kernels/dequantize.h`；CPU：`ggml-cpu/{ops.cpp,repack.h}`；SYCL/Vulkan/BLAS 也有 hadamard 相关路径。
- 模型实现：`src/models/qwen35.cpp` / `qwen35moe.cpp`（Qwen3.8-27B 混合注意力：~75% 线性 + ~25% 全注意力）。
- 测试：`tests/test-ptq1_0-element-map.cpp`、`test-ptq1_0-cuda-dot.cpp`、`test-quantize-fns.cpp`、`test-backend-ops.cpp`。

## 6. 格式生态（MODEL-FORMATS.md 要点）

| 格式 | ggml type id | group | 谁能读 |
|---|---|---|---|
| 旧版 Q2_0（弃用） | 42 | 128 | 仅 prism-v5 及更早 |
| 官方 Q2_0 | 42 | 64 | 主线 llama.cpp + prism-v7+ |
| PQ2_0 | 142 | 128 | prism-v7+ fork |
| PTQ1_0 | (fork 私有) | 128 | prism-v7+ fork |

- demo 选择逻辑：`Bonsai-demo/scripts/common.sh` `pq2_0_ready_backend`（mac/cpu/cuda/rocm/hip → 优先 PQ2_0，否则 _g64 官方文件）。
- 旧版文件在 v7 上直接报错并指向 MODEL-FORMATS.md（报错原文存在于二进制与源码中）。
- Bonsai 2 三档：PTQ1_0 5.93 GB / PQ2_0 7.25 GB / Q2_0 7.6 GB（Q2_0 仅测试用——主线能加载但输出乱码，因为权重在旋转基里，故单独发布且文件名标注 fork-required）。
- MLX 打包：类 PQ2_0 但每组多存一个冗余 FP16 bias（36 B vs 34 B/128），整包 8.49 GB（含全精度视觉塔）。

## 7. 关键数字（白皮书）

- 模型：基于 Qwen3.8-27B，24.35B 语言 + 0.47B 视觉 + 2.54B emb/head = 27.36B；262K 上下文。
- 基准：20 项套件平均 83.9（FP16 基线 85.4 的 98.2%；Qwen3.6 83.6；IQ2_XXS 75.2 且更大）。
- Agentic：Terminal-Bench 2.1 = 52.8、SWE-bench Verified = 60.8（约 FP16 的 3/4）。
- 吞吐：RTX 5090 142.5 tok/s（PQ2_0）；M5 Max 46.8、M5 Pro 27.7 tok/s；能耗最低 0.58 mWh/tok。
- 智能密度：0.444 1/GB（FP16 0.051，IQ2_XXS 0.276）。

## 8. 未公开部分（探索边界）与二次探索结论（2026-09-20 补充）

### 8.1 本地取证结论：秘密管线的"接口"完全可见

- 转换器**没有任何读取三值码的代码**（conversion/*.py 无 ternary/trit 命中）→ 上游检查点就是普通 bf16 safetensors，但每组 128 个权重只取 {−s, 0, +s} 三个值（旋转基下精确可表示），RTN 重打包无损。
- 上游随检查点输出 `hadamard_packing.json`：变换名、block_size、符号表（identity/explicit）、逐张量记录（`axis: -1` = 沿输入特征折叠；`role` = `fold-before-matmul` 或 `inverse-after-lookup`）。
- GGUF 契约 `prism.hadamard.*`：version / block_size / transform / axis=`input-last-dimension` / sign_mode / weight_names / sign_widths+sign_values / inverse_weight_names / gdn_v_grouped。
- 可折叠张量白名单（conversion/base.py `_HADAMARD_KINDS`）：lm_head(output)、全部注意力投影、全部 MLP/专家 MLP、ssm_out；`inverse-after-lookup` 只允许 token_embd.weight——**embedding 与 lm_head 共享同一张折叠张量**，查表后做逆变换还原。
- 已验证架构：LLAMA / QWEN3 / QWEN3MOE / QWEN35 / QWEN35MOE / QWEN3NEXT。
- 演进时间线：1-bit 8B（2026-03，无旋转）→ Image 4B（2026-05）→ 27B（2026-07，无旋转，参考文献无旋转类论文）→ Bonsai 2（2026-09，新增固定 Hadamard 基 + 全精度小岛；白皮书承认存在"pre-rotation build"）。旋转是第二代才加入的质量升级。
- 姊妹仓库（mflux-prism 图像模型）同样模式：标准 MLX affine 量化测试证明"权重到达时已量化，工具只打包"。
- PrismML-Eng org 全部仓库：mlx / llama.cpp / mlx-swift / mlx-c / sglang / Bonsai-demo / Bonsai-Image-Demo / mflux-prism / image-studio——**全是推理侧，无任何训练/打包仓库**。

### 8.2 身份与 IP（公开报道，已确认）

- 创始人 CEO **Babak Hassibi**：Caltech 教授，压缩理论史上 **Optimal Brain Surgeon（1992）的 "H"**——OBS→GPTQ 谱系的源头。联合研究负责人 Sahin Lale（Caltech 博士）、Omead Pooladzandi（优化/二阶方法）。顾问含 Ion Stoica。
- WSJ（2026-03-31）：**IP 归 Caltech 所有，PrismML 独家授权**。Hassibi："We spent years developing the mathematical theory required to compress a neural network without losing its reasoning capabilities"；框架 "can be applied to any of them"（架构无关）。
- 融资 $22.25M（Khosla、Cerberus、Caltech；Google/Caltech 算力资助）；CNBC 2026-07 报道 Apple 洽谈授权（用于 iPhone）。
- 无方法论文、无专利、无 arXiv 公司署名文章。

### 8.3 理论谱系（创始人公开 arXiv，推断与 IP 强相关）

| 论文 | 内容 | 与三值化的关联 |
|---|---|---|
| arXiv 2311.02270（2023-11）Akhtiamov, Ghane, Hassibi | 强 ℓ∞ 正则下回归权重"聚到两个反号值" | 1-bit 的理论源头 |
| arXiv 2402.10474（2024-02）同三人 | 1-bit 量化 + 稀疏化（两水平+零 ≈ 三值） | 三值 = 1-bit + 零态 |
| arXiv 2510.16250（2025-10） | random-features 模型 1-bit 无泛化损失 | 可量化性理论 |
| arXiv 2602.18997（2026-02）含 Pooladzandi | Matrix Stochastic Mirror Descent 的隐偏置与收敛 | **把训练权重"引向"离散结构的机制** |
| arXiv 2603.10485（2026-03，与 1-bit Bonsai 同月） | 过参数化 regime 的对偶空间预条件 GD | 训练效率 |

### 8.4 方法形态推断（标注：推断）

**几乎确定存在实质性的逐模型优化/训练阶段**，而非纯 PTQ 舍入：

1. 文献中纯旋转+RTN 的 2-bit（QuaRot 类）在同类评测上掉点严重，98.2% 保真度无公开方法可及——差值就是秘密训练阶段贡献的；
2. 官方措辞从不说 quantization-only，只说 "representation transformation / moves into a ternary representation"；
3. 招聘含大规模训练与 post-training 平台岗位（fine-tuning/RL）；中文媒体称其为"微调模型"（弱证据）；
4. 创始人理论线 = **镜像下降/强正则把权重驱向 {-1,0,+1}**——与出货物"旋转基下 RTN 无损的三值权重"严丝合缝。

**最可能的方法形态**：固定 Hadamard 基下做约束优化——把权重当作在旋转基中被训练的对象，用带离散结构诱导偏置的目标（镜像下降/正则化 + 大概率蒸馏原模型）直接优化三值码与组尺度；输出 = 三值权重 + `hadamard_packing.json`。**与 SpinQuant 互为镜像**：SpinQuant 学习旋转去适应权重，PrismML 固定旋转、训练权重去适应旋转。

学术 SOTA 对照（公开方法里最近的邻居）：表示侧 QuaRot（固定 Hadamard，4-bit RTN）/ QuIP#（RHT + 码本）；质量侧（1-2 bit 训练类）BitDistiller、LLM-QAT、EfficientQAT、PV-Tuning、OneBit、TernaryLLM——无一在 27B 规模、1.7 bpw 达到 98% 保真。

### 8.6 等效方案重建（2026-09-21，独立文档）

基于创始人五篇论文的数学 + QuaRot/SpinQuant/PV-Tuning 等公开工程，已写出可实现的等效管线与验证计划，见 [bonsai-ternarization-reconstruction.md](bonsai-ternarization-reconstruction.md)。核心结论：

- 方法内核 = 强正则/镜像下降的**隐偏置量化**（ℓ1→零、ℓ∞→二值、组合→三值；矩阵 SMD 收敛到"满足蒸馏约束、Bregman 最接近原权重"的类三值解）；
- **固定 Hadamard 基的作用是把权重制造成理论成立所需的近高斯非相干分布**——这解释了旋转为何在第二代才引入、以及对称三值（无 shift）为何成立；
- 与 SpinQuant 互为镜像：SpinQuant 学旋转适配权重，Bonsai 固定旋转训权重适配基；
- 等效管线：QuaRot 式基底插入 → 旋转基初始化 → SMD（三架势 ψ）或 PV-Tuning 交替法的离散约束蒸馏 → RTN 吸附 → 现成工具打包；27B 估计 200–800 A100·时；
- 可分级对拍验证：已发布的 Ternary-Bonsai-1.7B/4B/8B（无旋转一代）即地面真值。

### 8.5 调研死胡同（备查）

HF 模型卡（本网络不可达）、LinkedIn（451）、The Information（付费墙）、Google Patents（无结果）、web.archive.org、YouTube 字幕、Forbes、Semantic Scholar（限流）。WSJ 全文存 /tmp/wsj.txt（易失）。

## 10. 原型可行性评估与本地资产盘点（2026-09-21 深夜归档）

### 10.1 环境事实（经实测，修正两处此前误判）

- **GPU**：GTX 1060 6GB（Pascal sm_61）。**torch 2.7.1+cu126 实测可用**（fp16 matmul 冒烟通过）——此前"新版 wheel 放弃 Pascal"的判断错误。transformers 钉 4.57.1。栈在 `<VENV>/`。
- **HF 下载**：直连不通，但宿主机代理可用：`export https_proxy=<proxy>`（网关 IP，`ip route | awk '/default/ {print $3}'`）；ModelScope 直连可作备份通道。
- **Pascal 注意力事实**（引自 UncensoredReseach/ENVIRONMENT.md）：bf16 仅 math SDPA 后端；**fp16 可用 mem-efficient**——推理/训练用 fp16。6GB 显存可载 2B 级 bf16 模型（3.4–5.0GB）。
- **CUDA 13.3 预编译 fork 二进制在本机不可用**（Pascal 被 CUDA 13 移除 + 缺 libcudart）→ 端到端验证走 **CPU 编译 fork**（`Bonsai-demo/scripts/build_cpu_linux.sh`）。
- 12 核 / 15GB RAM / 47GB 空闲盘。

### 10.2 本地模型资产（<MODELS_DIR>/）

| 资产 | 用途 |
|---|---|
| `Qwen3-0.6B`（完整 safetensors） | L0/L1-微观 基座 |
| `Qwen3-1.7B`（完整 safetensors） | L1 完整对拍基座（上一代 Ternary-Bonsai-1.7B 同基座） |
| **`Ternary-Bonsai-2-27B/Ternary-Bonsai-2-27B-PQ2_0.gguf`（7.2GB）** | **地面真值 artifact**：契约验证 + 权重统计挖掘 |
| MiniCPM5-2B（Llama 架构）、Spark-X2.5-1.7B（绑定嵌入，对应 inverse-after-lookup 情形） | 备用基座/架构变体 |

### 10.3 真实 artifact 契约验证（2026-09-21，gguf-py 直读 PQ2_0 元数据）

全部命中此前从源码推得的契约：`prism.hadamard.version=1`、`block_size=1024`、`transform=normalized-sylvester-walsh-hadamard`、`axis=input-last-dimension`、`sign_mode=explicit`、`sign_widths=[17408]`（=MLP 中间维）、`gdn_v_grouped=True`、`general.architecture=qwen35`、`general.file_type=141`（PQ2_0）。KV 共 52 条。
待做：weight_names / inverse_weight_names 字符串数组的完整解码（快速脚本读取偏移有误，gguf-py 正规 API 可解）。

### 10.4 原型分级计划（明日接续点）

- **L0′ 真值挖掘**（纯 CPU，小时级；**明天的起点**）：完整解析 PQ2_0 的 Hadamard 元数据 + 反量化若干张量 → 统计检验：(a) 稀疏率（零占比，理论预期 ~12% RTN 高斯 vs 实测）；(b) scale == 组 amax？(c) 旋转域权重分布（直方图/峰度 vs 高斯）；(d) 逐张量折叠清单与角色；(e) embedding 是否真为旋转域存储。工具：fork gguf-py + numpy。
- **L0 机制验证**（小时级）：Qwen3-0.6B，PyTorch FWHT 插入旋转 → RTN 三值 → WikiText ppl 对比（FP / 无旋转 RTN / 旋转 RTN 三档）。
- **L1-微观**（天级）：0.6B 蒸馏（冻结主体、训组尺度 + PV 子空间吸附，数百步）→ 产出 `hadamard_packing.json` → fork conversion → PTQ1_0 GGUF → CPU 版 fork 加载生成。验收：端到端跑通 + 三值 ppl 显著优于裸 RTN。
- **L1 完整**（周级/云）：Qwen3-1.7B 充分训练，对拍已发布 Ternary-Bonsai-1.7B 分数（白皮书附录 C 有数，无需下载）。

### 10.5 相关文档索引

- 方法重建：[bonsai-ternarization-reconstruction.md](bonsai-ternarization-reconstruction.md)（四阶段管线 + 双引擎 + 验证计划）
- 环境坑位手册：ENVIRONMENT.md (local-only)
- 论文提取缓存：/tmp/papers/*.txt（易失，重启后需重新提取）

## 9. 参考文件位置

- 源码：`<FORK_DIR>/`（PrismML-Eng/llama.cpp，prism 分支 @ 9a9394a89）
- Demo 文档：`<DEMO_DIR>/{MODEL-FORMATS.md,README.md,SPECULATIVE.md}`
- 白皮书：`<DEMO_DIR>/*.pdf`（提取文本：/tmp/*whitepaper*.txt，易失）
- 预编译二进制：`<FORK_DIR>` 同级 tar.gz（注意：本机缺 libcudart.so.13，CUDA 13.3 构建暂不能直接跑）
