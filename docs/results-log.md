# Bonsai 三值化等效管线：实验结果登记

- 登记规范见 [bonsai-ternarization-implementation-plan.md](bonsai-ternarization-implementation-plan.md) §6。
- 配置哈希 = oneLLM 仓库 git commit（proto/ 代码纳入 oneLLM 管理，不再单独 git init——2026-09-22 起 oneLLM 已是 git 仓库，T0.6 据此调整）。

| 日期 | 级别 | 配置 | 指标 | 备注 |
|---|---|---|---|---|
| 2026-09-22 | M0 | cmake 4.4.3、gguf 0.19.0(fork editable)、WikiText-2(test 4358行)、fork CPU 构建 b10709 | llama-cli --version OK | 构建 ~25min@12核；产物 Bonsai-demo/bin/cpu/；T0.6 调整：proto/ 纳入 oneLLM 仓库 |
| 2026-09-22 | M1 (L0′) | dump_metadata.py + dequant_stats.py @ 27B PQ2_0 | 零占比 0.3278；scale==amax 0 违例；emb kurt 1.51→3.01 | 五项检验全过；修正笔记两处记载；详见 bonsai-l0prime-findings.md |
| 2026-09-22 | M2 (L0) | run_rotation_test.py @ Qwen3-0.6B, WikiText-2 77.3k tok, ctx1024/stride512, fp16 | FP 19.29 / naive RTN 1.12e9 / 旋转 RTN 1.24e8；恒等性 rel=0.24% argmax=99.61% | "QuaRot 效应"对权重-only 三值 RTN 证伪（逐层比值≈1）；验收门修订；详见 bonsai-l0-findings.md |
| 2026-09-22 | M3 消融 | train.py 40 步×2，512K token 教师缓存 | amax 起点 1.09e8→10758；zerofrac 起点 1.28e6→3123@20 | zerofrac 起点优 93×，选为主训练默认 |
| 2026-09-22 | M3 主训练 | 300 步 +V-step τ1% lr1e-3+clip | 1.28M→2804（振荡，最佳 2320@200） | V-step 末样本排序噪声是振荡源 |
| 2026-09-22 | M3 主训练 | 600 步 无V-step（尺度+小岛） | 1.28M→**166.8**（单调平滑，每百步 ×0.75） | 纯 P-step 微尺度有效；零占比保持 0.3224≈artifact 0.3278 |
| 2026-09-23 | M3 Phase3/4 | export→convert→PTQ1_0→运行时（600 步 ckpt） | C1–C5 全 PASS、maxdiff=0；运行时 ppl 87–211 vs PyTorch 166.8；1.75bpw 精确 | 端到端打通；质量门②（≤41）未达→续训计划见 findings §6；详见 bonsai-l1micro-findings.md |
| 2026-09-23 | M3 续训 | resume 1000 步 @ 2M token 缓存（总 1600 步） | 167→**91.6**（末 300 步打平） | 纯尺度训练 ~91 饱和（3.3×FP）；闭合需码迁移/E1，转 M5-A1 议程 |
| 2026-09-23 | M3 v3 链条 | export→convert→PTQ1_0→运行时（cont1000 ckpt） | 契约 PASS；运行时 ppl ≈85 vs 91.6；生成连贯英文句 | qwen3-0.6b-v3.ptq1_0.gguf（157MB）；生成从"the the the"变为连贯句 |
| 2026-09-23 | M5-A1 | V-step 三变体 @cont1000（91.6 起点） | EMA 排序 308 / 带符号 EMA 3054 / 接受回退闸门 88.5-91（全拒，flips 0 rej 19.7K/步） | 离散翻转在微预算不可行；闭合需 Z+STE（详见 bonsai-m5a1-findings.md） |
| 2026-09-23 | M5-A1 | E1 SMD 玩具（d=512≫n=64 三值稀疏真值） | 浅井 ν=4.1：SMD≈GD（34.6%/34.8% 三值）；深井 ν=14：卡死 loss 0.216 不插值 | 朴素损失梯度 mirror descent ≠ 2602.18997 的约束对偶形式；E1 须按原论文实现 |
| 2026-09-24 | M5-A1 | 逐层 Z+STE 三变体 @cont1000（91.6 起点） | 教师输入 31,970 / 序贯贪心 474,363 / 组级闸门 91.64（28/28 组回退，局部 MSE 全改善但全局全变差） | 局部目标与全局 LM 目标错位；本机质量上限 ≈3.3×FP；详见 bonsai-m5a1-findings.md §5-6 |
| 2026-09-24 | 工具 | notebooks/tritfold-qat-e2e-colab.ipynb（端到端 Z+STE QAT，T4 16GB 目标） | 核心冒烟：恒等性 4.6e-5、Z/theta/小岛梯度流、3 步 loss 14→10、含 Adam8bit 峰值估 ~11GB | 闭合 91.6→41 缺口的载体；STE 返 fp16 + 逐样本反传降峰 |
| 2026-09-26 | e2e QAT | Colab 战役全记录（T4 1000 步 + lr 阶梯 + v6 系列 + bf16 手术，详见 bonsai-e2e-qat-findings.md） | **48.08 @3500（1.77×FP）**；真值 80.75 破 91.6 天花板；moved 19.1% 无碍 | 平台主因=数据回收；fp16 测量通胀制造假退化/假里程碑；端到端 STE 配方=2e-4+大数据+温热优化器 |
| 2026-09-26 | e2e 收尾 | 本机 qat-v1 全链（convert→PTQ1_0→契约→运行时） | C1–C5 全 PASS、maxdiff=0、1.75bpw 精确；llama-perplexity 64.99±5.6（vs PyTorch 48.08，协议不同）；生成连贯 | M3-② 部分闭合（48.08 vs 门 41）；M4 上量已正当 |
| 2026-09-26 | M4 (L1) | Qwen3-1.7B 从零 5000 步 @ A100（M4 notebook，配方零调整） | **best 28.77 = 1.41×FP** @4700；step 2600 破 1.5× 质量门；skip 0/40 全程；断连热恢复零损失 | 规模效应：1.77×→1.41×；详见 bonsai-m4-findings.md |
| 2026-09-26 | M4 收尾 | 本机 m4 全链（convert→PTQ1_0→契约→运行时） | 契约 PASS（宽度自动推导 [2048,6144]）、maxdiff=0、424MB=9.13×；运行时 ppl 38.10±2.4（三协议分解：28.77 真质量 ×1.24 切片 ×1.07 运行时=38.1 闭合；FP 基线 20.44 两协议重合） | **M4 质量门达成，项目两规模全链闭环** |
| 2026-09-26 | M5′ 前置 | ARC-c 0-shot 基线（harness 本地验证 + M5′ 验收分母） | FP 0.357/0.378；三值 M4 0.195/0.224（**低于随机 0.25**，"自信地错"形态） | 验收线 = 70%×FP = acc_norm≥0.265；起点 0.224 |
| 2026-10-03 | M5′ 训练 | 1.7B 指令混合蒸馏 3000 步 @ A100（GGUF 自举，冷优化器） | wiki ppl 28.77→**25.73@800（1.26×FP 纪录）**→27.87 终态；护栏未触发；skip 0 全程 | 指令正迁移→后段温和竞争，两段式曲线 |
| 2026-10-03 | M5′ 评估 | 生成抽检 + ARC-c（FP 0.358/0.377） | 英文指令跟随质变（"Certainly!..."）；中文❌；ARC 0.187/0.234（门 0.265 未达，测位错配归因） | "学会回答，没学会知道"——详见 bonsai-m5p-findings.md |
| 2026-10-03 | M5′ 收尾 | v0.2 本机全链（convert→PTQ1_0→契约→运行时） | 契约 PASS、maxdiff=0、424MB；运行时 ppl **35.20±2.3**（v0.1 为 38.10——指令蒸馏连运行时口径 ppl 都改善 8%）；who-are-you 同框对照实拍 | 运行时 ppl 链闭环：25.73bf16→(协议链)→35.2 与 v0.1 分解一致 |
| 2026-10-03 | M5″ 训练 | v0.3 三流混合 3000 步 @ A100（v0.2 GGUF 链式自举） | wiki ppl best **25.20（1.24×FP 纪录）**终态 26.84；零工程事故 | 语言建模三连改善 28.77→25.73→25.20 |
| 2026-10-03 | M5″ 判卷 | ARC 三协议 + 抽检 | 似然 0.208/**0.261**（+11.5%，破随机线；门 0.265 差 0.004 记❌）；生成式 0.227（未解析 39%→4.7%）；红行星题无 Mars（"star in Pisces"） | **1.75bpw 知识天花板实证**；详见 bonsai-m5pp-findings.md |
| 2026-10-03 | M5″ 方法论 | Phase 0 评估伪影解剖 | FP chat "0.224" = OKAY→A 解析伪影 + think 截断双重假；fp16 裸格式 0.733 与 A100 一致（fp16 无罪） | 自动解析必带字母分布审计；enable_thinking kwarg 不可信须手动闭合 |
| 2026-10-03 | M5″ 判卷补全 | chat 修正版 ARC（强制闭合 think + 严格解析） | FP 0.573/**已解析 0.789**（字母知识真值）；v0.3 0.034/**86% 空输出**（空 think 伪影触发 EOS） | 三协议矩阵闭合：知识天花板证据链完整 |
| 2026-10-03 | M5″ 收尾 | v0.3 本机全链（convert→PTQ1_0→契约→运行时） | 契约 PASS、maxdiff=0、424MB；运行时 ppl **33.51±2.1**（38.10→35.20→33.51 三连降）；红行星运行时复现（"star in the constellation"，无 Mars） | **v0.3 发布完成：GitHub tag + HF tritfold-1.7b-knowledge-ptq1_0** |
| 2026-10-03 | 思想收官 | 中心问题实证回答成文（三层：纯转换灾难 / 蒸馏分层恢复 / 转换-训练合并） | README 英文版发布（60b54ec）+ 本档案 | 分布性能力 ~80% 可恢复 vs 信息性能力 1.58bit 天花板——项目核心论点的最终表述 |
| 2026-10-03 | 思想归档 | 基座遗产三项（骨架 81%/直觉/起点）+ BitNet 对比成文 | README 英文版（72eb99f）+ 本档案 | "基座贡献怎么说话（~80%），贡献不了知道什么"；产业答案 = 转换架构 + 为事实付算力 |
| 2026-10-03 | M7 zh 臂 | v0.4-zh 3000 步（Belle 3+wiki 3+ultra 2，自举 v0.3 GGUF） | wiki best 25.39/终态 27.61；中文成句质变但话题漂移（AI 题全对）；ARC 0.243 回落；sciq 0.363 | sciq 基线补齐：FP 0.75/v0.3 **0.40**（超随机 59%）→ 知识天花板修订为"非全有全无"；零和能力分配实证 |
| 2026-10-04 | M7 know 臂 | v0.4-know 3000 步（FWED ×2 + sciq 流直训，自举 v0.3 GGUF） | ARC 0.264（门差 0.001 噪声）；**sciq 0.402（vs v0.3 0.398，+0.004 平坦）** | **知识饱和定律确立：对症语料直训也装不进**；v0.4 不发模型只发结论；详见 bonsai-m7-findings.md |
| 2026-10-04 | 方法论归档 | 蒸馏三角色（语料=触发器，教师=知识源，学生=接收方）+ 闭源适配路线（B 两阶段为主） | 详见 bonsai-distillation-methodology.md | "学生学的不是语料，是教师对语料的理解"；三值化需白盒→PrismML 同 |
| 2026-10-04 | M8 训练 | v0.5 跨尺寸教师 3000 步 @ A100（8B 教师→1.7B 三值学生） | wiki ppl 28.84→**20.62（FP 的 101%！）**；700 步即到 best | **语言建模无损化**：1.75 bpw ≈ FP |
| 2026-10-04 | M8 判卷 | ARC + sciq + 抽检 | ARC acc 0.208→**0.355（FP 的 99%）**；acc_norm 0.245（锐化签名）；sciq 0.398→**0.439（FP 的 63%）**；红行星仍无 Mars | 双因素模型：语言/排序=教师瓶颈、深层知识=教师+容量共同瓶颈 |
| 2026-10-04 | 理论梳理 | 数学框架系统化：8 组件清单、信息瓶颈 9.36×、两类信息（低秩/密集）分离、优化景观不对称、综合框架"三值化=低通滤波+知识蒸发"、5 个可测假设推导 | 详见 bonsai-math-framework.md | Hadamard≈二值 Fourier 视角统一解释全部实验结果 |
| 2026-10-04 | 工具固化 | dryrun_notebook.py：本机干跑验证脚本（GTX 1060 / 0.6B / mock Colab+bitsandbytes+huggingface_hub） | cells 1-6 全绿（安装/env/config/math/STE）；数据 cell 本机 OOM（wiki-103 太大，非 notebook bug） | 以后每个 notebook 交付前先干跑，抓 NameError/Indentation/shape |
| 2026-10-04 | M9′ 1b 第一轮 | KL+cos wiki6+sciq2 无ultra | sciq 0.503 / 生成崩溃（think循环） | 零和代价实证 |
| 2026-10-04 | M9′ 1b 修正 | +ultra 2（4+2+2） | sciq 0.536/0.485（FP 72%）；ARC 0.246/0.279（FP 74%）；生成恢复；ppl 26.52 | **v0.6 发布**：HF tritfold-1.7b-feature-ptq1_0；运行时 GGUF 首次正确生成 Mars |
| 2026-10-04 | M9′ 本机验证 | v0.6 全链（convert→PTQ1_0→契约→运行时） | 契约 PASS、424MB、1.75bpw 精确 | 运行时 ppl 待补 |
| 2026-10-04 | M9′ 运行时 ppl | v0.6 llama-perplexity（同协议 wt2_30k c512） | **35.68±2.3**（v0.1 38.10→v0.2 35.20→v0.3 33.51→v0.6 35.68） | v0.6 比 v0.3 略高（ultra 加入的 ppl 代价），仍在最优区间 |
| 2026-10-05 | M9′ 1a 对照 | 1a 纯 KL 1500 步（同 4+2+2、同起点 v0.3 GGUF、同教师） | sciq 0.471/**0.458**（±0.034，n=845 去污染）；ARC 0.236/**0.268**（n=1172）；wiki ppl 25.44（best 25.32）；抽检 2/3 退化循环 | **余弦净贡献：sciq +0.027 acc_norm（噪声内）/+0.065 acc（≈2.7σ）、ARC +0.011（无贡献）**；1a 较 v0.3 +0.060 → 触发 M7 sciq 流 bug 复查（见下行勘误）；对症增益不外溢 ARC（语料特异）；**余弦第二角色：生成稳定器**（同 ultra 剂量下纯 KL rollout 塌缩、ppl 反而更优——teacher-forced 与 free-running 口径分离） |
| 2026-10-05 | M9′ 运行时对照 | 1a 全链本机重建（quantize 需 `--output/--token-embedding-type PTQ1_0`，否则 embd/output 落 Q4_K、711MB≠431MB）+ llama-server 双协议 × 双模型多次采样（完整配方含 repeat_penalty 1.1） | 契约 C1-C5 全 PASS maxdiff=0；**Mars：1a 1/3 vs 1b 0/9**（随机事件，非 v0.6 能力）；**1a chat 3/3 流畅成句**（A100 循环塌缩=缺 repeat_penalty 的伪影）；1a 字母枚举退化 1/9 | **稳定器假说降级**：余弦仅降低宽松采样下的循环倾向，正确配方下纯 KL 臂无实用缺陷；纪律确立——生成类结论须多协议×多次采样×双臂同场后才入档 |
| 2026-10-05 | M10 训练 | 三杠杆叠加（8B KL 教师[缓存阶段设计：常驻 8 min 出 top-50 后释放]+ sciq 全量流 + 18 层余弦 + ARC-C/E 对症流 + ultra 2，自举 v0.3，1500 步 @ A100 ~22GB 峰值） | wiki ppl best **20.70（1.014×FP，8B 杠杆复现）**；8B 教师 ARC 参考线 0.474/0.472 | 双教师常驻 35GB 静态必爆 40GB——缓存设计一劳永逸；干跑抓 k 泄漏必崩 bug |
| 2026-10-05 | M10 判卷 | sciq + ARC + 抽检（n=845 去污染 / n=1172） | **sciq 0.569/0.530（FP 76%）；ARC 0.296/0.319（FP 84%，超 0.30 门、随机线上方决定性）**；抽检 3/3 成句零循环；ARC 近重叠 25/1172 封死泄漏 | **可加性盲预测 0.526 vs 实测 0.530（三杠杆线性可加）；对症双向验证（ARC 流 +0.05~0.06）；两个"知识天花板"系统性改写**；四维同场无零和；详见 findings-12-stacked-levers.md |
| 2026-10-05 | **M7 勘误** | 本机复算 sciq 流（Qwen3 tokenizer + arrow 缓存） | v0.4/v0.5 通路逐条 ≥512 过滤：11679 条仅 81 条（0.7%）存活 → 池 0.041M tok/81 窗，know 臂 3000 draw=**每窗 37 遍、覆盖率 3.3%**；m9p 通路全量 1.27M tok/2485 窗（1a 覆盖 121%） | **"知识饱和定律"的对症语料半边撤回**（空流伪影）；存活：FWED 泛语料无效、零和分配；v0.5 的 +0.041 实为纯教师质量效应（sciq 流同样为空）；三杠杆分解：教师质量 +0.041 / 对症语料 +0.060 / 余弦特征 +0.027~0.065 |
