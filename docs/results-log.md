# Bonsai 三值化等效管线：实验结果登记

- 登记规范见 [implementation-plan.md](implementation-plan.md) §6。
- 配置哈希 = oneLLM 仓库 git commit（proto/ 代码纳入 oneLLM 管理，不再单独 git init——2026-09-22 起 oneLLM 已是 git 仓库，T0.6 据此调整）。

| 日期 | 级别 | 配置 | 指标 | 备注 |
|---|---|---|---|---|
| 2026-09-22 | M0 | cmake 4.4.3、gguf 0.19.0(fork editable)、WikiText-2(test 4358行)、fork CPU 构建 b10709 | llama-cli --version OK | 构建 ~25min@12核；产物 Bonsai-demo/bin/cpu/；T0.6 调整：proto/ 纳入 oneLLM 仓库 |
| 2026-09-22 | M1 (L0′) | dump_metadata.py + dequant_stats.py @ 27B PQ2_0 | 零占比 0.3278；scale==amax 0 违例；emb kurt 1.51→3.01 | 五项检验全过；修正笔记两处记载；详见 findings-1-truth-mining.md |
| 2026-09-22 | M2 (L0) | run_rotation_test.py @ Qwen3-0.6B, WikiText-2 77.3k tok, ctx1024/stride512, fp16 | FP 19.29 / naive RTN 1.12e9 / 旋转 RTN 1.24e8；恒等性 rel=0.24% argmax=99.61% | "QuaRot 效应"对权重-only 三值 RTN 证伪（逐层比值≈1）；验收门修订；详见 findings-2-rotation-study.md |
| 2026-09-22 | M3 消融 | train.py 40 步×2，512K token 教师缓存 | amax 起点 1.09e8→10758；zerofrac 起点 1.28e6→3123@20 | zerofrac 起点优 93×，选为主训练默认 |
| 2026-09-22 | M3 主训练 | 300 步 +V-step τ1% lr1e-3+clip | 1.28M→2804（振荡，最佳 2320@200） | V-step 末样本排序噪声是振荡源 |
| 2026-09-22 | M3 主训练 | 600 步 无V-step（尺度+小岛） | 1.28M→**166.8**（单调平滑，每百步 ×0.75） | 纯 P-step 微尺度有效；零占比保持 0.3224≈artifact 0.3278 |
| 2026-09-23 | M3 Phase3/4 | export→convert→PTQ1_0→运行时（600 步 ckpt） | C1–C5 全 PASS、maxdiff=0；运行时 ppl 87–211 vs PyTorch 166.8；1.75bpw 精确 | 端到端打通；质量门②（≤41）未达→续训计划见 findings §6；详见 findings-3-pipeline.md |
| 2026-09-23 | M3 续训 | resume 1000 步 @ 2M token 缓存（总 1600 步） | 167→**91.6**（末 300 步打平） | 纯尺度训练 ~91 饱和（3.3×FP）；闭合需码迁移/E1，转 M5-A1 议程 |
| 2026-09-23 | M3 v3 链条 | export→convert→PTQ1_0→运行时（cont1000 ckpt） | 契约 PASS；运行时 ppl ≈85 vs 91.6；生成连贯英文句 | qwen3-0.6b-v3.ptq1_0.gguf（157MB）；生成从"the the the"变为连贯句 |
| 2026-09-23 | M5-A1 | V-step 三变体 @cont1000（91.6 起点） | EMA 排序 308 / 带符号 EMA 3054 / 接受回退闸门 88.5-91（全拒，flips 0 rej 19.7K/步） | 离散翻转在微预算不可行；闭合需 Z+STE（详见 findings-4-discrete-movement.md） |
| 2026-09-23 | M5-A1 | E1 SMD 玩具（d=512≫n=64 三值稀疏真值） | 浅井 ν=4.1：SMD≈GD（34.6%/34.8% 三值）；深井 ν=14：卡死 loss 0.216 不插值 | 朴素损失梯度 mirror descent ≠ 2602.18997 的约束对偶形式；E1 须按原论文实现 |
| 2026-09-24 | M5-A1 | 逐层 Z+STE 三变体 @cont1000（91.6 起点） | 教师输入 31,970 / 序贯贪心 474,363 / 组级闸门 91.64（28/28 组回退，局部 MSE 全改善但全局全变差） | 局部目标与全局 LM 目标错位；本机质量上限 ≈3.3×FP；详见 findings-4-discrete-movement.md §5-6 |
| 2026-09-24 | 工具 | notebooks/tritfold-train-0p6b.ipynb（端到端 Z+STE QAT，T4 16GB 目标） | 核心冒烟：恒等性 4.6e-5、Z/theta/小岛梯度流、3 步 loss 14→10、含 Adam8bit 峰值估 ~11GB | 闭合 91.6→41 缺口的载体；STE 返 fp16 + 逐样本反传降峰 |
| 2026-09-26 | e2e QAT | Colab 战役全记录（T4 1000 步 + lr 阶梯 + v6 系列 + bf16 手术，详见 findings-5-e2e-qat.md） | **48.08 @3500（1.77×FP）**；真值 80.75 破 91.6 天花板；moved 19.1% 无碍 | 平台主因=数据回收；fp16 测量通胀制造假退化/假里程碑；端到端 STE 配方=2e-4+大数据+温热优化器 |
| 2026-09-26 | e2e 收尾 | 本机 qat-v1 全链（convert→PTQ1_0→契约→运行时） | C1–C5 全 PASS、maxdiff=0、1.75bpw 精确；llama-perplexity 64.99±5.6（vs PyTorch 48.08，协议不同）；生成连贯 | M3-② 部分闭合（48.08 vs 门 41）；M4 上量已正当 |
| 2026-09-26 | M4 (L1) | Qwen3-1.7B 从零 5000 步 @ A100（M4 notebook，配方零调整） | **best 28.77 = 1.41×FP** @4700；step 2600 破 1.5× 质量门；skip 0/40 全程；断连热恢复零损失 | 规模效应：1.77×→1.41×；详见 findings-6-1p7b.md |
| 2026-09-26 | M4 收尾 | 本机 m4 全链（convert→PTQ1_0→契约→运行时） | 契约 PASS（宽度自动推导 [2048,6144]）、maxdiff=0、424MB=9.13×；运行时 ppl 38.10±2.4（三协议分解：28.77 真质量 ×1.24 切片 ×1.07 运行时=38.1 闭合；FP 基线 20.44 两协议重合） | **M4 质量门达成，项目两规模全链闭环** |
