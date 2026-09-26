# 端到端 QAT 战报：Z 潜变量 + STE 路线的完整验证（Colab T4→A100）

- 日期：2026-09-25 ~ 09-26
- 载体：`notebooks/tritfold-train-0p6b.ipynb` + 会话内迭代 cell（v5b/v5c/v6/v6b/v6c/v6d/v6e）
- 结果：**0.6B 三值 ppl 80.75 → 48.08（1.77×FP），本机天花板 91.6 被真实突破**；artifact 经 fork 运行时全链验证（契约 PASS、打包值级无损）
- 接续文档：[findings-4-discrete-movement.md](findings-4-discrete-movement.md)（微预算下离散移动证伪 → 本轮端到端路线成功）

---

## 1. 实验阶梯（关键数字链）

| 阶段 | 配置 | 结果 | 结论 |
|---|---|---|---|
| T4 主循环 1000 步 | 2e-4，WikiText-2（2.5M tok 回收） | 平台 115–122 | 平台≠容量上限 |
| lr 阶梯 5e-4 / 3.5e-4 | 换档+回滚闸门 | 冲到 330 / 205，回滚 | 高档一律 excursion |
| v6（三变量齐改） | wikitext-103 + emb-Z 解封 + 2.5e-4 | 155→215 崩 | 教训：一次只动一个变量 |
| **v6b（数据单变量）** | wikitext-103 + 2e-4 + emb 冻结 | 1000 步即 95.36 | **平台主因 = 数据回收** |
| fp16 测量通胀暴露 | step 1200 权重 bf16/fp32 复测 | 真值 80.75（账面 95.36） | fp16 已系统性测高 ~18%；"1300–1500 退化"与"91.17 里程碑"分别是通胀与跳窗偏倚的假象 |
| **v6e（bf16 + 真测量）** | 从 1200/80.75 续 2300 步 | **48.08 @3500（终点仍创新低）** | skip 0/40 全程；moved 19.1% 无碍 |

## 2. 方法学发现（比数字更值钱）

1. **数据回收是微规模三值化的第一瓶颈**：2.5M token 循环训练制造的平台（115–122）在 32M token 下 100 步即被击穿；
2. **fp16 评估通胀**：权重漂移使激活动态范围增长，fp16 前向开始失真——表现为"评估退化"（95→103）与"跳窗偏倚"（skip 8/40 时 91.17 的假里程碑）。**bf16（A100）+ 逐窗 NLL + 零跳窗才准 best** 是完整解；fp32/bf16 双测交叉验证（80.75/80.84，差 0.1%）确认 bf16 无损；
3. **端到端 STE 是唯一可行的码迁移路线**（与 M5-A1 七项证伪互为印证）：有效配方 = 低 lr（2e-4）+ 大数据 + 温热优化器续训；`moved` 12% 警戒线被证实为 fp16 伪迹（19.1% 仍在改善）；
4. 工程坑全记录：IPython 串行执行（运行中 cell 后的排队 cell 不执行）、`load_state_dict` 会覆盖 lr、断连后 `/content` 全失、Colab 镜像 transformers 版本漂移、checkpoint 载 GPU 致 OOM（应 map_location="cpu"）。

## 3. 本机运行时验证（2026-09-26，fork CPU 版）

| 项 | 结果 |
|---|---|
| 契约 C1–C5 | **全 PASS**；sign_values 与训练种子逐字节一致；197 折叠清单精确 |
| 打包无损 | PTQ1_0 反量化 vs 导出 safetensors **maxdiff = 0**（抽样 5 张量） |
| 体积 | 157.04 MiB = 精确 1.75 bpw |
| 运行时 ppl | **64.99 ± 5.6**（llama-perplexity，wt2_30k，c512；chunk 26–65）vs PyTorch 侧 48.08（40 窗协议）——协议不同各自自洽，量级一致 |
| 生成 | 连贯英文（带重复）："in French, for the Frenchman , and the Frenchman is ..."（对比 v1 的 "the the the"） |

## 4. 资产位置

- Colab Drive：`bonsai_qat_milestone_48p08.pt`（终态，含优化器状态）、`bonsai_qat_milestone_80p75.pt`（破 91.6 那一刻）、`artifact_biased91.pt`（教训留档）
- 本机：`<WORK_DIR>/qat-v1.ptq1_0.gguf`（157MB）+ 导出目录 `qwen3-0.6b-qat-v1/`
- Windows：`Downloads/qwen3-0.6b-ternary-hd-qat.zip`（导出原件）

## 5. 未竟与后续

1. **M3 质量门（≤41 = 1.5×FP）未闭合**：48.08，差 15%；终点仍在创新低（每百步 ~-1），继续训（3501–5000，需重建缓存）+ 更大数据是明确路径；
2. **M4（1.7B）现在完全正当**：配方已验证，A100 40GB 装得下（Z 5.6+梯度 5.6+8bit 优化器 2.8+瞬态 ~6 ≈ 21GB）；1.7B 与已发布 Ternary-Bonsai-1.7B 同基座，可对拍白皮书附录 C；
3. emb-Z 解封在 v6 的三变量实验中未单独归因，待 1.7B 阶段用小 lr 独立组验证。
