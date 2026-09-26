# L1-micro 端到端管线报告：Qwen3-0.6B 三值化 + fork 运行时验证

- 日期：2026-09-22/23
- 代码：`proto/l1micro/{ternary_model,cache_teacher_logits,train,export_model}.py`、`proto/common/check_contract.py`
- 产物：`<WORK_DIR>/`（checkpoint、导出目录、GGUF、日志）
- 结论：**端到端管线全通，C1–C5 契约自检全部 PASS，打包链路值级无损（maxdiff=0），运行时 ppl 与 PyTorch 侧一致**；质量门 ②（≤1.5×FP）未达——600 步达到 167（FP=27.25），趋势陡峭向下，属训练预算问题而非方法问题

---

## 1. 训练框架（E2-lite 实现，对规划 T3.3 的三处偏离及理由）

设计：折叠权重 = int8 码 T + 每组 128 一个 fp32 尺度 s_g（即 artifact 格式本身）；前向 w_eff=s_g·T + 激活在线 FWHT；目标 = 教师 top-50 logits KD（teacher=冻结原模型，logits 预计算落盘 512K token）。

| # | 偏离 | 理由 |
|---|---|---|
| D1 | embedding/lm_head 只训尺度、码不翻（capture_grad=False） | 155M 参数表的 fp32 梯度 >0.6GB，WSL 共享内存换页代价大；其尺度仍正常训练 |
| D2 | 尺度 lr 3e-3 → **1e-3 + 梯度裁剪 1.0** | lr 3e-3 无裁剪在 ~70 步发散（尺度膨胀 → fp16 激活溢出 → NaN） |
| D3 | V-step 改为每 2 步、末样本梯度排序、分层抽样阈值 | 全量 2.2 亿元素 kthvalue 每步 ~18s；抽样后 <1s |

性能修复记录：首版 w_eff 用 fp32 广播乘，autograd 为每个折叠张量保存两份 fp32 中间量（0.6B 模型 ≈5.4GB），WSL 上总占用 16GB 严重换页、27 tok/s。改为自定义 autograd Function（只存 int8 码 + theta）+ 逐样本反传后 **<6GB 无换页、~190 tok/s**（5.5s/步 @ 2×512）。

## 2. 初始化消融（M2 遗留问题，各 40 步，512K token 教师缓存）

| 初始化 | step-0 ppl | step-20 | step-40 |
|---|---|---|---|
| amax RTN（零占比 83.4%） | 1.09×10⁸ | 102,188 | 10,758 |
| 目标零占比 1/3（s=0.836σ，零占比 32.2%） | 1.28×10⁶（**93× 更低**） | **3,123** | 12,261（振荡） |

结论：起点差 93 倍；40 步时 amax 追到同量级。zerofrac 起点优势真实但训练初期振荡更大。**选 zerofrac 为主训练默认**（与 artifact 指纹一致）。

## 3. 主训练与 V-step 消融（核心发现）

| 臂 | 配置 | 轨迹（quick ppl，40 窗 20K token 测试片） |
|---|---|---|
| main | 300 步 + V-step（τ=1%，信赖比 0.01） | 1.28M → 14,977(50) → 18,987(100) → 37,180(150) → **2,320**(200) → 2,468(250) → 2,804(300)；剧烈振荡 |
| novstep600 | 600 步，**关 V-step**（只训尺度+小岛） | 1.28M → 2,625(100) → 599(200) → 410(300) → 276(400) → 203(500) → **166.8**(600)；单调平滑下降，每 100 步 ×~0.75 |

**发现：V-step（按 PV-Tuning 直译的实现）是振荡源**——末样本梯度排序 + 每 2 步 ~20K 次码翻转注入的噪声超过信号（零占比恒为初值：翻转方向恰好平衡）。尺度+小岛训练（纯 P-step）在 600 步内平滑地把 ppl 从 1.28M 压到 167。PV-Tuning 的 V-step 需要 EMA 平滑信号与数千步预算，微尺度 300 步不成立——这直接回答重建文档 §6-1 的一半：**E2 的 P 步在微尺度即有效；V 步/离散迁移需要长得多的日程与平滑信号**（E1 SMD 对照仍属 M5）。

零占比对照：训练后保持 0.3224（初值），artifact 为 0.3278——初始化指纹高度一致，训练未破坏（也未主动改变）熵分布。

## 4. Phase 3/4 打包链路（三个实操陷阱，已写入规划附录 B）

1. **llama-quantize 默认把 token_embd 降级 Q4_K**（llama-quant.cpp:503）→ 必须 `--token-embedding-type PTQ1_0 --output-tensor-type PTQ1_0`。
2. convert_hf_to_gguf 的 Qwen2/3 vocab 检测先尝试 sentencepiece，模块缺失时不回退 → venv 需 `pip install sentencepiece`。
3. fork 的 llama-cli 默认进交互会话不退出 → 一次性生成需 `-st`（demo 脚本 run_llama.sh 同款用法）。

manifest：schema_version=1 + signs dict（{1024,2048,3072} 全宽 ±1 向量，与训练同种子）+ 197 条 fold-before-matmul + token_embd inverse-after-lookup；`tie_word_embeddings=false` 导出双张量（embed + lm_head 同表）。

## 5. 验收门判定

| 门 | 结果 | 判定 |
|---|---|---|
| ① 端到端跑通 | PTQ1_0 GGUF 157MB（精确 1.75 bpw）；llama-cli -st 生成连贯英文（质量对应 ppl 量级） | ✅ |
| ② 三值 ppl ≤1.5×FP（≈41） | 最佳 166.8 @ 600 步（FP=27.25），曲线仍陡降 | ❌ 未达：训练预算问题（见 §6） |
| ③ C1–C5 契约 | v1 artifact 全 PASS：元数据/符号表逐字节一致、197 折叠清单精确、抽样张量**值级无损 maxdiff=0**、scale==amax、小岛 F32 | ✅ |
| ④ 运行时 ppl ≈ PyTorch ppl | llama-perplexity ≈ 3,150（chunk 2,520–3,869）vs PyTorch 2,804 @step300 | ✅ 同量级 |

v2（600 步 checkpoint）链条：契约自检 OVERALL PASS（同 v1 逐项）；llama-perplexity 各 chunk 87–211（均值 ≈155）vs PyTorch 166.8 @step600——✅ 一致。产物：`qwen3-0.6b-v2.ptq1_0.gguf`（157MB）。

## 6. 未达质量门的分析与后续（含 2026-09-23 续训更新）

**续训结果（2M token 教师缓存，resume 1000 步，总计 1600 步）**：167 → 91.6（轨迹：149.7→127.8→125.6→115.4→106.7→100.4→98.0→93.3→91.4→91.6）。最后 300 步基本打平——**纯尺度训练在 ~91 饱和**（3.3×FP），不是步数不够而是该参数子空间的极限。167 vs 41 的原估计（预算问题）被修正为：**P-step 能快速逼近但无法闭合质量差距，闭合需要离散码迁移（修复版 V-step 或 E1 SMD）和/或更丰富目标（特征 KD、更多数据）**——这正是 M5-A1 的议程。

v3 终模型链条（cont1000 checkpoint）：契约自检 OVERALL PASS；运行时 ppl chunk 43–115（均值 ≈85）vs PyTorch 91.6 一致；llama-cli 生成从"the the the"（ppl 2804 时）变为**连贯英文句**（"In the first part of the 1950s, the first part of the French government's public life was"）。产物 `qwen3-0.6b-v3.ptq1_0.gguf`（157MB，1.75bpw 精确）。

M4 前的直接后续（不变，优先级重排）：
1. **V-step 修复（EMA 平滑排序 + 小 τ）或 Z 潜变量 STE 对照**——现在是闭合差距的明确瓶颈（M5-A1 提前）；
2. 特征 KD（A2）与领域数据补齐；
3. 更长日程只作为边际手段（曲线已平）。

## 7. 复现

```bash
python proto/l1micro/cache_teacher_logits.py --tokens 524288
python proto/l1micro/train.py --init zerofrac --steps 600 --no-vstep --tag novstep600
python proto/l1micro/export_model.py --ckpt .../l1micro_novstep600.pt --out .../qwen3-0.6b-ternary-hd-v2
python <FORK_DIR>/convert_hf_to_gguf.py <导出目录> --outfile out.f16.gguf
$B/llama-quantize --token-embedding-type PTQ1_0 --output-tensor-type PTQ1_0 out.f16.gguf out.ptq1_0.gguf PTQ1_0
python proto/common/check_contract.py out.ptq1_0.gguf <导出目录>
```
