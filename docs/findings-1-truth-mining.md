# L0′ 地面真值挖掘报告：Ternary-Bonsai-2-27B PQ2_0

- 日期：2026-09-22
- 对象：`<MODELS_DIR>/Ternary-Bonsai-2-27B/Ternary-Bonsai-2-27B-PQ2_0.gguf`（7.2GB，851 张量，52 条 KV）
- 工具：`proto/l0prime/dump_metadata.py`（gguf-py 正规 API）、`proto/l0prime/dequant_stats.py`（PQ2_0 numpy 反量化，对照 `ggml-quants.c:113/494`）
- 产物：`<WORK_DIR>/l0prime_metadata.json`、`l0prime_stats.json`
- 结论：**五项检验全部有结论，M1 验收门通过**；其中两处实测修正了笔记 §1/§10.3 的记载

---

## 1. 元数据契约（`prism.hadamard.*` 全解码）

| 键 | 值 |
|---|---|
| version | 1 |
| block_size | 1024 |
| transform | `normalized-sylvester-walsh-hadamard` |
| axis | `input-last-dimension` |
| sign_mode | `explicit` |
| **sign_widths** | **[5120, 6144, 17408]**（⚠️ 修正笔记 §10.3 的 `[17408]`——那是快速脚本偏移误读） |
| **sign_values** | **28672 个未打包 ±1 整数** = 5120+6144+17408，即**每个输入宽度一条共享符号向量**（非每tensor一条，非位打包） |
| weight_names | 401 个（完整清单见 JSON） |
| inverse_weight_names | `['token_embd.weight']`（唯一） |
| gdn_v_grouped | True |

其他：arch=qwen35、hidden=5120、block_count=64、heads=24/kv=4（head_dim=256，attn_q/k_norm 存在）、ffn=17408、vocab=248320、file_type=141（MOSTLY_PQ2_0=141 对应 ggml type 142）。

## 2. 折叠普查（输入宽 = ggml ne0 = `ReaderTensor.shape[0]`）

| 输入宽 | 张量数 | 成员 | 符号表 |
|---|---|---|---|
| 5120 | 273 | attn_qkv×48、attn_gate×48（GDN）；attn_q/k/v×16（softmax）；ffn_gate/up×64；output.weight×1 | ✅ |
| 6144 | 64 | ssm_out×48（GDN）；attn_output×16（softmax） | ✅ |
| 17408 | 64 | ffn_down×64 | ✅ |

块模式：48 个 GDN 块（attn_qkv+attn_gate+ssm_out+ffn×3），16 个 softmax 块（attn_q/k/v+attn_output+ffn×3），合计 401+embedding 逆变换 = 与 `_HADAMARD_KINDS` 白名单（base.py:700-711）完全一致。

**结构含义**：同一残差流的所有消费者（attn/ffn 输入侧）共享同一输入宽 5120 → 共用同一 R → 运行时 `hadamard_memo` 去重成立；softmax 块 attn_q 输出 12288=24×256×2（含 gate），attn_output 输入 6144=24×256。

## 3. 全精度小岛（C5 实测确认）

F32：全部归一化（output_norm/attn_norm/post_attention_norm/ssm_norm/attn_q_norm/attn_k_norm）、ssm_conv1d.weight、ssm_dt.bias、ssm_a；BF16：ssm_alpha.weight、ssm_beta.weight。与白皮书 Table 2 一致。

## 4. 五项统计检验

反量化对象：8 个代表性折叠张量全量 + token_embd/output 各 8192 行抽样。

| 检验 | 结果 | 判定 |
|---|---|---|
| (a) 零占比 | **0.3278 ± 0.0002**（所有张量一致，含 embedding/lm_head） | ≈**1/3**，见下方分析 |
| (b) scale == 组 amax | 所有组含 ±1 码（missPM1 = 0）、code 3（+2）零使用、无全零组 | **精确成立**（C3 无损 RTN 的直接证据） |
| (c) 旋转域分布 | ±1 平衡 0.999–1.003（完美对称 → C2 无 shift）；列零占比 p5/p95 = [0.30, 0.35]（位置无关 → 非结构化） | 符合 |
| (d) 折叠清单与角色 | 见 §2，与白名单一致；inverse 仅 token_embd | 符合 |
| (e) embedding 旋转域存储 | 存储域行峰度 **1.51**（均匀三值理论值 E[x⁴]/E[x²]² = (2/3)/(2/3)² = **1.5**）；逆 FWHT 后 3.01 ≈ 对照组 BF16 ssm_beta 的 3.11；行范数变换前后同为 0.826（正交性自检通过） | **确认旋转域存储** |

尺度 d：均值 0.011–0.018，CV 0.10–0.31（深层更紧），无异常。

### 关键发现：三值码是近最大熵的（≈均匀三值），不是 RTN 高斯的残影

- 实测零占比 0.3278、±1 各占 0.336：三个符号近等概 → 每 trit 熵 ≈ log₂3 = 1.585 bit，**顶满 base-3 打包的信息容量**。
- 对比参照：① 笔记 §1 的"RTN 对近高斯权重约 12% 置零"记载**被实测否定**；② 若真是"高斯 + amax 尺度 RTN"（组 128，E[amax]≈3.1σ，阈值 0.5·amax≈1.55σ），零占比应 ≈ **88%**——与实测差 55 个百分点。
- 结论：训练端把码**主动成形**为近均匀三值（信息密度最大化），这正是"秘密训练阶段"在 artifact 上留下的指纹；也意味着 RTN 基线与成品的差距比此前估计的更大，Phase 2 蒸馏的负担更重、更必要。
- 理论呼应：2602.18997 的 SMD 收敛解是"满足约束下 Bregman 最近原权重"，没有理由产生 88% 零；均匀三值与"强正则把权重驱向 {-1,0,+1} 且不偏袒零"的图景一致。

## 5. 对下游管线的修正（已回写实施规划精神）

1. **M3 manifest 写法确定**：符号表按输入宽共享。Qwen3-0.6B 对应宽度 = {1024（q/k/v/gate/up/lm_head/embd）、2048（o_proj）、3072（down_proj）}，sign_mode=explicit、三条符号向量按 sign_widths 顺序拼接为 sign_values。
2. **Phase 2 目标分布**：期望收敛到 ~1/3 零占比的近均匀三值；不要把 RTN 统计当先验。验收时可加"零占比落入 [0.30, 0.36]"作为训练成形的健康指标。
3. **s_g = amax 收尾**成立（实测每 407 万组无一例缺 ±1）；训练期可用 MSE 最优尺度，但 Phase 3 吸附必须回到 amax（C3）。
4. 笔记两处记载修正：零占比 ~12% → 实测 32.8%；sign_widths [17408] → [5120, 6144, 17408]。

## 6. 复现

```bash
python proto/l0prime/dump_metadata.py <gguf> l0prime_metadata.json
python proto/l0prime/dequant_stats.py <gguf> l0prime_stats.json
```
