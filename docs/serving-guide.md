# 三值模型运行指南（两份 artifact 归档）

- 日期：2026-09-26
- 对象：`<WORK_DIR>/qat-v1.ptq1_0.gguf`（Qwen3-0.6B 三值，157MB，1.77×FP）与 `<WORK_DIR>/m4.ptq1_0.gguf`（Qwen3-1.7B 三值，424MB，1.41×FP）
- 运行时：**PrismML fork（prism 分支）专用**——本机为 `Bonsai-demo/bin/cpu/` 预编译版

---

## 前提与硬性警告

1. **只能在 prism fork 上运行**。主线 llama.cpp 不认识 PTQ1_0（私有 ggml type 143）与 `prism.hadamard.*` 元数据；同目录的 `*.f16.gguf`（折叠权重的 F16 存储）虽能被主线读入，但会**静默跳过激活旋转、输出乱码**——MODEL-FORMATS.md 的 fork-required 标记即此意。
2. 本机（GTX 1060）只能跑 CPU：预编译 CUDA 13.3 包不支持 Pascal。
3. **采样必须用保守配方**（2026-09-26 实测定案）：三值分布尾部平坦，默认采样（temp≈0.8 无重复惩罚）会放大尾部噪声、陷入 token 循环（"5.5.5..." / "the case of the case of..."）。必备参数：

   ```
   --temp 0.5 --top-p 0.85 --top-k 20 --repeat-penalty 1.1
   ```

   官方 Bonsai 2 同样在模型元数据里固定了采样默认值，属同类设计。
4. 质量定位：**流畅但事实不可靠**（"Great Wall, a 1982 novel by Robert A. Ewell"——语法通顺、文体正确、事实瞎编）是 1.41×FP + 9.1× 压缩的诚实水位；中文问答与指令跟随需等 M5′ 指令蒸馏版。

## 方式一：一次性生成

```bash
BIN=<RUNTIME_BIN>
M=<WORK_DIR>/m4.ptq1_0.gguf     # 0.6B 换 qat-v1.ptq1_0.gguf
$BIN/llama-cli -m $M -p "The Great Wall of China was originally built to" -n 96 -t 12 -st \
    --temp 0.5 --top-p 0.85 --top-k 20 --repeat-penalty 1.1
```

（英文续写式 prompt；中文问答见 M5′。）

## 方式二：交互聊天

```bash
$BIN/llama-cli -m $M -t 12 -c 4096 --color \
    --temp 0.5 --top-p 0.85 --top-k 20 --repeat-penalty 1.1 \
    --chat-template-kwargs '{"enable_thinking": false}'
```

不带 `-p` 直接进入对话（自动使用模型内置 Qwen3 聊天模板）。

## 方式三：OpenAI 兼容 API

```bash
$BIN/llama-server -m $M --host 127.0.0.1 --port 8080 -t 12 -c 4096 \
    --temp 0.5 --top-p 0.85 --top-k 20 --repeat-penalty 1.1
```

标准 `/v1/chat/completions` 端点（服务端已带默认采样，请求里可再覆盖）：

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"介绍一下长城"}], "temperature": 0.5, "top_k": 20}'
```

任何 OpenAI API 客户端可直接指向该地址；OpenWebUI（`Bonsai-demo/scripts/start_openwebui.sh`）在设置中把 llama.cpp 连接指向它即可。

## 性能与质量预期

| 项 | 0.6B (157MB) | 1.7B (424MB) |
|---|---|---|
| CPU 生成速度（12 线程） | ~11–14 tok/s | ~5–7 tok/s |
| ppl（bf16 协议 / 运行时协议） | 48.08 / 65.0 | 28.77 / 38.1 |
| 质量 | 连贯短句、事实性弱 | 连贯有事实性，长程推理与专业知识会露馅（9.1× 压缩 + 5 小时蒸馏的预期水位） |

## GPU 加速（A100 serving）

见 [notebooks/tritfold-serving-a100.ipynb](notebooks/tritfold-serving-a100.ipynb)：Colab A100 上构建 fork CUDA 版（PTQ1_0 整数 GEMM + FWHT kernel 的原生主场，prompt 处理比 CPU 快一个量级），从 Drive 拉取 GGUF，起 llama-server 并经 cloudflared 隧道暴露公网 OpenAI 端点。**使用前先把两份 GGUF 上传到 Google Drive 根目录。**
