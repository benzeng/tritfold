#!/usr/bin/env python
"""Build Bonsai/colab/bonsai-serving-a100.ipynb: serve the ternary GGUFs on
Colab A100 with the PrismML fork's CUDA runtime + a public cloudflared tunnel.

Prereq (user): upload the PTQ1_0 GGUF(s) to Google Drive root.
  - m4.ptq1_0.gguf    (1.7B, 424MB)
  - qat-v1.ptq1_0.gguf (0.6B, 157MB)
"""
import json
from pathlib import Path

MD, CODE = "markdown", "code"


def md(src):
    return {"cell_type": MD, "metadata": {}, "source": src}


def code(src):
    return {"cell_type": CODE, "metadata": {}, "execution_count": None,
            "outputs": [], "source": src}


cells = []

cells.append(md("""# Bonsai 三值模型 Serving（Colab A100）

**目标**：在 A100 上用 PrismML fork 的 CUDA 运行时跑 PTQ1_0 三值模型，暴露 OpenAI 兼容 API（公网隧道）——PTQ1_0 的整数 GEMM + FWHT kernel 是这个格式的原生主场。

**前提**：
1. 运行时选 **A100**（菜单 → 更改运行时类型）；
2. **先把 GGUF 上传到 Google Drive 根目录**（文件页拖入即可）：
   - `m4.ptq1_0.gguf`（1.7B，424MB）
   - `qat-v1.ptq1_0.gguf`（0.6B，157MB，可选）

**时间预算**：CUDA 构建 ~15–20 分钟（一次性）→ 之后每次启动 <1 分钟。"""))

cells.append(code("""# 挂载 Drive + 检查模型 + GPU 架构探测
from google.colab import drive
drive.mount("/content/drive")
import os, torch
DRIVE = "/content/drive/MyDrive"
for f in ("m4.ptq1_0.gguf", "qat-v1.ptq1_0.gguf"):
    print(("OK  " if os.path.exists(os.path.join(DRIVE, f)) else "缺失 "), f)
assert os.path.exists(os.path.join(DRIVE, "m4.ptq1_0.gguf")), \\
    "Drive 根目录没有 m4.ptq1_0.gguf——先在左侧文件页上传到 MyDrive 根目录"
cap = torch.cuda.get_device_capability()
CUDA_ARCH = f"{cap[0]}{cap[1]}"
print("GPU:", torch.cuda.get_device_name(0), "| sm_" + CUDA_ARCH)"""))

cells.append(code("""# 构建 PrismML fork（CUDA，arch 按当前 GPU；只编 server/cli 省一半时间）
%pip -q install cmake
!test -d /content/llama.cpp || git clone -b prism --depth 1 https://github.com/PrismML-Eng/llama.cpp.git /content/llama.cpp
%cd /content/llama.cpp
!cmake -B build-cuda -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_CUDA_ARCH=$CUDA_ARCH > /tmp/cmake.log 2>&1 || tail -20 /tmp/cmake.log
!cmake --build build-cuda --target llama-server llama-cli -j$(nproc) 2>&1 | tail -3
!./build-cuda/bin/llama-server --version"""))

cells.append(md("""## 启动服务

`MODEL` 二选一后运行；随后运行自测 cell。日志在 `/content/server.log`。"""))

cells.append(code("""# 启动 llama-server（后台；-ngl 99 全量上 GPU，-fa on 为 Bonsai 模型验证过的组合）
MODEL = os.path.join(DRIVE, "m4.ptq1_0.gguf")     # 1.7B；0.6B: "qat-v1.ptq1_0.gguf"
PORT = 8080

!pkill -f llama-server || true
import time
get_ipython().system_raw(
    f"/content/llama.cpp/build-cuda/bin/llama-server "
    f"-m {MODEL} --host 127.0.0.1 --port {PORT} -ngl 99 -fa on -c 4096 "
    f"> /content/server.log 2>&1 &")
for _ in range(120):
    time.sleep(1)
    log = open("/content/server.log", errors="ignore").read()
    if "listening" in log or f":{PORT}" in log:
        print("server ready"); break
    if "error" in log.lower() and "listening" not in log:
        print(log[-2000:]); raise SystemExit("server failed to start")
else:
    print(open("/content/server.log", errors="ignore").read()[-2000:])
    raise SystemExit("timeout")
print("\\n".join(l for l in open("/content/server.log", errors="ignore") if "Hadamard" in l or "loaded" in l)[:1500])"""))

cells.append(code("""# 本机自测（OpenAI 兼容端点）
import json, urllib.request
req = urllib.request.Request(
    f"http://127.0.0.1:{PORT}/v1/chat/completions",
    data=json.dumps({"messages": [{"role": "user", "content": "用一句话介绍长城"}],
                     "temperature": 0.7, "max_tokens": 96,
                     "chat_template_kwargs": {"enable_thinking": False}}).encode(),
    headers={"Content-Type": "application/json"})
t0 = __import__("time").time()
r = json.loads(urllib.request.urlopen(req, timeout=120).read())
msg = r["choices"][0]["message"]["content"]
print(msg)
print(f"\\n[耗时 {__import__('time').time()-t0:.1f}s | ~{r['usage']['completion_tokens']/(__import__('time').time()-t0):.1f} tok/s]")"""))

cells.append(code("""# cloudflared 快速隧道（免注册，输出公网 URL）
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /usr/local/bin/cloudflared && chmod +x /usr/local/bin/cloudflared
import time, re
get_ipython().system_raw("/usr/local/bin/cloudflared tunnel --url http://127.0.0.1:%d > /content/tunnel.log 2>&1 &" % PORT)
url = None
for _ in range(60):
    time.sleep(1)
    m = re.search(r"https://[-a-z0-9]+\\.trycloudflare\\.com", open("/content/tunnel.log", errors="ignore").read())
    if m: url = m.group(0); break
assert url, open("/content/tunnel.log", errors="ignore").read()[-1000:]
print("公网 OpenAI 端点：", f"{url}/v1/chat/completions")
print("base_url（填入任意 OpenAI 客户端）：", f"{url}/v1")"""))

cells.append(md("""## 使用与安全

- 任意 OpenAI 客户端：`base_url = <上面的 /v1 地址>`，api_key 随便填；
- ⚠️ trycloudflare URL **公开无鉴权**——拿到地址的人都能用你的 A100。用完运行下面的停止 cell，或直接断开运行时；
- notebook 断开 = 服务终止（隧道 URL 也随之失效）；
- 换模型：改 `MODEL` 那行后重跑“启动服务”与隧道两个 cell。"""))

cells.append(code("""# 停止服务与隧道
!pkill -f llama-server || true
!pkill -f cloudflared || true
print("stopped")"""))

nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "colab": {"provenance": [], "gpuType": "A100"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    },
    "cells": cells,
}

out = Path("notebooks/tritfold-serving-a100.ipynb")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print("wrote", out)
