# Runtime Serving Engines & Workstation Architecture

Comparative evaluation, operational procedures, and configuration recipes for standing up local model serving engines across Windows 11, macOS, and Linux.

---

## 1. Serving Engine Architectural Comparison

| Runtime Engine | Platform Support | Underlying Engine | OpenAI API Compatibility | Memory Overhead | Best Suited For | Tradeoffs |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Ollama** | Native Windows, macOS, Linux | llama.cpp fork with dynamic loader | Native (`/v1/chat/completions`, `/v1/embeddings`, `/v1/models`) | Low (~150 MB process RAM) | **Primary Workstation Runtime**: Seamless model download, automatic keep-alive, concurrent slots. | Less granular control over custom layer splits or experimental KV cache quantization flags. |
| **llama.cpp (`llama-server`)** | Native Windows, macOS, Linux | Direct C++ CUDA / Metal / cuBLAS | Native (`/v1/chat/completions`, `/v1/models`) | Minimal (~80 MB process RAM) | **Power Users / Custom Offloading**: Manual layer splitting (`-ngl`), FlashAttention (`-fa`), quantized KV cache (`-ctk q8_0 -ctv q8_0`). | Requires manual GGUF file management and shell launch scripts. |
| **vLLM** | Linux / WSL2 only | Python / C++ PagedAttention | Native (`/v1/chat/completions`) | High (Reserves 90% GPU VRAM by default) | **Server / Multi-tenant**: High-concurrency continuous batching. | Windows friction; requires WSL2 or Docker; large idle VRAM footprint. |
| **LM Studio** | Native Windows, macOS, Linux | llama.cpp / MLX wrapper | Native (`/v1/chat/completions`) | Moderate (~300 MB GUI RAM) | **Interactive Prototyping**: Visual quant testing and prompt playground. | Extra GUI overhead; CLI daemon requires running background app. |

---

## 2. Recommended Primary Runtime: Ollama

Ollama provides zero-configuration local service management, automated model pulling from the official registry, native CUDA / Metal acceleration, and built-in OpenAI API compatibility.

### 2.1 Installation

- **Windows**: `winget install Ollama.Ollama` or download installer from [ollama.com](https://ollama.com).
- **macOS**: `brew install ollama` or direct `.zip` application.
- **Linux**: `curl -fsSL https://ollama.com/install.sh | sh`.

### 2.2 Environment Variables & Tuning

Configure the following environment variables to optimize concurrency, FlashAttention, and memory persistence:

| Variable | Recommended Value | Purpose |
| :--- | :--- | :--- |
| `OLLAMA_HOST` | `127.0.0.1:11434` | Binds locally, preventing external network exposure on developer machines. |
| `OLLAMA_NUM_PARALLEL` | `4` | Allows up to 4 concurrent inference slots for parallel subagent executions. |
| `OLLAMA_MAX_LOADED_MODELS` | `2` | Allows keeping a coding model and an embedding model hot simultaneously. |
| `OLLAMA_KEEP_ALIVE` | `30m` | Keeps models in VRAM for 30 minutes after last request to eliminate reload latency. |
| `OLLAMA_FLASH_ATTENTION` | `1` | Enables FlashAttention-2 kernels on modern Tensor Cores. |
| `OLLAMA_MODELS` | Path to fast NVMe SSD | Directs model weight cache to optimal storage path. |

### 2.3 Verification via OpenAI-Compatible Endpoint

Ollama exposes an OpenAI-compatible REST API at `http://127.0.0.1:11434/v1`:

```bash
# Verify running models
curl http://127.0.0.1:11434/v1/models

# Test Chat Completion
curl http://127.0.0.1:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-coder:14b",
    "messages": [
      {"role": "system", "content": "You are a fast coding assistant."},
      {"role": "user", "content": "Write a python function to compute fibonacci."}
    ],
    "temperature": 0.2
  }'
```

---

## 3. Alternative High-Performance Runtime: llama.cpp (`llama-server`)

For surgical control over custom layer splits between GPU and system RAM, `llama-server` provides precise CLI flags.

### 3.1 Startup Recipes

#### Scenario A: Full GPU Execution (Tier A / Tier B Workstations)
```bash
./llama-server \
  -m ./models/qwen2.5-coder-14b-instruct-q4_k_m.gguf \
  -ngl 99 \
  -c 32768 \
  -fa \
  -ctk q8_0 -ctv q8_0 \
  --port 8080 \
  --host 127.0.0.1
```
- `-ngl 99`: Offloads all 48 transformer layers to GPU VRAM.
- `-c 32768`: Configures 32k context window.
- `-fa`: Enables FlashAttention.
- `-ctk q8_0 -ctv q8_0`: Quantizes KV cache keys and values to 8-bit integers, halving KV cache VRAM usage.

#### Scenario B: Hybrid GPU+CPU Offload (32B Models on 16 GB GPUs)
```bash
./llama-server \
  -m ./models/qwen2.5-coder-32b-instruct-q4_k_m.gguf \
  -ngl 36 \
  -t 16 \
  -c 16384 \
  -fa \
  --port 8080 \
  --host 127.0.0.1
```
- `-ngl 36`: Offloads ~36 of 64 layers to GPU VRAM (~11.2 GB VRAM).
- `-t 16`: Assigns 16 high-performance CPU worker threads to process remaining layers in host system RAM.
- Throughput: ~14–18 tokens/second.

---

## 4. Security & Network Isolation

1. **Localhost Binding**: Runtime must bind exclusively to `127.0.0.1` or `localhost`. Never bind to `0.0.0.0` on developer workstations to prevent cross-network extraction.
2. **Zero Cloud Exfiltration**: Inference executes 100% locally on host GPU/CPU hardware. No prompt data, tokens, or code snippets leave the machine.
3. **Authentication**: Dummy tokens (e.g. `Authorization: Bearer local`) are accepted, eliminating secret management overhead for the local tier.
