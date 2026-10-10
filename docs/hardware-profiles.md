# Hardware Profiles & Memory Budget Mathematics

## 1. Workstation Hardware Profile Tiers

Local language model deployment depends fundamentally on available compute hardware: dedicated GPU VRAM, system RAM capacity and memory bandwidth, CPU vector extensions (AVX2 / AVX-512), and the host operating system.

To provide clear operational guidance without coupling to specific machines, workstations are categorized into five standard **Hardware Profiles**:

| Profile Tier | GPU / VRAM Class | System RAM & Bandwidth | Recommended Model Placement | Expected Throughput |
| :--- | :--- | :--- | :--- | :--- |
| **Tier 0 (System 1 Decision Layer)** | Co-exists with any GPU or CPU (<1 GB VRAM or RAM) | Shared with host OS | Laya (421M), ModernBERT (149M), SmolLM2-360M | **10–40 ms / decision**<br>(Non-autoregressive forward pass) |
| **Tier A (High-Performance GPU)** | 16 GB+ VRAM (e.g. Ada / Ampere / RDNA3 class; ~700+ GB/s) | 64–128 GB DDR5 (~70–90 GB/s) | 14B Q4_K_M (100% VRAM); 32B hybrid offload (GPU/CPU) | 14B: 70–85 tok/s<br>7B: 115–140 tok/s<br>32B: 14–18 tok/s |
| **Tier B (Standard Workstation GPU)** | 10–12 GB VRAM (e.g. RTX 4070 / 3080 class; ~500 GB/s) | 32–64 GB DDR4/DDR5 | 7B Q8_0 / 14B Q4_K_M (bounded context 8k–16k) | 14B: 65–75 tok/s<br>7B: 100–120 tok/s |
| **Tier C (Entry / Mobile GPU)** | 6–8 GB VRAM (e.g. RTX 4060 / Laptop class; ~280 GB/s) | 16–32 GB RAM | 7B Q4_K_M (100% VRAM) / 3B models | 7B: 80–110 tok/s<br>3B: 140+ tok/s |
| **Tier D (Unified Memory / Apple Silicon)** | 36–128 GB Unified Memory (M2/M3/M4 Pro/Max; ~150–400 GB/s) | Shared with OS | 14B–32B Q4_K_M / Q8_0 (Metal accelerated) | 14B: 40–60 tok/s<br>32B: 20–30 tok/s |
| **Tier E (CPU-Only / Vector Fallback)** | None / Integrated GPU | 32–128 GB DDR4/DDR5 (AVX2 / AVX-512) | 1.5B–7B quantized models (100% RAM) | 1.5B: 25–40 tok/s<br>7B: 8–15 tok/s |

---

## 2. Memory Footprint & VRAM Budget Mathematics

Deploying local models requires calculating two distinct memory pools: **Model Weights ($M_{\text{weights}}$)** and **KV Cache ($M_{\text{kv}}$)**.

### 2.1 Model Weights Memory

$$M_{\text{weights}} \approx P \times \frac{b}{8} \times 1.05$$

Where:
- $P$ is parameter count in billions.
- $b$ is average bits per weight:
  - **4-bit (Q4_K_M / AWQ / GPTQ-Int4)**: $b \approx 4.5 \implies \sim0.56\text{ GB per billion parameters}$.
  - **8-bit (Q8_0 / FP8 / INT8)**: $b \approx 8.0 \implies \sim1.0\text{ GB per billion parameters}$.
  - **16-bit (FP16 / BF16)**: $b \approx 16.0 \implies \sim2.0\text{ GB per billion parameters}$.
- $1.05$ accounts for model tensor metadata, embedding layers, and CUDA runtime initialization overhead.

### 2.2 KV Cache Memory (Grouped Query Attention)

Modern open-weight models (Qwen 2.5, Llama 3.1/3.3, DeepSeek) employ Grouped Query Attention (GQA) to drastically reduce KV cache size. Memory per sequence token is computed as:

$$M_{\text{kv}} = 2 \times L \times H_{\text{kv}} \times D_{\text{head}} \times C \times B_{\text{kv}}$$

Where:
- $L$ is number of transformer layers.
- $H_{\text{kv}}$ is number of key-value heads.
- $D_{\text{head}}$ is head dimension.
- $C$ is context length in tokens.
- $B_{\text{kv}}$ is bytes per KV cache element (FP16 = 2 bytes, Q8_0 = 1 byte).

**Example: Qwen 2.5 Coder 14B ($L=48, H_{\text{kv}}=8, D_{\text{head}}=128$):**
- Per token in FP16: $2 \times 48 \times 8 \times 128 \times 2 = 196,608\text{ bytes} \approx 0.187\text{ MB per 1,000 tokens}$.
- At 32,768 tokens (FP16): $\approx 6.1\text{ GB}$.
- At 32,768 tokens with **quantized Q8_0 KV cache** (`-ctk q8_0 -ctv q8_0`): $\approx 3.05\text{ GB}$.
- At 16,384 tokens with **quantized Q8_0 KV cache**: $\approx 1.53\text{ GB}$.

### 2.3 Operating Budget in Desktop Workstations

On desktop workstations driving graphical displays, operating systems (Windows DWM, macOS WindowServer, X11/Wayland) reserve a baseline portion of GPU VRAM:
- **Desktop Overhead ($VRAM_{\text{os}}$)**: Typically ~1.5 GB to 3.5 GB depending on display count and desktop applications.
- **Usable Budget ($VRAM_{\text{usable}}$)**:
  $$VRAM_{\text{usable}} = VRAM_{\text{total}} - VRAM_{\text{os}} - VRAM_{\text{headroom}}$$
  Where $VRAM_{\text{headroom}} \approx 0.5\text{–}1.0\text{ GB}$ prevents CUDA out-of-memory faults.
- For a 16 GB GPU, typical usable VRAM is $\sim12.5\text{–}13.5\text{ GiB}$.

---

## 3. Workstation Resource Allocation Overview

```text
+-------------------------------------------------------------------------------+
|                    WORKSTATION RESOURCE ALLOCATION OVERVIEW                   |
|                                                                               |
|   DEDICATED GPU VRAM (16 GB Class)                 HOST SYSTEM RAM (64-128 GB)|
|   +---------------------------------------+        +----------------------+   |
|   | OS / Desktop Display GUI| 2.0-3.5 GB  |        | OS & Core Services   |   |
|   +-------------------------+-------------+        | (8-16 GB used)       |   |
|   | Primary Model (14B Q4)  | ~9.0 GB     |        +----------------------+   |
|   | KV Cache (16k-32k tok)  | 1.5-2.0 GB  |        | Available for Hybrid |   |
|   | CUDA Overhead & Headroom| 1.0-1.5 GB  |        | Offload & 32B Models |   |
|   +-------------------------+-------------+        | (48-112 GB free)     |   |
|   | Total Usable VRAM Budget: ~13.0 GB    |        +----------------------+   |
|   +---------------------------------------+                                   |
+-------------------------------------------------------------------------------+
```
