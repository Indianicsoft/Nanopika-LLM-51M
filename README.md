# Nanopika-LLM-51M
A ~51M parameter language model that trains entirely in the browser — no Python, no server, no installation.

# 🧠 Nanopika LLM — 51M

> **The second variant of the Nanopika LLM family.**  
> A ~51 million parameter language model built from scratch in pure Vanilla JavaScript,  
> trainable entirely inside the browser using CPU or WebGPU acceleration.

---

## 🚀 What is Nanopika LLM 51M?

Nanopika LLM 51M is the second model in the **Nanopika LLM family** — a series of lightweight,
browser-native language models designed to be trained and run without any Python, External Libraries,
server infrastructure, or installation.

Everything runs inside a standard web browser tab.

---

## ✨ Key Features

- 🔢 **~51 Million Parameters** — Scaled transformer architecture
- 🌐 **100% Browser-Native** — Pure Vanilla JavaScript, no libraries, no frameworks
- ⚡ **Dual Training Engine** — Train on **WebGPU** (GPU acceleration) or **CPU fallback** (pure JS Float32Array)
- 🔤 **BPE Tokenizer** — Byte-Pair Encoding tokenizer built with TypedArrays (`Uint32Array`)
- 🧮 **WebGPU Matrix Multiplication** — Custom WGSL compute shaders for GEMM operations
- 🎛️ **Adam Optimizer** — Full Adam with bias correction (β1=0.9, β2=0.999)
- 📦 **IndexedDB Weight Persistence** — Weights saved/loaded automatically across sessions
- 🗜️ **Quantization Support** — Float16 / Int8 post-training quantization
- 🔁 **Gradient Accumulation** — Supports micro-batching for memory efficiency
- 💬 **SFT + Pre-Training Modes** — Auto-detects input format; routes to correct pipeline
- 🔒 **Fully Private** — No data ever leaves your browser tab

---

## 🧬 Nanopika LLM Family

| Variant | Parameters | Status |
|---|---|---|
| Nanopika LLM 15M | 15M | First variant (open source)|
| **Nanopika LLM 51M** | **~51M** | **Second Variant (open source)** |

---

## 🏗️ Architecture

| Component | Detail |
|---|---|
| Architecture | Decoder-only Transformer |
| Parameters | ~51 Million |
| Tokenizer | BPE (Byte-Pair Encoding) — `tokenizer.js` |
| Embedding | Token + Positional Embeddings |
| Attention | Scaled Dot-Product Multi-Head Attention |
| MLP | Feed-forward with ReLU activation (He initialization) |
| Normalization | Layer Norm (γ initialized to 1.0, β to 0.0) |
| Optimizer | Adam with bias correction |
| Weight Init | He init (MLP), Xavier (Attention/Embedding/Head) |
| Memory | Single pre-allocated `Float32Array` arena (no runtime allocation) |
| GPU Backend | Custom WebGPU WGSL GEMM compute shaders |
| CPU Backend | Pure JS Float32Array loops (fallback) |
| Quantization | Float16Array / Int8 post-train compression |
| Persistence | IndexedDB (auto-save/load on reload) |

---

## ⚙️ How to Use

### Train in the Browser

1. Clone or download this repository
2. Open `index.html` in a **Chromium-based browser** (Chrome, Edge, Brave) — WebGPU required for GPU mode
3. Paste your training data into the textarea
4. Select **GPU** or **CPU** engine via the toggle
5. Click **Train** — loss updates live without freezing the UI


## 🖥️ Browser Requirements

| Requirement | Minimum |
|---|---|
| Browser | Chrome 113+, Edge 113+, Brave |
| WebGPU | Required for GPU mode |
| RAM | 4 GB+ recommended for 51M params |
| CPU mode | Any modern browser (slower) |

---

## 📊 Training Details

- **Batch size:** Micro-batching with gradient accumulation
- **Learning rate schedule:** Warm-up + decay (configurable in `model_config.js`)
- **Loss:** Cross-entropy with SFT loss masking on padding tokens
- **Gradient tape:** Wengert tape (reset after each full batch)
- **Weight export:** Binary `ArrayBuffer` — no JSON serialization

---

## 🔭 Roadmap

- [ ] Hugging Face model card + weight upload
- [ ] WASM fallback for non-WebGPU browsers
- [ ] Multi-tab distributed training via SharedArrayBuffer
- [ ] Quantized GGUF-style export
- [ ] Fine-tuned chat checkpoint release

---

## 👤 Author

**Rohith Thirunahari**  
Full-Stack Developer & AI Builder · Hyderabad, India  
[GitHub](https://github.com/Indianicsoft) · [LinkedIn](https://www.linkedin.com/in/rohith-tofficial-knowey/)

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

> *Nanopika LLM 51M — because the browser is powerful enough.*
