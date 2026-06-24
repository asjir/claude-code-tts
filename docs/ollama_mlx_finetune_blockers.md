# Deep-research brief: serving a fine-tuned Qwen3.5-2B through Ollama's MLX engine

> **UPDATE — mostly resolved.** An **8-bit MLX** build is now serving in Ollama. The fix
> was to **pre-quantize with `mlx_lm` first** (text-only weights, `--q-bits 8`), then
> `ollama create --experimental` with **no `-q`** — Ollama imports a pre-quantized MLX
> folder fine; it only panics when *it* does the quantization. Result: arch `qwen3_5`,
> `quantization int8`, `requires 0.19.0`. See `docs/finetune_modal.md` for the full recipe.
>
> Still genuinely open for research (questions 1–3 below): producing native **mxfp8**
> (float8, like the stock `qwen3.5:2b-mlx`) *locally* — i.e. fixing/working around the
> `quantize.go` "no Stream(gpu, 1)" panic, or hand-building Ollama's mxfp8 tensor layers.
> The `int8` build we have is functionally fine, so this is an optimization, not a blocker.

## What I want out of this research

I have a LoRA-fine-tuned **Qwen3.5-2B** model that I need to serve **locally on Apple
Silicon through Ollama's MLX engine**, ideally as an **8-bit (mxfp8) build** matching the
stock `qwen3.5:2b-mlx` model I already run. I got a **bf16** MLX build working via a
workaround, but I cannot produce an **8-bit MLX** build locally — Ollama's import-time
quantizer panics. I need:

1. A way to produce an **8-bit MLX (mxfp8)** build of this model that Ollama 0.30.10 (or a
   newer/pre-release build) can serve locally — either by fixing/working around the
   `quantize.go` MLX-stream panic, or by importing a pre-quantized MLX model in a format
   Ollama accepts.
2. Confirmation of **how the registry `qwen3.5:2b-mlx` (mxfp8) was actually built**, and
   whether that pipeline is runnable locally.
3. The set of viable fallbacks (8-bit GGUF/llama.cpp; serving via `mlx_lm.server` instead
   of Ollama) with their tradeoffs.

## Environment

- **Hardware/OS:** Apple Silicon Mac, macOS Darwin 24.6.0.
- **Ollama:** version **0.30.10** (confirmed latest release on GitHub at time of writing).
  Installed via Homebrew cask `ollama-app` → `/Applications/Ollama.app`, symlinked at
  `/usr/local/bin/ollama`.
- **Python tooling:** `uv` for ephemeral envs; `transformers` (has a native `qwen3_5`
  module), `mlx-vlm`, `mlx-lm`, `safetensors`, `torch`.

## The model

- **Source:** LoRA fine-tune of `Qwen/Qwen3.5-2B` merged to a full **bf16 HF** checkpoint
  (Unsloth on a Modal A10G). Arch string `Qwen3_5ForConditionalGeneration`, `model_type:
  qwen3_5`, `text_config.model_type: qwen3_5_text`, plus a `vision_config`.
- **Single-file safetensors**, 632 tensors, ~4.55 GB. Tensor namespaces:
  - `model.language_model.*` — the LM. **Hybrid attention**: layers contain
    `linear_attn.A_log` (F32 [16]) and `linear_attn.norm.weight` (F32 [128]) — i.e.
    linear/SSM-style attention interleaved with full attention.
  - `model.visual.*` — vision tower (patch_embed, blocks, merger, pos_embed).
  - `mtp.*` — **multi-token-prediction** module (fc, layers, norms) for speculative decode.
- **Weights verified correct** independently of Ollama: loaded in `transformers`
  (dtype=float32, CPU), with `enable_thinking=False`, the chat prompt
  `"Reply with exactly: it works"` returns exactly `it works`. So any garbage output below
  is an Ollama import problem, not a bad checkpoint.
- Loading in transformers also emitted (non-fatal): a Mistral-regex tokenizer warning
  (`set fix_mistral_regex=True`), and "fast path is not available" for
  flash-linear-attention / causal-conv1d (so the hybrid linear-attn layers fell back to a
  pure-torch path).

## Reference model that already works

`ollama show qwen3.5:2b-mlx`:

```
architecture        qwen3_5
parameters          2.2B
quantization        mxfp8
requires            0.19.0
context length      262144
capabilities        completion, vision, thinking, tools
```

Its manifest stores the model as **hundreds of per-tensor `application/vnd.ollama.image.tensor`
layers** (the new MLX engine format), not a single GGUF blob. I believe this was **pulled
from the Ollama registry** (built server-side by Ollama/Qwen), not produced by a local
`ollama create`. Confirming that, and the exact build pipeline, is one of my questions.

## Problems encountered (each with exact errors)

### 1. Disk pressure + orphaned blobs

`ollama create` first **copies the raw source safetensors into its blob store**
(`~/.ollama/models/blobs`) before converting, so importing a ~4.5 GB model needs ~4.5 GB
free just for the copy, plus space for the converted output (peak ~7 GB). On a near-full
disk this failed with:

```
Error: write /Users/.../.ollama/models/blobs/sha256-3137154846: no space left on device
```

Every **failed** import left its raw-safetensors copy behind as an **orphan blob** (e.g. a
2.66 GB blob from a failed run). Ollama 0.30.10's `ollama create --help` exposes only
`-f/--file`, `-q/--quantize`, `--draft-quantize`, `--experimental` — **no prune/gc**
command. I had to GC manually by diffing `manifests/**` digests against `blobs/`.

- *Gap:* recommended way to prune unreferenced Ollama blobs; whether import can avoid the
  full raw-safetensors copy; how to bound peak disk during `ollama create`.

### 2. Pre-quantized MLX (mlx-vlm / mlx-lm) is rejected by Ollama

Converting to MLX 8-bit with `mlx_vlm.convert ... -q --q-bits 8` produces weights with
**U32-packed** quantized tensors (187 U32 + BF16 + F32). `ollama create` on that dir:

```
Error: unknown data type: U32
```

So Ollama's importer cannot read MLX's standard packed-quant format. (The mlx-vlm output
also wraps the LM under a `language_model.` prefix as a VLM.)

- *Gap:* What on-disk format does Ollama's MLX engine expect for **pre-quantized** tensors?
  Can I hand-produce the per-tensor `vnd.ollama.image.tensor` mxfp8 layers to match the
  registry model and skip Ollama's local quantizer entirely?

### 3. Legacy path (no `--experimental`) → GGUF, then vision crash at runtime

Plain `ollama create qwen3.5-2b-tts -f Modelfile` takes the **legacy GGUF/llama.cpp**
converter: `ollama show` reports `architecture qwen35` (note: no underscore) and
`quantization F16`. It imports fine but **crashes at first inference**:

```
Error: 500 Internal Server Error: llama-server process has terminated:
GGML_ASSERT(idx_mean >= 0 && "image_mean not found") failed
```

i.e. the llama.cpp runtime expects vision-preprocessor metadata (`image_mean`) that the
GGUF conversion didn't carry.

### 4. Experimental path (bf16) imports but outputs gibberish

`ollama create qwen3.5-2b-tts --experimental -f Modelfile` takes the **new safetensors →
MLX engine** path: `ollama show` reports `architecture qwen3_5`, `requires 0.19.0`,
`quantization bfloat16`, imported "with 637 layers". It runs without crashing but generates
**multilingual token salad** (random CJK/Cyrillic/Latin fragments) for any prompt — classic
symptom of a tensor mapping/order problem, not a chat-template issue.

**Cause I isolated:** the extra **`mtp.*`** and **`model.visual.*`** tensors confuse the
experimental importer. Stripping them fixes it (see workaround).

- *Gap:* Why does the experimental importer mis-map a full multimodal/MTP `qwen3_5`
  checkpoint? Is importing the **full** multimodal qwen3_5 (vision + MTP) via
  `--experimental` supported at all in 0.30.10? Should MTP go through `--draft-quantize`
  (speculative decoding) instead of being dropped?

### 5. Any `-q` on the experimental path → MLX quantizer panic (the core 8-bit blocker)

This is the blocker for an 8-bit MLX build. **All** quant targets panic identically:

```
ollama create ... --experimental -q mxfp8   -> panic
ollama create ... --experimental -q q8_0    -> panic
ollama create ... --experimental -q q4_K_M  -> panic
```

```
panic: mlx: There is no Stream(gpu, 1) in current thread.
  at .../_deps/mlx-c-src/mlx/c/transforms.cpp:73
github.com/ollama/ollama/x/create/client.loadAndQuantizeArray (quantize.go:106)
github.com/ollama/ollama/x/create/client.quantizeTensor          (quantize.go:140)
```

Two notable facts:

- The panic is **identical across mxfp8, q8_0, and q4_K_M** — on `--experimental`, Ollama
  appears to route **every** `-q` value through its **MLX** quantizer (mlx-c), so even
  GGUF-named schemes (`q8_0`, `q4_K_M`) hit the MLX path. The name only selects target
  precision; the underlying op is MLX, and that op is what fails.
- The error is an MLX **GPU-stream / threading** issue (`no Stream(gpu, 1) in current
  thread`) raised inside the **CLI client's** quantize routine (`x/create/client`), i.e. the
  Metal/MLX device stream isn't initialized in the thread doing the quantization.

- *Gap (primary):* Is this a known Ollama bug? Fixed in a newer release / pre-release /
  nightly? Any env var or invocation that initializes the MLX GPU stream for the importer
  (e.g. forcing a single stream, running quantization in the server process, a Metal device
  flag)? Any minimum macOS/Metal requirement?

## Workaround currently in production (not ideal)

1. Write a **text-only** copy of the weights: drop every tensor under `mtp.*` and
   `model.visual.*`, and remove `vision_config` from `config.json`. (632 → 320 tensors;
   ~4.55 GB → ~3.76 GB.)
2. `ollama create qwen3.5-2b-tts --experimental -f Modelfile` (FROM the text-only dir, **no
   `-q`**). Result: `architecture qwen3_5`, `requires 0.19.0`, `quantization bfloat16`,
   1.9B params. Coherent output; verified good summaries through the OpenAI-compatible
   endpoint with `reasoning_effort: none`.

This gives a **working bf16 MLX** model (~3.8 GB) but **no 8-bit**, and **no vision** (fine
for my text-only summarizer use case, but a limitation to note).

## Status matrix (Ollama 0.30.10, local)

| Want                                   | Route                                          | Result |
|----------------------------------------|------------------------------------------------|--------|
| 8-bit **MLX** (like `qwen3.5:2b-mlx`)  | `--experimental -q mxfp8`                       | ❌ MLX quantizer panic (`no Stream(gpu,1)`) |
| **MLX**, any size                      | `--experimental` (bf16), text-only weights     | ✅ shipped (~3.8 GB, no vision) |
| **MLX** from pre-quantized file        | import `mlx_vlm`/`mlx_lm` 8-bit                 | ❌ `unknown data type: U32` |
| 8-bit, **GGUF/llama.cpp**              | `-q q8_0` (no `--experimental`), text-only      | untested; legacy path otherwise crashed on vision (`image_mean`) — text-only may fix it |
| Full multimodal MLX (vision + MTP)     | `--experimental` on full checkpoint            | ❌ gibberish (importer mis-maps mtp/visual) |

## Concrete questions for deep research

1. **`quantize.go` MLX-stream panic:** Is `mlx: There is no Stream(gpu, 1) in current
   thread` during `ollama create --experimental -q ...` a known issue? Is it fixed in any
   released/pre-release Ollama? Is there a workaround (env var, single-stream mode, running
   the quantization server-side, a Metal/macOS requirement)?
2. **Pre-quantized import:** What exact on-disk layout does Ollama's MLX engine expect for
   quantized tensors (the `vnd.ollama.image.tensor` layers in `qwen3.5:2b-mlx`)? Is there a
   supported/community tool to convert an HF or MLX model into that mxfp8 layer format so I
   can `ollama create` from it without the local quantizer? Why is mlx-vlm/mlx-lm's U32
   format rejected, and can it be transcoded?
3. **Registry build provenance:** How was `qwen3.5:2b-mlx` (mxfp8, `requires 0.19.0`)
   actually produced — what tool/pipeline, and is it runnable locally? (Chain reportedly:
   `mlx-community/Qwen3.5-2B-MLX-8bit` → Ollama mxfp8.)
4. **Full multimodal import:** Can Ollama 0.30.10 import a full `qwen3_5`
   (`Qwen3_5ForConditionalGeneration`) with vision + MTP via `--experimental` and serve it
   correctly? If the importer mis-handles `mtp.*`/`model.visual.*`, is there a supported way
   (config flags, `--draft-quantize` for MTP) to keep them?
5. **GGUF vision metadata:** Is there a way to carry `image_mean`/vision-preprocessor
   metadata through the legacy GGUF conversion so the multimodal model runs on llama.cpp,
   or is text-only stripping the only option there?
6. **Alternative serving:** Tradeoffs of serving the MLX model via `mlx_lm.server`
   (OpenAI-compatible) instead of Ollama — does it support mxfp8, KV-cache reuse, idle
   unload, and `reasoning_effort`/think-disable equivalents? (My summarizer talks to
   Ollama's `/v1` OpenAI endpoint and relies on Ollama-specific KV reuse + idle unload.)
7. **Blob hygiene:** Recommended way to prune unreferenced Ollama blobs and to bound peak
   disk during `ollama create` (avoid the full raw-safetensors copy).

## Repro snippets

Strip to text-only:

```python
from safetensors import safe_open
from safetensors.torch import save_file
import json, os, shutil
src = "run1-model/model.safetensors-00001-of-00001.safetensors"
dst = "run1-text"; os.makedirs(dst, exist_ok=True)
keep = {}
with safe_open(src, framework="pt") as f:
    for k in f.keys():
        if k.startswith("mtp.") or k.startswith("model.visual."):
            continue
        keep[k] = f.get_tensor(k)
save_file(keep, f"{dst}/model.safetensors", metadata={"format": "pt"})
cfg = json.load(open("run1-model/config.json")); cfg.pop("vision_config", None)
json.dump(cfg, open(f"{dst}/config.json", "w"), indent=2)
# also copy tokenizer*/generation_config/chat_template/processor_config
```

Imports:

```sh
# works (bf16 MLX, text-only)
ollama create qwen3.5-2b-tts --experimental -f Modelfile      # FROM ./run1-text

# panics (any -q on experimental)
ollama create qwen3.5-2b-tts --experimental -q mxfp8 -f Modelfile

# legacy GGUF (crashes at runtime on full model: image_mean not found)
ollama create qwen3.5-2b-tts -f Modelfile
```
