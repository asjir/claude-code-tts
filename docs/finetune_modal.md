# Finetuning the TTS summarizer on Modal (CUDA)

The local MLX finetune (`scripts/finetune.py`) failed: NaN loss from iter 10 and a
pinned Mac. Root cause was the assistant-only loss combined with `max_seq_length=1024`
truncation — long chains had every loss token cut away, so the masked loss was `0/0`
(see `datasets/finetune/run1.log` and `docs/gradient_checkpointing.md`). We retrain on
real CUDA instead.

`scripts/finetune_modal.py` LoRA-finetunes the **HF base** `Qwen/Qwen3.5-2B` — the source
of the served `mlx-community/Qwen3.5-2B-MLX-8bit` / Ollama `qwen3.5:2b-mlx` build — in
bf16 on a Modal A10G via **Unsloth** (which has first-class Qwen3.5 support), merges the
adapter into a full bf16 model, and writes it to a Modal Volume. No quantization (so no
NaN) and seq 8192 (covers the whole corpus, max ~4133 tok, so nothing truncates).
Assistant-only loss comes from Unsloth's `train_on_responses_only` (marker-based masking
on `<|im_start|>assistant`), and chats render with `enable_thinking=False` to match the
served `reasoning_effort=none`.

## Prereqs

- `modal` CLI installed (`uv tool install modal`) and the `pa-paradysz` profile active
  (`modal profile list`).
- Modal secrets `hf-token` and `wandb` exist in the workspace (`modal secret list`). The
  function reads `HF_TOKEN`/`HF_TOKEN`-style and `WANDB_API_KEY` from them.
- `datasets/finetune/train.jsonl` present locally (regenerate with
  `uv run python scripts/export_finetune.py` if needed). It is baked into the image at run
  time, so no upload step.

## Run

Smoke test first (loads the model via Unsloth, applies response-only masking, runs a few
steps — proves the qwen3_5 hybrid arch trains and loss is finite, not NaN):

```sh
modal run scripts/finetune_modal.py --max-steps 5
```

Full run (3 epochs; ~705 chains, batch 1 × grad-accum 8):

```sh
modal run scripts/finetune_modal.py
```

Useful knobs (all have defaults): `--run-name`, `--epochs`, `--learning-rate`,
`--lora-r`, `--lora-alpha`, `--max-seq-length`, `--grad-accum`. Watch the loss curve in
Weights & Biases (project `tts-summarizer-finetune`).

## Fetch the merged model

A plain recursive `modal volume get tts-finetune-out run1/merged <dest>` trips on the
`.cache/` subdir Unsloth leaves in the output (`[Errno 21] Is a directory`). Pull the
model files individually instead — this also skips the useless cache:

```sh
dest=datasets/finetune/run1-model
for f in model.safetensors-00001-of-00001.safetensors model.safetensors.index.json \
         config.json generation_config.json tokenizer_config.json tokenizer.json \
         processor_config.json chat_template.jinja; do
  modal volume get tts-finetune-out "run1/merged/$f" "$dest/$f"
done
```

This is a full bf16 HF model + tokenizer (~4.3 GB, arch `Qwen3_5ForConditionalGeneration`)
— the portable artifact. Sanity-check it locally with transformers (load it, feed a
`summarizer.py`-style system+user prompt, confirm the spoken-summary style) before
converting.

## Serve it back through Ollama (local, manual)

The live summarizer (`scripts/summarizer.py`) serves via Ollama's MLX engine (the served
`qwen3.5:2b-mlx` is arch `qwen3_5`, `mxfp8`). The fine-tuned HF model has to land in that
same MLX engine. This was the fiddly part — the procedure below produces an **8-bit MLX**
build on Ollama 0.30.10. The dead ends are recorded after it so nobody re-walks them.

The winning shape: **pre-quantize with `mlx_lm` first, then `ollama create --experimental`
with no `-q`.** Ollama's experimental importer can ingest a pre-quantized MLX folder fine;
it only breaks when *it* has to do the quantization math (see dead end 4).

1. Write a **text-only** copy of the merged weights — drop `mtp.*` (multi-token-prediction)
   and `model.visual.*` (vision) tensors, and `vision_config` from `config.json`. The
   summarizer is text-only, and those extra tensors otherwise scramble the importer (dead
   end 3):

   ```sh
   uv run --with safetensors --with torch python - <<'PY'
   import json, os, shutil
   from safetensors import safe_open
   from safetensors.torch import save_file
   src = "datasets/finetune/run1-model/model.safetensors-00001-of-00001.safetensors"
   dst = "datasets/finetune/run1-text"; os.makedirs(dst, exist_ok=True)
   keep = {}
   with safe_open(src, framework="pt") as f:
       for k in f.keys():
           if k.startswith("mtp.") or k.startswith("model.visual."):
               continue
           keep[k] = f.get_tensor(k)
   save_file(keep, f"{dst}/model.safetensors", metadata={"format": "pt"})
   for fn in ("generation_config.json", "tokenizer_config.json", "tokenizer.json",
              "chat_template.jinja", "processor_config.json"):
       p = f"datasets/finetune/run1-model/{fn}"
       if os.path.exists(p): shutil.copy(p, f"{dst}/{fn}")
   cfg = json.load(open("datasets/finetune/run1-model/config.json"))
   cfg.pop("vision_config", None)
   json.dump(cfg, open(f"{dst}/config.json", "w"), indent=2)
   PY
   ```

2. Quantize to 8-bit with **`mlx_lm`** (not `mlx_vlm` — it wraps the LM under a
   `language_model.` prefix). ~8.5 bits/weight:

   ```sh
   uv run --with mlx-lm python -m mlx_lm convert \
     --hf-path ./datasets/finetune/run1-text \
     --mlx-path ./datasets/finetune/run1-mlx8 -q --q-bits 8
   ```

   `mlx_lm` rewrites `config.json` (adds `quantization_config` — keep it) but **strips
   `added_tokens_decoder` from `tokenizer_config.json`**. Restore the full one so special
   tokens stay registered:

   ```sh
   cp datasets/finetune/run1-text/tokenizer_config.json datasets/finetune/run1-mlx8/
   ```

3. Import the pre-quantized folder via the experimental MLX path (**no `-q`**):

   ```sh
   cd datasets/finetune
   printf 'FROM ./run1-mlx8\n' > Modelfile
   ollama create qwen3.5-2b-tts --experimental -f Modelfile
   ```

   `ollama show qwen3.5-2b-tts` should report `architecture qwen3_5`, `quantization int8`,
   `requires 0.19.0` (the MLX engine) — not `qwen35`/`F16`.

4. Sanity-check through the OpenAI endpoint the summarizer uses, thinking off:

   ```sh
   curl -s http://localhost:11434/api/generate \
     -d '{"model":"qwen3.5-2b-tts","prompt":"Reply with exactly: it works","think":false,"stream":false}'
   ```

   Coherent output = good; multilingual gibberish = the strip in step 1 didn't take. The
   reply will contain an empty `<think></think>` block — that's expected (the fine-tune was
   trained with `enable_thinking=False`, which bakes that marker in, and Ollama doesn't strip
   it under `reasoning_effort=none`). `summarizer.py` removes it via `_strip_think()` before
   TTS.

5. Point the summarizer at it: `MODEL = "qwen3.5-2b-tts"` in `scripts/summarizer.py` (done).

### Dead ends (Ollama 0.30.10, don't re-walk)

1. **Pre-quantizing with `mlx_vlm`** → MLX U32-packed weights; `ollama create` rejects with
   `Error: unknown data type: U32`. Use `mlx_lm` instead, and import with `--experimental`.
2. **Plain `ollama create` (no `--experimental`)** → legacy GGUF/llama.cpp path (arch
   `qwen35`, F16), which crashes at runtime with `GGML_ASSERT ... "image_mean not found"`
   (vision tower). `--experimental` is required for the MLX engine.
3. **Importing the full checkpoint with `mtp.*`/`model.visual.*`** → experimental import
   "succeeds" but emits multilingual gibberish. Strip those tensors first (step 1).
4. **`ollama create --experimental -q <anything>`** (mxfp8, q8_0, q4_K_M all the same) →
   `panic: mlx: There is no Stream(gpu, 1) in current thread` in `quantize.go`. On
   `--experimental`, every `-q` value routes through the MLX quantizer, which is broken in
   0.30.10 — hence pre-quantizing with `mlx_lm` and importing with no `-q`.

Note: this is **int8** (mlx_lm affine 8-bit), not the stock `mxfp8` (float8) of
`qwen3.5:2b-mlx`. Both are 8-bit MLX; revisit mxfp8 if a newer Ollama fixes the
`quantize.go` panic (then `--experimental -q mxfp8` from `run1-text` would work directly).

## Epochs vs. data — why v1 trains 1 epoch

v1 (`run1`) trains a **single epoch** on the 703 chains (loss 1.61 → ~0.26, avg 0.557,
~20 min on an A10G). The `--epochs` knob defaults to 3, but the deliberate stance is:
**don't chase epochs, grow the dataset.** Re-running the labeling loop in
`docs/dataset_refinement.md` to produce 3×/5× more curated chains is cheap, and more
*distinct* data is strictly better than more passes over the same data — it improves the
model without ever flirting with overfitting. So the playbook for v2+ is "generate more
labels, retrain 1 epoch," not "turn up epochs." Bump `--epochs` only if a future dataset
is genuinely small and underfitting at one pass.

## Notes / risks

- **Validation:** `valid.jsonl` currently holds only ~2 chains, so the run skips eval —
  matching the round-one "the real eval is listening, not val cross-entropy" stance in
  `docs/dataset_refinement.md` (and the epochs-vs-data note above). For a held-out metric,
  re-export with a larger split (`scripts/export_finetune.py --val-frac 0.15`) and wire
  eval into the trainer.
- **Versions:** Qwen3.5 needs `transformers` v5 (pinned in the image). Unsloth pulls the
  rest of the stack; if its torch/transformers pins ever conflict with that, relax the
  explicit pin and let Unsloth choose.
- The old MLX `scripts/finetune.py` is left in place as the superseded approach.
