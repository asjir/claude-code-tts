#!/usr/bin/env python3
"""CUDA LoRA finetune of the TTS summarizer on Modal, with Unsloth.

The earlier local MLX run (scripts/finetune.py) produced NaN loss and pinned the
Mac. Two causes: (1) max_seq_length=1024 truncated long chains so their
assistant-only loss tokens were cut away entirely -> sum(mask)=0 -> 0/0 = NaN;
(2) the 8-bit MLX base + Metal working-set limits on 16 GB unified memory, with a
hybrid (gated delta-net) arch that has no training backward in MLX.

This trains the real HF base `Qwen/Qwen3.5-2B` (the source of the served
mlx-community 8-bit / Ollama qwen3.5:2b-mlx build) in bf16 on an A10G via Unsloth,
which has first-class Qwen3.5 support. No quantization (no NaN) and seq 4096 — and
rather than cut a chain mid-response (the truncation that NaN'd the MLX run), the
few chains over 4096 tok have whole trailing (user, assistant) turn-pairs dropped
until they fit, so every surviving assistant turn is intact. A 2B LoRA at 4096 on a
24 GB A10G leaves ample VRAM, so gradient checkpointing is off (faster). It
LoRA-finetunes, merges the adapter into a full bf16 HF model, and writes it to a
Modal Volume. MLX re-quantization + `ollama create` happen locally afterward (see
docs/finetune_modal.md).

The dataset is scripts/export_finetune.py's `{"messages": [...]}` chat format,
assembled the same way summarizer.py builds its live history, so training matches
inference. Loss lands only on assistant turns via Unsloth's
`train_on_responses_only`, which masks everything up to each `<|im_start|>assistant`
marker — the robust, marker-based equivalent of the MLX `train_on_responses_only`.
Chats render with `enable_thinking=False` to match the served reasoning_effort=none.

Run it:
    modal run scripts/finetune_modal.py                 # full 3-epoch run
    modal run scripts/finetune_modal.py --max-steps 5   # smoke test (load + few steps)
"""

from __future__ import annotations

import modal

APP_NAME = "tts-summarizer-finetune"
BASE_MODEL = "Qwen/Qwen3.5-2B"
VOLUME_NAME = "tts-finetune-out"
DATA_DIR = "/data"  # where the dataset is baked into the image
OUT_DIR = "/out"  # where the output Volume is mounted

app = modal.App(APP_NAME)

# Cap transformers/trl to Unsloth's supported ranges (its pyproject pins
# transformers<=5.5.0, trl<=0.24.0). Without the caps, pip grabs the mirror's much
# newer transformers/trl, which forces it to backtrack to an ancient Unsloth whose
# unsloth_zoo breaks on `from trl.trainer.utils import ConstantLengthDataset`.
# Qwen3.5 still needs transformers v5, which sits inside that cap. Unsloth pulls
# unsloth_zoo/peft/torch itself.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "unsloth",
        "transformers>=5.2.0,<=5.5.0",
        "trl>=0.18.2,!=0.19.0,<=0.24.0",
        "bitsandbytes",
        "wandb",
    )
    # The datasets/ tree is gitignored/regenerable, so just bake the jsonl in
    # rather than staging a Volume upload.
    .add_local_dir(
        "datasets/finetune", DATA_DIR, ignore=["**/run*", "**/adapters", "*.log"]
    )
)

out_vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def render_chat(messages: list[dict], tokenizer, tokenize: bool):
    # enable_thinking=False stops the template injecting a <think> scaffold on the
    # last assistant turn, matching the served reasoning_effort=none.
    return tokenizer.apply_chat_template(
        messages,
        tokenize=tokenize,
        add_generation_prompt=False,
        enable_thinking=False,
    )


def fit_chat(messages: list[dict], tokenizer, max_len: int):
    """Drop whole trailing (user, assistant) pairs until the chain fits max_len.

    Trimming at turn granularity (never mid-message) keeps every surviving assistant
    turn intact, so its loss tokens always survive — the failure that NaN'd the MLX
    run was a chain cut mid-response. Trimming from the *end* also preserves the
    train/inference match for the kept turns (each still sees its full prior history).
    Returns (messages, n_dropped) or (None, n) if even the first turn overflows."""
    msgs = list(messages)
    dropped = 0
    while len(render_chat(msgs, tokenizer, tokenize=True)) > max_len:
        if len(msgs) <= 3:  # system + one user/assistant pair — can't trim further
            return None, dropped
        msgs = msgs[:-2]
        dropped += 1
    return msgs, dropped


@app.function(
    image=image,
    gpu="A10G",
    timeout=3600,
    volumes={OUT_DIR: out_vol},
    secrets=[
        modal.Secret.from_name("hf-token"),
        modal.Secret.from_name("wandb"),
    ],
)
def train(
    run_name: str = "run1",
    epochs: float = 3.0,
    learning_rate: float = 2e-4,
    lora_r: int = 16,
    lora_alpha: int = 16,
    max_seq_length: int = 4096,
    grad_accum: int = 4,
    max_steps: int = -1,
) -> str:
    import json
    import os
    from pathlib import Path

    # Unsloth must be imported before transformers/trl so its patches take hold;
    # isort: split keeps it ahead of the alphabetically-earlier datasets/trl.
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only

    # isort: split
    from trl import SFTConfig, SFTTrainer

    from datasets import Dataset

    use_wandb = bool(os.environ.get("WANDB_API_KEY"))
    if use_wandb:
        os.environ.setdefault("WANDB_PROJECT", APP_NAME)

    # --- model + LoRA -----------------------------------------------------
    print(f"[finetune] loading {BASE_MODEL} (bf16) via Unsloth...", flush=True)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=max_seq_length,
        load_in_4bit=False,
        load_in_16bit=True,
        full_finetuning=False,
    )
    # Qwen3.5-2B is a VLM, so Unsloth returns a multimodal processor whose
    # apply_chat_template expects list-of-parts content. We train text-only, so use
    # the processor's inner text tokenizer (plain-string content, standard chat path).
    text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        # Attention proj (full-attention layers) + MLP (every layer, incl. the
        # gated-delta-net blocks). Matches Unsloth's Qwen3.5 fine-tune guide.
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=lora_alpha,
        lora_dropout=0,
        bias="none",
        # 2B LoRA at seq 4096 fits an A10G with room to spare, so skip the
        # recompute-on-backward cost. Flip to "unsloth" if seq/batch grows.
        use_gradient_checkpointing=False,
        random_state=3407,
        max_seq_length=max_seq_length,
    )

    # --- data: fit each chain to max_seq_length at turn granularity, then render --
    rows = [
        json.loads(line)
        for line in (Path(DATA_DIR) / "train.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    texts, trimmed, dropped = [], 0, 0
    for row in rows:
        msgs, n = fit_chat(row["messages"], text_tokenizer, max_seq_length)
        if msgs is None:
            dropped += 1
            continue
        trimmed += n > 0
        texts.append(render_chat(msgs, text_tokenizer, tokenize=False))
    dataset = Dataset.from_list([{"text": t} for t in texts])
    print(
        f"[finetune] {len(dataset)} training chains "
        f"({trimmed} trimmed to fit {max_seq_length}, {dropped} dropped)",
        flush=True,
    )

    # --- trainer ----------------------------------------------------------
    out_path = Path(OUT_DIR) / run_name
    trainer = SFTTrainer(
        model=model,
        tokenizer=text_tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=grad_accum,
            num_train_epochs=epochs,
            max_steps=max_steps,
            learning_rate=learning_rate,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            bf16=True,
            logging_steps=10,
            optim="adamw_8bit",
            seed=3407,
            output_dir=str(out_path / "trainer"),
            report_to="wandb" if use_wandb else "none",
            run_name=run_name,
            dataset_num_proc=1,
        ),
    )
    # Loss only on assistant turns: mask everything up to each assistant marker.
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    print("[finetune] training...", flush=True)
    trainer.train()

    # --- merge + save -----------------------------------------------------
    print("[finetune] merging LoRA into base (bf16)...", flush=True)
    merged_dir = out_path / "merged"
    model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")
    out_vol.commit()
    print(f"[finetune] merged model -> {merged_dir}", flush=True)
    return run_name


@app.local_entrypoint()
def main(
    run_name: str = "run1",
    epochs: float = 3.0,
    learning_rate: float = 2e-4,
    lora_r: int = 16,
    lora_alpha: int = 16,
    max_seq_length: int = 4096,
    grad_accum: int = 4,
    max_steps: int = -1,
):
    run = train.remote(
        run_name=run_name,
        epochs=epochs,
        learning_rate=learning_rate,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        max_seq_length=max_seq_length,
        grad_accum=grad_accum,
        max_steps=max_steps,
    )
    print("\n[finetune] done. Fetch the merged model with:")
    print(
        f"    modal volume get {VOLUME_NAME} {run}/merged "
        f"./datasets/finetune/{run}-merged"
    )
