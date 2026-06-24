#!/usr/bin/env python3
"""First-round LoRA finetune of the TTS summarizer with mlx-tune.

Trains an MLX build of the served model (Qwen3.5-2B) on the curated
spoken-summary dataset produced by export_finetune.py. The dataset is already
one `{"messages": [...]}` chat conversation per chain, assembled the same way
summarizer.py builds its live history, so training matches inference.

mlx-tune is an Unsloth-compatible MLX trainer:
  - FastLanguageModel.from_pretrained / get_peft_model attach the LoRA adapter,
  - SFTTrainer + SFTConfig drive a native mlx-lm training loop,
  - train_on_responses_only masks the system/user turns so the loss lands only
    on the assistant summaries (multi-turn teacher forcing).

Round-one choices (see docs/dataset_refinement.md): no held-out validation set
(val_batches=0) — the corpus is small and the real eval is listening, not val
cross-entropy — so every chain trains. Frequent logging (logging_steps) gives a
visible loss trajectory across the first epoch.

Run it:
    uv run --with mlx-tune python scripts/finetune.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA = REPO_DIR / "datasets" / "finetune" / "train.jsonl"
DEFAULT_OUT = REPO_DIR / "datasets" / "finetune" / "run1"
# An already-MLX-quantized build of the served qwen3.5:2b-mlx model.
DEFAULT_MODEL = "mlx-community/Qwen3.5-2B-MLX-8bit"


def patch_gated_delta_scan(block_size: int):
    """Chunk Qwen3.5's gated delta-net training scan to bound its memory.

    Qwen3.5 is a hybrid model: 3 of every 4 layers are linear-attention (gated
    delta-net) blocks. mlx-lm has a fast Metal kernel for them, but the kernel has
    no backward, so in training mode the layer falls back to ``gated_delta_ops`` —
    a per-timestep Python loop (the source calls it a "reference implementation").
    Autodiff retains that loop's entire T-step graph for the layer, so activation
    memory grows *linearly with sequence length*. This, not the cross-entropy, is
    the real OOM: it walls out at ~512-1024 tokens on a 16 GB Mac (the run dies on
    the first long example regardless of how the loss is computed).

    Fix: process the sequence in ``block_size``-timestep blocks, each wrapped in
    ``mx.checkpoint``, carrying the recurrent state across blocks. A block's graph
    is freed after its forward and recomputed during backward, so peak scan memory
    is O(block_size) instead of O(T) — independent of sequence length. The carried
    state threads the gradient across blocks, so the result is unchanged.

    Patches the module global ``gated_delta.gated_delta_ops``; ``gated_delta_update``
    resolves it by name at call time, so the linear-attention layers pick it up.
    """
    import mlx.core as mx
    from mlx_lm.models import gated_delta

    original_ops = gated_delta.gated_delta_ops

    def chunked_ops(q, k, v, g, beta, state=None, mask=None):
        B, T = q.shape[0], q.shape[1]
        if state is None:
            Hv, Dv = v.shape[-2], v.shape[-1]
            Dk = q.shape[-1]
            state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)

        def run_block(qb, kb, vb, gb, bb, st, mb):
            # mx.checkpoint differentiates w.r.t. array args and recomputes the
            # block's per-timestep loop in the backward pass instead of storing it.
            return original_ops(qb, kb, vb, gb, bb, st, mb)

        checkpointed = mx.checkpoint(run_block)
        outs = []
        for start in range(0, T, block_size):
            sl = slice(start, min(start + block_size, T))
            mb = None if mask is None else mask[:, sl]
            y, state = checkpointed(
                q[:, sl], k[:, sl], v[:, sl], g[:, sl], beta[:, sl], state, mb
            )
            outs.append(y)
        return mx.concatenate(outs, axis=1), state

    gated_delta.gated_delta_ops = chunked_ops


def make_chunked_ce_loss(chunk_size: int):
    """Return an mlx-lm-compatible loss that never materializes full logits.

    mlx-lm's stock ``default_loss`` does ``logits = model(inputs)`` then a single
    ``cross_entropy`` over the whole ``[B, S, V]`` logit tensor. For this model
    V = 248,320, so at S=1024 that tensor alone is ~1 GB in fp32 (plus its
    gradient and the softmax temporary) — a sequence-linear spike on the ~11.5 GB
    Metal working set of a 16 GB Mac. This is *not* the primary OOM (that is the
    delta-net scan; see ``patch_gated_delta_scan``), but it is real headroom: a
    secondary sequence-linear term worth removing while we are at it. Apple's "Cut
    Cross-Entropy" names the problem; no MLX build of it is wired into
    mlx-lm/mlx-tune, so we chunk it ourselves.

    The model splits cleanly into a transformer body (whose layers mlx-lm already
    grad-checkpoints) and a frozen tied head. We run the body once for the small
    ``[B, S, d]`` hidden states, then walk the sequence in ``chunk_size`` slices,
    applying the head + cross-entropy to each slice inside ``mx.checkpoint`` so a
    chunk's ``[B, chunk_size, V]`` logits are freed immediately and recomputed in
    the backward pass. Peak logit memory drops from O(S·V) to O(chunk_size·V);
    the result is numerically identical to the stock loss.
    """
    import mlx.core as mx
    import mlx.nn as nn

    def loss(model, batch, lengths):
        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        lm = model.language_model
        body = lm.model  # layers already grad-checkpointed by mlx-lm
        head = lm.lm_head if getattr(lm, "lm_head", None) is not None else (
            body.embed_tokens.as_linear  # tied embeddings (this model)
        )

        hidden = body(inputs)  # [B, S, d] — cheap, kept resident

        # Same response-only mask mlx-lm's default_loss builds: loss lands only on
        # tokens in [lengths[:,0], lengths[:,1]) (assistant turns here).
        steps = mx.arange(1, targets.shape[1] + 1)
        mask = mx.logical_and(steps >= lengths[:, 0:1], steps <= lengths[:, 1:])

        def chunk_ce(h_chunk, t_chunk, m_chunk):
            logits = head(h_chunk)
            ce = nn.losses.cross_entropy(logits, t_chunk) * m_chunk
            return ce.astype(mx.float32).sum()

        checkpointed = mx.checkpoint(chunk_ce)
        seq_len = targets.shape[1]
        total = mx.zeros((), dtype=mx.float32)
        for start in range(0, seq_len, chunk_size):
            sl = slice(start, min(start + chunk_size, seq_len))
            total = total + checkpointed(hidden[:, sl], targets[:, sl], mask[:, sl])

        ntoks = mask.sum()
        return total / ntoks, ntoks

    return loss


def load_chats(path: Path) -> list[dict]:
    """Load the exported chat dataset as a list of {"messages": [...]} dicts."""
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise SystemExit(f"[finetune] no rows in {path}; run export_finetune.py first")
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--num-layers", type=int, default=8)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument(
        "--ce-chunk-size",
        type=int,
        default=128,
        help="Sequence chunk for the memory-efficient cross-entropy loss "
        "(caps peak logit memory at chunk_size x vocab). 0 disables chunking.",
    )
    p.add_argument(
        "--scan-chunk-size",
        type=int,
        default=256,
        help="Timestep block for the gated delta-net training scan (caps peak "
        "scan memory at O(block) instead of O(seq_len)). 0 disables chunking.",
    )
    args = p.parse_args()

    # The real OOM on this hybrid model is the gated delta-net training scan, not
    # the loss. Patch it to a chunked scan before any model/training code runs.
    if args.scan_chunk_size > 0:
        patch_gated_delta_scan(args.scan_chunk_size)

    # Imported lazily so --help works without the (heavy) mlx-tune install.
    from mlx_tune import (
        FastLanguageModel,
        SFTConfig,
        SFTTrainer,
        train_on_responses_only,
    )

    dataset = load_chats(args.data)
    iters_per_epoch = max(1, len(dataset) // args.batch_size)
    print(
        f"[finetune] {len(dataset)} chains | batch {args.batch_size} | "
        f"{iters_per_epoch} iters/epoch x {args.epochs} epochs "
        f"= {iters_per_epoch * args.epochs} iters",
    )

    print("[finetune] loading base model...", flush=True)
    model, tokenizer = FastLanguageModel.from_pretrained(
        args.model, max_seq_length=args.max_seq_length
    )
    print("[finetune] base model loaded; configuring LoRA...", flush=True)
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        # Trade compute for memory: recompute activations in the backward pass.
        # Needed to fit a 2.2B model on a 16GB-class unified-memory Mac.
        use_gradient_checkpointing="unsloth",
    )
    print("[finetune] LoRA configured; building trainer...", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    config = SFTConfig(
        output_dir=str(args.out),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        logging_steps=args.logging_steps,
        save_steps=iters_per_epoch,  # checkpoint at each epoch boundary
        max_seq_length=args.max_seq_length,
        num_layers=args.num_layers,
        grad_checkpoint=True,  # fit within ~16GB unified memory
        val_batches=0,  # round one: no held-out validation (see module docstring)
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        # A throwaway eval set just so the data loader has a non-empty valid
        # file; val_batches=0 means it is never actually evaluated.
        eval_dataset=dataset[:2],
        tokenizer=tokenizer,
        args=config,
        adapter_path=str(args.out / "adapters"),
    )
    # Loss only on assistant turns; markers auto-detected from the chat template.
    trainer = train_on_responses_only(trainer)

    # mlx-tune's SFTTrainer hands off to mlx-lm's train loop without exposing a
    # loss hook, and that loop hardcodes a full-logits cross-entropy that OOMs on
    # this 248k-vocab model. mlx-lm's train() *does* accept loss=, so inject our
    # chunked cross-entropy at the one call site mlx-tune holds a reference to.
    if args.ce_chunk_size > 0:
        import mlx_tune.sft_trainer as _sft

        _orig_train = _sft.mlx_train

        def _train_with_chunked_loss(*a, **kw):
            kw.setdefault("loss", make_chunked_ce_loss(args.ce_chunk_size))
            return _orig_train(*a, **kw)

        _sft.mlx_train = _train_with_chunked_loss

    trainer.train()
    print(f"[finetune] done -> {args.out / 'adapters'}")


if __name__ == "__main__":
    main()
