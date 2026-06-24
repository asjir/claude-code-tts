# Gradient checkpointing, sequence length, and our OOM

## What it trades

A forward pass produces an *activation* at every layer — the intermediate
tensors the backward pass needs to compute gradients. Normally all of them are
kept resident until backprop consumes them. For a transformer the activation
memory scales as

$$
M_\text{act} \;\sim\; O(L \cdot b \cdot s \cdot d)
$$

where $L$ = number of layers, $b$ = batch size, $s$ = sequence length, $d$ =
hidden size. (Attention adds an $O(b \cdot h \cdot s^2)$ term for the scores,
though MLX/flash-style kernels avoid materializing the full $s \times s$ matrix.)

**Gradient checkpointing** (a.k.a. activation recomputation) keeps only a few
*checkpoints* — typically the input to each transformer block — and throws away
the activations *inside* each block. During backward it re-runs the block's
forward to regenerate them on demand. Memory for the kept activations drops from
$O(L)$ to roughly $O(\sqrt{L})$ (checkpoint every $\sqrt{L}$ layers), or to a
single block's worth if you checkpoint every block ("unsloth" mode). The cost is
one extra forward pass — about **+30–35% compute**, no accuracy change.

## Where sequence length comes in

Both the activation term *and* checkpointing's savings live on the same factor
that is **linear in $s$**. So:

- Going $2048 \to 1024$ already **halves** $M_\text{act}$ on its own.
- Checkpointing then shrinks whatever is left of that term.

At $s = 1024$, $b = 1$, $d = 2048$, $L = 36$, the full activation footprint is
only on the order of a few hundred MB — and checkpointing makes a fraction of
*that* smaller. It is a rounding error against the things that **do not depend on
$s$ at all**:

| Term | Scales with | ~Size here (2.2B, 8-bit) |
|---|---|---|
| Model weights | params × bytes/param | ~2.2 GB (4.4 GB if upcast to bf16) |
| Optimizer state (LoRA only) | trainable params × 8 B | ~20 MB (negligible) |
| RoPE / mask buffers | context length | tiny **iff** sized to $s$, huge if sized to max ctx |
| Activations | $L\cdot b\cdot s\cdot d$ | a few hundred MB at $s{=}1024$ |

## Why our run still OOMs

The crash lands **before the training loop prints anything** — i.e. during
`from_pretrained` / `get_peft_model`, not during a forward/backward step. So the
peak that blows the ~10.6 GB Metal working-set limit on this 16 GB Mac is a
**load-time** allocation, dominated by weights (and possibly a transient upcast
or a context-sized buffer), **not** activations. Grad checkpointing and a shorter
$s$ only touch the activation term, which isn't what's overflowing — hence no
effect.

Levers that actually target the overflowing terms:

1. **4-bit base** (`mlx-community/Qwen3.5-2B-OptiQ-4bit` or `load_in_4bit`):
   halves weight memory to ~1.2 GB. Biggest single win. QLoRA-style; fine for a
   first round.
2. **Confirm the dtype on load** — if `from_pretrained` upcasts the 8-bit repo to
   bf16, weights double to ~4.4 GB. Keep it quantized.
3. **mlx_lm.lora directly** — already proven to reach `Iter 1` with this exact
   8-bit model on this machine; add `--mask-prompt` for response-only loss. This
   is the fallback if mlx-tune's loader is the memory hog.
4. **Raise the MLX memory limit / wired limit** as a last resort
   (`mx.set_memory_limit`) — lets MLX spill instead of aborting, at the cost of
   speed.
