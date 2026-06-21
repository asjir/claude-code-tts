# Dataset refinement workflow

How we curate gold spoken-summary labels to finetune the local TTS summarizer
(`qwen3.5:2b-mlx`, see `scripts/summarizer.py`). The corpus is the assistant
turns extracted from `~/.claude/projects/*/*.jsonl` — the same chains
`build_dataset.py` replays.

The bottleneck is **your labeling time**, not compute. So the loop is a
self-converging markdown file you edit in Obsidian: tables of
`input | output | id`, where **italic output = unconfirmed** and **plain output
= accepted gold**. You seed the style by hand; Claude fills the rest in
italics; the file converges to all-plain; then it exports to a finetuning set.

## Artifacts

All live under `datasets/` (gitignored — regenerable):

| File | Role |
|---|---|
| `labels.md` | Human-facing. One table per chain; you edit the **output** column in Obsidian. |
| `labels_raw.jsonl` | Source of truth: the exact turn text per row `{id, chain_key, session_id, chain_id, turn_index, prompt_text, assistant_text, is_final}`. The markdown flattens newlines/pipes for rendering, so the raw file is what export reads back. |
| `labels_work.json` | Scratch: the empty/italic cells Claude needs to fill, with per-chain accepted cells as style anchors. Produced by `fill_labels.py extract`. |
| `labels_filled.json` | Scratch: a flat `{id: "summary"}` map Claude writes, consumed by `fill_labels.py apply`. |
| `finetune/train.jsonl`, `finetune/valid.jsonl` | The mlx-lm chat dataset, split by chain. |

## The labeling protocol

Each chain is a heading, the human prompt as a blockquote, then a table:

```
## chain a1b2c3d4e5f6 — sess 9f3c… (claude-code-tts)
> User asked: patch the watcher to stream via ffplay

| input | output | id |
|---|---|---|
| [PROGRESS UPDATE] Reading the config and patching the watcher… |  | a1b2c3-00 |
| [FINAL REPLY] Done — the watcher now streams via ffplay…       |  | a1b2c3-01 |
```

- **input** = the `[PROGRESS UPDATE]`/`[FINAL REPLY]` tag + the full assistant
  text, flattened (newlines → `<br>`, `|` → `\|`). Code won't render but every
  character is preserved.
- **output** = the spoken summary. Empty to start.
- **id** = `{chain_id[:6]}-{turn_index:02d}`, with a `.N` suffix if two chains
  share a prompt (chain ids are `sha1(prompt)`, so identical prompts collide).
  The stable key joining a markdown row to its raw turn.

### State machine (one rule, applied each time Claude touches the file)

- output **empty** → Claude fills it with an *italic* proposal.
- output **\*italic\*** → unconfirmed; Claude **redoes** it (a fresh swing at
  different phrasing). If the italic text reads like an instruction
  (e.g. `*shorter, drop the file name*`), Claude treats it as a **hint** — a
  free "notes" channel, no 4th column needed.
- output **plain, non-empty** → **frozen/accepted**; Claude never touches it.
  This also covers your hand-written seed cells, which double as style anchors.

**Convergence** = no empty and no italic cells remain. For precise control,
rewrite a cell directly — plain text is accepted as-is.

## The loop

### 0. Build the document (once)

```sh
uv run python scripts/build_label_doc.py --all
```

Writes `datasets/labels.md` + `datasets/labels_raw.jsonl`. Open `labels.md` in
Obsidian. (Scope can be narrowed: a project query or `--sessions FILE …`
instead of `--all`.)

> Rebuilding overwrites your edits — build once, then refine in place.

### 1. Seed the style by hand (you, first)

Hand-write **plain** outputs for the first few chains. These are your style
anchors: they teach Claude the voice (laconic, no paths/code/URLs read aloud,
present-tense for progress, wrap-up for final). Spend your effort here — a
handful of strong seeds steers every proposal that follows.

### 2. Let Claude propose the rest (round-trip)

```sh
uv run python scripts/fill_labels.py extract          # -> datasets/labels_work.json
# Claude reads labels_work.json and writes datasets/labels_filled.json
uv run python scripts/fill_labels.py apply datasets/labels_filled.json
```

`extract` collects every empty/italic cell plus, per chain, the accepted plain
cells as in-context anchors. Claude proposes summaries (honoring the summarizer
rules and your anchors) into a flat `{id: summary}` map. `apply` rewrites only
those cells, each wrapped in `*italics*`; plain cells stay byte-identical, and a
cell you've since accepted is left frozen even if it slips into the map.

### 3. Review and converge (you)

In Obsidian, for each italic cell:

- **Good?** Delete the surrounding `*` → it's now plain/accepted.
- **Close?** Edit the text directly and drop the `*` → accepted as written.
- **Wrong direction?** Replace the italic text with an *instruction*
  (still italic), e.g. `*say what test passed, not how*`, and re-run step 2 —
  Claude redoes it using that hint.
- **Leave it italic** to have Claude take another fresh swing next round.

Repeat steps 2–3. Each pass only touches empty/italic cells, so the file
monotonically converges to all-plain. Check progress with:

```sh
grep -cE '\|  \| [0-9a-f.]+-[0-9]+ \|$' datasets/labels.md   # empty cells left
grep -cE '\| \*.*\* \|' datasets/labels.md                   # italic cells left
```

### 4. Export the finetuning set (when converged)

```sh
uv run python scripts/export_finetune.py        # -> datasets/finetune/{train,valid}.jsonl
```

Fails loudly if any cell is still empty or italic. For each chain it assembles
one mlx-lm chat conversation the same way the summarizer builds its live
history — system prompt, then per turn a user message (tagged body; turn 0 also
carries the prompt) and the **curated** summary as the assistant reply (teacher
forcing, so the training prefix matches inference). Chains are split ~15% to
`valid.jsonl` so no chain leaks across the split.

## Downstream finetuning (later)

- `mlx-lm` LoRA SFT on `datasets/finetune/` locally on the Mac.
- Eval = run held-out chains through the adapter and listen (reuse the
  `tts_watch` path).
- Optional DPO later: pairs come free as (pre-finetune model output, curated
  output).
