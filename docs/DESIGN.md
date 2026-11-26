# H3 Probe: design

Goal: see inside a MiniMax-H3 run while it samples, in ComfyUI, well enough to tell which blocks, layers and
tokens a LoRA (or a clock change) actually acts on, and save it so it can be analysed afterwards. First use
case: the H3 variable-clock LoRA (MATLOWAI/MiniMax-H3-Variable-Clock), where the useful temporal effect and
an unwanted softness both sit in blocks 35-49 and we want to keep the first and drop the second.

This is a separate pack on purpose (modular, opt-in). It is meant to fold into ComfyUI-MAINodes later.

## What is already known (from the clock-LoRA work, 2026-11-25)

- The clock LoRA adapts qkv / out / fc1 / fc2 in all 50 DiT blocks (no FinalLayer weights). Its weight change
  is 61 % in blocks 40-49; block 49 qkv is the largest single change.
- Block-range swaps with Pika 6-step: blocks 0-34 do almost nothing; blocks 35-49 carry both the steadier held
  copies and the softness.
- Turning the LoRA and the clock off for the last sampling step keeps most of the benefit with less softness
  (timing is decided early, detail late).
- ComfyUI's LoRA hook keyframes fail on the int8 convrot DiT, so step scheduling has to be done another way
  (split samplers, or this pack's own hooks).

## Lessons carried over from DeTangle (matlowai/DeTangle)

| DeTangle | here |
|---|---|
| triggered vs class prompts define "wanted" vs "bleed" responses | slowed (2x / 4x) and transition latents with the clock on = wanted; native-speed latents and clock-off runs = where any LoRA response is likely degradation |
| localise with cheap tensor swaps before solving anything | done for block ranges; the probe makes it per block, per module, per step and per token |
| A-side conditional refit `A' (Gc + mu Gt + eps I) = mu A Gt` from captured layer inputs, B fixed | capture layer inputs on the two token roles in DeTangle's capture schema, then refit the clock LoRA's A factors with DeTangle's solver |
| name gate: scale the adapter per patch by whether it reads the name | clock gate: scale the LoRA per token by its latent's speed, known exactly from the hold map |
| step scheduling beat searching for a cleaner fixed direction | per-step LoRA strength inside the probe's own hooks (works on int8) |
| a good local fit does not certify rendered behaviour | every edit is judged on rendered tiles and the video metrics, matched seeds, more than one prompt |
| record capture recipe, seeds, hashes; never overwrite outputs | every probe file carries a JSON manifest; files are never overwritten |

## Nodes

1. **H3 Video Metrics**: frames (+ reference frames, + hold map) to a report: sharpness and fine-detail energy
   vs the reference, PSNR, flicker vs the reference, brightness-flash p95, held-copy drift. Pure numpy.
2. **H3 Probe** (MODEL in, MODEL out): forward hooks on every LoRA-adapted linear (qkv, out, fc1, fc2 x 50).
   Given the LoRA file(s) to attribute, it recomputes each LoRA's low-rank contribution `s * B (A x)` from the
   layer input and records, per sampling step, block, module and token group, the contribution norm and the
   base output norm. Token groups: per latent frame (so native / 2x / 4x / transition come from the hold map)
   and a coarse spatial grid. It also records each block's residual update norm. Reductions only, so files
   stay small. Optional: reservoir-sample layer inputs per token role into DeTangle's capture schema.
3. **H3 Probe Save / Load**: writes `probes/<name>.npz` + `.json` manifest (model, LoRA hashes, seed, sigmas,
   hold map, prompt hash, settings); refuses to overwrite.
4. **H3 Probe Compare**: two probe files (clock on vs normal clock; LoRA on vs off) to per block x step x
   token-group deltas, as an IMAGE heatmap and a JSON table.
5. **H3 Probe Overlay**: decoded frames + probe to annotated video: where the LoRA pushes (early / mid / late
   blocks in R / G / B), direction (hue from a projection of the contribution), per-token temporal change,
   held-copy drift, flash markers.
6. Later: **H3 Clock Gate** (per-token LoRA scale by latent speed) and **refit** tooling around DeTangle's solver.

## Integration notes (ComfyUI core H3, verified by reading comfy/ldm/minimax/model.py)

- Blocks: `diffusion_model.blocks[i]` (`DiTBlock`), modules `attn.qkv_proj`, `attn.out_proj`, `mlp.fc1`,
  `mlp.fc2`. Token refiner blocks also exist (`token_refiner.blocks`).
- The packed sequence ends with the target audio then target video rows; the video segment is the trailing
  `T * (H/2) * (W/2)` rows. `transformer_options["minimax_h3_layout"]` carries the segment spans.
- Use plain module forward hooks, not `patches_replace["dit"]`: DyRoPE and streamed blocks already own that
  slot and only one patch per key is allowed.
- LoRAs are merged into the weights (or patched on the fly), so the LoRA's own contribution is recomputed from
  the LoRA file, not read from the model.
- Step index / sigma: a sampler-level wrapper (as MAINodes' True Clock does) or `transformer_options`.
