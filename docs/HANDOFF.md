# Handoff / status

Keep this file current; it is the entry point for the next session.

## Done

- `h3probe/metrics.py` + **H3 Video Metrics** node (`h3probe/nodes_metrics.py`), `tests/test_metrics.py`.
- `h3probe/probe.py` (Recorder: forward hooks on `blocks[i].attn.qkv_proj / attn.out_proj / mlp.fc1 / mlp.fc2`
  and on each block; per call, per block x module x latent frame x grid cell: LoRA energy `||s B A x||^2` via
  `G = s^2 B^T B`, module output energy, block residual energy; target video rows only) and `tests/test_probe.py`
  (matches a direct recomputation; detach removes all hooks).
- **H3 Probe** node (`h3probe/nodes_probe.py`): clones the MODEL, adds an OUTER_SAMPLE wrapper (attach hooks,
  save `output/h3probe/<name>_NNNNN.npz` + `.json` after each sampler run) and a DIFFUSION_MODEL wrapper (hands
  `transformer_options` with `minimax_h3_layout` and `sigmas` to the recorder per call).

## Not yet done

1. Live test in ComfyUI on the H3 int8 DiT. Concrete plan (2026-11-25): bamboo sword clip
   (`comfy_test/input/sw_src_bamboo.mp4`), graph `musubi-tuner/references/h3_motion_clock_phase2/option1_probe/workflows/swords/bamboo_Q2.api.json`
   with H3 Probe inserted on the model feeding the run-f guider (attribute the run f LoRA and the Pika LoRA),
   once with H3 True Clock and once without (normal clock), same seed. Check (hook inputs on int8 convrot Linear, VRAM/time overhead, a
   two-sampler split producing two files). Test install used so far: a separate ComfyUI with
   `--base-directory .../comfy_test`, port 8199, GPU 1 (never the operator's live install).
2. **H3 Probe Compare** node: two probe files to block x step x token-group deltas (heatmap IMAGE + JSON).
   Token groups from the hold map: native / 2x / 4x latents and transition latents.
3. **H3 Probe Overlay** node: per-token heat over decoded frames (LoRA push by early / mid / late blocks as
   R / G / B; per-token temporal change; held-copy drift; flash markers).
4. Optional layer-input reservoir capture in DeTangle's capture schema (`docs/captures.md` there) for an
   A-side refit of the clock LoRA (wanted role: slowed + transition tokens with the clock on; suppressed role:
   native tokens and clock-off runs).
5. Clock gate (per-token LoRA scale from latent speed) as an inference node.
6. README, then a registry release when the probe is proven on real runs.

## Known pitfalls

- The packed sequence is [text, ..., audio, video]; the probe takes the trailing video segment from
  `transformer_options["minimax_h3_layout"]` and checks its row count against `signature`.
- Do not register on `patches_replace["dit"]` (DyRoPE / streamed blocks own it); plain module hooks only.
- ComfyUI LoRA hook keyframes fail on the int8 convrot DiT (`'Linear' object has no attribute 'weight_scale'`).
- A two-sampler split (e.g. the S6 Pika workflow) triggers OUTER_SAMPLE twice: expect two probe files per run.

## Context

Background, numbers and the reasons for each design choice: `docs/DESIGN.md`. The experiment log lives in the
musubi-tuner repo (`H3_WEEKEND_LOOP.md`, branch `experiment/h3-clock-smear`).
