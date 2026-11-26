# ComfyUI-H3-Probe

Look inside MiniMax-H3 runs in ComfyUI: measure what a LoRA or a clock change does to a clip, and, while it
samples, which blocks, layers and tokens it acts on.

Status: early. Available now:

- **H3 Video Metrics**: sharpness, fine detail, PSNR, flicker and brightness-flash spikes against a reference
  clip, plus held-copy drift for time-smeared (de-rope) clips.

In progress (see [docs/DESIGN.md](docs/DESIGN.md)): **H3 Probe** (per block / module / step / token LoRA
attribution and layer-input captures), **H3 Probe Compare** and **H3 Probe Overlay** (heatmaps on the video).

This pack is separate from [ComfyUI-MAINodes](https://github.com/matlowai/ComfyUI-MAINodes) on purpose and is
intended to fold into it later. First use case: the
[H3 variable-clock LoRA](https://huggingface.co/MATLOWAI/MiniMax-H3-Variable-Clock).

Tests: `python tests/test_metrics.py`. MIT licensed.
