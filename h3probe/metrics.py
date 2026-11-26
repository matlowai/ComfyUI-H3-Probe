"""Video metrics for H3 runs (pure numpy, no ComfyUI imports).

Frames are float arrays in [0, 1] shaped [T, H, W, 3] (ComfyUI IMAGE layout) or uint8 [T, H, W, 3].
All measures are on luma, 0-255 scale.

  sharpness   mean |Laplacian|, reported as a ratio to the reference when one is given
  fine_detail share of spectral energy above 1/4 Nyquist (every 4th frame), ratio to the reference
  psnr        mean per-frame luma PSNR vs the reference (dB)
  flicker     mean |f[t+1] - f[t]| minus the reference's (0 = as much frame-to-frame change as the reference)
  flash_p95   95th percentile of |mean luma[t+1] - mean luma[t]| (global brightness jumps)
  copy_drift  generated (all-frames) clip + hold map: mean |copy - first copy| inside each held group
"""
from __future__ import annotations

import json

import numpy as np


def luma(frames) -> np.ndarray:
    f = np.asarray(frames, dtype=np.float32)
    if f.max() <= 1.0 + 1e-6:
        f = f * 255.0
    return 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]


def laplacian_mean(g: np.ndarray) -> float:
    return float(np.abs(4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]).mean())


def high_freq_share(g: np.ndarray, cutoff: float = 0.125) -> float:
    spec = np.abs(np.fft.rfft2(g - g.mean())) ** 2
    fy = np.abs(np.fft.fftfreq(g.shape[0]))[:, None]
    fx = np.fft.rfftfreq(g.shape[1])[None, :]
    total = spec.sum()
    return float(spec[np.sqrt(fx ** 2 + fy ** 2) > cutoff].sum() / total) if total > 0 else 0.0


def _stats(y: np.ndarray) -> dict:
    return {
        "sharpness": float(np.mean([laplacian_mean(f) for f in y])),
        "fine_detail": float(np.mean([high_freq_share(f) for f in y[::4]])),
        "change": float(np.abs(np.diff(y, axis=0)).mean()) if len(y) > 1 else 0.0,
        "flash_p95": float(np.percentile(np.abs(np.diff(y.mean(axis=(1, 2)))), 95)) if len(y) > 1 else 0.0,
    }


def parse_holds(hold_map: str | dict | list | None) -> list[int] | None:
    if hold_map is None or hold_map == "":
        return None
    h = json.loads(hold_map) if isinstance(hold_map, str) else hold_map
    if isinstance(h, dict):
        h = h.get("holds", h.get("hold_map_used"))
    return [int(x) for x in h]


def copy_drift(all_frames, holds: list[int]) -> float:
    y = luma(all_frames)
    if len(y) != sum(holds):
        raise ValueError(f"all-frames clip has {len(y)} frames, the hold map needs {sum(holds)}")
    starts = np.cumsum([0] + holds[:-1])
    vals = [np.abs(y[s + c] - y[s]).mean() for s, h in zip(starts, holds) for c in range(1, h)]
    return float(np.mean(vals)) if vals else 0.0


def video_metrics(frames, reference=None, all_frames=None, hold_map=None) -> dict:
    y = luma(frames)
    out = {"frames": int(len(y)), **_stats(y)}
    if reference is not None:
        r = luma(reference)
        if r.shape != y.shape:
            raise ValueError(f"reference is {r.shape}, frames are {y.shape}; give the matching window")
        rs = _stats(r)
        mse = ((y - r) ** 2).mean(axis=(1, 2))
        out.update({
            "sharpness_vs_ref": out["sharpness"] / rs["sharpness"] if rs["sharpness"] else None,
            "fine_detail_vs_ref": out["fine_detail"] / rs["fine_detail"] if rs["fine_detail"] else None,
            "psnr": float(np.mean(10 * np.log10(255.0 ** 2 / np.maximum(mse, 1e-6)))),
            "flicker_vs_ref": out["change"] - rs["change"],
            "reference": rs,
        })
    holds = parse_holds(hold_map)
    if all_frames is not None and holds is not None:
        out["copy_drift"] = copy_drift(all_frames, holds)
    return out


def report_text(m: dict) -> str:
    keys = [("sharpness_vs_ref", "sharpness vs ref", "{:.3f}"), ("fine_detail_vs_ref", "fine detail vs ref", "{:.3f}"),
            ("psnr", "PSNR dB", "{:.2f}"), ("flicker_vs_ref", "flicker vs ref", "{:+.2f}"),
            ("flash_p95", "flash p95", "{:.2f}"), ("copy_drift", "held-copy drift", "{:.2f}")]
    lines = [f"{m['frames']} frames"]
    for k, label, fmt in keys:
        if m.get(k) is not None:
            lines.append(f"{label}: {fmt.format(m[k])}")
    return "\n".join(lines)
