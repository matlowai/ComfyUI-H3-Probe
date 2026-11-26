"""H3 Probe core: record where LoRAs (and each block) act, per sampling step, block, module and token cell.

Reductions only (energies summed over a coarse token grid), so a probe file stays small. Everything here is
plain torch; the ComfyUI wiring lives in nodes_probe.py.

Per model call (one sampling step for a guider without CFG), for the TARGET VIDEO rows only:
  lora[k]  [blocks, modules, T, gy, gx]  sum over the cell of ||s_k B_k A_k x||^2  (each attributed LoRA k)
  base     [blocks, modules, T, gy, gx]  sum over the cell of ||module output||^2 (LoRAs included, as run)
  resid    [blocks, T, gy, gx]           sum over the cell of ||block output - block input||^2
T is the latent frame index, (gy, gx) a coarse spatial grid. sqrt(lora / base) is the LoRA's relative push.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import numpy as np
import torch

MODULES = ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")
_MUSUBI = re.compile(r"^lora_unet_blocks_(\d+)_(attn_qkv_proj|attn_out_proj|mlp_fc1|mlp_fc2)\.(lora_down|lora_up|alpha)(?:\.weight)?$")
_COMFY = re.compile(r"^(?:diffusion_model\.|transformer\.)?blocks\.(\d+)\.(attn\.qkv_proj|attn\.out_proj|mlp\.fc1|mlp\.fc2)\.(lora_A|lora_B|lora_down|lora_up|alpha)(?:\.weight)?$")


def parse_lora(state: dict, strength: float = 1.0) -> dict:
    """{(block, module): (A [r, in], G = s^2 B^T B [r, r])} from a musubi or ComfyUI-format LoRA state dict."""
    parts: dict = {}
    for k, v in state.items():
        m = _MUSUBI.match(k)
        if m:
            b, mod, kind = int(m.group(1)), m.group(2).replace("attn_", "attn.").replace("mlp_", "mlp."), m.group(3)
        else:
            m = _COMFY.match(k)
            if not m:
                continue
            b, mod, kind = int(m.group(1)), m.group(2), m.group(3)
        kind = {"lora_down": "A", "lora_A": "A", "lora_up": "B", "lora_B": "B", "alpha": "alpha"}[kind]
        parts.setdefault((b, mod), {})[kind] = v
    out = {}
    for key, p in parts.items():
        if "A" not in p or "B" not in p:
            continue
        A = p["A"].float()
        B = p["B"].float()
        r = A.shape[0]
        alpha = float(p["alpha"]) if "alpha" in p else float(r)
        s = strength * alpha / r
        out[key] = (A, (B.T @ B) * (s * s))
    return out


def video_rows(layout):
    """(start, stop, T, h, w) of the target video rows (frame-major, h x w patch rows per frame)."""
    for a, b, kind in layout.segments:
        if kind == "video":
            _, t, lh, lw, _ = layout.signature
            h, w = lh // 2, lw // 2
            if (b - a) != t * h * w:
                raise RuntimeError(f"H3 Probe: video segment has {b - a} rows, expected {t}x{h}x{w}")
            return a, b, t, h, w
    raise RuntimeError("H3 Probe: no video segment in the layout")


def pool(e: torch.Tensor, t: int, h: int, w: int, gy: int, gx: int) -> torch.Tensor:
    """Per-row energies [t*h*w] -> cell sums [t, gy, gx] (cells as even as integer splits allow)."""
    e = e.reshape(t, h, w)
    ys = torch.linspace(0, h, gy + 1).round().long().tolist()
    xs = torch.linspace(0, w, gx + 1).round().long().tolist()
    return torch.stack([torch.stack([e[:, ys[i]:ys[i + 1], xs[j]:xs[j + 1]].sum(dim=(1, 2)) for j in range(gx)], -1)
                        for i in range(gy)], -2)


@dataclass
class Recorder:
    loras: dict                     # name -> parse_lora(...)
    grid: tuple = (3, 4)
    n_blocks: int = 50
    calls: list = field(default_factory=list)      # one dict of arrays per model call
    meta: list = field(default_factory=list)       # sigma, layout info per call
    _cur: dict | None = None
    _to: dict | None = None
    _hooks: list = field(default_factory=list)
    _dev_cache: dict = field(default_factory=dict)

    # ---- call boundaries (driven by the diffusion-model wrapper) ----
    def begin_call(self, transformer_options: dict):
        self._to = transformer_options
        self._cur = None

    def _ensure(self):
        if self._cur is not None:
            return self._cur
        layout = self._to.get("minimax_h3_layout") if self._to else None
        if layout is None:
            return None
        a, b, t, h, w = video_rows(layout)
        gy, gx = self.grid
        shape = (self.n_blocks, len(MODULES), t, gy, gx)
        self._cur = {"rows": (a, b, t, h, w),
                     "base": np.zeros(shape, np.float32), "resid": np.zeros((self.n_blocks, t, gy, gx), np.float32),
                     **{f"lora:{n}": np.zeros(shape, np.float32) for n in self.loras}}
        sig = self._to.get("sigmas")
        frame_t = layout.position_ids[a:b:h * w, 0].tolist()
        self.meta.append({"sigma": float(sig.reshape(-1)[0]) if sig is not None else None, "T": t, "h": h, "w": w,
                          "frame_t": frame_t})
        self.calls.append(self._cur)
        return self._cur

    def end_call(self):
        self._cur = None
        self._to = None

    # ---- hooks ----
    def _lora_dev(self, name, key, device, dtype):
        ck = (name, key, device, dtype)
        if ck not in self._dev_cache:
            A, G = self.loras[name][key]
            self._dev_cache[ck] = (A.to(device=device, dtype=dtype), G.to(device=device, dtype=torch.float32))
        return self._dev_cache[ck]

    def module_hook(self, block: int, mod_i: int):
        key = (block, MODULES[mod_i])

        def hook(module, inputs, output):
            cur = self._ensure()
            if cur is None:
                return
            a, b, t, h, w = cur["rows"]
            x = inputs[0]
            x = x[0] if x.dim() == 3 else x
            y = output[0] if output.dim() == 3 else output
            xv, yv = x[a:b], y[a:b]
            gy, gx = self.grid
            with torch.no_grad():
                e = torch.cat([yv[i:i + 4096].float().pow(2).sum(-1) for i in range(0, yv.shape[0], 4096)])
                cur["base"][block, mod_i] += pool(e, t, h, w, gy, gx).cpu().numpy()
                for name, lo in self.loras.items():
                    if key not in lo:
                        continue
                    A, G = self._lora_dev(name, key, xv.device, xv.dtype)
                    z = (xv @ A.T).float()                           # [N, r]
                    ee = ((z @ G) * z).sum(-1)                        # ||s B A x||^2 per row
                    cur[f"lora:{name}"][block, mod_i] += pool(ee, t, h, w, gy, gx).cpu().numpy()
        return hook

    def block_hook(self, block: int):
        def pre(module, args, kwargs):
            module._h3probe_in = args[0]

        def post(module, args, kwargs, output):
            cur = self._ensure()
            xin = getattr(module, "_h3probe_in", None)
            module._h3probe_in = None
            if cur is None or xin is None:
                return
            a, b, t, h, w = cur["rows"]
            x0 = xin[0] if xin.dim() == 3 else xin
            x1 = output[0] if output.dim() == 3 else output
            with torch.no_grad():
                d = x1[a:b] - x0[a:b]
                e = torch.cat([d[i:i + 4096].float().pow(2).sum(-1) for i in range(0, d.shape[0], 4096)])
                cur["resid"][block] += pool(e, t, h, w, *self.grid).cpu().numpy()
        return pre, post

    def attach(self, diffusion_model):
        self.detach()
        blocks = diffusion_model.blocks
        self.n_blocks = len(blocks)
        for i, blk in enumerate(blocks):
            for j, name in enumerate(MODULES):
                sub = blk.get_submodule(name)
                self._hooks.append(sub.register_forward_hook(self.module_hook(i, j)))
            pre, post = self.block_hook(i)
            self._hooks.append(blk.register_forward_pre_hook(pre, with_kwargs=True))
            self._hooks.append(blk.register_forward_hook(post, with_kwargs=True))

    def detach(self):
        for hdl in self._hooks:
            hdl.remove()
        self._hooks = []
        self._dev_cache = {}

    def arrays(self) -> dict:
        out = {}
        if not self.calls:
            return out
        for k in self.calls[0]:
            if k == "rows":
                continue
            out[k] = np.stack([c[k] for c in self.calls])
        return out


def save(path_npz: str, rec: Recorder, manifest: dict):
    arrs = rec.arrays()
    np.savez_compressed(path_npz, **{k.replace(":", "__"): v for k, v in arrs.items()})
    manifest = dict(manifest, modules=list(MODULES), grid=list(rec.grid), calls=rec.meta,
                    arrays={k: list(v.shape) for k, v in arrs.items()})
    with open(path_npz[:-4] + ".json", "w") as f:
        json.dump(manifest, f, indent=1)
