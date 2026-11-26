"""python tests/test_probe.py  (synthetic: a tiny 2-block stand-in for the H3 DiT, no ComfyUI, CPU)"""
import os
import sys
import tempfile
import types

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from h3probe import probe as P  # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(("  PASS  " if ok else "  FAIL  ") + name + ("  " + detail if detail else ""))
    if not ok:
        FAILS.append(name)


torch.manual_seed(0)
D, R = 16, 4


class Attn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv_proj = torch.nn.Linear(D, 3 * D)
        self.out_proj = torch.nn.Linear(3 * D, D)


class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(D, 2 * D)
        self.fc2 = torch.nn.Linear(2 * D, D)


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn, self.mlp = Attn(), MLP()

    def forward(self, x):
        h = x + self.attn.out_proj(self.attn.qkv_proj(x))
        return h + self.mlp.fc2(self.mlp.fc1(h))


model = types.SimpleNamespace(blocks=torch.nn.ModuleList([Block(), Block()]))
T, LH, LW, TEXT, AUD = 3, 4, 8, 5, 6         # latent 4x8 -> 2x4 patch rows per frame
h, w = LH // 2, LW // 2
nv = T * h * w
layout = types.SimpleNamespace(segments=[(0, TEXT, "text"), (TEXT, TEXT + AUD, "audio"), (TEXT + AUD, TEXT + AUD + nv, "video")],
                               signature=(TEXT, T, LH, LW, AUD // 2),
                               position_ids=torch.arange(TEXT + AUD + nv, dtype=torch.float64)[:, None].repeat(1, 3))
S = TEXT + AUD + nv

lora_sd = {}
for b in range(2):
    for mod, (din, dout) in {"attn_qkv_proj": (D, 3 * D), "attn_out_proj": (3 * D, D), "mlp_fc1": (D, 2 * D), "mlp_fc2": (2 * D, D)}.items():
        lora_sd[f"lora_unet_blocks_{b}_{mod}.lora_down.weight"] = torch.randn(R, din) * 0.1
        lora_sd[f"lora_unet_blocks_{b}_{mod}.lora_up.weight"] = torch.randn(dout, R) * 0.1
        lora_sd[f"lora_unet_blocks_{b}_{mod}.alpha"] = torch.tensor(float(R) / 2)
lo = P.parse_lora(lora_sd)
check("parsed 8 modules", len(lo) == 8)

rec = P.Recorder(loras={"runf": lo}, grid=(1, 2))
rec.attach(model)
x = torch.randn(S, D)
rec.begin_call({"minimax_h3_layout": layout, "sigmas": torch.tensor([0.8])})
out = x
for blk in model.blocks:
    out = blk(out)
rec.end_call()
arr = rec.arrays()
check("one call recorded", arr["base"].shape == (1, 2, 4, T, 1, 2), str(arr["base"].shape))

# direct recomputation for block 1 fc1 (input = h after attention of block 1)
blk0, blk1 = model.blocks
with torch.no_grad():
    x1 = blk0(x)
    hh = x1 + blk1.attn.out_proj(blk1.attn.qkv_proj(x1))
A = lora_sd["lora_unet_blocks_1_mlp_fc1.lora_down.weight"]
B = lora_sd["lora_unet_blocks_1_mlp_fc1.lora_up.weight"]
s = (R / 2) / R
contrib = (hh[TEXT + AUD:] @ A.T @ B.T) * s
e = contrib.pow(2).sum(-1).reshape(T, h, w)
want = torch.stack([e[:, :, :2].sum(dim=(1, 2)), e[:, :, 2:].sum(dim=(1, 2))], -1).numpy()
got = arr["lora__runf" if "lora__runf" in arr else "lora:runf"][0, 1, 2, :, 0, :]
check("LoRA energy matches ||s B A x||^2 per cell", np.allclose(got, want, rtol=1e-4, atol=1e-6), f"max err {np.abs(got - want).max():.2e}")
with torch.no_grad():
    d = (blk1(x1) - x1)[TEXT + AUD:].pow(2).sum(-1).reshape(T, h, w)
wantr = torch.stack([d[:, :, :2].sum(dim=(1, 2)), d[:, :, 2:].sum(dim=(1, 2))], -1).numpy()
check("block residual energy per cell", np.allclose(arr["resid"][0, 1, :, 0, :], wantr, rtol=1e-4))
check("text/audio rows excluded", rec.meta[0]["T"] == T and abs(rec.meta[0]["sigma"] - 0.8) < 1e-6)

rec.detach()
n_hooks = sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for blk in model.blocks for m in blk.modules())
check("detach removes every hook", n_hooks == 0)

with tempfile.TemporaryDirectory() as d:
    P.save(os.path.join(d, "p.npz"), rec, {"note": "test"})
    z = np.load(os.path.join(d, "p.npz"))
    check("saved arrays reload", set(z.files) == {"base", "resid", "lora__runf"})
print("%d failure(s)" % len(FAILS))
sys.exit(1 if FAILS else 0)
