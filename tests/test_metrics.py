"""python tests/test_metrics.py  (no GPU, no ComfyUI)"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from h3probe import metrics as M  # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(("  PASS  " if ok else "  FAIL  ") + name + ("  " + detail if detail else ""))
    if not ok:
        FAILS.append(name)


rng = np.random.default_rng(0)
ref = rng.random((12, 64, 96, 3)).astype(np.float32)
m = M.video_metrics(ref, reference=ref)
check("identical clip: ratios 1, flicker 0", abs(m["sharpness_vs_ref"] - 1) < 1e-6 and abs(m["fine_detail_vs_ref"] - 1) < 1e-6
      and abs(m["flicker_vs_ref"]) < 1e-6)

blur = (ref + np.roll(ref, 1, 1) + np.roll(ref, -1, 1) + np.roll(ref, 1, 2) + np.roll(ref, -1, 2)) / 5
m = M.video_metrics(blur, reference=ref)
check("blurred clip is softer", m["sharpness_vs_ref"] < 0.8 and m["fine_detail_vs_ref"] < 0.9,
      f"{m['sharpness_vs_ref']:.3f} {m['fine_detail_vs_ref']:.3f}")
check("blurred clip has finite PSNR", np.isfinite(m["psnr"]))

holds = [1, 2, 4, 1]
world = rng.random((4, 32, 32, 3)).astype(np.float32)
still = np.repeat(world, holds, axis=0)
check("exact holds: drift 0", M.copy_drift(still, holds) == 0.0)
moving = still.copy()
moving[2] += 0.1
check("moved copy: drift > 0", M.copy_drift(np.clip(moving, 0, 1), holds) > 0)
check("hold map from MAINodes JSON", M.parse_holds(json.dumps({"holds": holds, "world_len": 4})) == holds)
try:
    M.copy_drift(still[:-1], holds)
    check("length mismatch raises", False)
except ValueError:
    check("length mismatch raises", True)

flash = ref.copy()
flash[6] = np.clip(flash[6] + 0.4, 0, 1)
check("flash raises flash_p95", M.video_metrics(flash)["flash_p95"] > M.video_metrics(ref)["flash_p95"])

print("%d failure(s)" % len(FAILS))
sys.exit(1 if FAILS else 0)
