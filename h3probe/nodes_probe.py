"""H3 Probe node: wraps a MODEL so every sampling run it takes part in is recorded to output/h3probe/."""
import hashlib
import os

from . import probe as P

LORA_NONE = "none"


def _lora_list():
    import folder_paths
    return [LORA_NONE] + folder_paths.get_filename_list("loras")


def _file_sha(path, n=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(n))
    return h.hexdigest()[:16] + "(first MiB)"


class H3Probe:
    DESCRIPTION = (
        "Records where LoRAs and each DiT block act while H3 samples: per sampling step, block (0-49), module "
        "(qkv / out / fc1 / fc2) and token cell (latent frame x a coarse spatial grid), target video rows only. "
        "Pick up to two LoRA files to attribute (they are NOT applied here; load them as usual, this only "
        "recomputes their contribution s*B*A*x from each layer's input). Every sampler run that uses this model "
        "writes output/h3probe/<name>_NNNNN.npz plus a .json manifest. Reductions only, so files stay small. "
        "Wire the MODEL output to the guider in place of the input model.")

    @classmethod
    def INPUT_TYPES(cls):
        loras = _lora_list()
        return {"required": {
            "model": ("MODEL",),
            "name": ("STRING", {"default": "probe"}),
            "lora_1": (loras, {"default": LORA_NONE}),
            "lora_1_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            "lora_2": (loras, {"default": LORA_NONE}),
            "lora_2_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            "grid_h": ("INT", {"default": 3, "min": 1, "max": 24}),
            "grid_w": ("INT", {"default": 4, "min": 1, "max": 48}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "wrap"
    CATEGORY = "H3 Probe"

    def wrap(self, model, name, lora_1, lora_1_strength, lora_2, lora_2_strength, grid_h, grid_w):
        import comfy.patcher_extension as px
        import comfy.utils
        import folder_paths

        loras, lora_info = {}, []
        for i, (ln, st) in enumerate(((lora_1, lora_1_strength), (lora_2, lora_2_strength)), 1):
            if ln == LORA_NONE:
                continue
            path = folder_paths.get_full_path("loras", ln)
            tag = f"{i}_" + os.path.splitext(os.path.basename(ln))[0][:40]
            loras[tag] = P.parse_lora(comfy.utils.load_torch_file(path, safe_load=True), float(st))
            lora_info.append({"tag": tag, "file": ln, "strength": float(st), "modules": len(loras[tag]), "sha": _file_sha(path)})
        out_dir = os.path.join(folder_paths.get_output_directory(), "h3probe")
        os.makedirs(out_dir, exist_ok=True)
        state = {"rec": None}

        def outer(executor, *args, **kwargs):
            guider = executor.class_obj
            dm = guider.model_patcher.model.diffusion_model
            rec = P.Recorder(loras=loras, grid=(int(grid_h), int(grid_w)))
            state["rec"] = rec
            rec.attach(dm)
            try:
                return executor(*args, **kwargs)
            finally:
                rec.detach()
                state["rec"] = None
                if rec.calls:
                    k = 1
                    while os.path.exists(os.path.join(out_dir, f"{name}_{k:05d}.npz")):
                        k += 1
                    sig = args[3] if len(args) > 3 else kwargs.get("sigmas")
                    P.save(os.path.join(out_dir, f"{name}_{k:05d}.npz"), rec, {
                        "name": name, "loras": lora_info,
                        "sigmas": [float(s) for s in sig] if sig is not None else None,
                        "note": "energies are sums over each cell; sqrt(lora/base) is the relative push"})

        def diffusion(executor, *args, **kwargs):
            rec = state["rec"]
            to = args[3] if len(args) > 3 else kwargs.get("transformer_options", {})
            if rec is not None:
                rec.begin_call(to)
            try:
                return executor(*args, **kwargs)
            finally:
                if rec is not None:
                    rec.end_call()

        m = model.clone()
        m.add_wrapper_with_key(px.WrappersMP.OUTER_SAMPLE, "h3probe", outer)
        m.add_wrapper_with_key(px.WrappersMP.DIFFUSION_MODEL, "h3probe", diffusion)
        return (m,)


NODE_CLASS_MAPPINGS = {"H3Probe": H3Probe}
NODE_DISPLAY_NAME_MAPPINGS = {"H3Probe": "H3 Probe (record LoRA / block activity)"}
