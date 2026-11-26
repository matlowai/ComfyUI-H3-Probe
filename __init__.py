"""ComfyUI-H3-Probe: look inside MiniMax-H3 runs (metrics, probes, overlays). See docs/DESIGN.md."""
from .h3probe.nodes_metrics import NODE_CLASS_MAPPINGS as _M, NODE_DISPLAY_NAME_MAPPINGS as _MD
from .h3probe.nodes_probe import NODE_CLASS_MAPPINGS as _P, NODE_DISPLAY_NAME_MAPPINGS as _PD

NODE_CLASS_MAPPINGS = {**_M, **_P}
NODE_DISPLAY_NAME_MAPPINGS = {**_MD, **_PD}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
