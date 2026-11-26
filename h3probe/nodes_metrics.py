"""H3 Video Metrics node."""
import json

from . import metrics as M


class H3ProbeVideoMetrics:
    DESCRIPTION = (
        "Sharpness, fine detail, fidelity (PSNR), flicker and brightness-flash spikes of a clip, against a "
        "reference clip of the same length when one is wired (e.g. the source window of a de-rope pass). Wire "
        "the all-frames output and its hold map to also get held-copy drift (0 = the copies of a held frame "
        "came out identical). Report as text and JSON.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE", {"tooltip": "the clip to measure (for de-rope: the recovered, real-time frames)"}),
        }, "optional": {
            "reference": ("IMAGE", {"tooltip": "same number of frames, e.g. the source window"}),
            "all_frames": ("IMAGE", {"tooltip": "the generated clip including held copies"}),
            "hold_map": ("STRING", {"forceInput": True, "tooltip": "hold map of all_frames (H3 Time Smear's hold_map_used)"}),
        }}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("report", "json")
    FUNCTION = "measure"
    CATEGORY = "H3 Probe"
    OUTPUT_NODE = True

    def measure(self, images, reference=None, all_frames=None, hold_map=None):
        m = M.video_metrics(images.cpu().numpy(),
                            None if reference is None else reference.cpu().numpy(),
                            None if all_frames is None else all_frames.cpu().numpy(), hold_map)
        text = M.report_text(m)
        return {"ui": {"text": [text]}, "result": (text, json.dumps(m))}


NODE_CLASS_MAPPINGS = {"H3ProbeVideoMetrics": H3ProbeVideoMetrics}
NODE_DISPLAY_NAME_MAPPINGS = {"H3ProbeVideoMetrics": "H3 Video Metrics"}
