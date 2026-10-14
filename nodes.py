import comfy.patcher_extension

from . import probe, tsam
from .wrapper import TSAMWrapper


class Krea2TSAMEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "clip": ("CLIP",),
            "text": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            "text_layers": ("INT", {"default": 36, "min": 1, "max": 64, "tooltip": "Number of text encoder layers averaged into the target."}),
        }}

    RETURN_TYPES = ("CONDITIONING", "TSAM_TARGET")
    FUNCTION = "encode"
    CATEGORY = "conditioning/krea2"

    def encode(self, clip, text, text_layers):
        tokens = clip.tokenize(text)
        clip.load_model(tokens)
        layers = clip.cond_stage_model.qwen3vl_4b.transformer.model.layers
        attn_probe = probe.AttentionProbe(text_layers)
        handles = attn_probe.install(layers)
        try:
            cond = clip.encode_from_tokens_scheduled(tokens)
        finally:
            for h in handles:
                h.remove()
        strip = attn_probe.attn_sum.shape[0] - cond[0][0].shape[1]
        target = tsam.build_text_target(attn_probe.attn_sum, attn_probe.count, strip).cpu()
        return (cond, target)


class Krea2TSAM:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("MODEL",),
            "target": ("TSAM_TARGET",),
            "alpha": ("FLOAT", {"default": 15.0, "min": 0.0, "max": 1000.0, "step": 0.1}),
            "gamma": ("FLOAT", {"default": 4.0, "min": 0.1, "max": 16.0, "step": 0.1}),
            "start_pct": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "end_pct": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            "iters_per_step": ("INT", {"default": 1, "min": 1, "max": 50}),
            "layer_start": ("INT", {"default": 9, "min": 0, "max": 63}),
            "layer_end": ("INT", {"default": 18, "min": 1, "max": 64}),
            "smooth": ("BOOLEAN", {"default": True}),
            "normalize_grad": ("BOOLEAN", {"default": False}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "model_patches/krea2"

    def patch(self, model, target, **cfg):
        m = model.clone()
        m.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, "krea2_tsam", TSAMWrapper(target, cfg))
        return (m,)


NODE_CLASS_MAPPINGS = {"Krea2TSAMEncode": Krea2TSAMEncode, "Krea2TSAM": Krea2TSAM}
NODE_DISPLAY_NAME_MAPPINGS = {"Krea2TSAMEncode": "Krea2 T-SAM Encode", "Krea2TSAM": "Krea2 T-SAM"}
