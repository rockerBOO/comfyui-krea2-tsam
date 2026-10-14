import torch

import comfy.model_management

from . import probe, tsam


class TSAMWrapper:
    def __init__(self, target, cfg):
        self.target = target
        self.cfg = cfg

    def _active(self, transformer_options):
        sample_sigmas = transformer_options["sample_sigmas"]
        steps = len(sample_sigmas) - 1
        step = int((sample_sigmas - transformer_options["sigmas"][0]).abs().argmin())
        return self.cfg["start_pct"] * steps <= step < self.cfg["end_pct"] * steps

    def __call__(self, executor, *args, **kwargs):
        x, timesteps, context, attention_mask, ref_latents, transformer_options = args
        cond_idx = [i for i, c in enumerate(transformer_options["cond_or_uncond"]) if c == 0]
        if not cond_idx or not self._active(transformer_options):
            return executor(*args, **kwargs)

        if self.target.shape[0] != context.shape[1]:
            raise ValueError(f"T-SAM target has {self.target.shape[0]} tokens but the conditioning has {context.shape[1]}; encode the same prompt with Krea2 T-SAM Encode.")

        cfg = self.cfg
        b = cond_idx[0]
        dit = executor.class_obj
        lo, hi = cfg["layer_start"], min(cfg["layer_end"], len(dit.blocks))
        sigma = transformer_options["sigmas"][b].to(x.dtype)
        x_new = x
        with torch.inference_mode(False):
            comfy.model_management.in_training = True
            try:
                for _ in range(cfg["iters_per_step"]):
                    xg = x_new.detach().clone().requires_grad_(True)
                    collector = probe.MapCollector(context.shape[1], lo, hi, b)
                    patches = {k: list(v) for k, v in transformer_options.get("patches", {}).items()}
                    patches.setdefault("attn1_patch", []).append(collector)
                    opts = {**transformer_options, "patches": patches}
                    executor.original(xg, timesteps, context, attention_mask, ref_latents, opts, **kwargs)
                    a = collector.mean()
                    if cfg["smooth"]:
                        a = tsam.smooth_maps(a, xg.shape[-2] // dit.patch, xg.shape[-1] // dit.patch)
                    loss = tsam.tsam_loss(self.target.to(a.device), a, cfg["gamma"])
                    grad, = torch.autograd.grad(loss, xg)
                    if cfg["normalize_grad"]:
                        grad = grad / grad.norm().clamp_min(1e-8)
                    x_new = (xg - cfg["alpha"] * grad).detach()
            finally:
                comfy.model_management.in_training = False
        out = executor(x_new, timesteps, context, attention_mask, ref_latents, transformer_options, **kwargs).clone()
        out[b] += ((x[b] - x_new[b]) / sigma).to(out.dtype)
        return out
