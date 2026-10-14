import contextlib

import torch

import comfy.ldm.modules.attention
import comfy.model_management
import comfy.ops

from . import probe, tsam


class BlockCheckpoint(torch.autograd.Function):
    """Per-block checkpoint that also returns the block's attention map; same idea as OffloadCheckpointFunction in nodes_train.py."""

    @staticmethod
    def forward(ctx, x, run):
        ctx.save_for_backward(x)
        ctx.run = run
        with torch.no_grad():
            return run(x)

    @staticmethod
    def backward(ctx, grad_out, grad_map):
        x, = ctx.saved_tensors
        run, ctx.run = ctx.run, None
        with torch.enable_grad():
            xd = x.detach().requires_grad_(True)
            out, attn_map = run(xd)
            pairs = [(t, g) for t, g in ((out, grad_out), (attn_map, grad_map)) if t is not None and g is not None]
            torch.autograd.backward([t for t, _ in pairs], [g for _, g in pairs])
        return xd.grad, None


class _Stop(Exception):
    pass


def _differentiable_attention(func, *args, **kwargs):
    return comfy.ldm.modules.attention.attention_pytorch(*args, **kwargs)


def _owned(t):
    return t.clone() if torch.is_tensor(t) else t


@contextlib.contextmanager
def _linear_clones_inference_weights():
    """Weights are inference tensors that autograd refuses to save, and offloaded ones can be reused once the layer returns, so backward gets its own copies."""
    linear = torch.nn.functional.linear
    quant_apply = comfy.ops.QuantLinearFunc.apply

    torch.nn.functional.linear = lambda x, weight, bias=None: linear(x, _owned(weight), _owned(bias))
    comfy.ops.QuantLinearFunc.apply = lambda input, weight, bias, *rest: quant_apply(input, _owned(weight), _owned(bias), *rest)
    try:
        yield
    finally:
        torch.nn.functional.linear = linear
        del comfy.ops.QuantLinearFunc.apply


class TSAMWrapper:
    def __init__(self, target, cfg):
        self.target = target
        self.cfg = cfg

    def _active(self, transformer_options):
        sample_sigmas = transformer_options["sample_sigmas"]
        steps = len(sample_sigmas) - 1
        step = int((sample_sigmas - transformer_options["sigmas"][0]).abs().argmin())
        return self.cfg["start_pct"] * steps <= step < self.cfg["end_pct"] * steps

    def _loss(self, a, h, w):
        target = self.target.to(a.device)
        total = 0
        for row in a:
            if self.cfg["smooth"]:
                row = tsam.smooth_maps(row, h, w)
            total = total + tsam.tsam_loss(target, row, self.cfg["gamma"])
        return total

    def _install(self, dit, lo, hi, txtlen, rows, maps):
        for idx, block in enumerate(dit.blocks[:hi]):
            block.forward = self._checkpointed(block.forward, idx, lo, hi, txtlen, rows, maps)

        def undo():
            for block in dit.blocks[:hi]:
                del block.forward
        return undo

    @staticmethod
    def _checkpointed(forward, idx, lo, hi, txtlen, rows, maps):
        def run_block(x, vec, freqs, mask=None, timestep_zero_index=None, transformer_options={}):
            def run(xi):
                collector = probe.MapCollector(txtlen, lo, hi, rows)
                patches = {**transformer_options.get("patches", {}), "attn1_patch": [collector]}
                out = forward(xi, vec, freqs, mask, timestep_zero_index=timestep_zero_index, transformer_options={**transformer_options, "patches": patches, "optimized_attention_override": _differentiable_attention})
                return out, (collector.maps[0] if collector.maps else None)
            out, attn_map = BlockCheckpoint.apply(x, run)
            if attn_map is not None:
                maps.append(attn_map)
            if idx == hi - 1:
                raise _Stop
            return out
        return run_block

    def __call__(self, executor, *args, **kwargs):
        x, timesteps, context, attention_mask, ref_latents, transformer_options = args
        cond_or_uncond = transformer_options["cond_or_uncond"]
        per_chunk = x.shape[0] // len(cond_or_uncond)
        rows = [chunk * per_chunk + i for chunk, c in enumerate(cond_or_uncond) if c == 0 for i in range(per_chunk)]
        if not rows or not self._active(transformer_options):
            return executor(*args, **kwargs)

        if self.target.shape[0] != context.shape[1]:
            raise ValueError(f"T-SAM target has {self.target.shape[0]} tokens but the conditioning has {context.shape[1]}; encode the same prompt with Krea2 T-SAM Encode.")

        cfg = self.cfg
        dit = executor.class_obj
        lo, hi = cfg["layer_start"], min(cfg["layer_end"], len(dit.blocks))
        sigma = transformer_options["sigmas"][rows].to(x.dtype).view(-1, *[1] * (x.ndim - 1))
        x_new = x
        with torch.inference_mode(False), _linear_clones_inference_weights():
            comfy.model_management.in_training = True
            try:
                timesteps_g, context_g = timesteps.clone(), context.clone()
                mask_g = None if attention_mask is None else attention_mask.clone()
                refs_g = None if ref_latents is None else [r.clone() for r in ref_latents]
                for _ in range(cfg["iters_per_step"]):
                    xg = x_new.detach().clone().requires_grad_(True)
                    maps = []
                    undo = self._install(dit, lo, hi, context.shape[1], rows, maps)
                    try:
                        executor.original(xg, timesteps_g, context_g, mask_g, refs_g, transformer_options, **kwargs)
                    except _Stop:
                        pass
                    finally:
                        undo()
                    loss = self._loss(torch.stack(maps).mean(0), xg.shape[-2] // dit.patch, xg.shape[-1] // dit.patch)
                    grad, = torch.autograd.grad(loss, xg)
                    comfy.model_management.synchronize()
                    if cfg["normalize_grad"]:
                        grad = grad / grad.flatten(1).norm(dim=1).clamp_min(1e-8).view(-1, *[1] * (grad.ndim - 1))
                    x_new = (xg - cfg["alpha"] * grad).detach()
            finally:
                comfy.model_management.in_training = False
        out = executor(x_new, timesteps, context, attention_mask, ref_latents, transformer_options, **kwargs).clone()
        out[rows] += ((x[rows] - x_new[rows]) / sigma).to(out.dtype)
        return out
