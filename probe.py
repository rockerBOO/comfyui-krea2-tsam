import torch

from comfy.ldm.flux.math import apply_rope


class AttentionProbe:
    def __init__(self, max_layers):
        self.max_layers = max_layers
        self.attn_sum = None
        self.count = 0

    def wrap(self, inner):
        def probed(q, k, v, heads, mask=None, **kwargs):
            if self.count < self.max_layers:
                self._add(q, k, mask)
            return inner(q, k, v, heads, mask=mask, **kwargs)
        return probed

    def _add(self, q, k, mask):
        k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
        logits = (q.float() @ k.float().transpose(-1, -2)) * q.shape[-1] ** -0.5
        if mask is not None:
            logits = logits + mask.float()
        p = logits.softmax(-1).mean(dim=1)[0]
        self.attn_sum = p if self.attn_sum is None else self.attn_sum + p
        self.count += 1

    def install(self, layers):
        def pre_hook(module, args, kwargs):
            kwargs["optimized_attention"] = self.wrap(kwargs["optimized_attention"])
            return args, kwargs
        return [layer.self_attn.register_forward_pre_hook(pre_hook, with_kwargs=True) for layer in layers]


class MapCollector:
    def __init__(self, txtlen, layer_lo, layer_hi, batch_index):
        self.txtlen = txtlen
        self.layer_lo = layer_lo
        self.layer_hi = layer_hi
        self.batch_index = batch_index
        self.maps = []

    def __call__(self, q, k, v, pe=None, attn_mask=None, extra_options=None):
        if not self.layer_lo <= extra_options["block_index"] < self.layer_hi:
            return {}
        img_end = extra_options["img_slice"][1] - sum(extra_options["reference_image_num_tokens"])
        b = self.batch_index
        qb, kb = q[b:b + 1], k[b:b + 1]
        if pe is not None:
            qb, kb = apply_rope(qb, kb, pe[b:b + 1] if pe.shape[0] > 1 else pe)
        kb = kb.repeat_interleave(qb.shape[1] // kb.shape[1], dim=1)
        logits = (qb[:, :, self.txtlen:img_end] @ kb[:, :, :self.txtlen].transpose(-1, -2)) * qb.shape[-1] ** -0.5
        self.maps.append(logits.float().softmax(-1).mean(dim=1)[0])
        return {}

    def mean(self):
        return torch.stack(self.maps).mean(0)
