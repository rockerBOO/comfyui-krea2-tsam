import pytest
import torch

import comfy.model_management as mm
import comfy.ops
import comfy.patcher_extension as pe
from comfy.ldm.krea2.model import SingleStreamDiT
from krea2_tsam import probe, tsam, wrapper

SAMPLE_SIGMAS = torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0])


def tiny_dit():
    torch.manual_seed(0)
    m = SingleStreamDiT(features=64, tdim=32, txtdim=16, heads=4, kvheads=2, multiplier=2, layers=4, patch=2,
                        channels=4, txtlayers=3, txtheads=2, txtkvheads=2, operations=comfy.ops.disable_weight_init)
    for p in m.parameters():
        torch.nn.init.normal_(p, std=0.05)
        p.requires_grad_(False)
    return m


def target(s=5):
    t = torch.tril(torch.rand(s, s, generator=torch.Generator().manual_seed(1)))
    return t / t.sum(-1, keepdim=True).clamp_min(1e-6)


def cfg(**kw):
    base = dict(alpha=5.0, gamma=2.0, start_pct=0.0, end_pct=0.5, iters_per_step=1, layer_start=1, layer_end=3, smooth=False, normalize_grad=False)
    return {**base, **kw}


def run(m, c, sigma, tgt=None):
    opts = {"cond_or_uncond": [0], "sigmas": torch.tensor([sigma]), "sample_sigmas": SAMPLE_SIGMAS}
    pe.add_wrapper_with_key(pe.WrappersMP.DIFFUSION_MODEL, "tsam", wrapper.TSAMWrapper(target() if tgt is None else tgt, c), opts)
    x = torch.randn(1, 4, 8, 8, generator=torch.Generator().manual_seed(2))
    ctx = torch.randn(1, 5, 48, generator=torch.Generator().manual_seed(3))
    return m(x, torch.tensor([sigma]), ctx, transformer_options=opts)


def test_active_step_depends_on_alpha():
    m = tiny_dit()
    assert not torch.allclose(run(m, cfg(alpha=5.0), 1.0), run(m, cfg(alpha=0.0), 1.0))


def test_inactive_step_is_untouched():
    m = tiny_dit()
    assert torch.allclose(run(m, cfg(alpha=5.0), 0.25), run(m, cfg(alpha=0.0), 0.25))


def test_training_flag_restored_and_output_finite():
    out = run(tiny_dit(), cfg(), 1.0)
    assert mm.in_training is False
    assert torch.isfinite(out).all() and out.shape == (1, 4, 8, 8)


def test_target_size_mismatch_raises():
    with pytest.raises(ValueError, match="Krea2 T-SAM Encode"):
        run(tiny_dit(), cfg(), 1.0, tgt=target(4))


def test_blocks_restored_after_step():
    m = tiny_dit()
    run(m, cfg(), 1.0)
    assert all("forward" not in b.__dict__ for b in m.blocks)


def test_checkpointed_gradient_matches_plain_autograd():
    m = tiny_dit()
    x = torch.randn(1, 4, 8, 8, generator=torch.Generator().manual_seed(2))
    ctx = torch.randn(1, 5, 48, generator=torch.Generator().manual_seed(3))
    tgt = target()
    lo, hi = 1, 3

    mm.in_training = True
    try:
        xg = x.clone().requires_grad_(True)
        col = probe.MapCollector(5, lo, hi, 0)
        opts = {"patches": {"attn1_patch": [col]}}
        m(xg, torch.tensor([1.0]), ctx, transformer_options=opts)
        plain, = torch.autograd.grad(tsam.tsam_loss(tgt, col.mean(), 2.0), xg)

        xg = x.clone().requires_grad_(True)
        maps = []
        w = wrapper.TSAMWrapper(tgt, cfg())
        undo = w._install(m, lo, hi, 5, 0, maps)
        try:
            m(xg, torch.tensor([1.0]), ctx)
        except wrapper._Stop:
            pass
        finally:
            undo()
        ckpt, = torch.autograd.grad(tsam.tsam_loss(tgt, torch.stack(maps).mean(0), 2.0), xg)
    finally:
        mm.in_training = False
    assert torch.allclose(plain, ckpt, atol=1e-5, rtol=1e-4)
