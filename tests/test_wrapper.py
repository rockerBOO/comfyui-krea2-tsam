import pytest
import torch

import comfy.model_management as mm
import comfy.ops
import comfy.patcher_extension as pe
from comfy.ldm.krea2.model import SingleStreamDiT
from krea2_tsam import probe, tsam, wrapper

_ORIG_LINEAR = torch.nn.functional.linear
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
        col = probe.MapCollector(5, lo, hi, [0])
        opts = {"patches": {"attn1_patch": [col]}}
        m(xg, torch.tensor([1.0]), ctx, transformer_options=opts)
        plain, = torch.autograd.grad(wrapper.TSAMWrapper(tgt, cfg())._loss(col.mean(), 4, 4), xg)

        xg = x.clone().requires_grad_(True)
        maps = []
        w = wrapper.TSAMWrapper(tgt, cfg())
        undo = w._install(m, lo, hi, 5, [0], maps)
        try:
            m(xg, torch.tensor([1.0]), ctx)
        except wrapper._Stop:
            pass
        finally:
            undo()
        ckpt, = torch.autograd.grad(w._loss(torch.stack(maps).mean(0), 4, 4), xg)
    finally:
        mm.in_training = False
    assert torch.allclose(plain, ckpt, atol=1e-5, rtol=1e-4)


def test_inference_mode_weights_and_inputs():
    with torch.inference_mode():
        m = tiny_dit()
        for p in m.parameters():
            assert p.is_inference()
        t = target()
        opts = {"cond_or_uncond": [0], "sigmas": torch.tensor([1.0]), "sample_sigmas": SAMPLE_SIGMAS}
        pe.add_wrapper_with_key(pe.WrappersMP.DIFFUSION_MODEL, "tsam", wrapper.TSAMWrapper(t, cfg()), opts)
        x = torch.randn(1, 4, 8, 8)
        ctx = torch.randn(1, 5, 48)
        out = m(x, torch.tensor([1.0]), ctx, transformer_options=opts)
        plain = m(x, torch.tensor([1.0]), ctx, transformer_options={"cond_or_uncond": [0], "sigmas": torch.tensor([1.0]), "sample_sigmas": SAMPLE_SIGMAS})
    assert torch.isfinite(out).all()
    assert not torch.allclose(out, plain)
    assert torch.nn.functional.linear is _ORIG_LINEAR


def run_batch(m, c, cond_or_uncond, x, ctx, sigma=1.0):
    opts = {"cond_or_uncond": cond_or_uncond, "sigmas": torch.full((x.shape[0],), sigma), "sample_sigmas": SAMPLE_SIGMAS}
    pe.add_wrapper_with_key(pe.WrappersMP.DIFFUSION_MODEL, "tsam", wrapper.TSAMWrapper(target(), c), opts)
    return m(x, torch.full((x.shape[0],), sigma), ctx, transformer_options=opts)


def batch_inputs(n):
    g = torch.Generator().manual_seed(5)
    return torch.randn(n, 4, 8, 8, generator=g), torch.randn(n, 5, 48, generator=g)


def test_every_image_in_the_batch_is_optimised_independently():
    m = tiny_dit()
    x, ctx = batch_inputs(2)
    both = run_batch(m, cfg(), [0], x, ctx)
    off = run_batch(m, cfg(alpha=0.0), [0], x, ctx)
    assert not torch.allclose(both[0], off[0])
    assert not torch.allclose(both[1], off[1])
    for i in range(2):
        alone = run_batch(m, cfg(), [0], x[i:i + 1], ctx[i:i + 1])
        assert torch.allclose(both[i:i + 1], alone, atol=1e-5, rtol=1e-4)


def test_stacked_cond_uncond_only_touches_cond_rows():
    m = tiny_dit()
    x, ctx = batch_inputs(4)
    for order, cond_rows in (([0, 1], [0, 1]), ([1, 0], [2, 3])):
        on = run_batch(m, cfg(), order, x, ctx)
        off = run_batch(m, cfg(alpha=0.0), order, x, ctx)
        for r in range(4):
            assert torch.allclose(on[r], off[r], atol=1e-6) == (r not in cond_rows)
