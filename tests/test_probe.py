import torch

from krea2_tsam import probe


def _ref_probs(q, k, mask):
    k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
    logits = q @ k.transpose(-1, -2) * q.shape[-1] ** -0.5 + mask
    return logits.softmax(-1).mean(1)[0]


def test_text_probe_matches_reference_and_delegates():
    q, k, v = torch.randn(1, 4, 5, 8), torch.randn(1, 2, 5, 8), torch.randn(1, 2, 5, 8)
    mask = torch.full((5, 5), -1e9).triu(1)
    p = probe.AttentionProbe(max_layers=2)
    called = []

    def inner(q, k, v, heads, mask=None, **kw):
        called.append(kw)
        return "out"

    assert p.wrap(inner)(q, k, v, 4, mask=mask, skip_reshape=True, enable_gqa=True) == "out"
    assert called == [{"skip_reshape": True, "enable_gqa": True}]
    assert torch.allclose(p.attn_sum, _ref_probs(q, k, mask), atol=1e-6)
    assert p.count == 1


def test_probe_respects_layer_limit():
    p = probe.AttentionProbe(max_layers=1)
    q = torch.randn(1, 2, 3, 4)
    w = p.wrap(lambda *a, **kw: None)
    w(q, q, q, 2, mask=None)
    w(q, q, q, 2, mask=None)
    assert p.count == 1


def test_map_collector_softmax_over_text_keys():
    txtlen, n_img = 3, 6
    L = txtlen + n_img
    q, k, v = torch.randn(1, 4, L, 8), torch.randn(1, 2, L, 8), torch.randn(1, 2, L, 8)
    c = probe.MapCollector(txtlen=txtlen, layer_lo=0, layer_hi=2, rows=[0])
    opts = {"block_index": 1, "img_slice": [txtlen, L], "reference_image_num_tokens": []}
    assert c(q, k, v, pe=None, attn_mask=None, extra_options=opts) == {}
    assert c.maps[0].shape == (1, n_img, txtlen)
    assert torch.allclose(c.maps[0].sum(-1), torch.ones(1, n_img), atol=1e-5)


def test_map_collector_skips_other_layers_and_reference_tokens():
    txtlen, n_img, n_ref = 3, 4, 5
    L = txtlen + n_img + n_ref
    q = torch.randn(1, 2, L, 8)
    c = probe.MapCollector(txtlen=txtlen, layer_lo=2, layer_hi=3, rows=[0])
    opts = {"img_slice": [txtlen, L], "reference_image_num_tokens": [n_ref]}
    c(q, q, q, extra_options={**opts, "block_index": 0})
    assert c.maps == []
    c(q, q, q, extra_options={**opts, "block_index": 2})
    assert c.maps[0].shape == (1, n_img, txtlen)
