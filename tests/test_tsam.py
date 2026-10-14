import torch

from krea2_tsam import tsam


def test_text_target_crops_and_renormalises():
    attn = torch.tensor([[1.0, 0, 0, 0], [0.5, 0.5, 0, 0], [0.4, 0.3, 0.3, 0], [0.4, 0.2, 0.2, 0.2]]) * 2
    t = tsam.build_text_target(attn, num_layers=2, strip=1)
    assert t.shape == (3, 3)
    assert torch.allclose(t.sum(-1), torch.ones(3))
    assert t[0, 1] == 0 and t[0, 0] == 1


def test_similarity_rows_sum_to_one():
    s = tsam.similarity_matrix(torch.rand(50, 5) + 0.1)
    assert s.shape == (5, 5)
    assert torch.allclose(s.sum(-1), torch.ones(5), atol=1e-6)


def test_loss_zero_when_target_matches():
    a = torch.rand(64, 6) + 0.1
    t = tsam.similarity_matrix(a).pow(1 / 4)
    assert tsam.tsam_loss(t, a, gamma=4).abs() < 1e-5


def test_loss_has_gradient():
    a = (torch.rand(64, 6) + 0.1).requires_grad_(True)
    t = torch.tril(torch.rand(6, 6))
    t = t / t.sum(-1, keepdim=True).clamp_min(1e-6)
    loss = tsam.tsam_loss(t, a, gamma=2)
    loss.backward()
    assert loss > 0 and a.grad.abs().sum() > 0


def test_smooth_maps_keeps_shape_and_mass_roughly():
    a = torch.rand(16 * 16, 4)
    out = tsam.smooth_maps(a, 16, 16)
    assert out.shape == a.shape
    assert torch.allclose(out.sum(0), a.sum(0), rtol=0.05)
