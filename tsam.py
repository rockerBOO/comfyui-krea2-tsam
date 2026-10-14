import torch
import torch.nn.functional as F


def build_text_target(attn_sum, num_layers, strip):
    t = attn_sum[strip:, strip:] / num_layers
    return t / t.sum(dim=-1, keepdim=True)


def smooth_maps(a, h, w, sigma=0.5):
    s = a.shape[1]
    x = torch.arange(-1, 2, device=a.device, dtype=a.dtype)
    k1 = torch.exp(-x ** 2 / (2 * sigma ** 2))
    k = k1[:, None] * k1[None, :]
    k = (k / k.sum()).reshape(1, 1, 3, 3)
    m = a.t().reshape(s, 1, h, w)
    m = F.conv2d(F.pad(m, (1, 1, 1, 1), mode="reflect"), k)
    return m.reshape(s, h * w).t()


def similarity_matrix(a):
    a = F.normalize(a, dim=0)
    c = a.t() @ a
    return c / c.sum(dim=-1, keepdim=True)


def tsam_loss(t, a, gamma):
    s = t.shape[0]
    rho = torch.arange(1, s + 1, device=a.device, dtype=a.dtype) / s
    diff = (t.to(a.dtype).pow(gamma) - similarity_matrix(a)).abs() * rho[:, None]
    return torch.tril(diff).sum()
