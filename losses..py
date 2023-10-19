import torch
import numpy as np

def cvar_value(p, v, reg):
    """Returns <p, v> - reg * KL(p, uniform) for Torch tensors"""
    m = p.shape[0]

    with torch.no_grad():
        idx = torch.nonzero(p)  # where is annoyingly backwards incompatible
        kl = np.log(m) + (p[idx] * torch.log(p[idx])).sum()

    return torch.dot(p, v) - reg * kl


class loss(torch.nn.Module):
    def __init__(self, v, reg):
        super(loss, self).__init__()
        self.v = v
        self.reg = reg

    def forward(self, p):
        return cvar_value(p, self.v, self.reg)