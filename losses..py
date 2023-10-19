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
    def __init__(self, alpha, tol, maxiter, reg):
        super(loss, self).__init__()
        self.alpha = alpha
        self.tol = tol
        self.maxiter = maxiter
        self.reg = reg

    def response(self, v):
        alpha = self.alpha
        reg = self.reg
        m = v.shape[0]

        if self.reg>0:
            if alpha == 1.0:
                return torch.ones_like(v) / m
            def p(eta):
                x = (v-eta)/reg
                return torch.min(torch.exp(x), torch.Tensor([1.0/alpha]).type(x.dtype))/m

    def forward(self, p):
        return cvar_value(p, self.v, self.reg)