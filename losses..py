import torch
import numpy as np

def cvar_value(p, v, reg):
    """Returns <p, v> - reg * KL(p, uniform) for Torch tensors"""
    m = p.shape[0]

    with torch.no_grad():
        idx = torch.nonzero(p)  # where is annoyingly backwards incompatible
        kl = np.log(m) + (p[idx] * torch.log(p[idx])).sum()

    return torch.dot(p, v) - reg * kl


def bisection(eta_min, eta_max, f, tol, maxiter):
    minimum = f(eta_min)
    maximum = f(eta_max)

    while minimum > 0 or maximum<0:
        length = eta_max - eta_min
        if minimum>0:
            eta_max = eta_min
            eta_min = eta_min-2*length

        if maximum<0:
            eta_min = eta_max
            eta_max = eta_max+2*length

        minimum = f(eta_min)
        maximum = f(eta_max)
    
    for _ in range(maxiter):
        eta = (eta_min+eta_max)/2
        value = f(eta)
        if abs(value)<=tol:
            return eta
        if value>0:
            eta_max = eta
        elif value<0:
            eta_min = eta
    
    return (eta_min+eta_max)/2

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
            
            def bisection_target(eta):
                return 1.0 - p(eta).sum()
            
            eta_min = reg * torch.logsumexp(v / reg - np.log(m), 0)
            eta_max = v.max()

            if abs(bisection_target(eta_min))<=self.tol:
                return p(eta_min)
        else:
            return 'regularizer is under zero'


    def forward(self, v, p):
        return cvar_value(p, v, self.reg)