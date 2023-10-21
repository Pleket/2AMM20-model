import torch
import numpy as np

def cvar_value(p, v, reg):
    """Returns <p, v> - reg * KL(p, uniform) for Torch tensors"""
    m = v.shape[0]
    
    #with torch.no_grad():
    idx = torch.nonzero(p)  # where is annoyingly backwards incompatible
    kl = np.log(m) + (p[idx] * torch.log(p[idx])).sum()    
        
    print(p.shape, v.shape, kl)
    return torch.dot(p.squeeze(), v.squeeze()) - reg * kl


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

class Loss(torch.nn.Module):
    def __init__(self, alpha = 0.1, reg = 0.01, tol = 1e-4, maxiter=50):
        super(Loss, self).__init__()
        self.alpha = alpha
        self.tol = tol
        self.maxiter = maxiter
        self.reg = reg

    def response(self, v):
        alpha = self.alpha
        reg = self.reg
        m = v.shape[0]

        if reg>0:
            if alpha == 1.0:
                return torch.ones_like(v) / m
            
            def p(eta):
                x = (v-eta)/reg
                new_x = torch.exp(x)
                alph = torch.Tensor([1.0/alpha]).type(x.dtype)
                m_tensor = torch.Tensor([m]).type(x.dtype)
                return torch.div(torch.min(new_x, alph), m_tensor)
            
            def bisection_target(eta):
                return 1.0 - p(eta).sum()
            
            eta_min = reg * torch.logsumexp(v / reg - np.log(m), 0)
            eta_min = eta_min.squeeze()
            eta_max = v.max()

            if abs(bisection_target(eta_min))<=self.tol:
                # print(p(eta_min))
                return p(eta_min)
            # else:
            #     eta = bisection(eta_min, eta_max, bisection_target, self.tol, self.maxiter)
            #     # print(p(eta))
            #     return p(eta)
        else:
            return 'regularizer is under zero'


    def forward(self, v):
        #with torch.no_grad():
        p = self.response(v)    
        return cvar_value(p, v, self.reg)