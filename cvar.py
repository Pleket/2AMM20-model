import numpy as np
import cvxpy as cp
from losses import cvar_value

def cvar(returns, lam, alpha):
    m = len(returns)
    cvar = cp.Variable(m, nonreg = True)
    objective = v@cvar + lam*cp.sum(cp.entr(p=cvar)) - lam*np.log(m)
    constraints = [cp.max(cvar) <= 1.0/(alpha*m),
                    cp.sum(cvar) == 1]
    prob = cp.Problem(cp.Maximize(objective), constraints)
    prob.solve(solver=cp.MOSEK)
    return cvar


