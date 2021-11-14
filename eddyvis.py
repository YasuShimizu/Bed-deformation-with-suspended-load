import numpy as np
import math
from numba import jit
@jit
def epcal(epsilon,usta,hs_up,kappa,nx):
    for i in np.arange(0,nx+1):
        epsilon[i]=kappa/6.*usta[i]*hs_up[i]
    return epsilon
