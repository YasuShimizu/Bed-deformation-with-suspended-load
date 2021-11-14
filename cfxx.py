import numpy as np
from numba import jit
@jit
def cfx_cal(cfx,nx,un,hs_up,ie,g,snm):
    for i in np.arange(0,nx+1):
        ie[i]=snm**2*un[i]*np.abs(un[i])/hs_up[i]**(4./3.)
        cfx[i]=-g*ie[i]
    return cfx,ie
