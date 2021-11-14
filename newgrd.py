import numpy as np
from numba import jit
@jit
def ng_u(gux,u,un,nx,dx):
    for i in np.arange(1,nx):
        gux[i]=gux[i]+(un[i+1]-un[i-1]-u[i+1]+u[i-1])*0.5/dx
    return gux
@jit
def gcxcal(c,cn,gcx,nx,dx):
    for i in np.arange(2,nx):
        gcx[i]=gcx[i]+(cn[i+1]-cn[i-1]-c[i+1]+c[i-1])*0.5/dx
    gcx[1]=gcx[2]
    gcx[nx]=gcx[nx-1]
    return gcx