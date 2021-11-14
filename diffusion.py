import numpy as np
import math
from numba import jit
@jit
def cdiff(cn,epsilon,dcdx,nx,dx,dt):
    for i in np.arange(1,nx):
        dcdx[i]=(cn[i+1]-cn[i])/dx
    dcdx[0]=dcdx[1]; dcdx[nx]=dcdx[nx-1]

    for i in np.arange(1,nx+1):
        cn[i]=cn[i]+(dcdx[i]*epsilon[i]-dcdx[i-1]*epsilon[i-1])/dx*dt
    return cn,dcdx
    
