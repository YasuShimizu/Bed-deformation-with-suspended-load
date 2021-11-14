import numpy as np
import math
from numba import jit
@jit
def cn1(c,cn,qsu_x,hs,alpha,wf,wfcb,nx,dt):
    for i in np.arange(1,nx+1):
        wfcb[i]=wf*alpha[i]*c[i]
        cn[i]=c[i]+dt/hs[i]*(qsu_x[i]-wfcb[i])
    return cn,wfcb