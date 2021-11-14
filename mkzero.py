import numpy as np
from numba import jit

@jit
def z0(fn,gxn,nx):
    fn = np.zeros([nx+2])
    gxn = np.zeros([nx+2])
 
    return fn, gxn
