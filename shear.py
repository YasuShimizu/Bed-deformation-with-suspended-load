import numpy as np
import math
from numba import jit
@jit
def usts(usta,tausta,ie,hs_up,g,sd,nx):
    for i in np.arange(0,nx+1):
        ri=hs_up[i]*ie[i]
        usta[i]=math.sqrt(g*ri)
        tausta[i]=ri/sd
    return usta,tausta