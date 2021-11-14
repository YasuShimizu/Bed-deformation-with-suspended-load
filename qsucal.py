import numpy as np
import phisval
import math
from numba import jit
@jit
def itakura(qsu,tausta,sgd_s,wf,nx):
    for i in np.arange(0,nx+1):
        if tausta[i]>0.:
            omega=phisval.omg_s(tausta[i])
            qsu[i]=0.008*(0.14/2.65*sgd_s*omega/math.sqrt(tausta[i])-wf)
            if qsu[i]<0.:
                qsu[i]=0.
        else:
            qsu[i]=0.
    return qsu

