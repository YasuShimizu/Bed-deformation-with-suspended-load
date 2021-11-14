import numpy as np
import math
from numba import jit
@jit
def qbcal(qb,tausta,tsc,sgd3_s,nx):
    for i in np.arange(1,nx+1):
        tse=tausta[i]-tsc
        if tse<0.:
            qb[i]=0.
        else:
            qb[i]=8.*tse**(3./2.)*sgd3_s
    return qb
@jit
def etacal_bl(eta,deta,qb,qsu_x,h,hs,b_up,b,nx,dx,wf,wfcb,rlam,cn,alpha,j_sediment):
    for i in np.arange(1,nx+1):
        deta1=0.; deta2=0.
        if j_sediment==1 or j_sediment==3:
            deta1=rlam*(qb[i-1]*b_up[i-1]-qb[i]*b_up[i])/(b[i]*dx)

        if j_sediment>=2:
            deta2=-rlam*(qsu_x[i]-wfcb[i])

        deta[i]=deta1+deta2
        eta[i]=eta[i]+deta[i]
        hs[i]=h[i]-eta[i]
    return eta,deta,hs