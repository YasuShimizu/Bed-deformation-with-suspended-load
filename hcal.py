import numpy as np
from numba import jit
@jit
def hh(hn,h,hs,eta,qu,b,alh,hmin,dx,nx,dt,err):
    err=0.
    for i in np.arange(1,nx+1):
        div=(qu[i-1]-qu[i])/b[i]
        hsta=h[i]+div/dx*dt
        serr=abs(hsta-hn[i])
        err=err+serr
        hn[i]=hsta*alh+hn[i]*(1.-alh)
        hs[i]=hn[i]-eta[i] 
        if hs[i]<hmin:
            hs[i]=hmin; hn[i]=eta[i]+hmin

    return hn,hs,err                