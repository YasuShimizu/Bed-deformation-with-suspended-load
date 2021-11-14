import numpy as np
import math
from numba import jit
@jit
def fallvel(sgd3,sgd_s,nu):
    wf1=math.sqrt(2./3.+36.*nu**2/sgd3)
    wf2=math.sqrt(36.*nu**2/sgd3)
    wf=(wf1-wf2)*sgd_s
    return wf
@jit
def iwgk(diam,sub,g):
    dcm=diam*100.
    gcm=g*100.
    if dcm>= 0.303:
        usc2=80.9*dcm
    elif dcm>= 0.118:
        usc2=134.6*dcm**(31./22)
    elif dcm>= 0.0565:
        usc2=55.*dcm
    elif dcm>=0.0065:
        usc2=8.41*dcm**(11./22.)
    else:
        usc2=226*dcm
    tsc=usc2/(sub*gcm*dcm)
    return tsc
@jit    
def omg(ts):
    pi=3.141592; b_sta=0.143
    a1=.4361836; a2=-.1201676; a3 = .937298
    if ts<=1e-6:
        omega=0.
    else:
        ad=b_sta/ts-2.
        if ad>0.:
            x= ad*math.sqrt(2.)
        else:
            x=-ad*math.sqrt(2.)
            t=1./(1.+0.33627*x)
            zx=1./math.sqrt(2.*pi)*math.exp(-x**2./2.)
            px=1.-zx*(a1*t+a2*t**2+a3*t**3)
            er1=2.-2.*px
            if ad>=0.:
                er=er1/2.
            else:
                er=2.-er1/2.
            if b_sta == 0. or er == 0. :
                omega=0.
            else:
                omega=ts/b_sta/(2.*math.sqrt(pi))*math.exp(-ad**2)/er \
                    +ts*2./b_sta-1.
            if omega<1e-10:
                omega=0.
    return omega    
@jit
def omg_s(ts):
    omega=14.*ts-0.9
    return omega
@jit
def alpha_beta(alpha,beta,usta_x,wf,nx):
    for i in np.arange(1,nx+1):
        beta[i]=6.*wf/(0.4*usta_x[i])
        alpha[i]=beta[i]/(1.-math.exp(-beta[i]))
    beta[0]=beta[1]
    alpha[0]=alpha[1]
    return alpha,beta
