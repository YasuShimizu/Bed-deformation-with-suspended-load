import numpy as np
# j_upstm, j_dwstm .... 1=Wall, 2=Free, 3=Fixed Value(Flowing)

def h_bound(h,hs,eta,h_upstm,h_dwstm,hs_dwstm,nx,j_upstm,j_dwstm):
    if j_upstm==2:
        hs[0]=hs[1]

    if j_dwstm==3:
        hs[nx+1]=h_dwstm-eta[nx+1]
    else:
        hs[nx+1]=hs_dwstm

    h[0]=eta[0]+hs[0]
    h[nx+1]=eta[nx+1]+hs[nx+1]
    return h,hs

def u_bound(u,hs_up,qp,b_up,nx,j_upstm,j_dwstm,u_upstm,u_dwstm):
    if j_upstm==1:
        u[0]=0.
    else:
        u[0]=qp/(hs_up[0]*b_up[0])

    if j_dwstm==1:
        u[nx]=0.
    return u

def hs_up_cal(hs,hs_up,nx,hs_upstm,hs_dwstm,j_upstm,j_dwstm):
    for i in np.arange(0,nx+1):
        hs_up[i]=(hs[i]+hs[i+1])*.5

#    if j_upstm==3: 
#        hs_up[0]=hs_upstm
#    if j_dwstm==3:
#        hs_up[nx]=hs_dwstm

    return hs_up

def h_up_cal(hs_up,eta_up,h_up,nx):
    for i in np.arange(0,nx+1):
        h_up[i]=eta_up[i]+hs_up[i]
    return h_up

def eta_up_cal(eta,eta_up,nx):
    for i in np.arange(1,nx):
        eta_up[i]=(eta[i]+eta[i+1])*.5
    eta_up[0]=eta[1]
    eta_up[nx]=eta[nx]
    return eta_up

def gbound_u(gux,nx):
    gux[0]=0.
    gux[nx]=0.
    return gux

def shear_bound(usta,tausta):
    tausta[0]=tausta[1]
    usta[0]=usta[1]
    return usta,tausta

def qb_bound(qb,alpha_up_sed):
    qb[0]=qb[1]*alpha_up_sed
    return qb

def qsu_bound(qsu):
    qsu[0]=qsu[1]
    return qsu

def qsu_x_cal(qsu,qsu_x,nx):
    for i in np.arange(1,nx+1):
        qsu_x[i]=(qsu[i]+qsu[i-1])*.5
    return qsu_x

def eta_bound(eta,h,hs,nx):
    
    eta[nx+1]=2.*eta[nx]-eta[nx-1]
    hs[nx+1]=h[nx+1]-eta[nx+1]

    eta[0]=2.*eta[1]-eta[2]
    hs[0]=hs[1]
    h[0]=eta[0]+hs[0]

    return eta,hs,h

def usta_xcal(usta,usta_x,nx):

    for i in np.arange(1,nx+1):
        usta_x[i]=(usta[i]+usta[i-1])*.5
    return usta_x

def cbound(cn,wfcb,nx):
    cn[0]=cn[1]; wfcb[0]=wfcb[1]
    cn[nx+1]=cn[nx]; wfcb[nx+1]=wfcb[nx]
    return cn,wfcb

def gbound_c(gcx,nx):
    gcx[0]=0.
    gcx[nx+1]=0.
    return gcx