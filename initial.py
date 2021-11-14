import numpy as np
from numpy.matrixlib import defmatrix
from numba import jit
@jit
def eta_init(eta,eta0,eta_up,eta_up0,nx,dx, \
    slope,xl,xb1,xb2,xb3,dbed):
    zb0=xl*slope
    for i in np.arange(0,nx+2):
        xx=dx*float(i)
        eta_up[i]=zb0-xx*slope
        eta_up0[i]=eta_up[i]
#        print(i,nx,eta_up[i])
        if xx>xb1 and xx<xb2:
            ss=xx-xb1
            deta=dbed*ss/(xb2-xb1)
            eta_up[i]=eta_up[i]+deta
        elif xx>=xb2 and xx<xb3:
            ss=xb3-xx
            deta=dbed*ss/(xb3-xb2)
            eta_up[i]=eta_up[i]+deta

    for i in np.arange(1,nx+2):
        eta[i]=(eta_up[i]+eta_up[i-1])*.5
        eta0[i]=(eta_up0[i]+eta_up0[i-1])*.5
    eta[0]=2.*eta[1]-eta[2]
    eta0[0]=2.*eta0[1]-eta0[2]
    eta[nx+1]=2.*eta[nx]-eta[nx-1]
    eta0[nx+1]=2.*eta0[nx]-eta0[nx-1]
    return eta,eta0,eta_up,eta_up0

@jit
def eta_init_2(eta,eta0,eta_up,eta_up0,nx,dx, \
    xl,x_slope,slope1,slope2):
    zb0=x_slope*slope1+(xl-x_slope)*slope2
    zb1=zb0-x_slope*slope1
    for i in np.arange(0,nx+2):
        xx=dx*float(i)
        if xx <= x_slope:
            eta_up[i]=zb0-xx*slope1
        else:
            eta_up[i]=zb1-(xx-x_slope)*slope2    
        eta_up0[i]=eta_up[i]

    for i in np.arange(1,nx+2):
        eta[i]=(eta_up[i]+eta_up[i-1])*.5
        eta0[i]=(eta_up0[i]+eta_up0[i-1])*.5
    eta[0]=2.*eta[1]-eta[2]
    eta0[0]=2.*eta0[1]-eta0[2]
    return eta,eta0,eta_up,eta_up0

@jit
def bslope_cal(nx,dx,bslope,bslope_up\
    ,j_channel,slope,x_slope,slope1,slope2):
    if j_channel==1:
        for i in np.arange(1,nx+1):
            bslope[i]=slope
            bslope_up[i]=slope
    else:
        for i in np.arange(1,nx+1):
            xx=dx*float(i)-dx*.5
            if xx <= x_slope:
                bslope[i]=slope1
            else:
                bslope[i]=slope2
            xx=dx*float(i)
            if xx <= x_slope:
                bslope_up[i]=slope1
            else:
                bslope_up[i]=slope2
            
    bslope_up[0]=bslope[1]
    bslope_up[nx]=bslope[nx]

    return bslope,bslope_up

@jit
def width_init(b,b_up,nx,dx,xw1,xw2,xw3,xl, \
        w0,w1,w2,w3,w4):
    for i in np.arange(0,nx+2):
        xx=dx*float(i)
        if xx<=xw1:
            dd=xx/xw1
            b_up[i]=w0+(w1-w0)*dd
        elif xx<=xw2:
            dd=(xx-xw1)/(xw2-xw1)
            b_up[i]=w1+(w2-w1)*dd
        elif xx<=xw3:
            dd=(xx-xw2)/(xw3-xw2)
            b_up[i]=w2+(w3-w2)*dd
        elif xx<=xl:
            dd=(xx-xw3)/(xl-xw3)
            b_up[i]=w3+(w4-w3)*dd

#        print(i,xx,b_up[i])

    for i in np.arange(1,nx+2):
        b[i]=(b_up[i]+b_up[i-1])*.5

    return b,b_up

@jit
def h_init(eta,eta0,eta_up,eta_up0,h,hs,h_up,hs_up,hs0_up,h_upstm,h_dwstm, \
        hs_upstm,hs_dwstm,nx,dx,xl,j_ini_prof):
#    print(h_upstm,h_dwstm,hs_upstm,hs_dwstm)
    h00=eta_up0[0]+hs_upstm
    h11=eta_up0[nx]+hs_dwstm
    for i in np.arange(0,nx+1):
        xx=float(i)*dx;ss=xx/xl
        if i==0:
            hs_up[i]=hs_upstm;h_up[i]=h00
        elif i==nx:
            hs_up[i]=hs_dwstm;h_up[i]=h11
        else:
            if j_ini_prof==1:
                hs_up[i]=hs0_up[i];h_up[i]=eta_up0[i]+hs_up[i]
            elif j_ini_prof==2:
                h_up[i]=h00-ss*(h00-h11)
                hs_up[i]=h_up[i]-eta_up0[i]
            elif j_ini_prof==3:
                hs_up[i]=hs0_up[i];h_up[i]=eta_up0[i]+hs_up[i]
                if h_up[i] < h_dwstm:
                    h_up[i]=h_dwstm
                    hs_up[i]=h_up[i]-eta_up0[i]
#        print(i,xx,h_up[i],hs_up[i])
#    exit()
    for i in np.arange(1,nx+1):
        hs[i]=(hs_up[i]+hs_up[i-1])*.5
        h[i]=eta0[i]+hs[i]

    hs[0]=hs_upstm; h[0]=eta[0]+hs_upstm
    hs[nx+1]=hs_dwstm; h[nx+1]=eta[nx+1]+hs_dwstm

    return h,hs,h_up,hs_up

@jit
def u_init(g,qp,u,qu,hs_up,b_up,fr,nx):
    for i in np.arange(0,nx+1):
        u[i]=qp/(b_up[i]*hs_up[i])
        fr[i]=u[i]/np.sqrt(g*hs_up[i])
        qu[i]=qp
#        print(i,hs_up[i],u[i],fr[i])

    return u,fr

@jit
def x_cell_init(x_cell,x,dx,nx):
    for i in np.arange(1,nx+1):
        x_cell[i]=(x[i]+x[i-1])*.5
    x_cell[0]=x_cell[1]-dx
    x_cell[nx+1]=x_cell[nx]+dx

    return x_cell

@jit
def h0_cal(nx,dx,qp,snm,eta_up,hs0_up,h0_up, \
    b_up,bslope_up,u0_up):
    
    for i in np.arange(0,nx+1):
        if bslope_up[i]<=0. :
            hs0_up[i]=0.
            u0_up[i]=0.
        else:
            hs0_up[i]=(qp/b_up[i]*snm/np.sqrt(bslope_up[i]))**(.6)
            u0_up[i]=qp/(b_up[i]*hs0_up[i])

        h0_up[i]=eta_up[i]+hs0_up[i]
    
    return hs0_up,h0_up,u0_up

@jit
def hc_cal(hc_up,qp,nx,b_up,g):
    for i in np.arange(0,nx+1):
        hc_up[i]=(qp**2/b_up[i]**2/g)**(1./3.)

#        print(i,hc_up[i])
    return hc_up

@jit
def c_init(c,qsu,wf,wfcb,alpha,nx):
    for i in np.arange(1,nx+1):
        c[i]=qsu[i]/(wf*alpha[i])
        wfcb[i]=c[i]*alpha[i]
    return c,wfcb

@jit
def gcx_init(c,gcx,nx,dx):
    for i in np.arange(2,nx):
        gcx[i]=(c[i+1]-c[i-1])/(2.*dx)
#        print(i,gcx[i])
    gcx[1]=(c[2]-c[1])/dx
    gcx[nx]=(c[nx]-c[nx-1])/dx
    return gcx
