import numpy as np
import copy
from numba import jit
@jit
def u_cal1(un,gux,u,fn,gxn,nx,dx,dt):
    for i in np.arange(1,nx+1):
        xx=-u[i]*dt
        isn=int(np.sign(u[i]))
        if isn==0:
            isn=1
        im=i-isn
        a1=((gux[im]+gux[i])*dx*isn-2.*(un[i]-un[im]))/(dx**3*isn)
        e1=(3.*(un[im]-un[i])+(gux[im]+2.*gux[i])*dx*isn)/dx**2
        fn[i]=((a1*xx+e1)*xx+gux[i])*xx+un[i]            
        gxn[i]=(3.*a1*xx+2.*e1)*xx+gux[i]
    return fn,gxn
@jit
def u_cal2(fn,gxn,u,un,gux,nx,dx,dt):
    un=fn.copy() 
    gux=gxn.copy()
    for i in np.arange(1,nx+1):
        gxo=(u[i+1]-u[i-1])*.5/dx
        gux[i]=gux[i]-(gxo*(u[i+1]-u[i-1]))*.5*dt/dx
    return un,gux
@jit
def c_cal1(cn,gcx,u,fn,gxn,nx,dx,dt):
    for i in np.arange(1,nx+1):
        u_cp=(u[i]+u[i-1])*.5
        xx=-u_cp*dt
        isn=int(np.sign(u_cp))
        if isn==0:
            isn=1
        im=i-isn
        a1=((gcx[im]+gcx[i])*dx*isn-2.*(cn[i]-cn[im]))/(dx**3*isn)
        e1=(3.*(cn[im]-cn[i])+(gcx[im]+2.*gcx[i])*dx*isn)/dx**2
        fn[i]=((a1*xx+e1)*xx+gcx[i])*xx+cn[i]            
        gxn[i]=(3.*a1*xx+2.*e1)*xx+gcx[i]
    return fn,gxn
@jit
def c_cal2(fn,gxn,c,cn,gcx,nx,dx,dt):
    cn=fn.copy() 
    gcx=gxn.copy()
    for i in np.arange(1,nx+1):
        gxo=(c[i+1]-c[i-1])*.5/dx
        gcx[i]=gcx[i]-(gxo*(c[i+1]-c[i-1]))*.5*dt/dx
    return cn,gcx