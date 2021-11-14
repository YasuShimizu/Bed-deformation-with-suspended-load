import numpy as np
import copy, subprocess, os, yaml, sys, math
import matplotlib.pyplot as plt
from random import randint
import matplotlib.animation as animation
from numpy.lib.type_check import nan_to_num
import initial, boundary, cfxx, rhs, hcal, newgrd, cip1d
import qbz,phisval,qsucal,shear,source,eddyvis,diffusion
from matplotlib._version import get_versions as mplv
from matplotlib.animation import PillowWriter


conf_file='config_res.yml' # 貯水池への堆砂
gifname='res.gif'

# Open Config File
with open(conf_file,'r', encoding='utf-8') as yml:   
    config = yaml.load(yml)
 
xl=float(config['xl']); nx=int(config['nx'])
j_channel=int(config['j_channel'])
slope=float(config['slope'])
x_slope=float(config['x_slope'])
slope1=float(config['slope1']); slope2=float(config['slope2'])
xb1=float(config['xb1']); xb2=float(config['xb2']); xb3=float(config['xb3'])
dbed=float(config['dbed'])
xw1=float(config['xw1']); xw2=float(config['xw2']); xw3=float(config['xw3'])
w0=float(config['w0']);w1=float(config['w1']);w2=float(config['w2'])
w3=float(config['w3']);w4=float(config['w4'])

qp=float(config['qp']); g=float(config['g']); snm=float(config['snm'])
alh=float(config['alh']); lmax=int(config['lmax']); errmax=float(config['errmax'])
hmin=float(config['hmin'])
j_upstm=int(config['j_upstm']); j_dwstm=int(config['j_dwstm'])
j_dwstm_cnd=int(config['j_dwstm_cnd']); h_downstream=float(config['h_downstream'])
j_ini_prof=int(config['j_ini_prof'])
etime=float(config['etime']); dt=float(config['dt']); tuk=float(config['tuk'])
b_start=float(config['b_start'])
alpha_up=float(config['alpha_up']); alpha_dw=float(config['alpha_dw'])
alpha_up_sed=float(config['alpha_up_sed'])

diam=float(config['diam']);sub=float(config['sub']);lam=float(config['lam'])
nu=float(config['nu']); kappa=float(config['kappa'])
j_sediment=int(config['j_sediment']);j_sus_ini=int(config['j_sus_ini'])

sgd=sub*g*diam; sgd3=sgd*diam*diam
sgd3_s=math.sqrt(sgd3); sgd_s=math.sqrt(sgd);sd=sub*diam

wf=phisval.fallvel(sgd3,sgd_s,nu)
tsc=phisval.iwgk(diam,sub,g)


if j_channel==1:
    slp=slope
else:
    slp=slope1

hs0_use=(snm*qp/(w0*math.sqrt(slp)))**(3./5.)
usta0=math.sqrt(g*hs0_use*slp)
uswf=usta0/wf
if uswf<1.08:
    kwd='bedload'
elif uswf<1.67:
    kwd='bedload+suspended load'
else:
    kwd='suspended load'

#print('us/wf=',uswf,kwd)


#for i in np.arange(1,5):
#    s_tausta=i*.3
#    s_usta=math.sqrt(s_tausta)*sgd_s
#    print(s_usta,wf,s_usta/wf)
#    ome_s=phisval.omg(s_tausta)
#    qsu_s=0.008*(0.14/2.65*sgd_s*ome_s/math.sqrt(s_tausta)-wf)
#    print(s_tausta,ome_s,qsu_s)
#exit()

nx1=nx+1; nx2=nx+2
dx=xl/nx; xct=xl/2.
rlam=dt/(1.-lam)

x=np.linspace(0,xl,nx+1)
xmid=(x[0]+x[nx])*.5; nxm=int(nx/2)
x_cell=np.zeros([nx2])
x_cell=initial.x_cell_init(x_cell,x,dx,nx)
it_out=int(tuk/dt)

# Make Array
hs=np.zeros([nx2]); hs_up=np.zeros([nx2]); fr=np.zeros([nx2])
h=np.zeros([nx2]); hn=np.zeros([nx2]); h_up=np.zeros([nx2])
u=np.zeros([nx2]); un=np.zeros([nx2])
eta=np.zeros([nx2]); eta0=np.zeros([nx2]); eta_up=np.zeros([nx2]);eta_up0=np.zeros([nx2])
eta_00=np.zeros([nx2])
deta=np.zeros([nx2])
cfx=np.zeros([nx2]); qu=np.zeros([nx2])
gux=np.zeros([nx2]); gux_n=np.zeros([nx2])
fn=np.zeros([nx2]);gxn=np.zeros([nx2])
h0_up=np.zeros([nx2]); hc_up=np.zeros([nx2]); hs0_up=np.zeros([nx2])
bslope=np.zeros([nx2]); bslope_up=np.zeros([nx2])
b=np.zeros([nx2]); b_up=np.zeros([nx2]); u0_up=np.zeros([nx2])

c=np.zeros([nx2]); cn=np.zeros([nx2]); cb=np.zeros([nx2]); gcx=np.zeros([nx2])
wfcb=np.zeros([nx2])
ie=np.zeros([nx2]);tausta=np.zeros([nx2]);qb=np.zeros([nx2])
qsu=np.zeros([nx2]); qsu_x=np.zeros([nx2])
usta=np.zeros([nx2]);usta_x=np.zeros([nx2])
alpha=np.zeros([nx2]); beta=np.zeros([nx2])
epsilon=np.zeros([nx2]); dcdx=np.zeros([nx2])

# Geometric Condition
if j_channel==1:
    eta,eta0,eta_up,eta_up0=initial.eta_init \
        (eta,eta0,eta_up,eta_up0,nx,dx, \
            slope,xl,xb1,xb2,xb3,dbed)
else:
    eta,eta0,eta_up,eta_up0=initial.eta_init_2 \
        (eta,eta0,eta_up,eta_up0,nx,dx, \
        xl,x_slope,slope1,slope2)
eta_00=copy.copy(eta)

b,b_up=initial.width_init(b,b_up,nx,dx,xw1,xw2,xw3,xl, \
    w0,w1,w2,w3,w4)
bslope,bslope_up=initial.bslope_cal(nx,dx,bslope,bslope_up\
    ,j_channel,slope,x_slope,slope1,slope2)

# Uniform Flow Depth and Critical Depth
hs0_up,h0_up,u0_up=initial.h0_cal(nx,dx,qp,snm,eta_up, \
    hs0_up,h0_up,b_up,bslope_up,u0_up)    
hc_up=initial.hc_cal(hc_up,qp,nx,b_up,g)

# Initial Depth and Water Surface Elevation
hs_upstm=hs0_up[0]*alpha_up ; h_upstm=eta[0]+hs_upstm
if j_dwstm_cnd ==1:
    hs_dwstm=hs0_up[nx]*alpha_dw
elif j_dwstm_cnd ==2:
    hs_dwstm=h_downstream 

h_dwstm=eta[nx+1]+hs_dwstm

h,hs,h_up,hs_up=initial.h_init \
        (eta,eta0,eta_up,eta_up0,h,hs,h_up,hs_up,hs0_up,h_upstm,h_dwstm, \
        hs_upstm,hs_dwstm,nx,dx,xl,j_ini_prof)

# Hydraulic and Physical Parameters
u_upstm=qp/(b_up[0]*hs_upstm)
u_dwstm=qp/(b_up[nx]*hs_dwstm)

h,hs=boundary.h_bound(h,hs,eta,h_upstm,h_dwstm,hs_dwstm,nx,j_upstm,j_dwstm)

hs_up=boundary.hs_up_cal(hs,hs_up,nx,hs_upstm,hs_dwstm,j_upstm,j_dwstm)
h_up=boundary.h_up_cal(hs_up,eta_up,h_up,nx)
hn=copy.copy(h)

u,fr=initial.u_init(g,qp,u,qu,hs_up,b_up,fr,nx)
u=boundary.u_bound(u,hs_up,qp,b_up,nx,j_upstm,j_dwstm,u_upstm,u_dwstm)
un=copy.copy(u)

# Initial ie and cfx
cfx,ie=cfxx.cfx_cal(cfx,nx,un,hs_up,ie,g,snm)

# Initial usta and tausta
usta,tausta=shear.usts(usta,tausta,ie,hs_up,g,sd,nx)
usta,tausta=boundary.shear_bound(usta,tausta)
usta_x=boundary.usta_xcal(usta,usta_x,nx)

# Initial alpha and beta
alpha,beta=phisval.alpha_beta(alpha,beta,usta_x,wf,nx)

# Initial qb  
if j_sediment==1 or j_sediment==3:
    qb=qbz.qbcal(qb,tausta,tsc,sgd3_s,nx)
    qb=boundary.qb_bound(qb,alpha_up_sed)

# Initial qsu
if j_sediment>=2:
    qsu=qsucal.itakura(qsu,tausta,sgd_s,wf,nx)
    qsu=boundary.qsu_bound(qsu)
    qsu_x=boundary.qsu_x_cal(qsu,qsu_x,nx)
# Initial c
    if j_sus_ini==2:
        c,wfcb=initial.c_init(c,qsu_x,wf,wfcb,alpha,nx)
        gcx=initial.gcx_init(c,gcx,nx,dx)

# Seting for Plot
y_h0=np.zeros([nx+1]); y_hc=np.zeros([nx+1])
for i in np.arange(0,nx+1):
    y_h0[i]=h0_up[i]; y_hc[i]=eta_up[i]+hc_up[i]

fig=plt.figure(figsize=(30,40))
ims=[]
flag_legend=True

# Upper Panel Left:Elevation Right: Width
ax1= fig.add_subplot(3,1,1)
im1= ax1.set_title("1D Open Channel Flow with Bed Deformation",fontsize=50)
im1= ax1.set_xlabel("x(m)",fontsize=30)
ax1.tick_params(axis="x", labelsize=30)
zmax=np.amax(h0_up)*1.2
zmin=np.amin(eta); zmin=zmin-(zmax-zmin)*.3
im1= ax1.set_ylim(zmin, zmax)
im1= ax1.set_ylabel("Elevation(m)",fontsize=30)
ax1r=ax1.twinx() # Right Hand Vertical Axis
bmax=np.amax(b_up)*1.5
bmin=0.
im1r=ax1r.set_ylim(bmin,bmax)
im1r=ax1r.set_ylabel("Width(m)",fontsize=30)
ax1.tick_params(axis="y", labelsize=30)
ax1r.tick_params(axis="y", labelsize=30)

# Mid Pannel: Velocity Right: Tausta
ax2=fig.add_subplot(3,1,2)
im2= ax2.set_xlabel("x(m)",fontsize=30)
umax=np.amax(u)*1.5
im2= ax2.set_ylim(0, umax)
im2= ax2.set_ylabel("Velocity(m/s)",fontsize=30)
ax2r=ax2.twinx() # Right Hand Vertical Axis
tausta_max=np.amax(tausta)*2.
tausta_min=0.
im2r=ax2r.set_ylim(tausta_min,tausta_max)
im2r=ax2r.set_ylabel("Tausta",fontsize=30)
ax2.tick_params(axis="x", labelsize=30)
ax2.tick_params(axis="y", labelsize=30)
ax2r.tick_params(axis="y", labelsize=30)

#Lower Panel Left:Discharge Right:Froude Number
ax3= fig.add_subplot(3,1,3)
im3= ax3.set_xlabel("x(m)",fontsize=30)
qmax=np.amax(qp)*2.
im3= ax3.set_ylim(0, qmax)
im3= ax3.set_ylabel("Discharge(m3/s)",fontsize=30)
ax4=ax3.twinx() # Right Vertical Axis
frmax=np.amax(fr)*2.5
frmin=np.amin(fr)*1.2
im4= ax4.set_ylim(frmin, frmax)
im4= ax4.set_ylabel("Froude Number",fontsize=30)
ax3.tick_params(axis="x", labelsize=30)
ax3.tick_params(axis="y", labelsize=30)
ax4.tick_params(axis="y", labelsize=30)

# Bottom Panel qsu and wf_c
#ax5= fig.add_subplot(4,1,4)
#im5= ax5.set_xlabel("x(m)",fontsize=30)
#qsumax=1e-3
#im5= ax5.set_ylim(0,qsumax)
#im5= ax5.set_ylabel("Qsu and Wf_C(m/s)",fontsize=30)
#ax5.tick_params(axis="x", labelsize=30)
#ax5.tick_params(axis="y", labelsize=30)


time=0.; err=0.; icount=0; nfile=0; l=0
################ Main #####################

while time<= etime:
    if icount%it_out==0:
        print('time=',np.round(time,3),l)

# Plot Calculated Values
        hs_up=boundary.hs_up_cal(hs,hs_up,nx,hs_upstm,hs_dwstm,j_upstm,j_dwstm)
        eta_up=boundary.eta_up_cal(eta,eta_up,nx)

        h_up=boundary.h_up_cal(hs_up,eta_up,h_up,nx)
        y=np.zeros([nx+1]); y1=np.zeros([nx+1]); y2=np.zeros([nx+1])
        y3=np.zeros([nx+1]); y4=np.zeros([nx+1]); yb=np.zeros([nx+1])
        yt=np.zeros([nx+1])
        y5=np.zeros([nx+1]); y6=np.zeros([nx+1])

        for i in np.arange(0,nx+1):
            y[i]=eta_up[i]; yb[i]=b_up[i]
            y1[i]=h_up[i]
            y2[i]=u[i]
            y3[i]=qu[i]
            y4[i]=u[i]/np.sqrt(g*hs_up[i])
            yt[i]=tausta[i]
            y5[i]=qsu[i]; y6[i]=alpha[i]*c[i]*wf

#            y_hc[i]=eta_up[i]+hc_up[i]
#            y_h0[i]=eta_up[i]+hs0_up[i]

#        im1= ax1.plot(x,y,'magenta',label='Bed',linewidth=5)
        im1= ax1.plot(x_cell,eta,'magenta',label='Bed',linewidth=5)
        im00= ax1.plot(x_cell,eta_00,linestyle="dashed",color='magenta')
        im1r=ax1r.plot(x,yb,'green',label='Width',linewidth=5)
#        im11= ax1.plot(x,y1,linestyle = "dashed", color='blue',label="WSE",linewidth=2) 
        im1w= ax1.plot(x_cell,h,'blue',label="WSE",linewidth=5) 
#        if np.abs(dbed)<0.001:
#            im_h0=ax1.plot(x,y_h0,linestyle = "dashed",color='green',label='h0')
#            im0=ax1.text(x[nx],y_h0[nx],'h0',size='30')
#        else:
#            im_h0=""
#            im0=""

#       im_hc=ax1.plot(x,y_hc,linestyle = "dashed",color='black',label='hc')
#       imc=ax1.text(x[nx],y_hc[nx],'hc',size='30')
        
        im2= ax2.plot(x,y2,'red',label='Velocity',linewidth=5)
        im2r=ax2r.plot(x,yt,'blue',label='Tausta',linewidth=5)
        
        text1= ax1.text(0.,zmin,"Time="+str(np.round(time,3))+"sec",size=20)
        lg0=ax1.text(0.,eta[1],'Bed Elevation',size=20)
        lg00=ax1.text(xmid,eta_00[nxm],'Initial Bed',size=20)
        lg0r=ax1r.text(0.,yb[0],'Width',size=20)

        lg1=ax1.text(0.,y1[0],'Water Surface',size=20)
        text2= ax2.text(0.,0.,"Time="+str(np.round(time,3))+"sec",size=20)
        lg2=ax2.text(0.,y2[0],'Velocity',size=20)
        lg2r=ax2r.text(0.,yt[0],'Tausta',size=20)

        im3= ax3.plot(x,y3,'green',label='Dicharge',linewidth=5)
        im4= ax4.plot(x,y4,'black',label='Froude Number',linewidth=5)
        text3= ax3.text(0.,0.,"Time="+str(np.round(time,3))+"sec",size=20)
        lg3=ax3.text(0.,y3[0],'Discharge',size=20)
        lg4=ax4.text(0.,y4[0],'Froude Number',size=20)

#        im5= ax5.plot(x_cell,qsu,'green',label='Qsu',linewidth=5)
#        im6= ax5.plot(x_cell,wfcb,'red',label='Wf_Cb',linewidth=5)
#        text4= ax5.text(0.,0.,"Time="+str(np.round(time,3))+"sec",size=20)
#        lg5=ax5.text(0.,y5[0],'Qsu',size=20)
#        lg6=ax5.text(0.,y6[0],'Wf_Cb',size=20)

        itot=im1+im00+im1w+im2+im3+im4+[text1]+[text2]+[text3]+ \
            [lg0]+[lg00]+[lg1]+[lg2]+[lg3]+[lg4]+im1r+[lg0r]+im2r+[lg2r]
#            im5+im6+[text4]+[lg5]+[lg6]    

        ims.append(itot)
        
    #        exit()

# Non-Advection Phase
    l=0
    while l<lmax:
        hs_up=boundary.hs_up_cal(hs,hs_up,nx,hs_upstm,hs_dwstm,j_upstm,j_dwstm)
        cfx,ie=cfxx.cfx_cal(cfx,nx,un,hs_up,ie,g,snm)
        un=rhs.un_cal(un,u,nx,dx,cfx,hn,g,dt)
        un=boundary.u_bound(un,hs_up,qp,b_up,nx,j_upstm,j_dwstm,u_upstm,u_dwstm)
        qu=rhs.qu_cal(qu,un,hs_up,b_up,nx)
        hn,hs,err=hcal.hh(hn,h,hs,eta,qu,b,alh,hmin,dx,nx,dt,err)
        hn,hs=boundary.h_bound(hn,hs,eta,h_upstm,hs_dwstm,h_dwstm,nx,j_upstm,j_dwstm)
#        print(time,h[nx+1],hn[nx+1])


        if err<errmax:
            break
        l=l+1



#Differentials in Non Advection Phase
    gux=newgrd.ng_u(gux,u,un,nx,dx)
    gux=boundary.gbound_u(gux,nx)

# Advection Phase
    fn=np.zeros([nx2]);gxn=np.zeros([nx2])
#    fn,gxn=mkzero.z0(fn,gxn,nx)
    fn,gxn=cip1d.u_cal1(un,gux,u,fn,gxn,nx,dx,dt)
    un,gux=cip1d.u_cal2(fn,gxn,u,un,gux,nx,dx,dt)
    un=boundary.u_bound(un,hs_up,qp,b_up,nx,j_upstm,j_dwstm,u_upstm,u_dwstm)
    gux=boundary.gbound_u(gux,nx)


# Update u and h
    h=copy.copy(hn); u=copy.copy(un)

# usta and tausta
    usta,tausta=shear.usts(usta,tausta,ie,hs_up,g,sd,nx)
    usta,tausta=boundary.shear_bound(usta,tausta)
    usta_x=boundary.usta_xcal(usta,usta_x,nx)

# Bedload Sediment Transport
    if j_sediment==1 or j_sediment==3:
        qb=qbz.qbcal(qb,tausta,tsc,sgd3_s,nx)
        qb=boundary.qb_bound(qb,alpha_up_sed)

# Supended Sediment
    if j_sediment>=2:
        alpha,beta=phisval.alpha_beta(alpha,beta,usta_x,wf,nx)
        qsu=qsucal.itakura(qsu,tausta,sgd_s,wf,nx)
        qsu=boundary.qsu_bound(qsu)
        qsu_x=boundary.qsu_x_cal(qsu,qsu_x,nx)
# Sediment Source, Diffusion and Advection
        cn,wfcb=source.cn1(c,cn,qsu_x,hs,alpha,wf,wfcb,nx,dt)
        cn,wfcb=boundary.cbound(cn,wfcb,nx)
        epsilon=eddyvis.epcal(epsilon,usta,hs_up,kappa,nx)
        cn,dcdx=diffusion.cdiff(cn,epsilon,dcdx,nx,dx,dt)
        gcx=newgrd.gcxcal(c,cn,gcx,nx,dx)
        fn=np.zeros([nx2]);gxn=np.zeros([nx2])
#        fn,gxn=mkzero.z0(fn,gxn,nx)
        fn,gxn=cip1d.c_cal1(cn,gcx,u,fn,gxn,nx,dx,dt)
        cn,gcx=cip1d.c_cal2(fn,gxn,c,cn,gcx,nx,dx,dt)
        cn,wfcb=boundary.cbound(cn,wfcb,nx)
        gcx=boundary.gbound_c(gcx,nx)

# Bed Deformation
    if j_sediment>=1 and time>=b_start:
        eta,deta,hs=qbz.etacal_bl(eta,deta,qb,qsu_x,h,hs,b_up,b,nx,dx,wf,wfcb,rlam,cn,alpha,j_sediment)
        eta,hs,h=boundary.eta_bound(eta,h,hs,nx)
        eta_up=boundary.eta_up_cal(eta,eta_up,nx)
# Update C
    if j_sediment>=2:
        c=copy.copy(cn)

#Time Step Update
    time=time+dt
    icount=icount+1

    


ani = animation.ArtistAnimation(fig, ims, interval=10)
#plt.show()
ani.save(gifname,writer='imagemagick')
ani.save('res.mp4',writer='ffmpeg')
