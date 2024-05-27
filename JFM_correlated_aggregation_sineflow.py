#%% RUN FIRST : SINE FLOW
# minimum stretching argument for concentration lambda2/(2sigma2) 

import os
# multiprocessing thread
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"


import numpy as np
import matplotlib.pyplot as plt
import pylab
import multiprocessing as mp
from multiprocessing import Pool
from scipy.ndimage.filters import maximum_filter
import time
import h5py 
from scipy import spatial
import matplotlib as mpl
import cv2
from scipy import interpolate
from scipy.interpolate import UnivariateSpline,interp1d,griddata
import glob
from scipy.special import gamma
from matplotlib.colors import ListedColormap





PLOT=False
PAR=False
#% Advection-Diffusion
# Reinitialize Random generator to a new value

def subscript(ax,i,color='k',bg='w',x=0.03,y=0.93,script=['a)','b)','c)','d)']):
	txt=ax.text(x,y,script[i],color=color,transform = ax.transAxes,backgroundcolor=bg)
	return txt


def epsilon_d1(D1):
	# Theoretical value of epsilon
	mu=1/(D1-1)
	sigma2=2*(2-D1)/(D1-1)
	if mu-2*sigma2>0:
		M2=2*mu-2*sigma2
	else:
		M2=mu**2/(2*sigma2)
	return -M2+2

PhaseX=np.array([3.5430681 , 2.88156791, 4.02254445, 4.21064634, 0.91716555,
       4.08047678, 6.04773956, 3.26275256, 1.05590903, 0.81186953,
       3.65448027, 5.61191843, 4.32894932, 0.1966716 , 3.99438952,
       3.50597587, 1.98620531, 2.02850394, 6.24718112, 0.61540386,
       4.75874573, 5.72373971, 6.08673841, 1.72534008, 3.24327037,
       2.06559054, 0.3285389 , 3.67837735, 3.53882218, 1.48451121])
PhaseY=np.array([0.64263038, 1.88206305, 6.10362157, 0.77383875, 3.11129562,
       2.58909493, 0.82151335, 4.5941832 , 4.37914997, 1.70752141,
       0.4360514 , 5.26177372, 0.05536045, 2.69177567, 1.03390734,
       5.2551587 , 4.74335551, 5.78771678, 4.3293486 , 6.27630944,
       1.4302293 , 3.13659403, 3.69292764, 5.8666593 , 3.65222406,
       4.36286177, 2.4452526 , 2.65813945, 2.18423964, 0.70094664])
Angle=np.array([2.49113633, 1.32521453, 1.87795714, 4.24857794, 0.81543564,
       4.60253157, 3.70064055, 5.49679107, 2.88251539, 6.1025857 ,
       6.03430832, 0.07868065, 4.03293744, 3.62125448, 4.72731584,
       5.88348636, 1.03274493, 0.84339187, 4.142372  , 1.43931433,
       6.20672598, 0.26317178, 2.31635786, 5.36595677, 1.19488396,
       3.13125832, 5.10234605, 4.64916775, 0.66600625, 6.04336036])

## SINE FLOW with random rotation, random phase and/or random amplitude at each half period
T=0.5 # Half Period
def vel(x,t,A):
	theta=Angle[int(t/(2.*T))]
	theta=0
	#A=1.2
	#A=1./np.sqrt(2)
	R=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
	Rinv=np.linalg.inv(R)
	x=np.dot(x,R)
	if np.mod(t,T*2)<T:
		vx=np.zeros(x.shape[0])
		vy=A*np.sin(2*np.pi*x[:,0]+PhaseY[int(t/(2.*T))])
		return np.dot(np.vstack((vx,vy)).T,Rinv)
	else:
		vx=A*np.sin(2*np.pi*x[:,1]+PhaseX[int(t/(2*T))])
		vy=np.zeros(x.shape[0])
		return np.dot(np.vstack((vx,vy)).T,Rinv)

def vel_sine(x,t,A):
	theta=0
	#A=1.2
	#A=1./np.sqrt(2)
	R=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
	Rinv=np.linalg.inv(R)
	x=np.dot(x,R)
	if np.mod(t,T*2)<T:
		vx=np.zeros(x.shape[0])
		vy=A*np.sin(2*np.pi*x[:,0]+PhaseY[int(t/(2.*T))])
		return np.dot(np.vstack((vx,vy)).T,Rinv)
	else:
		vx=A*np.sin(2*np.pi*x[:,1]+PhaseX[int(t/(2*T))])
		vy=np.zeros(x.shape[0])
		return np.dot(np.vstack((vx,vy)).T,Rinv)

def vel_sine_longtimes(x,t,aa,ss):
	np.random.seed(seed=ss)
	PhaseX=np.random.rand(200)*2*np.pi
	PhaseY=np.random.rand(200)*2*np.pi
	theta=0
	#A=1.2
	#A=1./np.sqrt(2)
	R=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
	Rinv=np.linalg.inv(R)
	x=np.dot(x,R)
	if np.mod(t,T*2)<T:
		vx=np.zeros(x.shape[0])
		vy=aa*np.sin(2*np.pi*x[:,0]+PhaseY[int(t/(2.*T))])
		return np.dot(np.vstack((vx,vy)).T,Rinv)
	else:
		vx=aa*np.sin(2*np.pi*x[:,1]+PhaseX[int(t/(2*T))])
		vy=np.zeros(x.shape[0])
		return np.dot(np.vstack((vx,vy)).T,Rinv)

def curvature(L,dist):
# Curvature via derivatives
	dx=0.001
	# Normalize distance by dx to avoid precision problems
	dist=dist/dx
	# Compute gradients
	Dx=(np.diff(L[:,0]).T/(dist)).T
	Dy=(np.diff(L[:,1]).T/(dist)).T
	Dxx=(np.diff(Dx).T/(dist[:-1])).T
	Dyy=(np.diff(Dy).T/(dist[:-1])).T
	Dxx1=(np.diff(Dx).T/(dist[1:])).T
	Dyy1=(np.diff(Dy).T/(dist[1:])).T
	k=np.abs(Dx[:-1]*Dyy-Dy[:-1]*Dxx)/(Dx[:-1]**2.+Dy[:-1]**2.)**(3/2.)
	k1=np.abs(Dx[1:]*Dyy1-Dy[1:]*Dxx1)/(Dx[1:]**2.+Dy[1:]**2.)**(3/2.)
	#return (np.hstack((0,k))+np.hstack((k1,0)))/2.
	return np.maximum(np.hstack((0,k)),np.hstack((k,0)))


def run_DSM(Lmax,aa,s,STOP_ON_LMAX=False):
	#STOP_ON_LMAX allows to stop simulation immediately when Lmax is reached (instead of finishing the time step)
	import h5py
	print('running DSM...')
	l0=0.3
	#l0=1
	#% Advection parameters
	INTERPOLATE='LINEAR'
	CURVATURE='DIFF'

	PLOT=False
	dx=0.001 # Maximum distance between points, above which the line will be refined
	alpha=200*dx/np.pi
	dt=0.25 # dt needs to be small compare to the CFL condition
	npar=6 # Number of parallel processors
	#tmax=Tmax # Maximum advection time

	#A=2

	# Initial segment position and distance
	x=np.linspace(0,2*np.pi,int(1.8e3))
	#L=np.array([radius*np.cos(x),radius*np.sin(x)],dtype=np.float64).T
	#L[0,:]=L[-1,:]
	n=int(1.8e3)
	L=np.array([np.linspace(-l0/2,l0/2,n),np.zeros(n)],dtype=np.float64).T

	L=np.array(L)
	weights=np.ones(L.shape[0]-1)
	weights=weights/np.sum(weights)
	#L=2*(np.random.rand(2)-0.5)+np.array([x,l0*np.sin(x)],dtype=np.float64).T

	dist_old=np.sum(np.diff(L,axis=0)**2,1)**0.5
	weights=np.ones(dist_old.shape)
	W=weights
	# Initial segment width
	S=np.ones(L.shape[0]-1)
	# Initial Wrapped Time
	wrapped_time=np.zeros(L.shape[0]-1)
	
	# Initialization
	t=0
	ct=time.time()

	# Prepare Curvature ==========================================================
	if CURVATURE=='SPLINE':
		tck, u = interpolate.splprep([L[:,0],L[:,1]],s=0,per=1)
		umid=(u[1:]+u[:-1])/2.
		Dx,Dy = interpolate.splev(umid,tck,der=1)
		DDx,DDy = interpolate.splev(umid, tck,der=2)
		kappa_old=np.abs(Dx*DDy-Dy*DDx)/(Dx**2.+Dy**2.)**(3/2.)
	if CURVATURE=='DIFF':
		kappa_old=curvature(L,dist_old)
	#=============================================================================
	if STOP_ON_LMAX:
		RUN=(len(L)<Lmax)
	else:
		RUN=(len(L)<Lmax)|(t!=int(t))
	# MAIN PARALLEL LOOP #######################
	while RUN:
		
		v=vel_sine_longtimes(L,t,aa,s)
		L+=v*dt
		
		# Compute stretching rates and elongations
		dist_new=np.sum(np.diff(L,axis=0)**2,1)**0.5
		gamma=dist_new/dist_old # elongation
		#gamma=np.maximum(dist_new/dist_old,dist_old/dist_new)
		S=S/gamma
		#wrapped_time=wrapped_time+dt*(1./S)**2.
		# Force positive elongation
		#rho1=np.abs(1./S-1.)+1.
		rho1=np.maximum(1/S,S)
		#rho1=1/S
		wrapped_time=wrapped_time+dt*(rho1)**2.
		#Force periodicity
		#L[0,:]=L[-1,:]
		
		# Compute new curvature
	# =============================================================================
		if CURVATURE=='SPLINE':
			tck, u = interpolate.splprep([L[:,0],L[:,1]],s=0,per=1)
			umid=(u[1:]+u[:-1])/2.
			Dx,Dy = interpolate.splev(umid,tck,der=1)
			DDx,DDy = interpolate.splev(umid, tck,der=2)
			kappa_new=np.abs(Dx*DDy-Dy*DDx)/(Dx**2.+Dy**2.)**(3/2.)
		if CURVATURE=='DIFF':
			kappa_new=curvature(L,dist_new)
	# =============================================================================
		
		# Statistics on curvature increments
		dkappa=np.log(kappa_new)-np.log(kappa_old)
		dlKMean=np.average(dkappa,weights=weights)
		dlKVar=np.average((dkappa-dlKMean)**2./(1.-np.sum(weights**2.)),weights=weights) # Variance of Elongation

	# =============================================================================
	# REFINEMENT METHODS
	# =============================================================================
		# No refinement
	# =============================================================================
		if INTERPOLATE=='NO':
			dist_old=np.sum(np.diff(L,axis=0)**2,1)**0.5
			kappa_old=kappa_new
			W=np.ones(kappa_new.shape)
			W=W/np.sum(W)
	# =============================================================================
	# New interpolation based on Meunier's rule
	# =============================================================================
		if INTERPOLATE=='SPLINE':
			#Dl=np.sqrt(Dx**2.+Dy**2.)*np.diff(u)
			Dl=np.sum(np.diff(L,axis=0)**2,1)**0.5
			F=np.hstack((0,np.cumsum((1.+alpha*kappa_new)*Dl)))
			Finter=np.arange(F[0],F[-1],dx)
			uinter=np.interp(Finter,F,u)
			x,y=interpolate.splev(uinter,tck,der=0)
			L=np.vstack((x,y)).T
			dist_old=np.sum(np.diff(L,axis=0)**2,1)**0.5
			S=np.interp(uinter[1:],u[1:],S)
			gamma=np.interp(uinter[1:],u[1:],gamma)
			dkappa=np.interp(uinter[1:],u[1:],dkappa)
			uintermid=(uinter[1:]+uinter[:-1])/2.
			Dx,Dy = interpolate.splev(uintermid,tck,der=1)
			DDx,DDy = interpolate.splev(uintermid, tck,der=2)
			kappa_old=np.abs(Dx*DDy-Dy*DDx)/(Dx**2.+Dy**2.)**(3/2.)
			wrapped_time=np.interp(uinter[1:],u[1:],wrapped_time)
		# Weights
			weights=weights+np.log(dx/np.diff(F))
			weights=np.interp(uinter[1:],u[1:],weights)							
			W=np.exp(weights)/np.sum(np.exp(weights))
			#W=S*dist_old/np.sum(S*dist_old)
			#W=S*dist_old/np.sum(S*dist_old)
	# =============================================================================
	# =============================================================================
	# Refinement of elongated regions only
	# =============================================================================
		if INTERPOLATE=='LINEAR':
			ref=np.where(dist_new>dx)[0]
			dist_new[ref]=dist_new[ref]/2. # So that the sum of distance stays the same when adding new points
			dist_old=np.insert(dist_new,ref+1,dist_new[ref],axis=0)
			weights[ref]=weights[ref]/2.
			weights=np.insert(weights,ref+1,weights[ref],axis=0)
			L=np.insert(L,ref+1,(L[ref+1,:]+L[ref,:])/2.,axis=0)
			S=np.insert(S,ref+1,S[ref],axis=0)
			gamma=np.insert(gamma,ref+1,gamma[ref],axis=0)
			kappa_old=curvature(L,dist_old)
			wrapped_time=np.insert(wrapped_time,ref+1,wrapped_time[ref],axis=0)
			dkappa=np.insert(dkappa,ref+1,dkappa[ref],axis=0)
			#W=S*dist_old/np.sum(S*dist_old) # points regularly spread in the curve have a weights inersely proportional to their elongation
			#W=S/np.sum(S)
			W=weights/np.sum(weights)
			#print np.sum(W)
	# =============================================================================
		# Update time
		t=t+dt
		if STOP_ON_LMAX:
			RUN=(len(L)<Lmax)
		else:
			RUN=(len(L)<Lmax)|(t!=int(t))
		#print('Time:',t,' - Npts:',len(L), '- Cmax:',np.mean(1./np.sqrt(1.+wrapped_time/1000)),np.max(kappa_new))
	# End of MAIN LOOOP #######################
	#print('Computation time:', time.time() -ct)
	return L,S,wrapped_time,W,t

from scipy.fft import fft, ifft,fftfreq

def run_DNS_spectral(a,n,ss,t,D,dt=0.5):
	np.random.seed(seed=ss)
	PhaseX=np.random.rand(200)*2*np.pi+np.pi
	PhaseY=np.random.rand(200)*2*np.pi+np.pi
#	n=int(2**12) # Number of grid points
	# =============================================================================
	tmax=t # number of periods to compute
	#dt=1/1  # Discretisation of time step
	X,Y=np.meshgrid(np.arange(n),np.arange(n))
	#sigma=np.sqrt(2*D)*n
	# =============================================================================
	# iNITIAL condition
	l0=0.3
	radius=0.01
	#C[int(n/2):,:]=0.99 #half_plane
	#C[(X-n/2)**2+(Y-n/2)**2<(n/10)**2]=1
	# Initial condition corresponding to a single diffusive strip
	# Read Random angles
	C=np.zeros((n,n))					
	IdX=np.where((X/n<l0/2)|(X/n>(1-l0/2)))
	C[IdX]=np.exp(-(Y[IdX]/n)**2/(radius**2))+np.exp(-((Y[IdX]-n)/n)**2/(radius**2))
	# Save Variance and mean of C
	# Save Variance and mean of C
	VarC=[]
	MeanC=[]
	VarC.append(np.var(C))
	
	# Wavevector
	ky=2*np.pi*np.tile(fftfreq(C.shape[0], d=1.0/C.shape[0]),(C.shape[1],1)).T
	kx=2*np.pi*np.tile(fftfreq(C.shape[1], d=1.0/C.shape[0]),(C.shape[0],1))
	k=np.sqrt(ky**2+kx**2)
	
	# Start from the fourier transform of concentration field
	fC=fft(fft(C,axis=0),axis=1)
	for t in range(tmax):
		print(t)
		vX=a*np.sin(Y/n*2*np.pi+PhaseX[t])
		vY=a*np.sin(X/n*2*np.pi+PhaseY[t])
		# Half period
		# 2nd Half period
		
		for t in np.arange(0,0.5,dt):
			fCy = ifft(fC,axis=1)
			fC=fft(np.exp(1j*ky*vY*dt)*fCy,axis=1)*np.exp(-D*k**2*dt)

		for t in np.arange(0,0.5,dt):
			fCx = ifft(fC,axis=0)
			dcx=np.exp(1j*kx*vX*dt)*fCx
			fC=fft(dcx,axis=0)*np.exp(-D*k**2*dt)#			# With source term

		VarC.append(np.mean(np.abs(fC)**2)/n**2)
		MeanC.append(np.abs(fC[0,0])/n)
	return np.real(ifft(ifft(fC,axis=0),axis=1)),np.array(MeanC),np.array(VarC)


#%% Figure 3 :  Run DSM and plot at multiple scales

A=0.5
L,S,wrapped_time,W,t=run_DSM(1e7, A, 3)

Lmod=np.hstack((np.mod(L[:,1],1).reshape(-1,1),np.mod(-L[:,0],1).reshape(-1,1)))
idmod=np.where(np.nansum(np.diff(Lmod,axis=0)**2.,axis=1)>0.01)[0]
Lmod[idmod,:]=np.nan

fig,ax=plt.subplots(1,3,figsize=(6.5,2))
x1=[0,1]
y1=[0,1]
#plt.plot(np.mod(L[:,1],1),np.mod(-L[:,0],1),'k.',alpha=1.0,linewidth=0.2,markersize=0.1)
ax[0].plot(Lmod[:,0],Lmod[:,1],'k-',linewidth=0.1)
ax[0].set_xlim(x1)
ax[0].set_ylim(y1)
ax[0].set_xticks(x1)
ax[0].set_yticks(y1)

x1=[0.3,0.4]
y1=[0.3,0.4]
ax[0].plot([x1[0],x1[1],x1[1],x1[0],x1[0]],[y1[0],y1[0],y1[1],y1[1],y1[0]],'r-',linewidth=1.5)


#plt.plot(np.mod(L[:,1],1),np.mod(-L[:,0],1),'k.',alpha=1.0,linewidth=0.2,markersize=0.1)
ax[1].plot(Lmod[:,0],Lmod[:,1],'k-',linewidth=0.1)
ax[1].set_xlim(x1)
ax[1].set_ylim(y1)

ax[1].set_xticks(x1)
ax[1].set_yticks(y1)


x1=[0.36,0.37]
y1=[0.32,0.33]
ax[1].plot([x1[0],x1[1],x1[1],x1[0],x1[0]],[y1[0],y1[0],y1[1],y1[1],y1[0]],'r-',linewidth=1.5)

ax[2].plot(Lmod[:,0],Lmod[:,1],'k-',linewidth=0.1)
ax[2].set_xlim(x1)
ax[2].set_ylim(y1)
ax[2].set_xticks(x1)
ax[2].set_yticks(y1)

fig.subplots_adjust(wspace=0.2)

fig.savefig('sine_c_fractalsA={:1.1f}.jpg'.format(A),bbox_inches='tight',dpi=600)



#%% Figure 4 : Comparison DNS DSM 1

A=0.8
seed=3
n=int(2**11)

L,S,wrapped_time,W,t=run_DSM(1e7, A, seed)
dist_old=np.sum(np.diff(L,axis=0)**2,1)**0.5

Lyap=np.loadtxt('Sine_Lyap.txt')
lyap=np.interp(A,Lyap[:,0],Lyap[:,1])


sB1=1/150
sB2=1/50

factor=1/np.sqrt(2*np.pi) # to link aggregation scale to Batchelor scale

D=(sB1*factor)**2*lyap/2
#C1,mC,vC=run_DNS(A,n,seed,int(t),D)
C1,mC,vC=run_DNS_spectral(A,n,seed,int(t),D)

D=(sB2*factor)**2*lyap/2
#C0,mC,vC=run_DNS(A,n,seed,int(t),D)
C0,mC,vC=run_DNS_spectral(A,n,seed,int(t),D)

# plt.figure(figsize=(2,2))
# plt.plot(np.mod(L[:,0],1),np.mod(L[:,1],1),'.')

# plt.figure()
# plt.imshow(C1)

# plt.figure()
# plt.imshow(C1s)
# plot


cmap=plt.cm.viridis
radius=0.01
factor=1.0
clim=[-7,-3]
fig,ax=plt.subplots(2,3,figsize=(6,4))
sB=sB1*factor

C1dsm=np.histogram2d(np.mod(L[1:,1],1),np.mod(L[1:,0],1),np.arange(0,1,sB),weights=S*dist_old/sB*np.sqrt(np.pi)/sB*radius)[0]
#C1dsm[C1dsm==0]=np.exp(clim[0])
i0=ax[0,1].imshow(np.log(C1dsm),clim=clim,cmap=cmap)
ax[0,0].axis('off')
# cc=fig.colorbar(i0,ax=ax[1,1],location='right',shrink=0.8)
# cc.set_label(r'$\log c$', color='w')
# # set colorbar tick color
# cc.ax.yaxis.set_tick_params(color='w')
# # set colorbar edgecolor 
# cc.outline.set_edgecolor('w')

# plt.setp(plt.getp(cc.ax.axes, 'yticklabels'), color='w')

sB=sB2*factor
C0dsm=np.histogram2d(np.mod(L[1:,1],1),np.mod(L[1:,0],1),np.arange(0,1,sB),weights=S*dist_old/sB*np.sqrt(np.pi)/sB*radius)[0]
#C0dsm[C0dsm==0]=np.exp(clim[0])
ax[1,1].imshow(np.log(C0dsm),clim=clim,cmap=cmap) #sb2 line 1
ax[1,0].axis('off')

ax[0,0].imshow(np.log(C1),clim=clim,cmap=cmap)
ax[0,1].axis('off')

ax[1,0].imshow(np.log(C0),clim=clim,cmap=cmap)
ax[1,1].axis('off')

xt=0.02
yt=1.05
ax[0,1].text(xt,yt,'a.2) Eq. (3.4)',c='k',transform = ax[0,0].transAxes)#,backgroundcolor='w')
ax[0,0].text(xt,yt,'a.1) DNS',c='k',transform = ax[0,1].transAxes)#,backgroundcolor='w')

ax[1,1].text(xt,yt,'b.2) Eq. (3.4)',c='k',transform = ax[1,0].transAxes)#,backgroundcolor='w')
ax[1,0].text(xt,yt,'b.1) DNS',c='k',transform = ax[1,1].transAxes)#,backgroundcolor='w')

ax[1,2].text(xt,yt,'a.3) pdf',c='k',transform = ax[0,2].transAxes)#,backgroundcolor='w')
ax[1,2].text(xt,yt,'b.3) pdf',c='k',transform = ax[1,2].transAxes)#,backgroundcolor='w')
# 3/ Compare pdf

C1dsm_hist=C1dsm.flatten()[C1dsm.flatten()>0]
hdsm,x1=np.histogram(np.log(C1dsm_hist),100,density=True)
h,x2=np.histogram(np.log(C1.flatten()),100,density=True)

#plt.figure()
ax[0,2].plot(x2[1:],h,'ko',label='DNS')
ax[0,2].plot(x1[1:],hdsm,'r*',label='Eq. (3.4)') #sb2
ax[0,2].set_yscale('log')

ax[0,2].set_xlabel(r'$\log c$')
#ax[1,2].set_ylabel(r'pdf')
ax[0,2].legend(loc=3,framealpha=1,fontsize=8)
ax[0,2].set_ylim([1e-4,2])


C0dsm_hist=C0dsm.flatten()[C0dsm.flatten()>0]
hdsm,x1=np.histogram(np.log(C0dsm_hist),100,density=True) # Pe sb2
h,x2=np.histogram(np.log(C0.flatten()),100,density=True)

ax[1,2].plot(x2[1:],h,'ko',label='DNS')
ax[1,2].plot(x1[1:],hdsm,'r*',label='Eq. (3.4)') #sb2
ax[1,2].set_yscale('log')

ax[1,2].set_ylabel('pdf')


ax[1,2].set_xlabel(r'$\log c$')
#ax[0,2].set_ylabel(r'pdf')
ax[1,2].legend(loc=3,framealpha=1,fontsize=8)
ax[1,2].set_ylim([1e-4,2])

ax[1,2].set_xlim([-12,0])
ax[0,2].set_xlim([-12,0])
ax[0,2].set_ylabel('pdf')

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.2)

plt.savefig('DNS-LAG.pdf',bbox_inches='tight')


#%% Figure 5a : Run DSM and plot at multiple scales

A=0.5
L,S,wrapped_time,W,t=run_DSM(1e7, A, 3)

Lmod=np.hstack((np.mod(L[:,1],1).reshape(-1,1),np.mod(-L[:,0],1).reshape(-1,1)))
idmod=np.where(np.nansum(np.diff(Lmod,axis=0)**2.,axis=1)>0.01)[0]
Lmod[idmod,:]=np.nan

fig,ax=plt.subplots(1,3,figsize=(6.5,2))
x1=[0,1]
y1=[0,1]
#plt.plot(np.mod(L[:,1],1),np.mod(-L[:,0],1),'k.',alpha=1.0,linewidth=0.2,markersize=0.1)
ax[0].plot(Lmod[:,0],Lmod[:,1],'k-',linewidth=0.1)
ax[0].set_xlim(x1)
ax[0].set_ylim(y1)
ax[0].set_xticks(x1)
ax[0].set_yticks(y1)

x1=[0.3,0.4]
y1=[0.3,0.4]
ax[0].plot([x1[0],x1[1],x1[1],x1[0],x1[0]],[y1[0],y1[0],y1[1],y1[1],y1[0]],'r-',linewidth=1.5)


#plt.plot(np.mod(L[:,1],1),np.mod(-L[:,0],1),'k.',alpha=1.0,linewidth=0.2,markersize=0.1)
ax[1].plot(Lmod[:,0],Lmod[:,1],'k-',linewidth=0.1)
ax[1].set_xlim(x1)
ax[1].set_ylim(y1)

ax[1].set_xticks(x1)
ax[1].set_yticks(y1)


x1=[0.36,0.37]
y1=[0.32,0.33]
ax[1].plot([x1[0],x1[1],x1[1],x1[0],x1[0]],[y1[0],y1[0],y1[1],y1[1],y1[0]],'r-',linewidth=1.5)

ax[2].plot(Lmod[:,0],Lmod[:,1],'k-',linewidth=0.1)
ax[2].set_xlim(x1)
ax[2].set_ylim(y1)
ax[2].set_xticks(x1)
ax[2].set_yticks(y1)

fig.subplots_adjust(wspace=0.2)

fig.savefig('sine_c_fractalsA={:1.1f}.jpg'.format(A),bbox_inches='tight',dpi=600)

#%% Figure 7 : Run DSM and plot Coarse grained N, Cmax

A=0.4
seed=3

L,S,wrapped_time,W,t=run_DSM(1e7, A, seed)

#DNS
D=(1/50)**2/2*0.1
#C0,mC,vC=run_DNS(A,int(2**11),seed,int(t),D)

clim=[-1,8]
fig,ax=plt.subplots(1,2,figsize=(4,2))
dist_old=np.sum(np.diff(L,axis=0)**2,1)**0.5
sB=1/200
N=np.histogram2d(np.mod(L[1:,0],1),np.mod(L[1:,1],1),np.arange(0,1,sB),weights=dist_old/sB*np.sqrt(2))[0]
N[N==0]=np.exp(clim[0])
ax[0].imshow(np.log(N),clim=clim)
ax[0].axis('off')

#
#plt.setp(plt.getp(cc.ax.axes, 'yticklabels'), color='k')
sB=1/50
N=np.histogram2d(np.mod(L[1:,0],1),np.mod(L[1:,1],1),np.arange(0,1,sB),weights=dist_old/sB*np.sqrt(2))[0]
N[N==0]=np.exp(clim[0])
i0=ax[1].imshow(np.log(N),clim=clim)
ax[1].axis('off')

cc=fig.colorbar(i0,ax=ax[:],location='right',shrink=0.7)
cc.set_label(r'$\log n$', color='k')
# set colorbar tick color
cc.ax.yaxis.set_tick_params(color='k')
# set colorbar edgecolor 
cc.outline.set_edgecolor('k')

subscript(ax[0], 0,color='k')
subscript(ax[1], 1,color='k')
#plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.02, hspace=0.02)

plt.savefig('N-LAG.pdf',bbox_inches='tight')
#%% Figure 9 :mean(log n) as a fct of time : Plot
l0=0.3
import scipy.special
VarA=np.loadtxt('Sine_N_t.txt')
Lyap=np.loadtxt('Sine_Lyap.txt')
D1=np.loadtxt('Sine_D1.txt')
AA=np.unique(VarA[:,0])
AA=np.array([0.4,0.6,0.8,1.2,1.8])
SB=np.unique(VarA[:,2])
M=['d','o','s','*']
fig=plt.figure(figsize=(2,2))
for j,A in enumerate(AA):
	idA=np.where(VarA[:,0]==A)[0]
	Var=VarA[idA,1:]
	lyap=np.interp(A,Lyap[:,0],Lyap[:,1])
	sigma2=np.interp(A,Lyap[:,0],Lyap[:,2])
	Var=np.array(Var)
	D2=np.interp(A,D1[:,0],D1[:,2])
	for ii,sB in enumerate(SB):
		idsB=np.where(Var[:,1]==sB)[0]
		tagg=1#(lyap+sigma2/2)#np.log(1/(sB*l0))/(lyap+sigma2/2)
	#	tagg=1
		ni=sB#np.log(1/sB)*(2*(d1max-d1))#/(d1-d1max+1)**4
		#nivar=2*(2-d1)/(d1-1)**1*sB**2*(-np.log(sB))
		nivar=sB**2.*(sB**(D2-1-1)-1)
		plt.plot(np.log(Var[idsB,2]),Var[idsB,3]/ni,M[ii]+'-',color=plt.cm.cool(((A-AA.min())/(AA.max()-AA.min()))))#,label=r'$s_a=1/{:1.0f}$'.format(1/sB))
		plt.plot(np.log(Var[idsB,2]),np.sqrt(Var[idsB,4]/nivar),M[ii]+'--',color=plt.cm.cool(((A-AA.min())/(AA.max()-AA.min()))))#,label=r'$s_a=1/{:1.0f}$'.format(1/sB))

t=np.linspace(1,14,100)
plt.yscale('log')
#plt.ylim([0,10])
plt.xlabel('$\log L(t)$')
plt.plot([],[],'--',color='k',label=r'$\sigma_n \mathcal{A} / s_a / \sqrt{s_a^{D_2-2}-1}$')
plt.plot([],[],'-',color='k',label=r'$\mu_{n} \mathcal{A} / s_a$')
plt.plot(t,np.exp(t),'-',color='k',label=r'$L(t)$',linewidth=2,zorder=-10)
[plt.plot([],[],M[k],color='k',label=r'$s_a=1/{:1.0f}$'.format(1/sb)) for k,sb in enumerate(SB)]

#plt.plot(t,((1+scipy.special.erf(t-2.5))/2)**0.2,'-',color='k',label=r'$\sim \mathrm{erf} (t)^{0.2}$')

subscript(plt.gca(),2,x=-0.05,y=0.95)

plt.legend(loc=2,fancybox='off',frameon=0,fontsize=6)
#plt.ylim(1e-3,5)
#plt.xlim(-3,8)

ax2 = fig.add_axes([0.5, 0.3, 0.4, 0.02])
import matplotlib as mpl
#norm = mpl.colors.Normalize(vmin=T_all[0]*u_pore/d, vmax=T_all[-1]*u_pore/d)
norm = mpl.colors.Normalize(vmin=AA.min(), vmax=AA.max())
cb1 = mpl.colorbar.ColorbarBase(ax2,cmap=plt.cm.cool,norm=norm,
								orientation='horizontal')
cb1.set_label(r'$A$', color='k',size=8,labelpad=0)
cb1.set_ticks([0.4,0.8,1.2,1.8])



plt.savefig('Sine_meanN_t.pdf',bbox_inches='tight')

plt.figure()
plt.plot(D1[:,1],Lyap[:,2]-Lyap[:,1]*2*(2-D1[:,1]))
plt.plot(D1[:,1],Lyap[:,2]/Lyap[:,1])
plt.plot(D1[:,1],-2*D1[:,1]+4.35)

#%% Figure10 c :p(N) for several times
from scipy import special

x0=np.array([0.1])
var_sa=0
nb=20
s0=0.05
D=1e-4
Lmax=np.array([1e5,1e6,1e7])
M=['d','o','s','*']

Lyap=np.loadtxt('Sine_Lyap.txt')
D1=np.loadtxt('Sine_D1.txt')
A=0.4
s=4

sB=1/100
#TT=np.arange(0,20,5)
hE=[]
Time=[]
Ltot=[]
for i,lmax in enumerate(Lmax):
	L1,S1,wrapped_time,W1,t1=run_DSM(lmax,A,s,STOP_ON_LMAX=True)
	Lmod=np.mod(L1,1)
	dist_old=np.sum(np.diff(L1,axis=0)**2,1)**0.5
	Ltot.append(np.sum(dist_old))
	he=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1+sB,sB),weights=dist_old/sB,density=False)[0]
	he=np.uint16(he)
# =================Eulerian =================================================
	bin_n=np.logspace(-1,np.log10(he.max()+5),nb)
	hE.append(np.histogram(he,bin_n,density=True))
	Time.append(t1)

plt.figure(figsize=(2,2))
Time=np.array(Time)

for i,lmax in enumerate(Lmax):
	plt.plot(hE[i][1][:-1],hE[i][0],M[i],color=plt.cm.viridis(Time[i]/Time.max()),label='$t={:1.0f}$'.format(Time[i]),fillstyle='full')
# =============================================================================
	lyap=np.interp(A,Lyap[:,0],Lyap[:,1])
	sigma2=np.interp(A,Lyap[:,0],Lyap[:,2])
	d1=np.interp(A,D1[:,0],D1[:,1])
	d2=d1
	# Theory
	mu_n=sB*Ltot[i]
	sigma2_n=mu_n**2.*(sB**(d2-2)-1)
	# gamma
	k=mu_n**2./sigma2_n
	theta=sigma2_n/mu_n
	n=np.logspace(0,5,200)
	pdfn=1/special.gamma(k)/theta**k*n**(k-1)*np.exp(-n/theta)
	plt.plot(n,pdfn,'-',color=plt.cm.viridis(Time[i]/Time.max()))
	
	plt.xscale('log')
	plt.yscale('log')
	plt.xlabel('$n$')
	plt.ylabel('$P(n,t)$')
plt.ylim([1e-5,1])
plt.xlim([1e0,1e3])
plt.legend(fontsize=8)
subscript(plt.gca(),2)
plt.savefig('figure10c.pdf',bbox_inches='tight')
#%%Fig 11a : % Run DSM and plot Coarse grained N, Cmax

A=0.4
seed=3

L,S,wrapped_time,W,t=run_DSM(1e7, A, seed)

#DNS
D=(1/50)**2/2*0.1
#C0,mC,vC=run_DNS(A,int(2**11),seed,int(t),D)

clim=[-1,8]
fig,ax=plt.subplots(1,2,figsize=(4,2))
dist_old=np.sum(np.diff(L,axis=0)**2,1)**0.5
sB=1/200
N=np.histogram2d(np.mod(L[1:,0],1),np.mod(L[1:,1],1),np.arange(0,1,sB),weights=dist_old/sB*np.sqrt(2))[0]
N[N==0]=np.exp(clim[0])
ax[0].imshow(np.log(N),clim=clim)
ax[0].axis('off')

plt.setp(plt.getp(cc.ax.axes, 'yticklabels'), color='k')
sB=1/50
N=np.histogram2d(np.mod(L[1:,0],1),np.mod(L[1:,1],1),np.arange(0,1,sB),weights=dist_old/sB*np.sqrt(2))[0]
N[N==0]=np.exp(clim[0])
i0=ax[1].imshow(np.log(N),clim=clim)
ax[1].axis('off')

cc=fig.colorbar(i0,ax=ax[:],location='right',shrink=0.7)
cc.set_label(r'$\log n$', color='k')
# set colorbar tick color
cc.ax.yaxis.set_tick_params(color='k')
# set colorbar edgecolor 
cc.outline.set_edgecolor('k')

subscript(ax[0], 0,color='k')
subscript(ax[1], 1,color='k')
#plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.02, hspace=0.02)


#%
N=np.histogram2d(np.mod(L[1:,0],1),np.mod(L[1:,1],1),np.arange(0,1,sB),weights=dist_old/sB*np.sqrt(2))[0]
logrho=np.histogram2d(np.mod(L[1:,0],1),np.mod(L[1:,1],1),np.arange(0,1,sB),weights=np.log(1/S)*dist_old/sB*np.sqrt(2))[0]
c=np.histogram2d(np.mod(L[1:,0],1),np.mod(L[1:,1],1),np.arange(0,1,sB),weights=S*dist_old/sB*np.sqrt(2))[0]

fig,ax=plt.subplots(1,3,figsize=(5,2))
i0=ax[0].imshow(np.log(N+1e-8),extent=[0,1,0,1],clim=[1,8])
cc0=fig.colorbar(i0,ax=ax[0],location='bottom',aspect=10,shrink=0.8,label=r'$\log n$',pad=0.05)
i1=ax[1].imshow(logrho/(N+1),extent=[0,1,0,1],clim=[10,16])
cc1=fig.colorbar(i1,ax=ax[1],location='bottom',aspect=10,shrink=.8,label=r'$\langle \log \rho | n\rangle $',pad=0.05)
i2=ax[2].imshow(np.log(c+1e-8),extent=[0,1,0,1],clim=[-7,-2])
cc2=fig.colorbar(i2,ax=ax[2],location='bottom',aspect=10,shrink=0.8,label=r'$\log c$',pad=0.05)
#i2=ax[3].imshow(np.log(C0),extent=[0,1,0,1],clim=[-7,-2])
#cc2=fig.colorbar(i2,ax=ax[3],location='bottom',aspect=10,shrink=0.8,label=r'$\log c$ (DNS)',pad=0.05)

ax[0].axis('off')
ax[1].axis('off')
ax[2].axis('off')
#ax[3].axis('off')

script=['b.1','b.2','b.3','b.4']
subscript(ax[0],0,color='k',script=script)#,x=0.5,y=1.02)
subscript(ax[1],1,color='k',script=script)#,x=0.5,y=1.02)
subscript(ax[2],2,color='k',script=script)#,x=0.5,y=1.02)
#subscript(ax[3],3,color='k',script=script,x=0.5,y=1.02)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, hspace=0.02, wspace=0.02)

fig.savefig('sine_N_LAG.pdf',bbox_inches='tight')

#%% FIgure 12 ** Grid based scalings



s=3
A=0.8

L,S,wrapped_time,W,t=run_DSM(1e7,A,s,STOP_ON_LMAX=True)

Lmod=np.mod(L,1)
dist_old=np.sum(np.diff(L,axis=0)**2,1)**0.5

sB=1/100
N=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1+sB,sB)
									 ,weights=dist_old/sB,density=False)[0]
S1=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1+sB,sB)
									 ,weights=S*dist_old/sB,density=False)[0]
#	C1=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1,sB)
#										 ,weights=Cmax*dist_old/sB,density=False)[0]
S2=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1+sB,sB)
									 ,weights=S**2.*dist_old/sB,density=False)[0]
#	S4=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1,sB)
#										 ,weights=S**4.*dist_old/sB,density=False)[0]
#	C2=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1,sB)
#										 ,weights=Cmax**2.*dist_old/sB,density=False)[0]
logrho=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1+sB,sB)
									 ,weights=-np.log(S)*dist_old/sB,density=False)[0]
logrho2=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1+sB,sB)
									 ,weights=np.log(S)**2.*dist_old/sB,density=False)[0]

nb=30

D1=np.loadtxt('Sine_D1.txt')
d1=np.interp(A,D1[:,0],D1[:,1])

from scipy.stats import linregress
Mlogrho=(logrho/N)
Vlogrho=logrho2/N-(Mlogrho)**2.
MS1=S1/N

n=1
Plogrho = linregress((Mlogrho[N>n].flatten()),np.log(N[N>n].flatten()))
#plt.plot(Mlogrho[N>n].flatten(),np.log(N[N>n].flatten()),'b.',alpha=0.01)
np.polyfit(Mlogrho[N>n].flatten(),np.log(N[N>n].flatten()),1)
plt.figure()
rx=np.linspace(0,20,10)

nagg=N[N>1].flatten()
inv_rho=MS1[N>1].flatten()
logrhoagg_mean=Mlogrho[N>1].flatten()

fig,ax=plt.subplots(2,2,figsize=(3,3),sharey=False,sharex=False)
subscript(ax[0,0], 0,y=0.88,script=['a.1','a.2','b.1','b.2'])
subscript(ax[0,1], 1,y=0.88,script=['a.1','a.2','b.1','b.2'])
subscript(ax[1,0], 2,y=0.88,script=['a.1','a.2','b.1','b.2'])
subscript(ax[1,1], 3,y=0.88,script=['a.1','a.2','b.1','b.2'])
nbin=np.log(np.unique(np.uint32(np.logspace(0,3.5,40)))-0.5)
rhobin=np.linspace(logrhoagg_mean.min(), logrhoagg_mean.max(),nb)
ax[1,1].hist2d(logrhoagg_mean,np.log(nagg),[rhobin,nbin],cmap=plt.cm.Greys)
#Plogrho = linregress(logrhoagg_mean,np.log(nagg))
#Plogrho = linregress(np.log(nagg),logrhoagg_mean)
ax[1,1].set_xlabel(r'$\langle \log  \rho | n \rangle $')
ax[1,1].patch.set_facecolor(plt.cm.Greys(0))
rx=np.linspace(5,15,3)
ax[1,1].plot(rx,rx-5.5,'r--',label=r'$1$')
ax[1,1].plot(rx,rx*(d1-1)-4.2,'r-',label=r'$D_1-1$')
ax[1,1].set_xlim([5,15])
ax[1,1].set_ylim([0,8])
legend = ax[1,1].legend(frameon=False,fontsize=6,loc=4)

nbin=np.log(np.unique(np.uint32(np.logspace(0,3,40)))-0.5)
invrhobin=np.log(np.logspace(np.log10(inv_rho.min()),0,50))
ax[1,0].hist2d(np.log(inv_rho),np.log(nagg),[invrhobin,nbin],cmap=plt.cm.Greys)
ax[1,0].set_ylabel(r'$\log n $')
ax[1,0].set_xlabel(r'$\log\langle \rho^{-1} | n \rangle$')
ax[1,0].patch.set_facecolor(plt.cm.Greys(0))
rx=np.linspace(-15,-5,3)
ax[1,0].plot(rx,-rx-6,'r-',label=r'$-1$')
legend = ax[1,0].legend(frameon=False,fontsize=6)
ax[1,0].set_xlim([-15,-5])
ax[1,0].set_ylim([0,8])
# Load Baker data to plot together

a=0.1
nu=1+2*np.log(2)/np.log(a**(-1)+(1-a)**(-1))
Df=nu


Baker=np.loadtxt('Baker_Scaling_N_invrho_logrho_a{:1.2f}.txt'.format(a))
nagg=Baker[:,0]
s_mean=Baker[:,1]
logrho=Baker[:,2]
k=nb

nbin=np.log(np.unique(np.uint32(np.logspace(np.log10(np.min(nagg))*0.9,np.log10(np.max(nagg))*1.1,k)))-0.5)
sbin=np.log(np.logspace(np.log10(s_mean.min())*1.1,np.log10(s_mean.max())*0.9,k))
ax[0,0].hist2d(np.log(s_mean),np.log(nagg),[sbin,nbin],norm=mpl.colors.LogNorm(),cmap=plt.cm.Greys)
ax[0,0].patch.set_facecolor(plt.cm.Greys(0))
rx=sbin
ax[0,0].plot(rx,-rx-4.5,'r-',label=r'$-1$')
legend=ax[0,0].legend(frameon=False,fontsize=6)
ax[0,0].set_xlim([-20,-5])
ax[0,0].set_ylim([0,20])
#plt.setp(legend.get_texts(), color='r')


nbin=np.log(np.unique(np.uint32(np.logspace(np.log10(np.min(nagg))*0.9,np.log10(np.max(nagg))*1.1,k)))-0.5)
rhobin=np.linspace(logrho.min()*0.9,logrho.max()*1.1,k)
ax[0,0].set_ylabel(r'$ \log n $')
ax[0,1].hist2d(logrho,np.log(nagg),[rhobin,nbin],norm=mpl.colors.LogNorm(),cmap=plt.cm.Greys)
# plt.ylim([0,6])
# plt.xlim([0,20])
ax[0,1].patch.set_facecolor(plt.cm.Greys(0))
rx=rhobin
ax[0,1].plot(rx,rx-3.0,'r--',label=r'$1$')
ax[0,1].plot(rx,rx*(Df-1)-2.0,'r-',label=r'$D_1-1$')
ax[0,1].set_xlim([0,30])
ax[0,1].set_ylim([0,20])
legend=ax[0,1].legend(frameon=False,fontsize=6,loc=4)
plt.subplots_adjust(hspace=0.2,wspace=0.2)
plt.savefig('agg1_sine.pdf',bbox_inches='tight')


#%% Figure 17 a b PDF of aggregated 1/s

s=3
lmax=1e7

A=[0.4,0.9,1.8]
L1,S1,wrapped_time,W1,t1=run_DSM(lmax,A[0],s,STOP_ON_LMAX=True)
L2,S2,wrapped_time,W2,t2=run_DSM(lmax,A[1],s,STOP_ON_LMAX=True)
L3,S3,wrapped_time,W3,t3=run_DSM(lmax,A[2],s,STOP_ON_LMAX=True)

L=[L1,L2,L3]
S=[S1,S2,S3]
W=[W1,W2,W3]
t=[t1,t2,t3]
#% various a

M=['d','o','s','*']
ms='none'
sB=1/50
radius=0.01

fig=plt.figure(figsize=(3,2))
nb=150
factor=[1e2,2e2,5e2]
for i in range(3):
	L1=L[i]
	S1=S[i]
	Lmod=np.mod(L1,1)
	dist_old=np.sum(np.diff(L1,axis=0)**2,1)**0.5
#	C=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1,sB)
#										 ,weights=S1*dist_old/sB*np.sqrt(2)*np.sqrt(np.pi)/sB*radius,density=False)[0]
	C=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1,sB)
										 ,weights=S1*dist_old/sB,density=False)[0]
	h,x=np.histogram(C,np.logspace(-6,0,nb),density=True)
	plt.plot(x[1:],h,M[i],color=plt.cm.cool(i/3),label=r'$A={:1.1f}$'.format(A[i]),fillstyle=ms)
	h,x=np.histogram(S1,np.logspace(-9,0,100),density=True)
	plt.plot(x[1:],factor[i]*h,'--',color=plt.cm.cool(i/3))

	# ax2 = fig.add_axes([0.25, 0.3, 0.3, 0.05])
	# import matplotlib as mpl
	# #norm = mpl.colors.Normalize(vmin=T_all[0]*u_pore/d, vmax=T_all[-1]*u_pore/d)
	# norm = mpl.colors.Normalize(vmin=0.4, vmax=1.8)
	# cb1 = mpl.colorbar.ColorbarBase(ax2,cmap=plt.cm.cool,norm=norm,
	# 								orientation='horizontal')
	# cb1.set_label(r'$A$', color='k',size=12,labelpad=0)
	# cb1.set_ticks([0.4,0.9,1.8])
THEORY=True

if THEORY:
	from scipy import special
	Lyap=np.loadtxt('Sine_Lyap.txt')
	Ksi=np.loadtxt('Sine_scaling_C|N_sB1_50.txt')
	D1=np.loadtxt('Sine_D1.txt')
	for i in range(0,3):
		L1=L[i]
		S1=S[i]
		Lmod=np.mod(L1,1)
		n=np.logspace(0,6,1000)
		dn=np.diff(np.log(n))[0]*n
		lyap=np.interp(A[i],Lyap[:,0],Lyap[:,1])
		sigma2=np.interp(A[i],Lyap[:,0],Lyap[:,2])
		d1=np.interp(A[i],D1[:,0],D1[:,1])
		# Theory
		dist_old=np.sum(np.diff(L1,axis=0)**2,1)**0.5
		Ltot=np.sum(dist_old)
		mu_n=sB*Ltot
		sigma2_n=mu_n**2.*(-np.log(sB))*2*(2-d1)/(d1-1)
		# gamma
		k=mu_n**2./sigma2_n
		theta=sigma2_n/mu_n
		pdf_n=1/special.gamma(k)/theta**k*n**(k-1)*np.exp(-n/theta)
		# Normalize
		pdf_n=pdf_n/np.sum(pdf_n*dn)
		l0=0.3
		epsilon=epsilon_d1(d1)
		ksi=1-epsilon
		Ksi=np.loadtxt('Sine_scaling_C|N_sB1_{:1.0f}.txt'.format(50))
#		ksi=-np.interp(d1,Ksi[:,0],Ksi[:,1])
		ksi=np.interp(d1,Ksi[:,0],Ksi[:,1])
		mu_c_n=l0*sB/1
		sigma2_c_n=(l0*sB/1)**2.*(n**(-ksi)-1/n)
		# pdf_c_n is gamma
		k_c_n=mu_c_n**2./sigma2_c_n
		theta_c_n=sigma2_c_n/mu_c_n
		def convolution(c):
			import scipy.integrate
			pdf_c_n=1/special.gamma(k_c_n)/theta**k_c_n*c**(k_c_n-1)*np.exp(-c/theta_c_n)
			return np.sum(pdf_c_n*pdf_n*dn)
		mu_logc_n=np.log(mu_c_n/np.sqrt(sigma2_c_n/mu_c_n**2.+1))
		sigma2_logc_n=np.log(sigma2_c_n/mu_c_n**2.+1)
		def convolution_lognormal(c):
			import scipy.integrate
			pdf_c_n=1/np.sqrt(2*np.pi*sigma2_logc_n)/c*np.exp(-(np.log(c)-mu_logc_n)**2./2/sigma2_logc_n)
			return np.nansum(pdf_c_n*pdf_n*dn)
		C=np.logspace(-5,-1,1000)
		pdf_c=np.array([convolution_lognormal(c) for c in C])
		pdf_c=pdf_c/np.sum(pdf_c*np.diff(np.log(C))[0]*C)
		plt.plot(C,pdf_c,'-',color=plt.cm.cool(i/3))
#		plt.plot(C,np.exp(-C*theta))
#		plt.yscale('log')

plt.yscale('log')
plt.xscale('log')

plt.ylabel(r'pdf')
plt.xlabel(r'$c / (\theta_0 s_0 / s_a)$')
plt.legend()
plt.xlim([1e-4,5e-1])
plt.ylim([1e-2,1e3])
subscript(plt.gca(),0)
plt.savefig('Sine_pdf_A.pdf',bbox_inches='tight')

# various sB
theta0=1
s0=1

ms='none'
THEORY=True
A=[0.4,0.9,1.8]
M=['d','o','s','*']
SB=[1/50,1/100,1/500]
plt.figure(figsize=(3,2))

id_a=1
L1=L[id_a]
S1=S[id_a]
lyap=np.interp(A[id_a],Lyap[:,0],Lyap[:,1])
sigma2=np.interp(A[id_a],Lyap[:,0],Lyap[:,2])
d1=np.interp(A[id_a],D1[:,0],D1[:,1])

xi=0.5
Lmod=np.mod(L1,1)
dist_old=np.sum(np.diff(L1,axis=0)**2,1)**0.5
for i in range(3):
	sB=SB[i]
	C=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1,sB)
										 ,weights=S1*dist_old/sB,density=False)[0]
#	N=np.round(np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1,sB)
#										 ,weights=dist_old/sB,density=False)[0])
#	h,x=np.histogram(N[N>0]**(-xi/2),np.logspace(-6,0,100),density=True)
#	h[h==0]=np.nan
#	plt.plot(x[1:],h,'-',color=plt.cm.viridis(i/3))
	h,x=np.histogram(S1,np.logspace(-9,0,100),density=True)
	plt.plot(x[1:],1e1*h,'k--',label=r'')
	h,x=np.histogram(C,np.logspace(-6,0,50),density=True)
	plt.plot(x[1:],h,M[i],color=plt.cm.viridis(i/3),label=r'$s_a=1/{:1.0f}$'.format(1/sB),fillstyle=ms)

if THEORY:
	from scipy import special
	Lyap=np.loadtxt('Sine_Lyap.txt')
	D1=np.loadtxt('Sine_D1.txt')
	for i in range(0,3):
		sB=SB[i]
		print(sB)
		n=np.logspace(0,6,1000)
		dn=np.diff(np.log(n))[0]*n
		# Theory
		dist_old=np.sum(np.diff(L1,axis=0)**2,1)**0.5
		Ltot=np.sum(dist_old)
		mu_n=sB*Ltot
		sigma2_n=mu_n**2.*(-np.log(sB))*2*(2-d1)/(d1-1)
		# gamma
		k=mu_n**2./sigma2_n
		theta=sigma2_n/mu_n
		pdf_n=1/special.gamma(k)/theta**k*n**(k-1)*np.exp(-n/theta)
		# Normalize
		pdf_n=pdf_n/np.sum(pdf_n*dn)
		l0=0.3
		epsilon=epsilon_d1(d1)
		ksi=-1+epsilon
		# true ksi
# 		Ksi=np.loadtxt('Sine_scaling_C|N_sB1_{:1.0f}.txt'.format(50))
# 		ksi=np.interp(d1,Ksi[:,0],Ksi[:,1])
		mu_c_n=l0*sB/1
		#sigma2_c_n=(l0*sB/1)**2.*(n**(-ksi)-1/n)
		sigma2_c_n=(l0*sB/1)**2.*(n**(ksi)) 
		# pdf_c_n is gamma
		k_c_n=mu_c_n**2./sigma2_c_n
		theta_c_n=sigma2_c_n/mu_c_n
		def convolution(c):
			import scipy.integrate
			pdf_c_n=1/special.gamma(k_c_n)/theta**k_c_n*c**(k_c_n-1)*np.exp(-c/theta_c_n)
			return np.sum(pdf_c_n*pdf_n*dn)
		mu_logc_n=np.log(mu_c_n/np.sqrt(sigma2_c_n/mu_c_n**2.+1))
		sigma2_logc_n=np.log(sigma2_c_n/mu_c_n**2.+1)
		def convolution_lognormal(c):
			import scipy.integrate
			pdf_c_n=1/np.sqrt(2*np.pi*sigma2_logc_n)/c*np.exp(-(np.log(c)-mu_logc_n)**2./2/sigma2_logc_n)
			return np.nansum(pdf_c_n*pdf_n*dn)
		C=np.logspace(-5,-1,1000)
		pdf_c=np.array([convolution_lognormal(c) for c in C])
		pdf_c=pdf_c/np.sum(pdf_c*np.diff(np.log(C))[0]*C)
		plt.plot(C,pdf_c,'-',color=plt.cm.viridis(i/3))

plt.yscale('log')
plt.xscale('log')

plt.ylabel('pdf')
plt.xlim([1e-5,1e-1])
plt.ylim([1e-5,1e4])
plt.xlabel(r'$c / (\theta_0 s_0 / s_a)$')
plt.legend(loc=3)

subscript(plt.gca(),1)
plt.savefig('Sine_pdf_sB.pdf',bbox_inches='tight')


#%% Figure 18 Theory various time
theta0=1
s0=1
A=0.8
s=3

Lmax=[1e6,1e7,1e8]
L1,S1,wrapped_time,W1,t1=run_DSM(Lmax[0],A,s,STOP_ON_LMAX=True)
L2,S2,wrapped_time,W2,t2=run_DSM(Lmax[1],A,s,STOP_ON_LMAX=True)
L3,S3,wrapped_time,W3,t3=run_DSM(Lmax[2],A,s,STOP_ON_LMAX=True)


L=[L1,L2,L3]
S=[S1,S2,S3]
W=[W1,W2,W3]
t=[t1,t2,t3]

nb=100 
l0=0.3
ms='none'
factor=[1,1,1]
THEORY=True
THEORY_RANDOM=True
M=['d','o','s','*']
SB=[1/50,1/100,1/500]
plt.figure(figsize=(3,2))
sB=1/50

for i in range(3):
	L1=L[i]
	S1=S[i]
	Lmod=np.mod(L1,1)
	dist_old=np.sum(np.diff(L1,axis=0)**2,1)**0.5
	C=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1,sB)
										 ,weights=S1*dist_old/sB,density=False)[0]
	h,x=np.histogram(C,np.logspace(-6,0,nb),density=True)
	plt.plot(x[1:],h,M[i],color=plt.cm.cool(i/3),label=r'$t={:1.1f}$'.format(t[i]),fillstyle=ms)
	h,x=np.histogram(S1,np.logspace(-9,0,100),density=True)
#	plt.plot(x[1:],factor[i]*h,'--',color=plt.cm.cool(i/3))

if THEORY:
	from scipy import special
	Lyap=np.loadtxt('Sine_Lyap.txt')
	D1=np.loadtxt('Sine_D1.txt')
	for i in range(0,3):
		L1=L[i]
		S1=S[i]
		Lmod=np.mod(L1,1)
#			sB=SB[i]
		n=np.logspace(0.1,6,1000)
		lyap=np.interp(A,Lyap[:,0],Lyap[:,1])
		sigma2=np.interp(A,Lyap[:,0],Lyap[:,2])
		d1=np.interp(A,D1[:,0],D1[:,1])
		# Theory
		dist_old=np.sum(np.diff(L1,axis=0)**2,1)**0.5
		Ltot=np.sum(dist_old)
		Ltot=l0*np.exp((lyap+sigma2/2)*t[i])
		#print(Ltot)
		mu_n=sB*Ltot
		sigma2_n=mu_n**2.*(-np.log(sB))*2*(2-d1)/(d1-1)
		# gamma
		k=mu_n**2./sigma2_n
		theta=sigma2_n/mu_n
		print(theta**2*k)
		pdf_n=1/special.gamma(k)/theta**k*n**(k-1)*np.exp(-n/theta)
		# Normalize
		dn=np.diff(np.log(n))[0]*n
		pdf_n=pdf_n/np.sum(pdf_n*dn)
		l0=0.3
		epsilon=epsilon_d1(d1)
		ksi=-1+epsilon
		# true ksi
		Ksi=np.loadtxt('Sine_scaling_C|N_sB1_{:1.0f}.txt'.format(50))
# 		ksi=-np.interp(d1,Ksi[:,0],Ksi[:,1])
		mu_c_n=l0*sB/1
		#sigma2_c_n=(l0*sB/1)**2.*(n**(ksi)-1/n)
		sigma2_c_n=(l0*sB/1)**2.*(n**(ksi)) 
		# pdf_c_n is gamma
		k_c_n=mu_c_n**2./sigma2_c_n
		theta_c_n=sigma2_c_n/mu_c_n
		
		def convolution_gamma(c):
			import scipy.integrate
			pdf_c_n=1/special.gamma(k_c_n)/theta**k_c_n*c**(k_c_n-1)*np.exp(-c/theta_c_n)
			high_k=k_c_n>20
			pdf_c_n[high_k]=1/np.sqrt(2*np.pi*sigma2_c_n[high_k])*np.exp(-(c-mu_c_n)**2./2/sigma2_c_n[high_k])
			return np.sum(pdf_c_n*pdf_n*dn)
		def convolution_normal(c):
			import scipy.integrate
			pdf_c_n=1/np.sqrt(2*np.pi*sigma2_c_n)*np.exp(-(c-mu_c_n)**2./2/sigma2_c_n)
			return np.nansum(pdf_c_n*pdf_n*dn)
		mu_logc_n=np.log(mu_c_n/np.sqrt(sigma2_c_n/mu_c_n**2.+1))
		sigma2_logc_n=np.log(sigma2_c_n/mu_c_n**2.+1)
		def convolution_lognormal(c):
			import scipy.integrate
			pdf_c_n=1/np.sqrt(2*np.pi*sigma2_logc_n)/c*np.exp(-(np.log(c)-mu_logc_n)**2./2/sigma2_logc_n)
			return np.nansum(pdf_c_n*pdf_n*dn)
		
		C=np.logspace(-5,-1,100)
		pdf_c=np.array([convolution_lognormal(c) for c in C])
		pdf_c=pdf_c/np.sum(pdf_c*np.diff(np.log(C))[0]*C)
		plt.plot(C,pdf_c,'-',color=plt.cm.cool(i/3))
		#print(pdf_c)

if THEORY_RANDOM:
	from scipy import special
	Lyap=np.loadtxt('Sine_Lyap.txt')
	D1=np.loadtxt('Sine_D1.txt')
	for i in range(0,3):
		Ltot=l0*np.exp((lyap+sigma2/2)*t[i])
		print(Ltot)
		mu_n=sB*Ltot
		C=np.logspace(-5,-1,100)
		k_random=mu_n
		mu_c=l0*sB/1
		sigma2_random=mu_c**2./k_random
		if k_random<50:
			pdf_random=1/special.gamma(k_random)/(mu_c/k_random)**k_random*C**(k_random-1)*np.exp(-C/mu_c*k_random)
		else:
			pdf_random=1/np.sqrt(2*sigma2_random*np.pi)*np.exp(-(C-mu_c)**2/(2*sigma2_random))
		plt.plot(C,pdf_random,'--',color=plt.cm.cool(i/3))

plt.yscale('log')
plt.xscale('log')
plt.ylabel('pdf')
plt.plot([],[],'k--',label='Fully Random')
plt.plot([],[],'k-',label='Correlated')
plt.xlim([1e-5,5e-1])
plt.ylim([1e-2,1e4])
plt.xlabel(r'$c / (\theta_0 s_0 / s_a)$')
plt.ylabel(r'pdf')
plt.legend(loc=3,fontsize=8,frameon=False)

#subscript(plt.gca(),1)
plt.savefig('Sine_pdf_time.pdf',bbox_inches='tight')


#%% Figure 19 kn as a function of D2


from scipy.ndimage import gaussian_filter1d

from matplotlib.ticker import (AutoLocator, AutoMinorLocator)

def epsilon_d1(D1):
	# Theoretical value of epsilon
	mu=1/(D1-1)
	sigma2=2*(2-D1)/(D1-1)
	if mu-2*sigma2>0:
		M2=2*mu-2*sigma2
	else:
		M2=mu**2/(2*sigma2)
	return -M2+2

D=np.loadtxt('Sine_D1.txt')

D1f=gaussian_filter1d(D[:,1],3,mode='nearest')
D2f=gaussian_filter1d(D[:,2],3,mode='nearest')

D2=np.linspace(1.7,1.9,200)
Sa=np.logspace(-4.9,-0.1,1000)

fig, ax1 = plt.subplots(figsize=(3,2.5),constrained_layout=True)

s=[]
E=[]
for d2 in D2:
	#d1=np.interp(d2,D[:,2],D[:,1])
	d1=np.interp(d2,D2f,D1f)
	xi=-epsilon_d1(d1)+1
	print(xi)
	kn=1/(Sa**(d2-2)-1)
	s.append(Sa[np.where(kn>xi)[0][0]])
	exponent=kn
	exponent[np.where(kn>xi)[0]]=xi
	E.append(exponent)

E=np.array(E)
CS=ax1.contour(D2,np.log10(Sa),E.T,8,linewidth=1.5)

ax1.clabel(CS, inline=True, fontsize=9)
ax1.plot(D2,np.log10(s),'k--',linewidth=2)
ax1.set_ylim([-5,0.1])
ax1.set_xlim([1.7,1.9])
#ax1.set_yscale('log')
#ax1.imshow(E,extent=(1.7,1.9,-5,1))
ax1.set_xlabel('$D_2$')
ax1.set_ylabel('$\log_{10} s_A$')


def sAtoPe(x):
	return -2*x


def PetosA(x):
	return -x/2


def DtoA(x):
	return np.interp(x,D[:,2],D[:,0])


def AtoD(x):
	return np.interp(x,D[:,0],D[:,2])

secay = ax1.secondary_yaxis('right', functions=(sAtoPe, PetosA))
#secay.yaxis.set_minor_locator(AutoMinorLocator())
secay.set_ylabel(r'$\log_{10} Pe$')

# secax = ax1.secondary_xaxis('top', functions=(DtoA, AtoD))
# secax.xaxis.set_minor_locator(AutoMinorLocator())
# secax.set_xlabel('$A$')
ticks=np.linspace(1.7,1.9,5)
Aticks=np.interp(ticks,D2f,D[:,0])
[plt.text(t,0.2,'${:1.2f}$'.format(Aticks[i]),horizontalalignment='center',fontsize=9) for i,t in enumerate(ticks)]
plt.text(1.8,0.5,'$A$')

plt.text(1.8,-1,r'$\xi <k_n$',backgroundcolor='w',fontsize=9)
plt.text(1.8,-4,r'$k_n <\xi$',backgroundcolor='w',fontsize=9)
plt.text(1.72,-1.8,r'$k_n =\xi$',backgroundcolor='w',fontsize=9)

plt.savefig('xi_kn_sine.pdf')

#%% Figure 20b 
M=['d','o','s','*']
Gamma=np.loadtxt('Sine_gamma2_6.txt')
AA=np.unique(Gamma[:,0])
SB=np.unique(Gamma[:,1])
D1=np.loadtxt('Sine_D1.txt')

p=np.polyfit(D1[:,0],D1[:,1],1)
#D1=AA*p[0]+p[1]

fig=plt.figure(figsize=(2,2))
for i,sB in enumerate([SB[0]]):
	for A in AA:
		idseed=np.where((Gamma[:,0]==A)&(Gamma[:,1]==sB))[0]
		d1=A*p[0]+p[1]
		#d1=np.interp(A,D1[:,0],D1[:,1])
#		plt.plot(d1,-Gamma[idseed,-1].mean(),M[i],color=plt.cm.cool(i/3),fillstyle='full')
		plt.plot(d1,-Gamma[idseed,3].mean(),'ko',zorder=100)
#!!!
#[plt.plot([],[],M[i],color=plt.cm.cool(i/3),label='$s_a=1/{:1.0f}$'.format(1/sB)) for i,sB in enumerate(SB)]
plt.legend()
d1=AA*p[0]+p[1]


p=np.polyfit(D1[:,0],D1[:,2],1)
d2=AA*p[0]+p[1]
kn=1/(sB**(d2-2)-1)


# plt.figure()
# plt.plot(P[:,0],-P[:,1],'k'+M[i],label='$s_a=1/{:1.0f}$'.format(1/sB))
Lyap=np.loadtxt('Sine_Lyap.txt')
#Lyap=Gamma[:,[0,-2,-1]]
lyap=np.interp(AA,Lyap[:,0],Lyap[:,1])
sigma2=np.interp(AA,Lyap[:,0],Lyap[:,2])
id1=sigma2*1.2>=lyap

Var=np.loadtxt('Sine_N_t.txt')
idsB=np.where(Var[:,2]==sB)[0]
a=Var[idsB,0]
La=Var[idsB,3]


lyap=np.interp(AA,Lyap[:,0],Lyap[:,1])
sigma2=np.interp(AA,Lyap[:,0],Lyap[:,2])

l=np.interp(AA,Gamma[:,0],Gamma[:,4])

isol=np.interp(AA,Gamma[:,0],Gamma[:,-1])

#plt.plot(d1,lyap**2/(2*sigma2),'r--')
plt.plot([],[],'ko',label=r'Simulation')
#plt.plot([],[],'-',color='w',label=r'\textbf{Lagrangian model :}',linewidth=1.2)
plt.plot(d1,lyap-sigma2/2,'-',color='seagreen',label=r'Isolated lamella',linewidth=1.2)
#plt.plot([],[],'-',color='w',label=r'\textbf{Aggregation models :}',linewidth=1.2)


#plt.plot(d1,-isol,'-',color='seagreen',label=r'Isolated strip',linewidth=1.2))
fully_cor=2*(lyap-sigma2)
fully_cor[lyap<2*sigma2]=lyap[lyap<2*sigma2]**2/(2*sigma2[lyap<2*sigma2])
#plt.plot(d1,fully_cor,'--',color='darkorange',label=r'Fully correlated',linewidth=1.2)
plt.plot(d1,l,':',color='indianred',label=r'Random aggregation',linewidth=1.2)

subscript(plt.gca(),0,x=-0.12)

plt.xlabel(r'$D_1$')
plt.ylabel(r'$\gamma_{2,c}$')
#plt.yscale('log')

#plt.xlim([1.7,2])
plt.ylim([1e-2,3])
#G=np.loadtxt('sine_gamma2_theory.txt')

Ksi=np.loadtxt('Sine_scaling_C|N_sB1_50.txt')
p=np.polyfit(Ksi[:,0],Ksi[:,1],1)
ksi=d1*p[0]+p[1]

#Model
nu=np.linspace(1.7,2,1000)
mu=1/(nu-1)
sigma2=2*(2-nu)/(nu-1)
M2=2*mu-2*sigma2
M2[mu-2*sigma2<0]=mu[mu-2*sigma2<0]**2/(2*sigma2[mu-2*sigma2<0])

#ksi=np.interp(d1,nu,M2-1)

ksi[kn<ksi]=kn[kn<ksi]
plt.plot(d1,(l)*ksi,'-',color='blueviolet',label=r'Correlated aggregation',linewidth=1.2) #kn>ksi
plt.legend(fontsize=6)

plt.savefig('gamma2_sine.pdf',bbox_inches='tight')

#%% Compurtation for Figure 13 -  <1/rho^2 n>_B Parrallel
from scipy.stats import linregress
import multiprocessing

def POD_fit(x,y):
	# Make linear fitting with PCA/POD decomposition
	# Return slope and confidence interval (ratio between small axis/large axis)
	cov=np.cov(np.vstack((x,y)))
	evals, evecs = np.linalg.eig(cov)
	sort_indices = np.argsort(evals)[::-1]
	x_v1, y_v1 = evecs[:, sort_indices[0]]
	l1,l2=evals[sort_indices]
	return y_v1/x_v1, l2/l1
	
sB=1/50
AA=np.linspace(0.3,1.8,15)
Pall=[]
plt.figure()
Pallseeds=[]
def parrallel(seed):
	Pall=[]
	for a in AA:
		print('Seed=',seed,'A=',a)
		L,S,wrapped_time,W,t=run_DSM(1e7, a, seed)
		
		s0=0.01
		D=sB**2*0.5/2
		Tau=D/s0**2*wrapped_time
		Cmax=1./np.sqrt(1.+4*D/s0**2*wrapped_time)
		Sc=s0*np.sqrt(1.+4*D/s0**2*wrapped_time)*S
		print(np.mean(Sc))
		
		Lmod=np.mod(L,1)
		dist_old=np.sum(np.diff(L,axis=0)**2,1)**0.5
		N=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1,sB)
											 ,weights=dist_old/sB,density=False)[0]
		S1=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1,sB)
											 ,weights=S*dist_old/sB,density=False)[0]
		C1=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1,sB)
											 ,weights=Cmax*dist_old/sB,density=False)[0]
		Sb1=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1,sB)
											 ,weights=Sc*dist_old/sB,density=False)[0]
		S2=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1,sB)
											 ,weights=S**2.*dist_old/sB,density=False)[0]
	#	S4=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1,sB)
	#										 ,weights=S**4.*dist_old/sB,density=False)[0]
	#	C2=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1,sB)
	#										 ,weights=Cmax**2.*dist_old/sB,density=False)[0]
		logrho=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1,sB)
											 ,weights=-np.log(S)*dist_old/sB,density=False)[0]
		logrho2=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1,sB)
											 ,weights=np.log(S)**2.*dist_old/sB,density=False)[0]
		#	VM.append(np.mean(S2))
		
		MS=S1/N
		VS=S2/N-(MS)**2.
		Mlogrho=(logrho/(N))
		Vlogrho=logrho2/(N)-(Mlogrho)**2.
		
		n=5
		PS1=POD_fit(np.log(N[N>n].flatten()),np.log(MS[N>n].flatten()))
#		PS2=linregress(np.log(N[N>n].flatten()),np.log(VS[N>n].flatten()))[:2]
		PS2=linregress(np.log(N[N>n].flatten()),np.log((S2/N)[N>n].flatten()))[:2]
		Plogrho=POD_fit(np.log(N[N>n].flatten()),Mlogrho[N>n].flatten())
		Plogrho2=linregress(np.log(N[N>n].flatten()),Vlogrho[N>n].flatten())[:2]
# =================================X=f(N)====================================
# 		PS1 = linregress(np.log(N[N>n].flatten()),np.log(MS[N>n].flatten()))
# 		PS2 = linregress(np.log(N[N>n].flatten()),np.log(VS[N>n].flatten()))
# 		Plogrho = linregress(np.log(N[N>n].flatten()),(Mlogrho[N>n].flatten()))
# 		Plogrho2 = linregress(np.log(N[N>n].flatten()),(Vlogrho[N>n].flatten()))
#=================================N=f(X)=========================================

	#	plt.plot(np.log(nbin[1:]),np.log(nbin[1:])*P2[0]+P2[1],label=r'${:1.2f} \pm {:1.2f} (r^2={:1.2f})$ '.format(P2[0],P2[4]*1.96,P2[2]**2))
	# 	plt.plot(np.log(nbin[1:]),np.log(nbin[1:])*(-2)+P1[0],'k--')
	# 	plt.plot(np.log(nbin[1:]),np.log(nbin[1:])*(-1)+P1[0],'k--')
	#	plt.legend()
# 		n=20
# 		plt.figure()
# 		plt.hist2d(np.log(N[N>n]).flatten(),np.log(VS[N>n]).flatten(),20)
# 		PS2=POD_fit(np.log(N[N>n].flatten()),np.log(VS[N>n].flatten()))
# 		PS2=linregress(np.log(N[N>n].flatten()),np.log(VS[N>n].flatten()))
# 		x=np.linspace(0,10,10)
# 		plt.plot(x,x*PS2[0]-5,'w--')
# 		
# 		plt.figure()
# 		plt.hist2d(np.log(N[N>n]).flatten(),np.log(MS[N>n]).flatten(),20)
# 		PS1=POD_fit(np.log(N[N>n].flatten()),np.log(MS[N>n].flatten()))
# 		x=np.linspace(0,10,10)
# 		plt.plot(x,x*PS1[0]-5,'w--')
# 		PS1=linregress(np.log(N[N>n].flatten()),np.log(MS[N>n].flatten()))
# 		x=np.linspace(0,10,10)
# 		plt.plot(x,x*PS1[0]-5,'r--')
# #		plt.plot(x,-x*1.2-15,'w-')

# 		plt.figure()
# 		plt.hist2d(np.log(N[N>n]).flatten(),Vlogrho[N>n].flatten(),20)
# 		Plogrho2=POD_fit(np.log(N[N>n].flatten()),Vlogrho[N>n].flatten())
# 		plt.plot(x,x*Plogrho2[0]+2,'r--')
# 		Plogrho2=linregress(np.log(N[N>n].flatten()),Vlogrho[N>n].flatten())
# 		x=np.linspace(0,10,10)
# 		plt.plot(x,x*Plogrho2[0]+2,'w--')

# 		plt.figure()
# 		plt.hist2d(np.log(N[N>n]).flatten(),Mlogrho[N>n].flatten(),20)
# 		Plogrho=POD_fit(np.log(N[N>n].flatten()),Mlogrho[N>n].flatten())
# 		x=np.linspace(0,10,10)
# 		plt.plot(x,x*Plogrho[0]+5,'w--')
#		plt.hist2d(np.log(N[N>n]).flatten(),(logrho2[N>n]/N[N>n]).flatten(),40)
# 		plt.hist2d(np.log(N[N>n]).flatten(),(logrho[N>n]/N[N>n]).flatten(),40)
# 		x=np.linspace(1,8,10)
# 		plt.plot(x,x*Plogrho[0]+Plogrho[1],'w-')
		Pall.append([PS1,PS2,Plogrho,Plogrho2])
	return Pall

Seeds=np.arange(100)

pool = multiprocessing.Pool(np.minimum(len(Seeds),32))
Pallseeds=pool.map(parrallel, Seeds)
pool.close()
pool.join()

P=np.array(Pallseeds)
Pm=np.mean(P,axis=0)

D1=np.loadtxt('Sine_D1.txt')

plt.figure()
plt.plot(D1[:,1],-Pm[:,1,0],'rd')
plt.plot(D1[:,1],-Pm[:,0,0],'kd')
plt.xlabel("$D_1$")
plt.ylabel("$\gamma_1,\gamma_2$")
plt.savefig('Sine_gamma.pdf',bbox_inches='tight')

plt.figure()
plt.plot(D1[:,1],Pm[:,1,1],'rd')
plt.xlabel("$D_1$")
plt.ylabel("$\omega_2$")
plt.ylim([-15,0])
plt.savefig('Sine_omega.pdf',bbox_inches='tight')


plt.figure()
plt.plot(D1[:,1],Pm[:,2,0],'kd')
plt.plot(D1[:,1],Pm[:,3,0],'rd')
plt.xlabel(r"$D_1$")
plt.ylabel(r"$\alpha,\beta$")
plt.savefig('Sine_alpha_beta.pdf',bbox_inches='tight')

np.savetxt('Sine_scaling_N_sB1_{:1.0f}.txt'.format(1/sB),np.vstack((D1[:,1],-Pm[:,0,0],-Pm[:,1,0],Pm[:,2,0],Pm[:,3,0],Pm[:,1,1])).T,header='#D_1,\gamma_1,\gamma_2,\alpha,\beta,\omega_2')

#%% Computation for Figure 20b # Scalar decay of gridbased DSM parrallel
import multiprocessing

AA=np.linspace(0.3,1.8,20)
SS=np.arange(150,200)
SB=[1/50]

#Generate parameter list
P=[]
for sB in SB:
	for A in AA:
		for seed in SS:
			P.append([A,sB,seed])

def parrallel(par):
	A,sB,seed=par
	Time,C,L,gamma,rho_1,N=run_DSM_grid(sB,3e7,A,seed)
#	print(Time[-1],A,seed)
	varC=np.var(C.reshape(C.shape[0],-1),axis=1)
	lt2=int(len(Time)/2)
	Time=np.array(Time)
	p=np.polyfit(Time[lt2:],np.log(varC[lt2:]),1)
# 	plt.figure()
# 	plt.plot(Time,np.log(varC),'ko')
# 	plt.plot(Time,Time*p[0]+p[1],label='$A={:1.2f}, \gamma_2={:1.2f}$'.format(A,p[0]))
# 	plt.legend()
# 	plt.ylim([-16,-4])
# 	plt.savefig('Sine_decayrate_a{:1.1f}_sB1_{:1.0f}_seed{:1.0f}.jpg'.format(A,1/sB,seed))
	l=np.polyfit(Time[lt2:],np.log(L[lt2:]),1)[0]
	N=np.array(N)
	nmean=np.polyfit(Time[lt2:],np.log(N[lt2:,0]),1)[0]
	nvar=np.polyfit(Time[lt2:],np.log(N[lt2:,1]),1)[0]
	gamma=np.array(gamma)
	lyap=np.polyfit(Time[lt2:],gamma[lt2:,0],1)[0]
	sigma2=np.polyfit(Time[lt2:],gamma[lt2:,1],1)[0]
	rho_1=np.array(rho_1)
	r_1=np.polyfit(Time[lt2:],np.log(rho_1[lt2:]),1)[0]
	result=[A,sB,seed,p[0],l,nmean,nvar,lyap,sigma2,r_1]
	print(result)
	return result

# Lyap=np.loadtxt('Sine_Lyap.txt')
# plt.plot(Time,np.array(L)/0.3);
# lyap=np.interp(A,Lyap[:,0],Lyap[:,1])
# sigma2=np.interp(A,Lyap[:,0],Lyap[:,2])
# plt.plot(Time,np.exp((lyap+sigma2/2)*Time),'k--');
# plt.yscale('log')

pool = multiprocessing.Pool(processes=32)
Var=pool.map(parrallel, P)
pool.close()
pool.join()
VarA=np.array(Var).reshape(-1,10)

np.savetxt('Sine_gamma2_6.txt',VarA)

#%% Computation figures 18 & 20  <c^2 | n>_B Parrallel
from scipy.stats import linregress
import multiprocessing

def POD_fit(x,y):
	# Make linear fitting with PCA/POD decomposition
	# Return slope and confidence interval (ratio between small axis/large axis)
	cov=np.cov(np.vstack((x,y)))
	evals, evecs = np.linalg.eig(cov)
	sort_indices = np.argsort(evals)[::-1]
	x_v1, y_v1 = evecs[:, sort_indices[0]]
	l1,l2=evals[sort_indices]
	return y_v1/x_v1, l2/l1
	
sB=1/50
AA=np.linspace(0.3,1.8,15)
Pall=[]
plt.figure()
Pallseeds=[]
def parrallel(seed):
	Pall=[]
	for a in AA:
		print('Seed=',seed,'A=',a)
		L,S,wrapped_time,W,t=run_DSM(1e7, a, seed)
		
		s0=0.01
		D=sB**2*0.5/2
		Tau=D/s0**2*wrapped_time
		Cmax=1./np.sqrt(1.+4*D/s0**2*wrapped_time)
		Sc=s0*np.sqrt(1.+4*D/s0**2*wrapped_time)*S
		print(np.mean(Sc))
		
		Lmod=np.mod(L,1)
		dist_old=np.sum(np.diff(L,axis=0)**2,1)**0.5
		N=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1,sB)
											 ,weights=dist_old/sB,density=False)[0]
		C=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1,sB)
											 ,weights=S*dist_old/sB,density=False)[0]
		V=(C-np.mean(C))**2
		#	VM.append(np.mean(S2))
		nb=5
		p=np.polyfit(np.log(N[N>nb].flatten()),np.log(V[N>nb].flatten()),1)
# 		plt.plot(N,V,'.',alpha=0.3)
# #		plt.plot(N,C,'.',alpha=0.3)
# 		plt.plot(N,N**(-1),'k-')
# 		plt.plot(N,N**(-2),'r-')
# 		plt.yscale('log')
# 		plt.xscale('log')

		Pall.append([p[0],p[1]])
	return Pall

Seeds=np.arange(100)

try:
	import mkl
	mkl.set_num_threads(1)
except:
	pass


pool = multiprocessing.Pool(len(Seeds))
Pallseeds=pool.map(parrallel, Seeds)
pool.close()
pool.join()

P=np.array(Pallseeds)
Pm=np.mean(P,axis=0)

D1=np.loadtxt('Sine_D1.txt')
		
plt.figure()
plt.plot(D1[:,1],-Pm[:,0],'rd')
plt.xlabel("$D_1$")
plt.ylabel("$\gamma_1,\gamma_2$")
plt.savefig('Sine_gamma.pdf',bbox_inches='tight')

plt.figure()
plt.plot(D1[:,1],Pm[:,1],'rd')
plt.xlabel("$D_1$")
plt.ylabel("$\omega_2$")
plt.ylim([-15,0])
plt.savefig('Sine_omega.pdf',bbox_inches='tight')


np.savetxt('Sine_scaling_C|N_sB1_{:1.0f}.txt'.format(1/sB),np.vstack((D1[:,1],-Pm[:,0],Pm[:,1])).T,header='#D_1,\gamma_2,\omega_2')

#%% Computations Figure 16 

seed=3
l0=0.3
sB=1/100
s0=0.01
D=sB**2*0.5/2

def parrallel(seed):
	AA=np.linspace(0.3,1.8,15)
	R=[]
	for a in AA:
		print(a)
		L,S,wrapped_time,W,t=run_DSM(1e8, a, seed)
		Lmod=np.mod(L,1)
		dist_old=np.sum(np.diff(L,axis=0)**2,1)**0.5
		N=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1,sB)
											 ,weights=dist_old/sB,density=False)[0]
		C=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1,sB)
											 ,weights=S*dist_old/sB,density=False)[0]
		S2=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1,sB)
											 ,weights=S**2*dist_old/sB,density=False)[0]
		VS2=S2/N-(C/N)**2
		V=(C-np.mean(C))**2
	# 	
	# 	caxis=[-15,-5]
	# 	fig,ax=plt.subplots(1,4,figsize=(8,2))
	# 	ax[0].imshow(np.log(VS2),clim=caxis)
	# 	ax[1].imshow(np.log(VS2*N*N/(N-1)),clim=caxis)
	# 	ax[2].imshow(np.log(V),clim=caxis)
	# 	ax[3].imshow(np.log(N**-0.5))
	# 	
	# 	plt.figure()
	# 	plt.imshow(np.log(VS2*N*N/(N-1))-np.log(V),clim=[-6,6])
	# 	
		
		n=np.linspace(0,10,10)
		plt.figure()
		
		x=np.log(N[N>1].flatten())
		y=np.log(VS2[N>1].flatten())
		plt.plot(x,y,'r+',alpha=0.3,label=r'$\sigma^2_{\rho^{-1}|n}$')
		p=np.polyfit(x,y,1)
		p0=np.log((l0*sB)**2)
		p1=np.sum(x*(y-p0))/np.sum(x**2)
		plt.plot(n,p1*n+p0,'r-')
		ps2=p1
		x=np.log(N[N>1].flatten())
		y=np.log(VS2[N>1]*N[N>1]*N[N>1]/(N[N>1]-1)).flatten()
		plt.plot(x,y,'ko',alpha=0.3,label=r'$n \sigma^2_{\rho^{-1}|n}$')
		p=np.polyfit(x,y,1)
		p0=np.log((l0*sB)**2)
		p1=np.sum(x*(y-p0))/np.sum(x**2)
		plt.plot(n,p1*n+p0,'k-')
		pns2=p1
		x=np.log(N[N>1].flatten())
		y=np.log(V[N>1].flatten())
		plt.plot(x,y,'b*',alpha=0.1,label=r'$(\sum \rho^{-1} - \langle \sum \rho^{-1} \rangle)^2$')
		p=np.polyfit(x,y,1)
		p0=np.log((l0*sB)**2)
		p1=np.sum(x*(y-p0))/np.sum(x**2)
		plt.plot(n,p1*n+p0,'b-')
		pc2=p1
		
		plt.ylim([-30,-5])
		plt.xlim([0,12])
		plt.legend()
		plt.xlabel(r'$n$')
		plt.savefig('/ine_c2_n_a{:1.1f}_s{:1.0f}.png'.format(a,seed))
		
		R.append([seed,a,ps2,pns2,pc2])
		np.savetxt('Sine_scaling_with_n_seed{:1.0f}.txt'.format(seed),R)
	return R

import multiprocessing

Seeds=np.arange(5)
try:
	import mkl
	mkl.set_num_threads(1)
except:
	pass
pool = multiprocessing.Pool(len(Seeds))
Rallseeds=pool.map(parrallel, Seeds)
pool.close()
pool.join()

Rallseeds=np.array(Rallseeds)
Rm=np.mean(Rallseeds,axis=0)

np.savetxt('Sine_scaling_with_n.txt',Rm)
# R=np.array(R)
# plt.plot(AA,R[:,1])
# plt.plot(AA,R[:,2])
