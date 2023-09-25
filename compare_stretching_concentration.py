#%% RUN FIRST : SINE FLOW
# minimum stretching argument for concentration lambda2/(2sigma2) 

figdir='/home/joris/Dropbox/Articles/AggregationBatchelor/Figures/'

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
#Fire=np.loadtxt('/home/joris/.config/matplotlib/LUT_Fire.csv',delimiter=',',skiprows=1)
#Fire=Fire/255.
#cm_fire=ListedColormap(Fire[:,1:], name='Fire', N=None)

#
plt.style.use('~/.config/matplotlib/joris.mplstyle')




PLOT=False
PAR=False
#% Advection-Diffusion
# Reinitialize Random generator to a new value

def subscript(ax,i,color='k',bg='w',x=0.03,y=0.93,script=['a)','b)','c)','d)']):
	txt=ax.text(x,y,script[i],color=color,transform = ax.transAxes,backgroundcolor=bg)
	return txt

#% Check cmax of bundles
def bin_operation(x,y,xi,op):
	r=np.zeros(xi.shape[0]-1)
	for i in range(xi.shape[0]-1):
		idx=np.where((x<=xi[i+1])&(x>=xi[i]))[0]
		r[i]=op(y[idx])
	return r

def epsilon_d1(D1):
	# Theoretical value of epsilon
	mu=1/(D1-1)
	sigma2=2*(2-D1)/(D1-1)
	if mu-2*sigma2>0:
		M2=2*mu-2*sigma2
	else:
		M2=mu**2/(2*sigma2)
	return -M2+2

def lin_reg(xt,yt):
	# Linear regression with CI interval on slope and intercept
	# y =a + b x
	from scipy import stats
	y=yt[np.isfinite(yt)&np.isfinite(xt)]
	x=xt[np.isfinite(yt)&np.isfinite(xt)]
	b=np.nansum((x-np.nanmean(x))*(y-np.nanmean(y)))/np.nansum((x-np.nanmean(x))**2.)
	a=np.nanmean(y)-b*np.nanmean(x)
	ymod=a+b*x
	stdy=np.nanstd(ymod-y)
	n=np.nansum(~np.isnan(y))
	CIa=stdy/n
	CIb=stdy*np.sqrt(n)/np.sqrt(n-2.)/np.sqrt(np.nansum((x-np.nanmean(x))**2.))
	R2=np.nansum((a+b*x-np.nanmean(y))**2.)/np.nansum((y-np.nanmean(y))**2.)
	return a,b,CIa,CIb,R2

def bin_operation2(x,y,xi,op):
	r=np.zeros((xi.shape[0]-1,y.shape[1]))
	for i in range(xi.shape[0]-1):
		idx=np.where((x<=xi[i+1])&(x>=xi[i]))[0]
		r[i,:]=op(y[idx,:],axis=0)
	return r

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

#PhaseX=np.zeros(50)
# #PhaseY=np.zeros(50)

# np.random.seed(seed=10)
# PhaseX=np.random.rand(100)*2*np.pi
# PhaseY=np.random.rand(100)*2*np.pi
# Angle=np.random.rand(100)*2*np.pi

#plt.style.use('~/.config/matplotlib/joris.mplstyle')

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
	
def vel_shear(x,t,A):
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
		vy=A*np.sin(2*np.pi*x[:,0]+PhaseX[int(t/(2*T))])
		vx=np.zeros(x.shape[0])
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


from scipy.ndimage import gaussian_filter1d
def vel_shear(x,t,A):
	theta=0
	xi=np.linspace(0,1,100)
	v=10**gaussian_filter1d(A*np.random.randn(len(xi)),mode='wrap',sigma=3.)
	v=v-np.mean(v)
	vy=np.interp(np.mod(x[:,0],1),xi,v)
	vx=np.zeros(x.shape[0])
	return np.vstack((vx,vy)).T

def vel_double(x,t,A):
	theta=Angle[int(t/(2.*T))]
	theta=0
	#A=1.2
	#A=1./np.sqrt(2)
	R=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
	Rinv=np.linalg.inv(R)
	x=np.dot(x,R)
	if np.mod(t,T*2)<T:
		vx=np.zeros(x.shape[0])
		vy=A/2*np.sin(2*np.pi*x[:,0]+PhaseY[int(t/(2.*T))])+A/2*np.sin(4*np.pi*x[:,0]+PhaseX[int(t/(2.*T))])
		return np.dot(np.vstack((vx,vy)).T,Rinv)
	else:
		vx=A/2*np.sin(2*np.pi*x[:,1]+PhaseX[int(t/(2*T))])+A/2*np.sin(4*np.pi*x[:,1]+PhaseY[int(t/(2*T))])
		vy=np.zeros(x.shape[0])
		return np.dot(np.vstack((vx,vy)).T,Rinv)

def vel_single(x,t,A):
	theta=Angle[int(t/(2.*T))]
	theta=0
	#A=1.2
	#A=1./np.sqrt(2)
	R=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
	Rinv=np.linalg.inv(R)
	x=np.dot(x,R)
	if np.mod(t,T*2)<T:
		vx=np.zeros(x.shape[0])
		vy=A*np.abs(np.sin(np.pi*x[:,0]+PhaseY[int(t/(2.*T))]))
		return np.dot(np.vstack((vx,vy)).T,Rinv)
	else:
		vx=A*np.abs(np.sin(np.pi*x[:,1]+PhaseX[int(t/(2*T))]))
		vy=np.zeros(x.shape[0])
		return np.dot(np.vstack((vx,vy)).T,Rinv)

def vel_single_lin(x,t,A):
	theta=Angle[int(t/(2.*T))]
	theta=0
	#A=1.2
	#A=1./np.sqrt(2)
	R=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
	Rinv=np.linalg.inv(R)
	x=np.dot(x,R)
	if np.mod(t,T*2)<T:
		vx=np.zeros(x.shape[0])
		k=2*np.pi*x[:,0]+PhaseY[int(t/(2.*T))]
		vy=A*((np.mod(k,2*np.pi)<2*np.pi/3)*3*x[:,0]+
				(np.mod(k,2*np.pi)>4*np.pi/3)*(3-3*x[:,0])+
				(np.mod(k,2*np.pi)<=4*np.pi/3)*(np.mod(k,2*np.pi)>=2*np.pi/3))
		return np.dot(np.vstack((vx,vy)).T,Rinv)
	else:
		k=2*np.pi*x[:,1]+PhaseX[int(t/(2.*T))]
		vx=A*((np.mod(k,2*np.pi)<2*np.pi/3)*3*x[:,1]+
				(np.mod(k,2*np.pi)>4*np.pi/3)*(3-3*x[:,1])+
				(np.mod(k,2*np.pi)<4*np.pi/3)*(np.mod(k,2*np.pi)>2*np.pi/3))
		vy=np.zeros(x.shape[0])
		return np.dot(np.vstack((vx,vy)).T,Rinv)

def vel_half(x,t,A):
	theta=Angle[int(t/(2.*T))]
	theta=0
	#A=1.2
	#A=1./np.sqrt(2)
	R=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
	Rinv=np.linalg.inv(R)
	x=np.dot(x,R)
	if np.mod(t,T*2)<T:
		vx=np.zeros(x.shape[0])
		vy=A*(np.sin(2*np.pi*x[:,0]+PhaseY[int(t/(2.*T))])+
				np.abs(np.sin(2*np.pi*x[:,0]+PhaseY[int(t/(2.*T))])))/2.
		return np.dot(np.vstack((vx,vy)).T,Rinv)
	else:
		vx=A*(np.sin(2*np.pi*x[:,1]+PhaseX[int(t/(2*T))])+
				np.abs(np.sin(2*np.pi*x[:,1]+PhaseX[int(t/(2*T))])))/2.
		vy=np.zeros(x.shape[0])
		return np.dot(np.vstack((vx,vy)).T,Rinv)
		
def vel_halfsmooth(x,t,A):
	theta=Angle[int(t/(2.*T))]
	theta=0
	#A=1.2
	#A=1./np.sqrt(2)
	R=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
	Rinv=np.linalg.inv(R)
	x=np.dot(x,R)
	if np.mod(t,T*2)<T:
		vx=np.zeros(x.shape[0])
		vy=A*((np.sin(2*np.pi*x[:,0]+PhaseY[int(t/(2.*T))])+
				np.abs(np.sin(2*np.pi*x[:,0]+PhaseY[int(t/(2.*T))])))/2.)**2.
		return np.dot(np.vstack((vx,vy)).T,Rinv)
	else:
		vx=A*((np.sin(2*np.pi*x[:,1]+PhaseX[int(t/(2*T))])+
				np.abs(np.sin(2*np.pi*x[:,1]+PhaseX[int(t/(2*T))])))/2.)**2.
		vy=np.zeros(x.shape[0])
		return np.dot(np.vstack((vx,vy)).T,Rinv)
		
def vel_cubic(x,t,A):
	theta=Angle[int(t/(2.*T))]
	theta=0
	#A=1.2
	#A=1./np.sqrt(2)
	R=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
	Rinv=np.linalg.inv(R)
	x=np.dot(x,R)
	if np.mod(t,T*2)<T:
		vx=np.zeros(x.shape[0])
		vy=A*np.sin(2*np.pi*x[:,0]+PhaseY[int(t/(2.*T))])**3.
		return np.dot(np.vstack((vx,vy)).T,Rinv)
	else:
		vx=A*np.sin(2*np.pi*x[:,1]+PhaseX[int(t/(2*T))])**3.
		vy=np.zeros(x.shape[0])
		return np.dot(np.vstack((vx,vy)).T,Rinv)

def vel_standard(x,t,A):
	theta=Angle[int(t/(2.*T))]
	theta=0
	#A=1.2
	#A=1./np.sqrt(2)
	R=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
	Rinv=np.linalg.inv(R)
	x=np.dot(x,R)
	vx=A*x[:,1]
	vy=A*np.sin(2*np.pi*x[:,0])
	return np.dot(np.vstack((vx,vy)).T,Rinv)


def vel_bifreq(x,t,A):
	freq=2
	theta=Angle[int(t/(2.*T))]
	theta=0
	#A=1.2
	#A=1./np.sqrt(2)
	R=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
	Rinv=np.linalg.inv(R)
	x=np.dot(x,R)
	if np.mod(t,T*2)<T:
		vx=np.zeros(x.shape[0])
		vy=A*np.sin(2*np.pi*x[:,0]+PhaseY[int(t/(2.*T))])+A/freq*np.sin(freq*2*np.pi*x[:,0]+PhaseY[int(t/(2.*T))])
		return np.dot(np.vstack((vx,vy)).T,Rinv)
	else:
		vx=A*np.sin(2*np.pi*x[:,1]+PhaseX[int(t/(2*T))])**3.+A/freq*np.sin(freq*2*np.pi*x[:,1]+PhaseX[int(t/(2*T))])**3.
		vy=np.zeros(x.shape[0])
		return np.dot(np.vstack((vx,vy)).T,Rinv)
dir_out='./Compare_stretching_concentration/'
if not os.path.exists(dir_out):
	os.makedirs(dir_out)

def vel_nophase(x,t,A):
	theta=Angle[int(t/(2.*T))]
	theta=0
	#A=1.2
	#A=1./np.sqrt(2)
	R=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
	Rinv=np.linalg.inv(R)
	x=np.dot(x,R)
	if np.mod(t,T*2)<T:
		vx=np.zeros(x.shape[0])
		vy=A*np.sin(2*np.pi*x[:,0])
		return np.dot(np.vstack((vx,vy)).T,Rinv)
	else:
		vx=A*np.sin(2*np.pi*x[:,1])
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

def plot_per(L):
	Lmod=np.hstack((np.mod(L[:,1],1).reshape(-1,1),np.mod(-L[:,0],1).reshape(-1,1)))
	idmod=np.where(np.nansum(np.diff(Lmod,axis=0)**2.,axis=1)>0.01)[0]
	Lmod[idmod,:]=np.nan
#plt.plot(np.mod(L[:,1],1),np.mod(-L[:,0],1),'k.',alpha=1.0,linewidth=0.2,markersize=0.1)
	plt.plot(Lmod[:,0],Lmod[:,1],'k-',alpha=0.8,linewidth=0.1,markersize=0.1)

def plot_per2(L):
	Lmod=np.hstack((np.mod(L[:,0],1).reshape(-1,1),np.mod(L[:,1],1).reshape(-1,1)))
	idmod=np.where(np.nansum(np.diff(Lmod,axis=0)**2.,axis=1)>0.01)[0]
	Lmod[idmod,:]=np.nan
#plt.plot(np.mod(L[:,1],1),np.mod(-L[:,0],1),'k.',alpha=1.0,linewidth=0.2,markersize=0.1)
	plt.plot(Lmod[:,0],Lmod[:,1],'k-',alpha=0.8,linewidth=0.1,markersize=0.1)

def plot_wave(keyword):
	A=1
	n=1000
	t=0
	x=np.vstack((np.linspace(0,1,n),np.zeros(n))).T
#	xo=np.copy(x)
#	xo[:,0]=xo[:,0]-PhaseY[int(t/(2*T))]/2./np.pi
	v=globals()['vel_'+keyword](x,t,A)
	plt.plot(x[:,0],v[:,1],'k-')
	plt.plot(x[:,0],np.zeros(n),'k-')
	plt.box(False)
#	plt.ylim([-A,A])
	[plt.arrow(x[k,0], 0, 0, v[k,1],length_includes_head='True',
						head_width=0.03,width=0.001,color='k') for k in np.uint16(np.linspace(0,n-1,20))]

def find_pinch(L,dist_old,th_pinch):
# Maximum curvature finder
	kappa=curvature(L,dist_old)
	Mkappa=maximum_filter(kappa,50) # Maximum filter on a box of 50dx
	return np.where((Mkappa==kappa)&(kappa>th_pinch))[0]

def bin_operation(x,y,xi,op):
        r=np.zeros(xi.shape[0]-1)
        for i in range(xi.shape[0]-1):
                idx=np.where((x<=xi[i+1])&(x>=xi[i]))[0]
                r[i]=op(np.array(y[idx]))
        return r
#np.savetxt('./Compare_stretching_concentration/random_angles.txt',np.vstack((Angle,PhaseX,PhaseY)).T,header='Angle, PhaseX, PhaseY')
#% Advection parameters
fig=plt.figure()
Tmax_vec=np.array([2.,4.,6.,8.,10.,12.,14.])


def fractal(keyword):
	if keyword=='sine':
		return 1.75
	if keyword=='half':
		return 1.55
	if keyword=='cubic':
		return 1.68
	if keyword=='single':
		return 1.74
	return 1.7


def p_c(c,p_cmax,cmax,cm,c0):
	log_cm= np.log(cmax*(1+cm*(1/cmax-1/c0)))
	cv= 0.5
	p_c_cmax=1/c/np.sqrt(2*np.pi*cv**2)*np.exp(-(np.log(c)-log_cm)**2/(2*cv**2))
	return  np.trapz(p_cmax*p_c_cmax,cmax)


def pdf_selfconvolve(p,x,n):
	conv_pmf=p
	xc=2*x
	for i in range(n):
		conv_pmf = conv_pmf/np.sum(conv_pmf)
		conv_pmf = scipy.signal.fftconvolve(conv_pmf,p,'full')
#		conv_pmf = np.convolve(conv_pmf,conv_pmf)
		conv_pmf = np.interp(x,xc,conv_pmf[::2])
		conv_pmf = conv_pmf/np.sum(conv_pmf*np.diff(x)[0])
	return conv_pmf


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

def run_DSM_grid(sB,Lmax,aa,s,STOP_ON_LMAX=False):
	#STOP_ON_LMAX allows to stop simulation immediately when Lmax is reached (instead of finishing the time step)
	Cgrid=[]
	Lall=[]
	Gamma_all=[]
	Nall=[]
	rho_1_all=[]
	
	import h5py
	print('running DSM...')
	l0=0.3
	#l0=1
	#% Advection parameters
	INTERPOLATE='LINEAR'
	CURVATURE='None'

	PLOT=False
	dx=0.001 # Maximum distance between points, above which the line will be refined
	alpha=200*dx/np.pi
	dt=0.125 # dt needs to be small compare to the CFL condition
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
	Time=[t]

	Lmod=np.mod(L,1)
	C=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1+sB,sB),weights=dist_old/sB*np.minimum(S,1),density=False)[0]
	N=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1+sB,sB),weights=dist_old/sB,density=False)[0]
	Cgrid.append(C)
	
	Lall.append(np.sum(dist_old))
	lyap=np.average(-np.log(S),weights=W)
	sigma_lyap=np.average((-np.log(S)-lyap)**2,weights=W)
	Gamma_all.append([lyap,sigma_lyap])
	Nall.append([np.mean(N),np.var(N)])
	rho_1_all.append(np.average(S,weights=W))
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
#		rho1=np.maximum(1/S,S)
		#rho1=1/S
#		wrapped_time=wrapped_time+dt*(rho1)**2.
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
# 		dkappa=np.log(kappa_new)-np.log(kappa_old)
# 		dlKMean=np.average(dkappa,weights=weights)
# 		dlKVar=np.average((dkappa-dlKMean)**2./(1.-np.sum(weights**2.)),weights=weights) # Variance of Elongation

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
#			gamma=np.insert(gamma,ref+1,gamma[ref],axis=0)
#			kappa_old=curvature(L,dist_old)
#			wrapped_time=np.insert(wrapped_time,ref+1,wrapped_time[ref],axis=0)
#			dkappa=np.insert(dkappa,ref+1,dkappa[ref],axis=0)
			#W=S*dist_old/np.sum(S*dist_old) # points regularly spread in the curve have a weights inersely proportional to their elongation
			#W=S/np.sum(S)
			W=weights/np.sum(weights)
			#print np.sum(W)
	# =============================================================================
	
		# Map on a grid
		Lmod=np.mod(L,1)
		C=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1+sB,sB),weights=dist_old/sB*np.minimum(S,1),density=False)[0]
		N=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1+sB,sB),weights=dist_old/sB,density=False)[0]
		Cgrid.append(C)
		
		Lall.append(np.sum(dist_old))
		lyap=np.average(-np.log(S),weights=W)
		
		rho_1_all.append(np.average(S,weights=W))
		
		sigma_lyap=np.average((-np.log(S)-lyap)**2,weights=W)
		Gamma_all.append([lyap,sigma_lyap])
		Nall.append([np.mean(N),np.var(N)])
		
		# Update time
		t=t+dt
		Time.append(t)
		if STOP_ON_LMAX:
			RUN=(len(L)<Lmax)
		else:
			RUN=(len(L)<Lmax)|(t!=int(t))
		#print('Time:',t,' - Npts:',len(L), '- Cmax:',np.mean(1./np.sqrt(1.+wrapped_time/1000)),np.max(kappa_new))
	# End of MAIN LOOOP #######################
	#print('Computation time:', time.time() -ct)

	return Time,np.array(Cgrid),Lall,Gamma_all,rho_1_all,Nall
# Diffusion step
def diffusion_fourier(C,sigma):
	import scipy.ndimage
# 	Multidimensional Gaussian fourier filter.
# 	The array is multiplied with the fourier transform of a Gaussian kernel.
	input_ = np.fft.fft2(C)
	result = scipy.ndimage.fourier_gaussian(input_, sigma=sigma)
	return np.fft.ifft2(result).real

def run_DNS(a,n,ss,t,D):
	np.random.seed(seed=ss)
	PhaseX=np.random.rand(200)*2*np.pi
	PhaseY=np.random.rand(200)*2*np.pi
#	n=int(2**12) # Number of grid points
	# =============================================================================
	tmax=t # number of periods to compute
	dt=1/1  # Discretisation of time step
	X,Y=np.meshgrid(np.arange(n),np.arange(n))
	sigma=np.sqrt(2*D)*n
	# =============================================================================
	# iNITIAL condition
	l0=0.3
	radius=0.01
	#C[int(n/2):,:]=0.99 #half_plane
	#C[(X-n/2)**2+(Y-n/2)**2<(n/10)**2]=1
	# Initial condition corresponding to a single diffusive strip
	# Read Random angles
	C=np.zeros((n,n))					
	IdX=np.where((Y/n<l0/2)|(Y/n>(1-l0/2)))
	C[IdX]=np.exp(-(X[IdX]/n)**2/(2*radius**2))+np.exp(-((X[IdX]-n)/n)**2/(2*radius**2))
	# Save Variance and mean of C
	VarC=[]
	MeanC=[]
	VarC.append(np.var(C))
	MeanC.append(np.mean(C))
	for k in np.arange(tmax):
	# Choose flow map
		MapX=np.uint32(np.mod(np.round(X+0.5*dt*a*n*np.sin(Y/n*2*np.pi+PhaseX[k])),n))
		MapY=np.uint32(np.mod(np.round(Y+0.5*dt*a*n*np.sin(X/n*2*np.pi+PhaseY[k])),n))
		print('t=',k)
		# Advection
		for t in range(int(1/dt)):
			C[X.flatten(),MapY.flatten()]=C[X.flatten(),Y.flatten()]
			#C=diffusion_fourier(C, sigma*np.sqrt(dt))
			#C=diffusion(C, sigma*np.sqrt(dt))
		for t in range(int(1/dt)):
			C[MapX.flatten(),Y.flatten()]=C[X.flatten(),Y.flatten()]
			C=diffusion_fourier(C, sigma*np.sqrt(dt))
			#C=diffusion(C, sigma*np.sqrt(dt))
		VarC.append(np.var(C))
		MeanC.append(np.mean(C))
	return C,MeanC,VarC

#%% Run Simulations
#%%% Run Particle tracking and save (obsolet, use Fourier code)
Tmax=12.
#for Tmax in Tmax_vec:
Brownian=1e-3



PAR=True
PLOT=True
PLOT=False
dx=0.005 # Maximum distance between points, above which the line will be refined
alpha=200*dx/np.pi
dt=0.25 # dt needs to be small compare to the CFL condition
npar=12 # Number of parallel processors
tsave=0.1 # Time period between saving steps
Lmax=1e7 # Maximum number of points (prevent blow up)
th_pinch=100 # Curvature minimum (in nb of dx) to find a peak
Pe=1e2 # Peclet

l0=0.1
#Brownian=1e-4
Pe=1./np.sqrt(2.)/(Brownian**2./(2*dt))
N=int(5e6)

radius =0.01
l0=0.3
#dir_out='./Compare_stretching_concentration/{:1.0e}'.format(Brownian)
dir_out='./Compare_stretching_concentration/{:1.0e}periodic'.format(Brownian)
if not os.path.exists(dir_out):
	os.makedirs(dir_out)

def advect_sin(par):
	tmax,seed=par
	# new random seed
	np.random.seed(seed=seed)
	#Lr=(np.random.rand(N))**0.5*radius
	#Lt=(np.random.rand(N)-0.5)*2*np.pi
	#L=np.zeros((N,2))+0.5+(np.vstack((np.cos(Lt)*Lr,np.sin(Lt)*Lr))).T
	#L=np.zeros((N,2))+(np.vstack((np.cos(Lt)*Lr,np.sin(Lt)*Lr))).T
	Lx=(np.random.rand(N)-0.5)*l0
	Ly=(np.random.randn(N))*radius
	L=np.zeros((N,2))+(np.vstack((Lx,Ly))).T
	#L0=np.copy(L)
	n_T=int(tmax/dt)
	# Initialization
	t=0.0
	n=0
	
	Nbin=4096
	# Non Peridoic 
	#xi=np.linspace(-1,1,Nbin+1)
	# Peridodic
	xi=np.linspace(0,1,Nbin+1)
	
#		C=np.histogram2d(L[:,0],L[:,1],[xi,xi])[0]
#		cv2.imwrite(dir_out+'/{:04d}_{:02d}.tif'.format(int(0),int(seed)),np.uint16(C*2**16/200.))
	# MAIN PARALLEL LOOP #########	Lx=(np.random.rand(N)-0.5)*l0
	Ly=(np.random.randn(N))*radius##############
	C=np.histogram2d(L[:,0],L[:,1],[xi,xi])[0]
	maxC=np.max(C)*1.5
	cv2.imwrite(dir_out+'/{:04d}_{:02d}.tif'.format(int(t*10),int(seed)),np.uint16((np.float64(C)/(maxC)*2**16)))
	while t<tmax:
		# Push phase, reach Tmax
		
		L+=vel(L,t,1/np.sqrt(2))*dt+Brownian*np.random.randn(L.shape[0],L.shape[1])
		#tpull=t
		#Lpull=np.copy(L)
		# Pull phase
		t+=dt
		print(seed,t)
#			while tpull>0:
#				print(t,tpull)
#				Lpull+=-vel(Lpull,tpull)*dt+Brownian*np.random.randn(L.shape[0],L.shape[1])
#				tpull=tpull-dt
		# test Lpull=ML0
		#print(np.sum(np.sqrt(np.sum((Lpull-L0)**2.,axis=1))))
		# save results
		#Xmid=np.mean(L,axis=0)
		#C=np.histogram2d(L[:,0]-Xmid[0],L[:,1]-Xmid[1],[xi,xi])[0]
		# Noperiodic
#			C=np.histogram2d(Lpull[:,0],Lpull[:,1],[xi,xi])[0]
		#plt.imshow(C)
#			cv2.imwrite(dir_out+'/{:04d}_{:02d}.tif'.format(int(t*10),int(seed)),np.uint16(C*2**16/200.))
	#plt.imshow(C)
#			C=np.histogram2d(Lpull[:,0],Lpull[:,1],[xi,xi])[0]
	#plt.imshow(C)
		#Non Periodic
		#C=np.histogram2d(L[:,0],L[:,1],[xi,xi])[0]
		# Periodic
		C=np.histogram2d(np.mod(L[:,0],1),np.mod(L[:,1],1),[xi,xi])[0]
		cv2.imwrite(dir_out+'/{:04d}_{:02d}.tif'.format(int(t*10),int(seed)),np.uint16((np.float64(C)/(maxC)*2**16)))
	return 0
		#cv2.imwrite('{:04d}.tif'.format(int(tmax*10)),np.uint16(C*2**16))
	#%matplotlib inline
	#%matp2lotlib auto
	#np.savetxt(INTERPOLATE+'_{:d}PTS.txt'.format(L.shape[0]),np.vstack((KappaMean,logKappaMean,logKappaVar,Rhomean,logRhomean,logRhovar)).T)
npar=10

# Run pool of processes
if 'mpool' in locals():
	mpool.close()
if PAR:
	mpool = mp.Pool(processes=npar) # Pool of parallel workers
k=mpool.map(advect_sin,[[Tmax,i] for i in range(npar)])

# Merge pool images into 1
for t in np.arange(0,Tmax+dt,dt):
	flist=glob.glob(dir_out+'/{:04d}_*.tif'.format(int(t*10)))
	if len(flist)>0:
		Call=np.mean(np.array([np.float64(cv2.imread(f,2)) for f in flist]),axis=0)
		cv2.imwrite(dir_out+'/{:04d}.tif'.format(int(t*10)),np.uint16(Call))
		[os.remove(f) for f in flist]
	
	#xi=np.linspace(0,1,Nbin+1)
	#C=np.histogram2d(L[1:,0],L[1:,1],[xi,xi],weights=weights)
	#plt.imshow(np.log(C[0]+1e-4))
	
#%%%  Run DSM and save hdf5
import h5py

keyword='sine'
#keyword='shear'
#keyword='double'
#keyword='cubic'
#keyword='bifreq'
#keyword='standard'
#keyword='half'
#keyword='halfsmooth'
#keyword='single'
#keyword='single'
dir_out='./Compare_stretching_concentration/'+keyword+'/'
if not os.path.exists(dir_out):
	os.makedirs(dir_out)

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
Lmax=5e7 # Maximum number of points (prevent blow up)

A=1.2 # Ampluitude
#A=0.5
A=1/np.sqrt(2)
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

try:
	f.close()
except:
	print('No file open')
f = h5py.File('./Compare_stretching_concentration/'+keyword+'/DSM'+keyword+'_A{:1.1f}_l{:1.1f}.hdf5'.format(A,l0), 'w')
f.create_dataset('L_{:04d}'.format(int(t*10)), data=L)
f.create_dataset('wrapped_time_{:04d}'.format(int(t*10)), data=wrapped_time)
f.create_dataset('S_{:04d}'.format(int(t*10)), data=S)
f.create_dataset('Weights_{:04d}'.format(int(t*10)), data=W)

# MAIN PARALLEL LOOP #######################
#while (len(L)<Lmax):
while (t<14):
	
#	v=vel(L,t,A)
#	v=vel_cubic(L,t,A)
#	v=vel_bifreq(L,t,A)
#	v=vel_standard(L,t,A)
#	v=vel_half(L,t,A)
#	v=vel_nophase(L,t,A)

	v=locals()['vel_'+keyword](L,t,A)
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
	print('Time:',t,' - Npts:',len(L), '- Cmax:',np.mean(1./np.sqrt(1.+wrapped_time/1000)),np.max(kappa_new))
	
	if np.mod(t,1)==0:
		f.create_dataset('L_{:04d}'.format(int(t*10)), data=L)
		f.create_dataset('wrapped_time_{:04d}'.format(int(t*10)), data=wrapped_time)
		f.create_dataset('S_{:04d}'.format(int(t*10)), data=S)
		f.create_dataset('Weights_{:04d}'.format(int(t*10)), data=W)
		f.attrs['tmax']=t
# End of MAIN LOOOP #######################
print('Computation time:', time.time() -ct)
#plt.ylim([0,6])
#plt.xlim([0,20])
plt.ylim([0,20])
plt.xlim([-12,0])
plt.ylabel(r'log $\langle \rho \rangle_{s_B}$')

f.close()

plt.figure()
plt.plot(L[:,0],L[:,1],'.')
#%%% Compare variance between Lagrangian Eulerian
keyword='sine'

np.random.seed(seed=20)

Brownian = 1e-2# Diffusion strength
l0=0.3
radius=0.01
c0=1
s0=radius
A=0.2 # Ampluitude
#A=0.5
A=1/np.sqrt(2)
A=0.5

PhaseX=np.random.rand(100)*2*np.pi
PhaseY=np.random.rand(100)*2*np.pi
Angle=np.random.rand(100)*2*np.pi


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

#keyword='double'
#keyword='cubic'
#keyword='bifreq'
#keyword='standard'
#keyword='half'
#keyword='halfsmooth'
#keyword='single'
#keyword='single'
dir_out='./Compare_stretching_concentration/'+keyword+'/'
if not os.path.exists(dir_out):
	os.makedirs(dir_out)

#% Advection parameters
INTERPOLATE='LINEAR'
CURVATURE='DIFF'

PLOT=False
dx=0.001 # Maximum distance between points, above which the line will be refined
alpha=200*dx/np.pi
dt=0.25 # dt needs to be small compare to the CFL condition
npar=6 # Number of parallel processors
#tmax=Tmax # Maximum advection time
Lmax=1e7 # Maximum number of points (prevent blow up)


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

cmean=l0*s0*np.sqrt(np.pi)

varC_lag,varC_lag_2,varC_agg_rand,varC_agg_cor,Time,Time_eul=[],[],[],[],[],[]
meanC_lag,meanC_agg_cor,meanC_agg_rand=[],[],[]

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
# =============================================================================
# Save variables
# =============================================================================
D=(Brownian)**2./(2*dt)#+1e-08 # numerical diffusion ?
s0=radius#*np.sqrt(2)
dist=np.sum(np.sqrt(np.sum(np.diff(L,axis=0)**2.,axis=1)))
#Pe=1e7
Tau=D/s0**2*wrapped_time
Cmax=c0/np.sqrt(1.+4*Tau) #Gaussian
Si=S*np.sqrt(1+4*Tau)*s0

meanC_lag.append(l0*s0*np.sqrt(np.pi)*c0)
varC_lag_2.append(l0*np.average(1/S,weights=W)*np.mean(Si*Cmax**2.)*np.sqrt(np.pi/2))
varC_lag.append(l0*s0*c0*np.average(Cmax,weights=W)*np.sqrt(np.pi/2))
sB=np.mean(Si)
# Prediction of N
N=dist*sB/1*np.sqrt(np.pi/2)
#Lmod=np.mod(L,1)
# True N
#N=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1,sB),weights=dist_old/sB,density=False)[0]
#N=np.mean(N)
#print(np.sqrt(np.pi/2)*sB*l0*np.mean(Cmax)*N)
meanC_agg_rand.append(np.mean(Cmax)*N)
varC_agg_rand.append(np.var(Cmax)*N)
mC=np.average(Cmax,weights=W)
vC=np.average(Cmax**2.,weights=W)

meanC_agg_cor.append(cmean+np.mean(Cmax)*(1-cmean/c0))
varC_agg_cor.append(vC-mC**2.)
Time.append(t)

# =============================================================================
# MAIN PARALLEL LOOP #######################
while (len(L)<Lmax)|(t!=int(t)):
#while (t<12):
	
#	v=vel(L,t,A)
#	v=vel_cubic(L,t,A)
#	v=vel_bifreq(L,t,A)
#	v=vel_standard(L,t,A)
#	v=vel_half(L,t,A)
#	v=vel_nophase(L,t,A)

	v=locals()['vel_'+keyword](L,t,A)
	L+=v*dt
	
	# Compute stretching rates and elongations
	dist_new=np.sum(np.diff(L,axis=0)**2,1)**0.5
	gamma=dist_new/dist_old # elongation
	#gamma=np.maximum(dist_new/dist_old,dist_old/dist_new)
	S=S/gamma
	#wrapped_time=wrapped_time+dt*(1./S)**2.
	# Force positive elongation
	#rho1=np.abs(1./S-1.)+1.
	#rho1=np.maximum(1/S,S)
	rho1=1/S
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
	print('Time:',t,' - Npts:',len(L), '- Cmax:',np.mean(1./np.sqrt(1.+wrapped_time/1000)),np.max(kappa_new))
# =============================================================================
# Save variables
# =============================================================================
	D=(Brownian)**2./(2*dt)#+1e-08 # numerical diffusion ?
	s0=radius#*np.sqrt(2)
	dist=np.sum(np.sqrt(np.sum(np.diff(L,axis=0)**2.,axis=1)))
	#Pe=1e7
	Tau=D/s0**2*wrapped_time
	Cmax=c0/np.sqrt(1.+4*Tau) #Gaussian
	Si=S*np.sqrt(1+4*Tau)*s0
	meanC_lag.append(np.average(1/S,weights=W)*l0*np.mean(Si*Cmax)*np.sqrt(np.pi))
	varC_lag_2.append(np.average(1/S,weights=W)*l0*np.mean(Si*Cmax**2.)*np.sqrt(np.pi/2))
	varC_lag.append(l0*s0*c0*np.average(Cmax,weights=W)*np.sqrt(np.pi/2))
	sB=np.mean(Si)
	
#	N=dist*sB/1
	#N=cmean/np.mean(Cmax)
	Lmod=np.mod(L,1)
	N=np.mean(np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.linspace(0,1,int(1/sB))
									 ,weights=dist_old/sB*np.sqrt(2),density=False)[0])
	meanC_agg_rand.append(np.mean(Cmax)*N)
	varC_agg_rand.append(np.var(Cmax)*N)
	mC=np.average(Cmax,weights=W)
	vC=np.average(Cmax**2.,weights=W)
	varC_agg_cor.append(vC-mC**2.)
	meanC_agg_cor.append(cmean+np.mean(Cmax)*(1-cmean/c0))
	#varC_agg_cor.append(vC)
	Time.append(t)
# End of MAIN LOOOP #######################
print('Computation time:', time.time() -ct)
plt.style.use('~/.config/matplotlib/joris.mplstyle')


# Lyapunov
lyap=0.65
sigma2=0.5
#%
lyap=np.average(np.log(1/S),weights=W)/Time[-1]
sigma2=(np.average((np.log(1/S))**2./Time[-1],weights=W)-lyap**2.)/Time[-1]

# Eulerian
keyword='sine'
# Type of sine flow maps (classical is vel_sine)
def vel_sine_eul(X,Y,k,dt,a,n):
# 	MapX=np.mod(np.uint32(X+0.5*dt*a*n*np.sin(Y/n*2*np.pi+PhaseX[k])),n)
# 	MapY=np.mod(np.uint32(Y+0.5*dt*a*n*np.sin(X/n*2*np.pi+PhaseY[k])),n)
	MapX=np.uint32(np.mod(np.round(X+0.5*dt*a*n*np.sin(Y/n*2*np.pi+PhaseX[k])),n))
	MapY=np.uint32(np.mod(np.round(Y+0.5*dt*a*n*np.sin(X/n*2*np.pi+PhaseY[k])),n))
	return MapX,MapY

from scipy import ndimage, misc
import numpy.fft
from scipy.ndimage import gaussian_filter

# Diffusion step
def diffusion_fourier(C,sigma):
	input_ = np.fft.fft2(C)
	result = ndimage.fourier_gaussian(input_, sigma=sigma)
	return numpy.fft.ifft2(result).real

n=int(2**10) # Number of grid points
# =============================================================================
tmax=int(Time[-1]) # number of periods to compute
dt=1/1  # Discretisation of time step
X,Y=np.meshgrid(np.arange(n),np.arange(n))
D=Brownian**2./(2*0.25) # equivalent diffusion coeff
sigma=np.sqrt(2*D)*n
Pe=1./np.sqrt(2.)/D

# =============================================================================
# iNITIAL condition
#C[int(n/2):,:]=0.99 #half_plane
#C[(X-n/2)**2+(Y-n/2)**2<(n/10)**2]=1
# Initial condition corresponding to a single diffusive strip
# Read Random angles
C=np.zeros((n,n),dtype=np.float128)
IdX=np.where((Y/n<l0/2)|(Y/n>(1-l0/2)))
s0=radius#*np.sqrt(2)
C[IdX]=np.exp(-(X[IdX]/n)**2/(s0**2))+np.exp(-((X[IdX]-n)/n)**2/(s0**2))

# =============================================================================
dir_out='./'
if not os.path.exists(dir_out):
	os.makedirs(dir_out)
import scipy.stats
# Save Variance and mean of C
BINC=[]
PDFC=[]
PDFC_lin=[]
MeanC=[]
VarC=[]
binC=np.logspace(-10,0,100)
binC_lin=np.linspace(0,1,1000)
#PDFC.append(np.histogram(np.abs(C),binC,density=True)[0])
#PDFC_lin.append(np.histogram(C,binC_lin,density=True)[0])
Tv= np.arange(30)
Tv= np.arange(tmax)
binC=np.logspace(np.maximum(-10,np.log10(C.min())),np.log10(C.max()),100)
BINC.append(binC)
PDFC.append(np.histogram(np.abs(C),binC,density=True)[0])
PDFC_lin.append(np.histogram(C,binC_lin,density=True)[0])

MeanC.append(np.mean(C))
VarC.append(np.var(C))
Time_eul.append(0)

for k in Tv:
	print('t=',k)
	# Choose flow map
	MapX,MapY=locals()['vel_'+keyword+'_eul'](X,Y,k,dt,A,n)
	# Advection
	for t in range(int(1/dt)):
		C[X.flatten(),MapY.flatten()]=C[X.flatten(),Y.flatten()]
		#C=diffusion_fourier(C, sigma*np.sqrt(dt))
		#C=diffusion(C, sigma*np.sqrt(dt))
	for t in range(int(1/dt)):
		C[MapX.flatten(),Y.flatten()]=C[X.flatten(),Y.flatten()]
		C=diffusion_fourier(C, sigma*np.sqrt(dt))
		#C=diffusion(C, sigma*np.sqrt(dt))
	#C=C-np.mean(C)
	binC=np.logspace(np.maximum(-10,np.log10(np.abs(C).min())),np.log10(C.max()),100)
	BINC.append(binC)
	PDFC.append(np.histogram(np.abs(C),binC,density=True)[0])
#	print(PDFC[-1])
	PDFC_lin.append(np.histogram(C,binC_lin,density=True)[0])
	MeanC.append(np.mean(C))
	VarC.append(np.var(C))
	Time_eul.append(k+1)
	#fit_alpha, fit_loc, fit_beta=scipy.stats.gamma.fit(C.flatten()/C.mean())
#%

#plt.ylim([0,6])
#plt.xlim([0,20])
plt.ylim([0,20])
plt.xlim([-12,0])
plt.ylabel(r'log $\langle \rho \rangle_{s_B}$')


sBp=np.sqrt(D/(lyap+sigma2))

VarC=np.array(VarC)
VarC_lag=np.array(varC_lag)
VarC_lag_2=np.array(varC_lag_2)
VarC_agg_rand=np.array(varC_agg_rand)
VarC_agg_cor=np.array(varC_agg_cor)

MeanC=np.array(MeanC)
meanC_lag=np.array(meanC_lag)
meanC_agg_rand=np.array(meanC_agg_rand)
meanC_agg_cor=np.array(meanC_agg_cor)

tt=np.linspace(5,15,100)
lm=lyap+sigma2/2
cm=np.mean(C)

tagg=1/lm*np.log(1/(sB*l0))
Tagg=np.where(Time>tagg)[0][0]

factor_corr_agg=VarC_lag[Tagg]/VarC_agg_cor[Tagg]
plt.figure()
# plt.plot(tt,np.exp(-lyap**2./(2.*sigma2)*tt)*0.2,'k--',label=r"$-\mu^2/(2\sigma^2)$")
# plt.plot(tt,np.exp(-(lyap-sigma2/2)*tt)*0.2,'k-',label=r"$-(\mu-\sigma^2/2)$")
plt.plot([1/lm*np.log(1/(sB*l0)),1/lm*np.log(1/(sB*l0))],[1e-5,1e0],'0.5',linewidth=2)
plt.plot([1/lm*np.log(s0/sB),1/lm*np.log(s0/sB)],[1e-5,1e0],'0.75',linewidth=2)

plt.plot(Time_eul,VarC,'ko-',label='DNS',fillstyle='full')
plt.plot(Time,VarC_lag,'r*-',label='Isolated strip model')
#plt.plot(Time,VarC_lag_2,'g*',label='Isolated strip model')
plt.plot(Time,VarC_agg_rand,'rd-',label='Random aggregation model')
plt.plot(Time,factor_corr_agg*VarC_agg_cor,'b^-',label='Correlated aggregation model')
plt.yscale('log')
plt.ylim([1e-9,1e-1])
plt.legend(fontsize=8,ncol=1)
plt.xlabel('$t$')
plt.title(r'Sine Flow, $A={:1.1f}, s_B=1/{:1.0f}$'.format(A,1/sB))

plt.ylabel(r'$\sigma^2_c$')
plt.savefig('/home/joris/Dropbox/Articles/AggregationBatchelor/Figures/Sine_variance_A{:1.2f}_sB-1{:1.0f}.pdf'.format(A,1/sB),bbox_inches='tight')

#Mean
plt.figure()
# plt.plot(tt,np.exp(-lyap**2./(2.*sigma2)*tt)*0.2,'k--',label=r"$-\mu^2/(2\sigma^2)$")
# plt.plot(tt,np.exp(-(lyap-sigma2/2)*tt)*0.2,'k-',label=r"$-(\mu-\sigma^2/2)$")
# plt.plot([1/lm*np.log(1/(sB*l0)),1/lm*np.log(1/(sB*l0))],[1e-5,1e0],'0.5',linewidth=2)
# plt.plot([1/lm*np.log(s0/sB),1/lm*np.log(s0/sB)],[1e-5,1e0],'0.75',linewidth=2)

plt.plot(Time_eul,VarC/MeanC**2.,'ko-',label='DNS')
plt.plot(Time,VarC_lag/meanC_lag**2,'r*-',label='Isolated strip model')
#plt.plot(Time,VarC_lag_2,'g*',label='Isolated strip model')
plt.plot(Time,VarC_agg_rand/meanC_agg_rand**2.,'rd-',label='Random aggregation model')
plt.plot(Time,VarC_agg_cor/meanC_agg_cor**2.,'b^-',label='Correlated aggregation model')
plt.yscale('log')
plt.ylim([1e-3,1e3])
plt.legend(fontsize=8)
plt.xlabel('$t$')
plt.title('Sine Flow, A={:1.1f}, Pe={:1.0f}'.format(A,Pe))

plt.ylabel(r'$\sigma^2_c/\mu^2_c$')

#Mean
plt.figure()
# plt.plot(tt,np.exp(-lyap**2./(2.*sigma2)*tt)*0.2,'k--',label=r"$-\mu^2/(2\sigma^2)$")
# plt.plot(tt,np.exp(-(lyap-sigma2/2)*tt)*0.2,'k-',label=r"$-(\mu-\sigma^2/2)$")
# plt.plot([1/lm*np.log(1/(sB*l0)),1/lm*np.log(1/(sB*l0))],[1e-5,1e0],'0.5',linewidth=2)
# plt.plot([1/lm*np.log(s0/sB),1/lm*np.log(s0/sB)],[1e-5,1e0],'0.75',linewidth=2)

plt.plot(Time_eul,MeanC,'ko-',label='DNS')
plt.plot(Time,meanC_lag,'r*-',label='Isolated strip model')
#plt.plot(Time,VarC_lag_2,'g*',label='Isolated strip model')
plt.plot(Time,meanC_agg_rand,'rd-',label='Random aggregation model')
plt.plot(Time,meanC_agg_cor,'b^-',label='Correlated aggregation model')
plt.yscale('log')
plt.ylim([1e-3,1e0])
plt.legend(fontsize=8)
plt.xlabel('$t$')
plt.title('Sine Flow, A={:1.1f}, Pe={:1.0f}'.format(A,Pe))

plt.ylabel(r'$\sigma^2/\mu^2$')

I=C
# plt.figure(figsize=(10,10))
# #plt.imshow(np.log(I+I[I>0].min()),extent=[0,1,0,1],cmap=cm_fire)
# plt.imshow(I,extent=[0,1,0,1],cmap=cm_fire,alpha=0.5)
# #plt.imshow(np.log((I>0.2*np.max(I))*I**2.),extent=[0,1,0,1],cmap=cm_fire,alpha=0.5)
# Lmod=np.hstack((np.mod(L[:,1],1).reshape(-1,1),np.mod(-L[:,0],1).reshape(-1,1)))
# idmod=np.where(np.nansum(np.diff(Lmod,axis=0)**2.,axis=1)>0.01)[0]
# Lmod[idmod,:]=np.nan
# #plt.plot(np.mod(L[:,1],1),np.mod(-L[:,0],1),'k.',alpha=1.0,linewidth=0.2,markersize=0.1)
# plt.plot(Lmod[:,0],Lmod[:,1],'k-',alpha=0.8,linewidth=0.1,markersize=0.1)
# plt.axis('off')
# np.sum((I-I.mean())**2.*(I>0.1*np.max(I)))

# np.sum((I-I.mean())**2.)


# np.var(I)
# Icut=np.copy(I)
# Icut[I<0.2*np.max(I)]=I.mean()
# np.var(Icut)

# plt.figure(figsize=(10,10))
# #plt.imshow(np.log(I+I[I>0].min()),extent=[0,1,0,1],cmap=cm_fire)
# #plt.imshow(I,extent=[0,1,0,1],cmap=cm_fire)
# plt.imshow((np.abs(I-I.mean())<0.9*np.std(I)),extent=[0,1,0,1],alpha=0.5)
# Lmod=np.hstack((np.mod(L[:,1],1).reshape(-1,1),np.mod(-L[:,0],1).reshape(-1,1)))
# idmod=np.where(np.nansum(np.diff(Lmod,axis=0)**2.,axis=1)>0.01)[0]
# Lmod[idmod,:]=np.nan
# #plt.plot(np.mod(L[:,1],1),np.mod(-L[:,0],1),'k.',alpha=1.0,linewidth=0.2,markersize=0.1)
# plt.plot(Lmod[:,0],Lmod[:,1],'k-',alpha=0.8,linewidth=0.1,markersize=0.1)
# plt.axis('off')

plt.figure(figsize=(10,10))
#plt.imshow(np.log(I+I[I>0].min()),extent=[0,1,0,1],cmap=cm_fire)
plt.imshow(I,extent=[0,1,0,1],cmap=cm_fire,alpha=0.5)
#plt.imshow((np.abs(I-I.mean())<0.9*np.std(I)),extent=[0,1,0,1])
Lmod=np.hstack((np.mod(L[:,1],1).reshape(-1,1),np.mod(-L[:,0],1).reshape(-1,1)))
idmod=np.where(np.nansum(np.diff(Lmod,axis=0)**2.,axis=1)>0.01)[0]
Lmod[idmod,:]=np.nan
#plt.plot(np.mod(L[:,1],1),np.mod(-L[:,0],1),'k.',alpha=1.0,linewidth=0.2,markersize=0.1)
plt.plot(Lmod[:,0],Lmod[:,1],'k-',alpha=0.8,linewidth=0.1,markersize=0.1)
plt.axis('off')
plt.text(0.005,0.96,'$A={:1.2f}, s_B=1/{:1.0f}$'.format(A,1/sB),color='k',fontsize=30,backgroundcolor='w')
plt.savefig('/home/joris/Dropbox/Articles/AggregationBatchelor/Figures/Sine_variance_A{:1.2f}_sB-1{:1.0f}.png'.format(A,1/sB),bbox_inches='tight')
#%%% *  Compare variance between Lagrangian Eulerian, various A
keyword='sine'

fig,ax=plt.subplots(1,3,figsize=(8,3),sharey=True)

Brownian = 1e-3# Diffusion strength
l0=1.0
radius=0.1
c0=1
s0=radius
A=0.2 # Ampluitude
#A=0.5

np.random.seed(seed=13)
PhaseX=np.random.rand(100)*2*np.pi
PhaseY=np.random.rand(100)*2*np.pi
Angle=np.random.rand(100)*2*np.pi

for i,A in enumerate([0.5,0.8,1.5]):
#A=0.5
	
	
	
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
	
	#keyword='double'
	#keyword='cubic'
	#keyword='bifreq'
	#keyword='standard'
	#keyword='half'
	#keyword='halfsmooth'
	#keyword='single'
	#keyword='single'
	dir_out='./Compare_stretching_concentration/'+keyword+'/'
	if not os.path.exists(dir_out):
		os.makedirs(dir_out)
	
	#% Advection parameters
	INTERPOLATE='LINEAR'
	CURVATURE='DIFF'
	
	PLOT=False
	dx=0.001 # Maximum distance between points, above which the line will be refined
	alpha=200*dx/np.pi
	dt=0.25 # dt needs to be small compare to the CFL condition
	npar=6 # Number of parallel processors
	#tmax=Tmax # Maximum advection time
	Lmax=1e7 # Maximum number of points (prevent blow up)
	
	
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
	
	cmean=l0*s0*np.sqrt(np.pi)
	
	varC_lag,varC_lag_2,varC_agg_rand,varC_agg_cor,Time,Time_eul=[],[],[],[],[],[]
	meanC_lag,meanC_agg_cor,meanC_agg_rand=[],[],[]
	
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
	# =============================================================================
	# Save variables
	# =============================================================================
	D=(Brownian)**2./(2*dt)#+1e-08 # numerical diffusion ?
	s0=radius#*np.sqrt(2)
	dist=np.sum(np.sqrt(np.sum(np.diff(L,axis=0)**2.,axis=1)))
	#Pe=1e7
	Tau=D/s0**2*wrapped_time
	Cmax=c0/np.sqrt(1.+4*Tau) #Gaussian
	Si=S*np.sqrt(1+4*Tau)*s0
	
	meanC_lag.append(l0*s0*np.sqrt(np.pi)*c0)
	varC_lag_2.append(l0*np.average(1/S,weights=W)*np.mean(Si*Cmax**2.)*np.sqrt(np.pi/2))
	varC_lag.append(l0*s0*c0*np.average(Cmax,weights=W)*np.sqrt(np.pi/2))
	sB=np.mean(Si)
	# Prediction of N
	N=dist*sB/1*np.sqrt(np.pi/2)
	#Lmod=np.mod(L,1)
	# True N
	#N=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1,sB),weights=dist_old/sB,density=False)[0]
	#N=np.mean(N)
	#print(np.sqrt(np.pi/2)*sB*l0*np.mean(Cmax)*N)
	meanC_agg_rand.append(np.mean(Cmax)*N)
	varC_agg_rand.append(np.var(Cmax)*N)
	mC=np.average(Cmax,weights=W)
	vC=np.average(Cmax**2.,weights=W)
	
	meanC_agg_cor.append(cmean+np.mean(Cmax)*(1-cmean/c0))
	varC_agg_cor.append(vC-mC**2.)
	Time.append(t)
	
	# =============================================================================
	# MAIN PARALLEL LOOP #######################
	while (len(L)<Lmax)|(t!=int(t)):
	#while (t<12):
		
	#	v=vel(L,t,A)
	#	v=vel_cubic(L,t,A)
	#	v=vel_bifreq(L,t,A)
	#	v=vel_standard(L,t,A)
	#	v=vel_half(L,t,A)
	#	v=vel_nophase(L,t,A)
	
		v=locals()['vel_'+keyword](L,t,A)
		L+=v*dt
		
		# Compute stretching rates and elongations
		dist_new=np.sum(np.diff(L,axis=0)**2,1)**0.5
		gamma=dist_new/dist_old # elongation
		#gamma=np.maximum(dist_new/dist_old,dist_old/dist_new)
		S=S/gamma
		#wrapped_time=wrapped_time+dt*(1./S)**2.
		# Force positive elongation
		#rho1=np.abs(1./S-1.)+1.
		#rho1=np.maximum(1/S,S)
		rho1=1/S
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
		print('Time:',t,' - Npts:',len(L), '- Cmax:',np.mean(1./np.sqrt(1.+wrapped_time/1000)),np.max(kappa_new))
	# =============================================================================
	# Save variables
	# =============================================================================
		D=(Brownian)**2./(2*dt)#+1e-08 # numerical diffusion ?
		s0=radius#*np.sqrt(2)
		dist=np.sum(np.sqrt(np.sum(np.diff(L,axis=0)**2.,axis=1)))
		#Pe=1e7
		Tau=D/s0**2*wrapped_time
		Cmax=c0/np.sqrt(1.+4*Tau) #Gaussian
		Si=S*np.sqrt(1+4*Tau)*s0
		meanC_lag.append(np.average(1/S,weights=W)*l0*np.mean(Si*Cmax)*np.sqrt(np.pi))
		varC_lag_2.append(np.average(1/S,weights=W)*l0*np.mean(Si*Cmax**2.)*np.sqrt(np.pi/2))
		varC_lag.append(l0*s0*c0*np.average(Cmax,weights=W)*np.sqrt(np.pi/2))
		sB=np.mean(Si)
		
	#	N=dist*sB/1
		#N=cmean/np.mean(Cmax)
		Lmod=np.mod(L,1)
		N=np.mean(np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.linspace(0,1,int(1/sB))
										 ,weights=dist_old/sB*np.sqrt(2),density=False)[0])
		meanC_agg_rand.append(np.mean(Cmax)*N)
		varC_agg_rand.append(np.var(Cmax)*N)
		mC=np.average(Cmax,weights=W)
		vC=np.average(Cmax**2.,weights=W)
		varC_agg_cor.append(vC-mC**2.)
		meanC_agg_cor.append(cmean+np.mean(Cmax)*(1-cmean/c0))
		#varC_agg_cor.append(vC)
		Time.append(t)
	# End of MAIN LOOOP #######################
	print('Computation time:', time.time() -ct)
	plt.style.use('~/.config/matplotlib/joris.mplstyle')
	
	
	# Lyapunov
	lyap=0.65
	sigma2=0.5
	#%
	lyap=np.average(np.log(1/S),weights=W)/Time[-1]
	sigma2=(np.average((np.log(1/S))**2./Time[-1],weights=W)-lyap**2.)/Time[-1]
	
	# Eulerian
	keyword='sine'
	# Type of sine flow maps (classical is vel_sine)
	def vel_sine_eul(X,Y,k,dt,a,n):
	# 	MapX=np.mod(np.uint32(X+0.5*dt*a*n*np.sin(Y/n*2*np.pi+PhaseX[k])),n)
	# 	MapY=np.mod(np.uint32(Y+0.5*dt*a*n*np.sin(X/n*2*np.pi+PhaseY[k])),n)
		MapX=np.uint32(np.mod(np.round(X+0.5*dt*a*n*np.sin(Y/n*2*np.pi+PhaseX[k])),n))
		MapY=np.uint32(np.mod(np.round(Y+0.5*dt*a*n*np.sin(X/n*2*np.pi+PhaseY[k])),n))
		return MapX,MapY
	
	from scipy import ndimage, misc
	import numpy.fft
	from scipy.ndimage import gaussian_filter
	
	# Diffusion step
	def diffusion_fourier(C,sigma):
		input_ = np.fft.fft2(C)
		result = ndimage.fourier_gaussian(input_, sigma=sigma)
		return numpy.fft.ifft2(result).real
	
	n=int(2**11) # Number of grid points
	# =============================================================================
	tmax=int(Time[-1]) # number of periods to compute
	dt=1/1  # Discretisation of time step
	X,Y=np.meshgrid(np.arange(n),np.arange(n))
	D=Brownian**2./(2*0.25) # equivalent diffusion coeff
	sigma=np.sqrt(2*D)*n
	Pe=1./np.sqrt(2.)/D
	
	# =============================================================================
	# iNITIAL condition
	#C[int(n/2):,:]=0.99 #half_plane
	#C[(X-n/2)**2+(Y-n/2)**2<(n/10)**2]=1
	# Initial condition corresponding to a single diffusive strip
	# Read Random angles
	C=np.zeros((n,n),dtype=np.float128)
	IdX=np.where((Y/n<=l0/2)|(Y/n>=(1-l0/2)))
	s0=radius#*np.sqrt(2)
	C[IdX]=np.exp(-(X[IdX]/n)**2/(s0**2))+np.exp(-((X[IdX]-n)/n)**2/(s0**2))
	
	# =============================================================================
	dir_out='./'
	if not os.path.exists(dir_out):
		os.makedirs(dir_out)
	import scipy.stats
	# Save Variance and mean of C
	BINC=[]
	PDFC=[]
	PDFC_lin=[]
	MeanC=[]
	VarC=[]
	binC=np.logspace(-10,0,100)
	binC_lin=np.linspace(0,1,1000)
	#PDFC.append(np.histogram(np.abs(C),binC,density=True)[0])
	#PDFC_lin.append(np.histogram(C,binC_lin,density=True)[0])
	Tv= np.arange(30)
	Tv= np.arange(tmax)
	binC=np.logspace(np.maximum(-10,np.log10(C.min())),np.log10(C.max()),100)
	BINC.append(binC)
	PDFC.append(np.histogram(np.abs(C),binC,density=True)[0])
	PDFC_lin.append(np.histogram(C,binC_lin,density=True)[0])
	
	MeanC.append(np.mean(C))
	VarC.append(np.var(C))
	Time_eul.append(0)
	
	for k in Tv:
		print('t=',k)
		# Choose flow map
		MapX,MapY=locals()['vel_'+keyword+'_eul'](X,Y,k,dt,A,n)
		# Advection
		for t in range(int(1/dt)):
			C[X.flatten(),MapY.flatten()]=C[X.flatten(),Y.flatten()]
			#C=diffusion_fourier(C, sigma*np.sqrt(dt))
			#C=diffusion(C, sigma*np.sqrt(dt))
		for t in range(int(1/dt)):
			C[MapX.flatten(),Y.flatten()]=C[X.flatten(),Y.flatten()]
			C=diffusion_fourier(C, sigma*np.sqrt(dt))
			#C=diffusion(C, sigma*np.sqrt(dt))
		#C=C-np.mean(C)
		binC=np.logspace(np.maximum(-10,np.log10(np.abs(C).min())),np.log10(C.max()),100)
		BINC.append(binC)
		PDFC.append(np.histogram(np.abs(C),binC,density=True)[0])
	#	print(PDFC[-1])
		PDFC_lin.append(np.histogram(C,binC_lin,density=True)[0])
		MeanC.append(np.mean(C))
		VarC.append(np.var(C))
		Time_eul.append(k+1)
		#fit_alpha, fit_loc, fit_beta=scipy.stats.gamma.fit(C.flatten()/C.mean())
	#%
	
	#plt.ylim([0,6])
	#plt.xlim([0,20])
# 	plt.ylim([0,20])
# 	plt.xlim([-12,0])
# 	plt.ylabel(r'log $\langle \rho \rangle_{s_B}$')
	
	
	sBp=np.sqrt(D/(lyap+sigma2))
	
	VarC=np.array(VarC)
	VarC_lag=np.array(varC_lag)
	VarC_lag_2=np.array(varC_lag_2)
	VarC_agg_rand=np.array(varC_agg_rand)
	VarC_agg_cor=np.array(varC_agg_cor)
	
	MeanC=np.array(MeanC)
	meanC_lag=np.array(meanC_lag)
	meanC_agg_rand=np.array(meanC_agg_rand)
	meanC_agg_cor=np.array(meanC_agg_cor)
	
	tt=np.linspace(5,15,100)
	lm=lyap+sigma2/2
	cm=np.mean(C)
	
	tagg=1/lm*np.log(1/(sB*l0))
	Tagg=np.where(Time>tagg)[0]
	if len(Tagg)==0:
		Tagg=int(len(Time)-1)
	else:
		Tagg=Tagg[0]
	
	factor_corr_agg=VarC_lag[Tagg]/VarC_agg_cor[Tagg]
	# plt.plot(tt,np.exp(-lyap**2./(2.*sigma2)*tt)*0.2,'k--',label=r"$-\mu^2/(2\sigma^2)$")
	# plt.plot(tt,np.exp(-(lyap-sigma2/2)*tt)*0.2,'k-',label=r"$-(\mu-\sigma^2/2)$")
	ax[i].plot([1/lm*np.log(1/(sB*l0)),1/lm*np.log(1/(sB*l0))],[1e-5,1e0],'0.5',linewidth=2)
	ax[i].plot([1/lm*np.log(s0/sB),1/lm*np.log(s0/sB)],[1e-5,1e0],'0.75',linewidth=2)
	
	ax[i].plot(Time_eul,VarC,'ko',label='DNS',fillstyle='full')
	ax[i].plot(Time,VarC_lag,'r*',label='Isolated strip')
	#plt.plot(Time,VarC_lag_2,'g*',label='Isolated strip model')
	ax[i].plot(Time,VarC_agg_rand,'rd',label='Random aggregation')
	ax[i].plot(Time,factor_corr_agg*VarC_agg_cor,'b^',label='Correlated aggregation')
	ax[i].set_yscale('log')
	ax[i].set_ylim([1e-4,1e-1])
	ax[i].set_xlabel('$t$')
ax[0].legend(fontsize=8,ncol=1)
#ax[i].title(r'Sine Flow, $A={:1.1f}, s_B=1/{:1.0f}$'.format(A,1/sB))

ax[0].set_ylabel(r'$\sigma^2_c$')
plt.savefig('/home/joris/Dropbox/Articles/AggregationBatchelor/Figures/Sine_variance_A_sB1_{:1.0f}.pdf'.format(1/sB),bbox_inches='tight')
#%%% *  Compare mean over threshold between Lagrangian Eulerian, various A
keyword='sine'

epsilon = 0.0001

fig,ax=plt.subplots(1,3,figsize=(8,3),sharey=True)

Brownian = 1e-2# Diffusion strength
l0=1.0
radius=0.1
c0=1
s0=0.1

np.random.seed(seed=16)
PhaseX=np.random.rand(100)*2*np.pi
PhaseY=np.random.rand(100)*2*np.pi
Angle=np.random.rand(100)*2*np.pi

for i,A in enumerate([0.5,0.8,1.5]):
#A=0.5
	
	
	
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
	
	#keyword='double'
	#keyword='cubic'
	#keyword='bifreq'
	#keyword='standard'
	#keyword='half'
	#keyword='halfsmooth'
	#keyword='single'
	#keyword='single'
	dir_out='./Compare_stretching_concentration/'+keyword+'/'
	if not os.path.exists(dir_out):
		os.makedirs(dir_out)
	
	#% Advection parameters
	INTERPOLATE='LINEAR'
	CURVATURE='DIFF'
	
	PLOT=False
	dx=0.001 # Maximum distance between points, above which the line will be refined
	alpha=200*dx/np.pi
	dt=0.25 # dt needs to be small compare to the CFL condition
	npar=6 # Number of parallel processors
	#tmax=Tmax # Maximum advection time
	Lmax=1e5 # Maximum number of points (prevent blow up)
	
	
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
	
	cmean=l0*s0*np.sqrt(np.pi)
	
	varC_lag,varC_lag_2,varC_agg_rand,varC_agg_cor,Time,Time_eul=[],[],[],[],[],[]
	meanC_lag,meanC_agg_cor,meanC_agg_rand=[],[],[]
	
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
	# =============================================================================
	# Save variables
	# =============================================================================
	D=(Brownian)**2./(2*dt)#+1e-08 # numerical diffusion ?
#	s0=radius#*np.sqrt(2)
	dist=np.sum(np.sqrt(np.sum(np.diff(L,axis=0)**2.,axis=1)))
	#Pe=1e7
	Tau=D/s0**2*wrapped_time
	Cmax=c0/np.sqrt(1.+4*Tau) #Gaussian
	Si=S*np.sqrt(1+4*Tau)*s0
	
	meanC_lag.append(np.mean(Cmax)*0.3)
	varC_lag_2.append(l0*np.average(1/S,weights=W)*np.mean(Si*Cmax**2.)*np.sqrt(np.pi/2))
	varC_lag.append(np.mean(Cmax**2.)*0.12)
	sB=np.mean(Si)
	# Prediction of N
	N=dist*sB/1*np.sqrt(np.pi/2)
	#Lmod=np.mod(L,1)
	# True N
	#N=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1,sB),weights=dist_old/sB,density=False)[0]
	#N=np.mean(N)
	#print(np.sqrt(np.pi/2)*sB*l0*np.mean(Cmax)*N)
	meanC_agg_rand.append(np.mean(Cmax)*N)
	varC_agg_rand.append(np.var(Cmax)*N)
	mC=np.average(Cmax,weights=W)
	vC=np.average(Cmax**2.,weights=W)
	
	meanC_agg_cor.append(cmean+np.mean(Cmax)*(1-cmean/c0))
	varC_agg_cor.append(vC-mC**2.)
	Time.append(t)
	
	# =============================================================================
	# MAIN PARALLEL LOOP #######################
	while (len(L)<Lmax)|(t!=int(t)):
	#while (t<12):
		
	#	v=vel(L,t,A)
	#	v=vel_cubic(L,t,A)
	#	v=vel_bifreq(L,t,A)
	#	v=vel_standard(L,t,A)
	#	v=vel_half(L,t,A)
	#	v=vel_nophase(L,t,A)
	
		v=locals()['vel_'+keyword](L,t,A)
		L+=v*dt
		
		# Compute stretching rates and elongations
		dist_new=np.sum(np.diff(L,axis=0)**2,1)**0.5
		gamma=dist_new/dist_old # elongation
		#gamma=np.maximum(dist_new/dist_old,dist_old/dist_new)
		S=S/gamma
		#wrapped_time=wrapped_time+dt*(1./S)**2.
		# Force positive elongation
		#rho1=np.abs(1./S-1.)+1.
		#rho1=np.maximum(1/S,S)
		rho1=1/S
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
		print('Time:',t,' - Npts:',len(L), '- Cmax:',np.mean(1./np.sqrt(1.+wrapped_time/1000)),np.max(kappa_new))
	# =============================================================================
	# Save variables
	# =============================================================================
		D=(Brownian)**2./(2*dt)#+1e-08 # numerical diffusion ?
#		s0=radius#*np.sqrt(2)
		dist=np.sum(np.sqrt(np.sum(np.diff(L,axis=0)**2.,axis=1)))
		#Pe=1e7
		Tau=D/s0**2*wrapped_time
		Cmax=c0/np.sqrt(1.+4*Tau) #Gaussian
		Si=S*np.sqrt(1+4*Tau)*s0
		meanC_lag.append(np.mean(Cmax)*0.3)
		varC_lag_2.append(np.average(1/S,weights=W)*l0*np.mean(Si*Cmax**2.)*np.sqrt(np.pi/2))
		varC_lag.append(np.mean(Cmax**2.)*0.12)
		sB=np.mean(Si)
		
	#	N=dist*sB/1
		#N=cmean/np.mean(Cmax)
		Lmod=np.mod(L,1)
		N=np.mean(np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.linspace(0,1,int(1/sB))
										 ,weights=dist_old/sB*np.sqrt(2),density=False)[0])
		meanC_agg_rand.append(np.mean(Cmax)*N)
		varC_agg_rand.append(np.var(Cmax)*N)
		mC=np.average(Cmax,weights=W)
		vC=np.average(Cmax**2.,weights=W)
		varC_agg_cor.append(vC-mC**2.)
		meanC_agg_cor.append(cmean+np.mean(Cmax)*(1-cmean/c0))
		#varC_agg_cor.append(vC)
		Time.append(t)
	# End of MAIN LOOOP #######################
	print('Computation time:', time.time() -ct)
	plt.style.use('~/.config/matplotlib/joris.mplstyle')
	
	
	# Lyapunov
	lyap=0.65
	sigma2=0.5
	#%
	lyap=np.average(np.log(1/S),weights=W)/Time[-1]
	sigma2=(np.average((np.log(1/S))**2./Time[-1],weights=W)-lyap**2.)/Time[-1]
	
	# Eulerian
	keyword='sine'
	# Type of sine flow maps (classical is vel_sine)
	def vel_sine_eul(X,Y,k,dt,a,n):
	# 	MapX=np.mod(np.uint32(X+0.5*dt*a*n*np.sin(Y/n*2*np.pi+PhaseX[k])),n)
	# 	MapY=np.mod(np.uint32(Y+0.5*dt*a*n*np.sin(X/n*2*np.pi+PhaseY[k])),n)
		MapX=np.uint32(np.mod(np.round(X+0.5*dt*a*n*np.sin(Y/n*2*np.pi+PhaseX[k])),n))
		MapY=np.uint32(np.mod(np.round(Y+0.5*dt*a*n*np.sin(X/n*2*np.pi+PhaseY[k])),n))
		return MapX,MapY
	
	from scipy import ndimage, misc
	import numpy.fft
	from scipy.ndimage import gaussian_filter
	
	# Diffusion step
	def diffusion_fourier(C,sigma):
		input_ = np.fft.fft2(C)
		result = ndimage.fourier_gaussian(input_, sigma=sigma)
		return numpy.fft.ifft2(result).real
	
	n=int(2**10) # Number of grid points
	# =============================================================================
	tmax=int(Time[-1]) # number of periods to compute
	dt=1/1  # Discretisation of time step
	X,Y=np.meshgrid(np.arange(n),np.arange(n))
	D=Brownian**2./(2*0.25) # equivalent diffusion coeff
	sigma=np.sqrt(2*D)*n
	Pe=1./np.sqrt(2.)/D
	
	# =============================================================================
	# iNITIAL condition
	#C[int(n/2):,:]=0.99 #half_plane
	#C[(X-n/2)**2+(Y-n/2)**2<(n/10)**2]=1
	# Initial condition corresponding to a single diffusive strip
	# Read Random angles
	C=np.zeros((n,n),dtype=np.float128)
	IdX=np.where((Y/n<=l0/2)|(Y/n>=(1-l0/2)))
#	s0=radius#*np.sqrt(2)
	C[IdX]=np.exp(-(X[IdX]/n)**2/(s0**2))+np.exp(-((X[IdX]-n)/n)**2/(s0**2))
	
	# =============================================================================
	dir_out='./'
	if not os.path.exists(dir_out):
		os.makedirs(dir_out)
	import scipy.stats
	# Save Variance and mean of C
	BINC=[]
	PDFC=[]
	PDFC_lin=[]
	MeanC=[]
	VarC=[]
	binC=np.logspace(-10,0,100)
	binC_lin=np.linspace(0,1,1000)
	#PDFC.append(np.histogram(np.abs(C),binC,density=True)[0])
	#PDFC_lin.append(np.histogram(C,binC_lin,density=True)[0])
	Tv= np.arange(30)
	Tv= np.arange(tmax)
	binC=np.logspace(np.maximum(-10,np.log10(C.min())),np.log10(C.max()),100)
	BINC.append(binC)
	PDFC.append(np.histogram(np.abs(C),binC,density=True)[0])
	PDFC_lin.append(np.histogram(C,binC_lin,density=True)[0])
	
	MeanC.append(np.mean(C[C>epsilon]))
	VarC.append(np.var(C[C>epsilon]))
	Time_eul.append(0)
	
	for k in Tv:
		print('t=',k)
		# Choose flow map
		MapX,MapY=locals()['vel_'+keyword+'_eul'](X,Y,k,dt,A,n)
		# Advection
		for t in range(int(1/dt)):
			C[X.flatten(),MapY.flatten()]=C[X.flatten(),Y.flatten()]
			#C=diffusion_fourier(C, sigma*np.sqrt(dt))
			#C=diffusion(C, sigma*np.sqrt(dt))
		for t in range(int(1/dt)):
			C[MapX.flatten(),Y.flatten()]=C[X.flatten(),Y.flatten()]
			C=diffusion_fourier(C, sigma*np.sqrt(dt))
			#C=diffusion(C, sigma*np.sqrt(dt))
		#C=C-np.mean(C)
		binC=np.logspace(np.maximum(-10,np.log10(np.abs(C).min())),np.log10(C.max()),100)
		BINC.append(binC)
		PDFC.append(np.histogram(np.abs(C),binC,density=True)[0])
	#	print(PDFC[-1])
		PDFC_lin.append(np.histogram(C,binC_lin,density=True)[0])
		MeanC.append(np.mean(C[C>epsilon]))
		VarC.append(np.var(C[C>epsilon]))
		Time_eul.append(k+1)
		#fit_alpha, fit_loc, fit_beta=scipy.stats.gamma.fit(C.flatten()/C.mean())
	#%
	
	#plt.ylim([0,6])
	#plt.xlim([0,20])
# 	plt.ylim([0,20])
# 	plt.xlim([-12,0])
# 	plt.ylabel(r'log $\langle \rho \rangle_{s_B}$')
	
	
	sBp=np.sqrt(D/(lyap+sigma2))
	
	VarC=np.array(VarC)
	VarC_lag=np.array(varC_lag)
	VarC_lag_2=np.array(varC_lag_2)
	VarC_agg_rand=np.array(varC_agg_rand)
	VarC_agg_cor=np.array(varC_agg_cor)
	
	MeanC=np.array(MeanC)
	meanC_lag=np.array(meanC_lag)
	meanC_agg_rand=np.array(meanC_agg_rand)
	meanC_agg_cor=np.array(meanC_agg_cor)
	
	tt=np.linspace(5,15,100)
	lm=lyap+sigma2/2
	cm=np.mean(C)
	
	tagg=1/lm*np.log(1/(sB*l0))
	Tagg=np.where(Time>tagg)[0]
	if len(Tagg)==0:
		Tagg=int(len(Time)-1)
	else:
		Tagg=Tagg[0]
	
	factor_corr_agg=VarC_lag[Tagg]/VarC_agg_cor[Tagg]
	# plt.plot(tt,np.exp(-lyap**2./(2.*sigma2)*tt)*0.2,'k--',label=r"$-\mu^2/(2\sigma^2)$")
	# plt.plot(tt,np.exp(-(lyap-sigma2/2)*tt)*0.2,'k-',label=r"$-(\mu-\sigma^2/2)$")
	ax[i].plot([1/lm*np.log(1/(sB*l0)),1/lm*np.log(1/(sB*l0))],[1e-5,1e0],'0.5',linewidth=2)
	ax[i].plot([1/lm*np.log(s0/sB),1/lm*np.log(s0/sB)],[1e-5,1e0],'0.75',linewidth=2)
	
	ax[i].plot(Time_eul,MeanC,'ko',label='DNS',fillstyle='full')
	ax[i].plot(Time,meanC_lag,'r*',label='Isolated strip')
	#plt.plot(Time,VarC_lag_2,'g*',label='Isolated strip model')
# 	ax[i].plot(Time,VarC_agg_rand,'rd',label='Random aggregation')
# 	ax[i].plot(Time,factor_corr_agg*VarC_agg_cor,'b^',label='Correlated aggregation')
	ax[i].set_yscale('log')
	ax[i].set_ylim([1e-2,1e0])
	ax[i].set_xlabel('$t$')
ax[i].legend(fontsize=8,ncol=1)
#ax[i].title(r'Sine Flow, $A={:1.1f}, s_B=1/{:1.0f}$'.format(A,1/sB))

ax[0].set_ylabel(r'$\langle c|c>\epsilon \rangle$')
plt.savefig('/home/joris/Dropbox/Articles/AggregationBatchelor/Figures/Sine_mean_A_sB1_{:1.0f}_overepsilon.pdf'.format(1/sB),bbox_inches='tight')

plt.figure()
plt.imshow(C>epsilon)
plt.figure()
plt.imshow(C)
#%%% *  Compare variance over threshold between Lagrangian Eulerian, various A
keyword='sine'

epsilon = 0.0001

fig,ax=plt.subplots(1,3,figsize=(8,3),sharey=True)

Brownian = 1e-3# Diffusion strength
l0=1.0
radius=0.1
c0=1
s0=0.1

np.random.seed(seed=16)
PhaseX=np.random.rand(100)*2*np.pi
PhaseY=np.random.rand(100)*2*np.pi
Angle=np.random.rand(100)*2*np.pi

for i,A in enumerate([0.5,0.8,1.5]):
#A=0.5
	
	
	
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
	
	#keyword='double'
	#keyword='cubic'
	#keyword='bifreq'
	#keyword='standard'
	#keyword='half'
	#keyword='halfsmooth'
	#keyword='single'
	#keyword='single'
	dir_out='./Compare_stretching_concentration/'+keyword+'/'
	if not os.path.exists(dir_out):
		os.makedirs(dir_out)
	
	#% Advection parameters
	INTERPOLATE='LINEAR'
	CURVATURE='DIFF'
	
	PLOT=False
	dx=0.001 # Maximum distance between points, above which the line will be refined
	alpha=200*dx/np.pi
	dt=0.25 # dt needs to be small compare to the CFL condition
	npar=6 # Number of parallel processors
	#tmax=Tmax # Maximum advection time
	Lmax=1e7 # Maximum number of points (prevent blow up)
	
	
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
	
	cmean=l0*s0*np.sqrt(np.pi)
	
	varC_lag,varC_lag_2,varC_agg_rand,varC_agg_rand2,varC_agg_cor,Time,Time_eul=[],[],[],[],[],[],[]
	meanC_lag,meanC_agg_cor,meanC_agg_rand=[],[],[]
	
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
	# =============================================================================
	# Save variables
	# =============================================================================
	D=(Brownian)**2./(2*dt)#+1e-08 # numerical diffusion ?
#	s0=radius#*np.sqrt(2)
	dist=np.sum(np.sqrt(np.sum(np.diff(L,axis=0)**2.,axis=1)))
	#Pe=1e7
	Tau=D/s0**2*wrapped_time
	Cmax=c0/np.sqrt(1.+4*Tau) #Gaussian
	Si=S*np.sqrt(1+4*Tau)*s0
	
	meanC_lag.append(np.mean(Cmax)*0.25)
	varC_lag_2.append(l0*np.average(1/S,weights=W)*np.mean(Si*Cmax**2.)*np.sqrt(np.pi/2))
	varC_lag.append(np.mean(Cmax**2.)*0.12)
	sB=np.mean(Si)
	# Prediction of N
	N=dist*sB/1*np.sqrt(np.pi/2)
	#Lmod=np.mod(L,1)
	# True N
	#N=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1,sB),weights=dist_old/sB,density=False)[0]
	#N=np.mean(N)
	#print(np.sqrt(np.pi/2)*sB*l0*np.mean(Cmax)*N)
	meanC_agg_rand.append(np.mean(Cmax)*N)
	varC_agg_rand.append(np.var(Cmax)*N)
	varC_agg_rand2.append(np.var(Cmax)*N)
	mC=np.average(Cmax,weights=W)
	vC=np.average(Cmax**2.,weights=W)
	
	meanC_agg_cor.append(cmean+np.mean(Cmax)*(1-cmean/c0))
	varC_agg_cor.append(vC-mC**2.)
	Time.append(t)
	
	# =============================================================================
	# MAIN PARALLEL LOOP #######################
	while (len(L)<Lmax)|(t!=int(t)):
	#while (t<12):
		
	#	v=vel(L,t,A)
	#	v=vel_cubic(L,t,A)
	#	v=vel_bifreq(L,t,A)
	#	v=vel_standard(L,t,A)
	#	v=vel_half(L,t,A)
	#	v=vel_nophase(L,t,A)
	
		v=locals()['vel_'+keyword](L,t,A)
		L+=v*dt
		
		# Compute stretching rates and elongations
		dist_new=np.sum(np.diff(L,axis=0)**2,1)**0.5
		gamma=dist_new/dist_old # elongation
		#gamma=np.maximum(dist_new/dist_old,dist_old/dist_new)
		S=S/gamma
		#wrapped_time=wrapped_time+dt*(1./S)**2.
		# Force positive elongation
		#rho1=np.abs(1./S-1.)+1.
		#rho1=np.maximum(1/S,S)
		rho1=1/S
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
		print('Time:',t,' - Npts:',len(L), '- Cmax:',np.mean(1./np.sqrt(1.+wrapped_time/1000)),np.max(kappa_new))
	# =============================================================================
	# Save variables
	# =============================================================================
		D=(Brownian)**2./(2*dt)#+1e-08 # numerical diffusion ?
#		s0=radius#*np.sqrt(2)
		dist=np.sum(np.sqrt(np.sum(np.diff(L,axis=0)**2.,axis=1)))
		#Pe=1e7
		Tau=D/s0**2*wrapped_time
		Cmax=c0/np.sqrt(1.+4*Tau) #Gaussian
		Si=S*np.sqrt(1+4*Tau)*s0
		meanC_lag.append(np.average(1/S,weights=W)*l0*np.mean(Si*Cmax)*np.sqrt(np.pi))
		varC_lag_2.append(np.average(1/S,weights=W)*l0*np.mean(Si*Cmax**2.)*np.sqrt(np.pi/2))
		varC_lag.append(np.mean(Cmax**2.)*0.12)
		sB=np.mean(Si)
		
	#	N=dist*sB/1
		#N=cmean/np.mean(Cmax)
		Lmod=np.mod(L,1)
		N2=np.exp(np.average(np.log(1/S),weights=W))
		N=np.mean(np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.linspace(0,1,int(1/sB))
										 ,weights=dist_old/sB*np.sqrt(2),density=False)[0])
		meanC_agg_rand.append(np.mean(Cmax)*N)
		varC_agg_rand.append(np.var(Cmax)*N)
		varC_agg_rand2.append(1/N2*1e-1)
		mC=np.average(Cmax,weights=W)
		vC=np.average(Cmax**2.,weights=W)
		varC_agg_cor.append(vC-mC**2.)
		meanC_agg_cor.append(cmean+np.mean(Cmax)*(1-cmean/c0))
		#varC_agg_cor.append(vC)
		Time.append(t)
	# End of MAIN LOOOP #######################
	print('Computation time:', time.time() -ct)
	plt.style.use('~/.config/matplotlib/joris.mplstyle')
	
	
	# Lyapunov
	lyap=0.65
	sigma2=0.5
	#%
	lyap=np.average(np.log(1/S),weights=W)/Time[-1]
	sigma2=(np.average((np.log(1/S))**2./Time[-1],weights=W)-lyap**2.)/Time[-1]
	
	# Eulerian
	keyword='sine'
	# Type of sine flow maps (classical is vel_sine)
	def vel_sine_eul(X,Y,k,dt,a,n):
	# 	MapX=np.mod(np.uint32(X+0.5*dt*a*n*np.sin(Y/n*2*np.pi+PhaseX[k])),n)
	# 	MapY=np.mod(np.uint32(Y+0.5*dt*a*n*np.sin(X/n*2*np.pi+PhaseY[k])),n)
		MapX=np.uint32(np.mod(np.round(X+0.5*dt*a*n*np.sin(Y/n*2*np.pi+PhaseX[k])),n))
		MapY=np.uint32(np.mod(np.round(Y+0.5*dt*a*n*np.sin(X/n*2*np.pi+PhaseY[k])),n))
		return MapX,MapY
	
	from scipy import ndimage, misc
	import numpy.fft
	from scipy.ndimage import gaussian_filter
	
	# Diffusion step
	def diffusion_fourier(C,sigma):
		input_ = np.fft.fft2(C)
		result = ndimage.fourier_gaussian(input_, sigma=sigma)
		return numpy.fft.ifft2(result).real
	
	n=int(2**10) # Number of grid points
	# =============================================================================
	tmax=int(Time[-1]) # number of periods to compute
	dt=1/1  # Discretisation of time step
	X,Y=np.meshgrid(np.arange(n),np.arange(n))
	D=Brownian**2./(2*0.25) # equivalent diffusion coeff
	sigma=np.sqrt(2*D)*n
	Pe=1./D
	
	# =============================================================================
	# iNITIAL condition
	#C[int(n/2):,:]=0.99 #half_plane
	#C[(X-n/2)**2+(Y-n/2)**2<(n/10)**2]=1
	# Initial condition corresponding to a single diffusive strip
	# Read Random angles
	C=np.zeros((n,n),dtype=np.float128)
	IdX=np.where((Y/n<l0/2)|(Y/n>(1-l0/2)))
#	s0=radius#*np.sqrt(2)
	C[IdX]=np.exp(-(X[IdX]/n)**2/(s0**2))+np.exp(-((X[IdX]-n)/n)**2/(s0**2))
	
	# =============================================================================
	dir_out='./'
	if not os.path.exists(dir_out):
		os.makedirs(dir_out)
	import scipy.stats
	# Save Variance and mean of C
	BINC=[]
	PDFC=[]
	PDFC_lin=[]
	MeanC=[]
	VarC=[]
	binC=np.logspace(-10,0,100)
	binC_lin=np.linspace(0,1,1000)
	#PDFC.append(np.histogram(np.abs(C),binC,density=True)[0])
	#PDFC_lin.append(np.histogram(C,binC_lin,density=True)[0])
#	Tv= np.arange(30)
	Tv= np.arange(tmax+10)
	binC=np.logspace(np.maximum(-10,np.log10(C.min())),np.log10(C.max()),100)
	BINC.append(binC)
	PDFC.append(np.histogram(np.abs(C),binC,density=True)[0])
	PDFC_lin.append(np.histogram(C,binC_lin,density=True)[0])
	
	MeanC.append(np.mean(C[C>epsilon]))
	VarC.append(np.var(C[C>epsilon]))
	Time_eul.append(0)
	
	for k in Tv:
		print('t=',k)
		# Choose flow map
		MapX,MapY=locals()['vel_'+keyword+'_eul'](X,Y,k,dt,A,n)
		# Advection
		for t in range(int(1/dt)):
			C[X.flatten(),MapY.flatten()]=C[X.flatten(),Y.flatten()]
			
			#C=diffusion_fourier(C, sigma*np.sqrt(dt))
			#C=diffusion(C, sigma*np.sqrt(dt))
		for t in range(int(1/dt)):
			C[MapX.flatten(),Y.flatten()]=C[X.flatten(),Y.flatten()]
			C=diffusion_fourier(C, sigma*np.sqrt(dt))
			#C=diffusion(C, sigma*np.sqrt(dt))
		#C=C-np.mean(C)
		binC=np.logspace(np.maximum(-10,np.log10(np.abs(C).min())),np.log10(C.max()),100)
		BINC.append(binC)
		PDFC.append(np.histogram(np.abs(C),binC,density=True)[0])
	#	print(PDFC[-1])
		PDFC_lin.append(np.histogram(C,binC_lin,density=True)[0])
		MeanC.append(np.mean(C[C>epsilon]))
		VarC.append(np.var(C[C>epsilon]))
		Time_eul.append(k+1)
		#fit_alpha, fit_loc, fit_beta=scipy.stats.gamma.fit(C.flatten()/C.mean())
	#%
	
	#plt.ylim([0,6])
	#plt.xlim([0,20])
# 	plt.ylim([0,20])
# 	plt.xlim([-12,0])
# 	plt.ylabel(r'log $\langle \rho \rangle_{s_B}$')
	
	
	sBp=np.sqrt(D/(lyap+sigma2))
	
	VarC=np.array(VarC)
	VarC_lag=np.array(varC_lag)
	VarC_lag_2=np.array(varC_lag_2)
	VarC_agg_rand=np.array(varC_agg_rand)
	VarC_agg_rand2=np.array(varC_agg_rand2)
	VarC_agg_cor=np.array(varC_agg_cor)
	
	MeanC=np.array(MeanC)
	meanC_lag=np.array(meanC_lag)
	meanC_agg_rand=np.array(meanC_agg_rand)
	meanC_agg_cor=np.array(meanC_agg_cor)
	
	tt=np.linspace(5,15,100)
	lm=lyap+sigma2/2
	cm=np.mean(C)
	
	tagg=1/lm*np.log(1/(sB*l0))
	Tagg=np.where(Time>tagg)[0]
	if len(Tagg)==0:
		Tagg=int(len(Time)-1)
	else:
		Tagg=Tagg[0]
	
	factor_corr_agg=VarC_agg_rand[Tagg]/VarC_agg_cor[Tagg]
	# plt.plot(tt,np.exp(-lyap**2./(2.*sigma2)*tt)*0.2,'k--',label=r"$-\mu^2/(2\sigma^2)$")
	# plt.plot(tt,np.exp(-(lyap-sigma2/2)*tt)*0.2,'k-',label=r"$-(\mu-\sigma^2/2)$")
	ax[i].plot([1/lm*np.log(1/(sB*l0)),1/lm*np.log(1/(sB*l0))],[1e-7,1e0],'b--',linewidth=1)
	ax[i].plot([1/lm*np.log(s0/sB),1/lm*np.log(s0/sB)],[1e-7,1e0],'b-',linewidth=1)
	ax[i].plot(Time_eul,VarC,'ro',label='DNS (Pe$={:1.0e}$)'.format(Pe),fillstyle='full')
	ax[i].plot(Time,VarC_lag,'k--',label='Isolated strip')
	#plt.plot(Time,VarC_lag_2,'g*',label='Isolated strip model')
	ax[i].plot(Time,VarC_agg_rand,'k:',label='Random aggregation')
	ax[i].plot(Time,VarC_agg_rand2,'k-.',label='Gamma aggregation')
	ax[i].plot(Time,factor_corr_agg*VarC_agg_cor,'k-',label='Correlated aggregation')
	ax[i].set_yscale('log')
	ax[i].set_ylim([1e-7,1e0])
	ax[i].set_xlabel('$t$')
ax[0].legend(fontsize=8,ncol=1)
#ax[i].title(r'Sine Flow, $A={:1.1f}, s_B=1/{:1.0f}$'.format(A,1/sB))

ax[0].set_ylabel(r'$\sigma^2_{c|c>\epsilon}$')
plt.savefig('/home/joris/Dropbox/Articles/AggregationBatchelor/Figures/Sine_variance_A_sB1_{:1.0f}_overepsilon.pdf'.format(1/sB),bbox_inches='tight')

plt.figure()
plt.imshow(C>epsilon)
plt.figure()
plt.imshow(C)


#%%% *  Compare variance logC over threshold between Lagrangian Eulerian, various A
keyword='sine'

epsilon = 0.0001

Brownian = 1e-3# Diffusion strength
l0=1.0
radius=0.1
c0=1
s0=0.1

np.random.seed(seed=16)
PhaseX=np.random.rand(100)*2*np.pi
PhaseY=np.random.rand(100)*2*np.pi
Angle=np.random.rand(100)*2*np.pi

for i,A in enumerate([0.6]):
#A=0.5
	
	
	
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
	
	#keyword='double'
	#keyword='cubic'
	#keyword='bifreq'
	#keyword='standard'
	#keyword='half'
	#keyword='halfsmooth'
	#keyword='single'
	#keyword='single'
	dir_out='./Compare_stretching_concentration/'+keyword+'/'
	if not os.path.exists(dir_out):
		os.makedirs(dir_out)
	
	#% Advection parameters
	INTERPOLATE='LINEAR'
	CURVATURE='DIFF'
	
	PLOT=False
	dx=0.001 # Maximum distance between points, above which the line will be refined
	alpha=200*dx/np.pi
	dt=0.25 # dt needs to be small compare to the CFL condition
	npar=6 # Number of parallel processors
	#tmax=Tmax # Maximum advection time
	Lmax=1e7 # Maximum number of points (prevent blow up)
	
	
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
	
	cmean=l0*s0*np.sqrt(np.pi)
	
	varC_lag,varC_lag_2,varC_agg_rand,varC_agg_rand2,varC_agg_cor,Time,Time_eul=[],[],[],[],[],[],[]
	meanC_lag,meanC_agg_cor,meanC_agg_rand=[],[],[]
	
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
	# =============================================================================
	# Save variables
	# =============================================================================
	D=(Brownian)**2./(2*dt)#+1e-08 # numerical diffusion ?
#	s0=radius#*np.sqrt(2)
	dist=np.sum(np.sqrt(np.sum(np.diff(L,axis=0)**2.,axis=1)))
	#Pe=1e7
	Tau=D/s0**2*wrapped_time
	Cmax=c0/np.sqrt(1.+4*Tau) #Gaussian
	Si=S*np.sqrt(1+4*Tau)*s0
	
	meanC_lag.append(np.mean(Cmax)*0.25)
	varC_lag_2.append(l0*np.average(1/S,weights=W)*np.mean(Si*Cmax**2.)*np.sqrt(np.pi/2))
	varC_lag.append(np.var(np.log(Cmax)))
	sB=np.mean(Si)
	# Prediction of N
	N=dist*sB/1*np.sqrt(np.pi/2)
	#Lmod=np.mod(L,1)
	# True N
	#N=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1,sB),weights=dist_old/sB,density=False)[0]
	#N=np.mean(N)
	#print(np.sqrt(np.pi/2)*sB*l0*np.mean(Cmax)*N)
	meanC_agg_rand.append(np.mean(Cmax)*N)
	varC_agg_rand.append(np.var(Cmax)*N)
	varC_agg_rand2.append(np.var(Cmax)*N)
	mC=np.average(np.log(Cmax),weights=W)
	vC=np.average(np.log(Cmax)**2.,weights=W)
	
	meanC_agg_cor.append(cmean+np.mean(Cmax)*(1-cmean/c0))
	varC_agg_cor.append(vC-mC**2.)
	Time.append(t)
	
	# =============================================================================
	# MAIN PARALLEL LOOP #######################
	while (len(L)<Lmax)|(t!=int(t)):
	#while (t<12):
		
	#	v=vel(L,t,A)
	#	v=vel_cubic(L,t,A)
	#	v=vel_bifreq(L,t,A)
	#	v=vel_standard(L,t,A)
	#	v=vel_half(L,t,A)
	#	v=vel_nophase(L,t,A)
	
		v=locals()['vel_'+keyword](L,t,A)
		L+=v*dt
		
		# Compute stretching rates and elongations
		dist_new=np.sum(np.diff(L,axis=0)**2,1)**0.5
		gamma=dist_new/dist_old # elongation
		#gamma=np.maximum(dist_new/dist_old,dist_old/dist_new)
		S=S/gamma
		#wrapped_time=wrapped_time+dt*(1./S)**2.
		# Force positive elongation
		#rho1=np.abs(1./S-1.)+1.
		#rho1=np.maximum(1/S,S)
		rho1=1/S
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
		print('Time:',t,' - Npts:',len(L), '- Cmax:',np.mean(1./np.sqrt(1.+wrapped_time/1000)),np.max(kappa_new))
	# =============================================================================
	# Save variables
	# =============================================================================
		D=(Brownian)**2./(2*dt)#+1e-08 # numerical diffusion ?
#		s0=radius#*np.sqrt(2)
		dist=np.sum(np.sqrt(np.sum(np.diff(L,axis=0)**2.,axis=1)))
		#Pe=1e7
		Tau=D/s0**2*wrapped_time
		Cmax=c0/np.sqrt(1.+4*Tau) #Gaussian
		Si=S*np.sqrt(1+4*Tau)*s0
		meanC_lag.append(np.average(1/S,weights=W)*l0*np.mean(Si*Cmax)*np.sqrt(np.pi))
		varC_lag_2.append(np.average(1/S,weights=W)*l0*np.mean(Si*Cmax**2.)*np.sqrt(np.pi/2))
		varC_lag.append(np.var(np.log(Cmax)))
		sB=np.mean(Si)
		
	#	N=dist*sB/1
		#N=cmean/np.mean(Cmax)
		Lmod=np.mod(L,1)
		N2=np.exp(np.average(np.log(1/S),weights=W))
		N=np.mean(np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.linspace(0,1,int(1/sB))
										 ,weights=dist_old/sB*np.sqrt(2),density=False)[0])
		meanC_agg_rand.append(np.mean(Cmax)*N)
		varC_agg_rand.append(np.var(Cmax)*N)
		varC_agg_rand2.append(1/N2*1e-1)
		mC=np.average(np.log(Cmax),weights=W)
		vC=np.average(np.log(Cmax)**2.,weights=W)
		varC_agg_cor.append(vC-mC**2.)
		meanC_agg_cor.append(cmean+np.mean(Cmax)*(1-cmean/c0))
		#varC_agg_cor.append(vC)
		Time.append(t)
	# End of MAIN LOOOP #######################
	print('Computation time:', time.time() -ct)
	plt.style.use('~/.config/matplotlib/joris.mplstyle')
	
	
	# Lyapunov
	lyap=0.65
	sigma2=0.5
	#%
	lyap=np.average(np.log(1/S),weights=W)/Time[-1]
	sigma2=(np.average((np.log(1/S))**2./Time[-1],weights=W)-lyap**2.)/Time[-1]
	
	# Eulerian
	keyword='sine'
	# Type of sine flow maps (classical is vel_sine)
	def vel_sine_eul(X,Y,k,dt,a,n):
	# 	MapX=np.mod(np.uint32(X+0.5*dt*a*n*np.sin(Y/n*2*np.pi+PhaseX[k])),n)
	# 	MapY=np.mod(np.uint32(Y+0.5*dt*a*n*np.sin(X/n*2*np.pi+PhaseY[k])),n)
		MapX=np.uint32(np.mod(np.round(X+0.5*dt*a*n*np.sin(Y/n*2*np.pi+PhaseX[k])),n))
		MapY=np.uint32(np.mod(np.round(Y+0.5*dt*a*n*np.sin(X/n*2*np.pi+PhaseY[k])),n))
		return MapX,MapY
	
	from scipy import ndimage, misc
	import numpy.fft
	from scipy.ndimage import gaussian_filter
	
	# Diffusion step
	def diffusion_fourier(C,sigma):
		input_ = np.fft.fft2(C)
		result = ndimage.fourier_gaussian(input_, sigma=sigma)
		return numpy.fft.ifft2(result).real
	
	n=int(2**10) # Number of grid points
	# =============================================================================
	tmax=int(Time[-1]) # number of periods to compute
	dt=1/1  # Discretisation of time step
	X,Y=np.meshgrid(np.arange(n),np.arange(n))
	D=Brownian**2./(2*0.25) # equivalent diffusion coeff
	sigma=np.sqrt(2*D)*n
	Pe=1./D
	
	# =============================================================================
	# iNITIAL condition
	#C[int(n/2):,:]=0.99 #half_plane
	#C[(X-n/2)**2+(Y-n/2)**2<(n/10)**2]=1
	# Initial condition corresponding to a single diffusive strip
	# Read Random angles
	C=np.zeros((n,n),dtype=np.float128)
	IdX=np.where((Y/n<l0/2)|(Y/n>(1-l0/2)))
#	s0=radius#*np.sqrt(2)
	C[IdX]=np.exp(-(X[IdX]/n)**2/(s0**2))+np.exp(-((X[IdX]-n)/n)**2/(s0**2))
	
	# =============================================================================
	dir_out='./'
	if not os.path.exists(dir_out):
		os.makedirs(dir_out)
	import scipy.stats
	# Save Variance and mean of C
	BINC=[]
	PDFC=[]
	PDFC_lin=[]
	MeanC=[]
	VarC=[]
	binC=np.logspace(-10,0,100)
	binC_lin=np.linspace(0,1,1000)
	#PDFC.append(np.histogram(np.abs(C),binC,density=True)[0])
	#PDFC_lin.append(np.histogram(C,binC_lin,density=True)[0])
#	Tv= np.arange(30)
	Tv= np.arange(tmax+20)
	binC=np.logspace(np.maximum(-10,np.log10(C.min())),np.log10(C.max()),100)
	BINC.append(binC)
	PDFC.append(np.histogram(np.abs(C),binC,density=True)[0])
	PDFC_lin.append(np.histogram(C,binC_lin,density=True)[0])
	
	MeanC.append(np.mean(C[C>epsilon]))
	VarC.append(np.nanvar(np.log(np.abs(C-C.mean()))))
	Time_eul.append(0)
	
	for k in Tv:
		print('t=',k)
		# Choose flow map
		MapX,MapY=locals()['vel_'+keyword+'_eul'](X,Y,k,dt,A,n)
		# Advection
		for t in range(int(1/dt)):
			C[X.flatten(),MapY.flatten()]=C[X.flatten(),Y.flatten()]
			
			#C=diffusion_fourier(C, sigma*np.sqrt(dt))
			#C=diffusion(C, sigma*np.sqrt(dt))
		for t in range(int(1/dt)):
			C[MapX.flatten(),Y.flatten()]=C[X.flatten(),Y.flatten()]
			C=diffusion_fourier(C, sigma*np.sqrt(dt))
			#C=diffusion(C, sigma*np.sqrt(dt))
		#C=C-np.mean(C)
		binC=np.logspace(np.maximum(-10,np.log10(np.abs(C).min())),np.log10(C.max()),100)
		BINC.append(binC)
		PDFC.append(np.histogram(np.abs(C),binC,density=True)[0])
	#	print(PDFC[-1])
		PDFC_lin.append(np.histogram(C,binC_lin,density=True)[0])
		MeanC.append(np.mean(C[C>epsilon]))
		VarC.append(np.nanvar(np.log(np.abs(C-C.mean()))))
		Time_eul.append(k+1)
		#fit_alpha, fit_loc, fit_beta=scipy.stats.gamma.fit(C.flatten()/C.mean())
	#%
	
	#plt.ylim([0,6])
	#plt.xlim([0,20])
# 	plt.ylim([0,20])
# 	plt.xlim([-12,0])
# 	plt.ylabel(r'log $\langle \rho \rangle_{s_B}$')
	
	
	sBp=np.sqrt(D/(lyap+sigma2))
	
	VarC=np.array(VarC)
	VarC_lag=np.array(varC_lag)
	VarC_lag_2=np.array(varC_lag_2)
	VarC_agg_rand=np.array(varC_agg_rand)
	VarC_agg_rand2=np.array(varC_agg_rand2)
	VarC_agg_cor=np.array(varC_agg_cor)
	
	MeanC=np.array(MeanC)
	meanC_lag=np.array(meanC_lag)
	meanC_agg_rand=np.array(meanC_agg_rand)
	meanC_agg_cor=np.array(meanC_agg_cor)
	
	tt=np.linspace(5,15,100)
	lm=lyap+sigma2/2
	cm=np.mean(C)
	
	tagg=1/lm*np.log(1/(sB*l0))
	Tagg=np.where(Time>tagg)[0]
	if len(Tagg)==0:
		Tagg=int(len(Time)-1)
	else:
		Tagg=Tagg[0]
	

fig,ax=plt.subplots(1,1,figsize=(3,3),sharey=True)
factor_corr_agg=1
sigma0=0.6
# plt.plot(tt,np.exp(-lyap**2./(2.*sigma2)*tt)*0.2,'k--',label=r"$-\mu^2/(2\sigma^2)$")
# plt.plot(tt,np.exp(-(lyap-sigma2/2)*tt)*0.2,'k-',label=r"$-(\mu-\sigma^2/2)$")

#ax.plot([1/lm*np.log(1/(sB*l0)),1/lm*np.log(1/(sB*l0))],[1e-7,5],'b--',linewidth=1)
#ax.plot([1/lm*np.log(s0/sB),1/lm*np.log(s0/sB)],[1e-7,5],'b-',linewidth=1)
ax.plot(Time_eul,VarC,'ro',label='DNS (Pe$={:1.0e}$)'.format(Pe),fillstyle='full')
#ax.plot(Time,VarC_lag+sigma0,'k--',label='Isolated strip')
#plt.plot(Time,VarC_lag_2,'g*',label='Isolated strip model')
ax.plot(Time_eul,np.zeros(len(Time_eul))+np.pi**2/8,'k:',label=r'Gaussian ($\pi^2/8$)')
ax.plot(Time_eul,np.zeros(len(Time_eul))+np.pi**2/6,'k-.',label=r'Exponential ($\pi^2/6$)')
#	ax[i].plot(Time,VarC_agg_rand2,'k-.',label='Gamma aggregation')
ax.plot(Time,factor_corr_agg*VarC_agg_cor+sigma0,'k-',label='Correlated aggregation')
#	ax.set_yscale('log')
#	ax.set_ylim([1e-7,1e0])
ax.set_xlabel('$t$')
ax.legend(fontsize=8,ncol=1)
#ax[i].title(r'Sine Flow, $A={:1.1f}, s_B=1/{:1.0f}$'.format(A,1/sB))

ax.set_ylabel(r'$\sigma^2_{\log |c-\langle c \rangle |}$')
plt.savefig('/home/joris/Dropbox/Articles/AggregationBatchelor/Figures/Sine_variance_Logc_A_sB1_{:1.0f}_overepsilon.pdf'.format(1/sB),bbox_inches='tight')

plt.figure()
plt.imshow(C>epsilon)
plt.figure()
plt.imshow(C)

#%%% log of normal distribut
from scipy import integrate

muc=0.1
sigma2c=1e-4
x=np.log(np.abs(np.random.randn(1000000)*np.sqrt(sigma2c)))

np.var(x)


def varlogc(y,sigma2):
	return y**2*2/np.sqrt(2*np.pi*sigma2)*np.exp(y-np.exp(2*y)/(2*sigma2))

def meanlogc(y,sigma2):
	return y*2/np.sqrt(2*np.pi*sigma2)*np.exp(y-np.exp(2*y)/(2*sigma2))

def plogc(y,sigma2):
	return 2/np.sqrt(2*np.pi*sigma2)*np.exp(y-np.exp(2*y)/(2*sigma2))

sigma2=1e-2
Var0=integrate.quad(varlogc,-100,100,args=(sigma2))[0]
M0=integrate.quad(meanlogc,-100,100,args=(sigma2))[0]
P0=integrate.quad(plogc,-100,100,args=(sigma2))[0]
print(M0,M0*np.sqrt(sigma2))
print(Var0,Var0*(sigma2))
print(Var0-M0**2)
#%%% Plot for various Péclet 
keyword='sine'

np.random.seed(seed=20)

Brownian_vec = np.logspace(-3,-2,10)# Diffusion strength
D=Brownian_vec**2./(2*0.25) # equivalent diffusion coeff
Sigma=np.sqrt(2*D)*n
Pe=1./np.sqrt(2.)/D

l0=0.3
radius=0.01
c0=1
s0=radius
A=0.2 # Ampluitude
#A=0.5
A=1/np.sqrt(2)
#A=1.5

PhaseX=np.random.rand(100)*2*np.pi
PhaseY=np.random.rand(100)*2*np.pi
Angle=np.random.rand(100)*2*np.pi

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

#keyword='double'
#keyword='cubic'
#keyword='bifreq'
#keyword='standard'
#keyword='half'
#keyword='halfsmooth'
#keyword='single'
#keyword='single'
dir_out='./Compare_stretching_concentration/'+keyword+'/'
if not os.path.exists(dir_out):
	os.makedirs(dir_out)

#% Advection parameters
INTERPOLATE='LINEAR'
CURVATURE='DIFF'

PLOT=False
dx=0.001 # Maximum distance between points, above which the line will be refined
alpha=200*dx/np.pi
dt=0.25 # dt needs to be small compare to the CFL condition
npar=6 # Number of parallel processors
#tmax=Tmax # Maximum advection time
Lmax=2e5# Maximum number of points (prevent blow up)


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

cmean=l0*s0*np.sqrt(np.pi)

varC_lag,varC_lag_2,varC_agg_rand,varC_agg_cor,Time,Time_eul=[],[],[],[],[],[]
meanC_lag,meanC_agg_cor,meanC_agg_rand=[],[],[]

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
# =============================================================================
# Save variables
# =============================================================================
D=(Brownian)**2./(2*dt)#+1e-08 # numerical diffusion ?
s0=radius#*np.sqrt(2)
dist=np.sum(np.sqrt(np.sum(np.diff(L,axis=0)**2.,axis=1)))
#Pe=1e7
Tau=D/s0**2*wrapped_time
Cmax=c0/np.sqrt(1.+4*Tau) #Gaussian
Si=S*np.sqrt(1+4*Tau)*s0

meanC_lag.append(l0*s0*np.sqrt(np.pi)*c0)
varC_lag_2.append(l0*np.average(1/S,weights=W)*np.mean(Si*Cmax**2.)*np.sqrt(np.pi/2))
varC_lag.append(l0*s0*c0*np.average(Cmax,weights=W)*np.sqrt(np.pi/2))
sB=np.mean(Si)
# Prediction of N
N=dist*sB/1*np.sqrt(np.pi/2)
#Lmod=np.mod(L,1)
# True N
#N=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1,sB),weights=dist_old/sB,density=False)[0]
#N=np.mean(N)
#print(np.sqrt(np.pi/2)*sB*l0*np.mean(Cmax)*N)
meanC_agg_rand.append(np.mean(Cmax)*N)
varC_agg_rand.append(np.var(Cmax)*N)
mC=np.average(Cmax,weights=W)
vC=np.average(Cmax**2.,weights=W)

meanC_agg_cor.append(cmean+np.mean(Cmax)*(1-cmean/c0))
varC_agg_cor.append(vC-mC**2.)
Time.append(t)

# =============================================================================
# MAIN PARALLEL LOOP #######################
while (len(L)<Lmax)|(t!=int(t)):
#while (t<12):
	
#	v=vel(L,t,A)
#	v=vel_cubic(L,t,A)
#	v=vel_bifreq(L,t,A)
#	v=vel_standard(L,t,A)
#	v=vel_half(L,t,A)
#	v=vel_nophase(L,t,A)

	v=locals()['vel_'+keyword](L,t,A)
	L+=v*dt
	
	# Compute stretching rates and elongations
	dist_new=np.sum(np.diff(L,axis=0)**2,1)**0.5
	gamma=dist_new/dist_old # elongation
	#gamma=np.maximum(dist_new/dist_old,dist_old/dist_new)
	S=S/gamma
	#wrapped_time=wrapped_time+dt*(1./S)**2.
	# Force positive elongation
	#rho1=np.abs(1./S-1.)+1.
	#rho1=np.maximum(1/S,S)
	rho1=1/S
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
	print('Time:',t,' - Npts:',len(L), '- Cmax:',np.mean(1./np.sqrt(1.+wrapped_time/1000)),np.max(kappa_new))
# =============================================================================
# Save variables
# =============================================================================
	D=(Brownian)**2./(2*dt)#+1e-08 # numerical diffusion ?
	s0=radius#*np.sqrt(2)
	dist=np.sum(np.sqrt(np.sum(np.diff(L,axis=0)**2.,axis=1)))
	#Pe=1e7
	Tau=D/s0**2*wrapped_time
	Cmax=c0/np.sqrt(1.+4*Tau) #Gaussian
	Si=S*np.sqrt(1+4*Tau)*s0
	meanC_lag.append(np.average(1/S,weights=W)*l0*np.mean(Si*Cmax)*np.sqrt(np.pi))
	varC_lag_2.append(np.average(1/S,weights=W)*l0*np.mean(Si*Cmax**2.)*np.sqrt(np.pi/2))
	varC_lag.append(l0*s0*c0*np.average(Cmax,weights=W)*np.sqrt(np.pi/2))
	sB=np.mean(Si)
	
#	N=dist*sB/1
	#N=cmean/np.mean(Cmax)
	Lmod=np.mod(L,1)
	N=np.mean(np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.linspace(0,1,int(1/sB))
									 ,weights=dist_old/sB*np.sqrt(2),density=False)[0])
	meanC_agg_rand.append(np.mean(Cmax)*N)
	varC_agg_rand.append(np.var(Cmax)*N)
	mC=np.average(Cmax,weights=W)
	vC=np.average(Cmax**2.,weights=W)
	varC_agg_cor.append(vC-mC**2.)
	meanC_agg_cor.append(cmean+np.mean(Cmax)*(1-cmean/c0))
	#varC_agg_cor.append(vC)
	Time.append(t)
# End of MAIN LOOOP #######################
print('Computation time:', time.time() -ct)
plt.style.use('~/.config/matplotlib/joris.mplstyle')


# Lyapunov
lyap=0.65
sigma2=0.5
#%
lyap=np.average(np.log(1/S),weights=W)/Time[-1]
sigma2=(np.average((np.log(1/S))**2./Time[-1],weights=W)-lyap**2.)/Time[-1]

for Brownian in Brownian_vec: # for lagrangian
	# Eulerian
	keyword='sine'
	# Type of sine flow maps (classical is vel_sine)
	def vel_sine_eul(X,Y,k,dt,a,n):
	# 	MapX=np.mod(np.uint32(X+0.5*dt*a*n*np.sin(Y/n*2*np.pi+PhaseX[k])),n)
	# 	MapY=np.mod(np.uint32(Y+0.5*dt*a*n*np.sin(X/n*2*np.pi+PhaseY[k])),n)
		MapX=np.uint32(np.mod(np.round(X+0.5*dt*a*n*np.sin(Y/n*2*np.pi+PhaseX[k])),n))
		MapY=np.uint32(np.mod(np.round(Y+0.5*dt*a*n*np.sin(X/n*2*np.pi+PhaseY[k])),n))
		return MapX,MapY
	
	from scipy import ndimage, misc
	import numpy.fft
	from scipy.ndimage import gaussian_filter
	
	# Diffusion step
	def diffusion_fourier(C,sigma):
		input_ = np.fft.fft2(C)
		result = ndimage.fourier_gaussian(input_, sigma=sigma)
		return numpy.fft.ifft2(result).real
	
	n=int(2**10) # Number of grid points
	# =============================================================================
	tmax=int(Time[-1]) # number of periods to compute
	dt=1/1  # Discretisation of time step
	X,Y=np.meshgrid(np.arange(n),np.arange(n))
	D=Brownian**2./(2*0.25) # equivalent diffusion coeff
	sigma=np.sqrt(2*D)*n
	Pe=1./np.sqrt(2.)/D
	
	# =============================================================================
	# iNITIAL condition
	#C[int(n/2):,:]=0.99 #half_plane
	#C[(X-n/2)**2+(Y-n/2)**2<(n/10)**2]=1
	# Initial condition corresponding to a single diffusive strip
	# Read Random angles
	C=np.zeros((n,n),dtype=np.float128)
	IdX=np.where((Y/n<l0/2)|(Y/n>(1-l0/2)))
	s0=radius#*np.sqrt(2)
	C[IdX]=np.exp(-(X[IdX]/n)**2/(s0**2))+np.exp(-((X[IdX]-n)/n)**2/(s0**2))
	
	# =============================================================================
	dir_out='./'
	if not os.path.exists(dir_out):
		os.makedirs(dir_out)
	import scipy.stats
	
	Tv= np.arange(tmax)

	for k in Tv:
		print('t=',k)
		# Choose flow map
		MapX,MapY=locals()['vel_'+keyword+'_eul'](X,Y,k,dt,A,n)
		# Advection
		for t in range(int(1/dt)):
			C[X.flatten(),MapY.flatten()]=C[X.flatten(),Y.flatten()]
			#C=diffusion_fourier(C, sigma*np.sqrt(dt))
			#C=diffusion(C, sigma*np.sqrt(dt))
		for t in range(int(1/dt)):
			C[MapX.flatten(),Y.flatten()]=C[X.flatten(),Y.flatten()]
			C=diffusion_fourier(C, sigma*np.sqrt(dt))
			#C=diffusion(C, sigma*np.sqrt(dt))

	I=C
	plt.figure(figsize=(10,10))
	#plt.imshow(np.log(I+I[I>0].min()),extent=[0,1,0,1],cmap=cm_fire)
	plt.imshow(I,extent=[0,1,0,1],cmap=cm_fire,alpha=0.7)
	#plt.imshow((np.abs(I-I.mean())<0.9*np.std(I)),extent=[0,1,0,1])
	Lmod=np.hstack((np.mod(L[:,1],1).reshape(-1,1),np.mod(-L[:,0],1).reshape(-1,1)))
	idmod=np.where(np.nansum(np.diff(Lmod,axis=0)**2.,axis=1)>0.01)[0]
	Lmod[idmod,:]=np.nan
	#plt.plot(np.mod(L[:,1],1),np.mod(-L[:,0],1),'k.',alpha=1.0,linewidth=0.2,markersize=0.1)
	plt.plot(Lmod[:,0],Lmod[:,1],'k-',alpha=0.5,linewidth=0.1,markersize=0.1)
	plt.axis('off')
	plt.text(0.005,0.96,'$A={:1.2f}, s_B=1/{:1.0f}$'.format(A,1/np.sqrt(D/0.5)),color='k',fontsize=30,backgroundcolor='w')
	plt.savefig('/data/Simulations/SineFlow/Peclet/Sine_Pe_A{:1.2f}_sB-1{:1.0f}.png'.format(A,1/np.sqrt(D/0.5)),bbox_inches='tight')


#%%% Variance decay as a function of Péclet

Brownian = 1e-2# Diffusion strength
l0=0.3
radius=0.01
c0=1

from scipy import ndimage, misc
import numpy.fft
from scipy.ndimage import gaussian_filter

# Diffusion step
def diffusion_fourier(C,sigma):
	input_ = np.fft.fft2(C)
	result = ndimage.fourier_gaussian(input_, sigma=sigma)
	return numpy.fft.ifft2(result).real

n=int(2**10) # Number of grid points
# =============================================================================
a=1/np.sqrt(2) # Sine flow amplitude
tmax=30 # number of periods to compute
dt=1/1  # Discretisation of time step
X,Y=np.meshgrid(np.arange(n),np.arange(n))
D=Brownian**2./(2*0.25) # equivalent diffusion coeff
sigma=np.sqrt(2*D)*n
Pe=1./np.sqrt(2.)/D

# =============================================================================
# iNITIAL condition
#C[int(n/2):,:]=0.99 #half_plane
#C[(X-n/2)**2+(Y-n/2)**2<(n/10)**2]=1
# Initial condition corresponding to a single diffusive strip
# Read Random angles
C=np.zeros((n,n),dtype=np.float128)
IdX=np.where((Y/n<l0/2)|(Y/n>(1-l0/2)))
s0=radius#*np.sqrt(2)
C[IdX]=np.exp(-(X[IdX]/n)**2/(s0**2))+np.exp(-((X[IdX]-n)/n)**2/(s0**2))

# =============================================================================
dir_out='./'
if not os.path.exists(dir_out):
	os.makedirs(dir_out)
import scipy.stats
# Save Variance and mean of C
BINC=[]
PDFC=[]
PDFC_lin=[]
MeanC=[]
VarC=[]
th=1e-14
binC=np.logspace(-10,0,100)
binC_lin=np.linspace(0,1,1000)
#PDFC.append(np.histogram(np.abs(C),binC,density=True)[0])
#PDFC_lin.append(np.histogram(C,binC_lin,density=True)[0])
Tv= np.arange(30)
Sigma=np.array([2,4,8,16,32,64])
Seeds=[12,19,65,14,321,45]
A=[0.5 ,0.8,1.2,1.5]
Gamma_all=np.zeros((len(A),len(Sigma)))
GammaLog_all=np.zeros((len(A),len(Sigma)))
for seed in Seeds:
	gamma_all=[]
	gammaLog_all=[]
	np.random.seed(seed=seed)
	PhaseX=np.random.rand(100)*2*np.pi
	PhaseY=np.random.rand(100)*2*np.pi
	Angle=np.random.rand(100)*2*np.pi


	l0=0.3
	# Eulerian
	keyword='sine'
	# Type of sine flow maps (classical is vel_sine)
	def vel_sine_eul(X,Y,k,dt,a,n):
	# 	MapX=np.mod(np.uint32(X+0.5*dt*a*n*np.sin(Y/n*2*np.pi+PhaseX[k])),n)
	# 	MapY=np.mod(np.uint32(Y+0.5*dt*a*n*np.sin(X/n*2*np.pi+PhaseY[k])),n)
		MapX=np.uint32(np.mod(np.round(X+0.5*dt*a*n*np.sin(Y/n*2*np.pi+PhaseX[k])),n))
		MapY=np.uint32(np.mod(np.round(Y+0.5*dt*a*n*np.sin(X/n*2*np.pi+PhaseY[k])),n))
		return MapX,MapY

	for a in A:
		gamma=[]
		gammaLog=[]
		print('a=',a,'seeds =',seed)
		for sigma in Sigma:
			# =============================================================================
			# iNITIAL condition
			#C[int(n/2):,:]=0.99 #half_plane
			#C[(X-n/2)**2+(Y-n/2)**2<(n/10)**2]=1
			# Initial condition corresponding to a single diffusive strip
			# Read Random angles
			D=(sigma/n)**2./2
			s0=4*np.sqrt(D/0.5)
			C=np.zeros((n,n),dtype=np.float128)
			IdX=np.where((Y/n<l0/2)|(Y/n>(1-l0/2)))
			s0=radius#*np.sqrt(2)
			C[IdX]=np.exp(-(X[IdX]/n)**2/(s0**2))+np.exp(-((X[IdX]-n)/n)**2/(s0**2))
			VarC=[]
			VarLogC=[]
			Time_eul=[]
			VarC.append(np.var(C))
			Cm=np.mean(C)
			VarLogC.append(np.nanvar(np.log(np.abs(C-Cm))))
			Time_eul.append(0)
			for k in Tv:
				# Choose flow map
				MapX,MapY=locals()['vel_'+keyword+'_eul'](X,Y,k,dt,a,n)
				# Advection
				for t in range(int(1/dt)):
					C[X.flatten(),MapY.flatten()]=C[X.flatten(),Y.flatten()]
					#C=diffusion_fourier(C, sigma*np.sqrt(dt))
					#C=diffusion(C, sigma*np.sqrt(dt))
				for t in range(int(1/dt)):
					C[MapX.flatten(),Y.flatten()]=C[X.flatten(),Y.flatten()]
					C=diffusion_fourier(C, sigma*np.sqrt(dt))
					#C=diffusion(C, sigma*np.sqrt(dt))
				Cm=np.mean(C)
				VarLogC.append(np.nanvar(np.log(np.abs(C-Cm))))
				VarC.append(np.var(C))
				Time_eul.append(k+1)
				#fit_alpha, fit_loc, fit_beta=scipy.stats.gamma.fit(C.flatten()/C.mean())
			#%
			VarC=np.array(VarC,dtype=np.float32)
			VarLogC=np.array(VarLogC,dtype=np.float32)
			Time_eul=np.array(Time_eul)
			plt.plot(Time_eul[20:],np.log(VarC[20:]))
			gamma.append(np.polyfit(Time_eul[20:],np.log(VarC[20:]),1)[0])
			gammaLog.append(np.polyfit(Time_eul[20:],np.log(VarLogC[20:]),1)[0])
		gamma_all.append(gamma)
		gammaLog_all.append(gammaLog)
	Gamma_all+=np.array(gamma_all)/len(Seeds)
	GammaLog_all+=np.array(gammaLog_all)/len(Seeds)


Lyap=np.array([0.3,0.76,1.41,1.8])
Sigma2=np.array([0.3,0.67,1.0,1.14])

D=(Sigma/n)**2./2
Pe=a/D

gamma_all=np.array(gamma_all)
plt.figure()
[plt.plot(D,(Gamma_all[i,:]/Gamma_all[i,0]),'o-',fillstyle='full',color=plt.cm.jet(A[i]/2),label=r'$A={:1.1f}$'.format(A[i])) for i in range(gamma_all.shape[0])]
plt.legend()
plt.xlabel(r'$\kappa$')
plt.xscale('log')
plt.ylabel(r'$\gamma_2/\gamma_{2,\infty}$')
plt.title('Sine flow')


plt.figure()
[plt.plot(1/D,-(Gamma_all[i,:]-Gamma_all[i,0])/Sigma2[i],'o-',fillstyle='full',color=plt.cm.jet(A[i]/2),label=r'$A={:1.1f}$'.format(A[i])) for i in range(gamma_all.shape[0])]
plt.plot(Pe,20*Pe**-0.5,'k--',label='$20\kappa^{-1/2}$')
plt.xlabel(r'$1/\kappa$')
plt.xscale('log')
plt.yscale('log')
plt.ylabel(r'$(\gamma_2-\gamma_{2,\infty})/\sigma^2$')
plt.title('Sine flow')
plt.legend()
plt.savefig('/home/joris/Dropbox/Articles/AggregationBatchelor/Figures/gamma2_sine_Peclet.pdf',bbox_inches='tight')


[plt.plot(1/D,-(GammaLog_all[i,:]-GammaLog_all[i,0]),'o-',fillstyle='full',color=plt.cm.jet(A[i]/2),label=r'$A={:1.1f}$'.format(A[i])) for i in range(GammaLog_all.shape[0])]
plt.plot(Pe,20*Pe**-0.5,'k--',label='$20\kappa^{-1/2}$')
plt.xlabel(r'$1/\kappa$')
plt.xscale('log')
plt.yscale('log')
plt.ylabel(r'$(\gamma_2-\gamma_{2,\infty})/\sigma^2$')
plt.title('Sine flow')
plt.legend()
plt.savefig('/home/joris/Dropbox/Articles/AggregationBatchelor/Figures/gamma2log_sine_Peclet.pdf',bbox_inches='tight')


plt.figure()
[plt.plot(1/np.log(D)**2.,-(Gamma_all[i,:]-Gamma_all[i,0]),'o-',fillstyle='full',color=plt.cm.jet(A[i]/2),label=r'$A={:1.1f}$'.format(A[i])) for i in range(gamma_all.shape[0])]
plt.xlabel(r'$1/ (\log \kappa)^2$')
#plt.xscale('log')
#plt.yscale('log')
plt.ylabel(r'$(\gamma_2-\gamma_{2,\infty})$')
plt.title('Sine flow')
plt.legend()

#%%% Variance LogC as a function of Péclet

Brownian = 1e-2# Diffusion strength
l0=0.3
radius=0.01
c0=1

from scipy import ndimage, misc
import numpy.fft
from scipy.ndimage import gaussian_filter

# Diffusion step
def diffusion_fourier(C,sigma):
	input_ = np.fft.fft2(C)
	result = ndimage.fourier_gaussian(input_, sigma=sigma)
	return numpy.fft.ifft2(result).real

n=int(2**10) # Number of grid points
# =============================================================================
a=1/np.sqrt(2) # Sine flow amplitude
tmax=30 # number of periods to compute
dt=1/1  # Discretisation of time step
X,Y=np.meshgrid(np.arange(n),np.arange(n))
D=Brownian**2./(2*0.25) # equivalent diffusion coeff
sigma=np.sqrt(2*D)*n
Pe=1./np.sqrt(2.)/D

# =============================================================================
# iNITIAL condition
#C[int(n/2):,:]=0.99 #half_plane
#C[(X-n/2)**2+(Y-n/2)**2<(n/10)**2]=1
# Initial condition corresponding to a single diffusive strip
# Read Random angles
C=np.zeros((n,n),dtype=np.float128)
IdX=np.where((Y/n<l0/2)|(Y/n>(1-l0/2)))
s0=radius#*np.sqrt(2)
C[IdX]=np.exp(-(X[IdX]/n)**2/(s0**2))+np.exp(-((X[IdX]-n)/n)**2/(s0**2))

# =============================================================================
dir_out='./'
if not os.path.exists(dir_out):
	os.makedirs(dir_out)
import scipy.stats
# Save Variance and mean of C
BINC=[]
PDFC=[]
PDFC_lin=[]
MeanC=[]
VarC=[]
th=1e-14
binC=np.logspace(-10,0,100)
binC_lin=np.linspace(0,1,1000)
#PDFC.append(np.histogram(np.abs(C),binC,density=True)[0])
#PDFC_lin.append(np.histogram(C,binC_lin,density=True)[0])
Tv= np.arange(30)
Sigma=np.array([2,4,8,12])
Seeds=[12]
A=[0.8]
Gamma_all=np.zeros((len(A),len(Sigma)))
GammaLog_all=np.zeros((len(A),len(Sigma)))
for seed in Seeds:
	gamma_all=[]
	gammaLog_all=[]
	np.random.seed(seed=seed)
	PhaseX=np.random.rand(100)*2*np.pi
	PhaseY=np.random.rand(100)*2*np.pi
	Angle=np.random.rand(100)*2*np.pi


	l0=0.3
	# Eulerian
	keyword='sine'
	# Type of sine flow maps (classical is vel_sine)
	def vel_sine_eul(X,Y,k,dt,a,n):
	# 	MapX=np.mod(np.uint32(X+0.5*dt*a*n*np.sin(Y/n*2*np.pi+PhaseX[k])),n)
	# 	MapY=np.mod(np.uint32(Y+0.5*dt*a*n*np.sin(X/n*2*np.pi+PhaseY[k])),n)
		MapX=np.uint32(np.mod(np.round(X+0.5*dt*a*n*np.sin(Y/n*2*np.pi+PhaseX[k])),n))
		MapY=np.uint32(np.mod(np.round(Y+0.5*dt*a*n*np.sin(X/n*2*np.pi+PhaseY[k])),n))
		return MapX,MapY

	for a in A:
		gamma=[]
		gammaLog=[]
		print('a=',a,'seeds =',seed)
		for sigma in Sigma:
			# =============================================================================
			# iNITIAL condition
			#C[int(n/2):,:]=0.99 #half_plane
			#C[(X-n/2)**2+(Y-n/2)**2<(n/10)**2]=1
			# Initial condition corresponding to a single diffusive strip
			# Read Random angles
			D=(sigma/n)**2./2
			s0=4*np.sqrt(D/0.5)
			C=np.zeros((n,n),dtype=np.float128)
			IdX=np.where((Y/n<l0/2)|(Y/n>(1-l0/2)))
			s0=radius#*np.sqrt(2)
			C[IdX]=np.exp(-(X[IdX]/n)**2/(s0**2))+np.exp(-((X[IdX]-n)/n)**2/(s0**2))
			VarC=[]
			VarLogC=[]
			Time_eul=[]
			VarC.append(np.var(C))
			Cm=np.mean(C)
			VarLogC.append(np.nanvar(np.log(np.abs(C-Cm))))
			Time_eul.append(0)
			for k in Tv:
				# Choose flow map
				MapX,MapY=locals()['vel_'+keyword+'_eul'](X,Y,k,dt,a,n)
				# Advection
				for t in range(int(1/dt)):
					C[X.flatten(),MapY.flatten()]=C[X.flatten(),Y.flatten()]
					#C=diffusion_fourier(C, sigma*np.sqrt(dt))
					#C=diffusion(C, sigma*np.sqrt(dt))
				for t in range(int(1/dt)):
					C[MapX.flatten(),Y.flatten()]=C[X.flatten(),Y.flatten()]
					C=diffusion_fourier(C, sigma*np.sqrt(dt))
					#C=diffusion(C, sigma*np.sqrt(dt))
				Cm=np.mean(C)
				VarLogC.append(np.nanvar(np.log(np.abs(C-Cm))))
				VarC.append(np.var(C))
				Time_eul.append(k+1)
				#fit_alpha, fit_loc, fit_beta=scipy.stats.gamma.fit(C.flatten()/C.mean())
			#%
			VarC=np.array(VarC,dtype=np.float32)
			VarLogC=np.array(VarLogC,dtype=np.float32)
			Time_eul=np.array(Time_eul)
			plt.plot(Time_eul[:],VarLogC[:],label='Pe={:1.0e}'.format(a/((sigma/n)**2./2)))
			gamma.append(np.polyfit(Time_eul[20:],np.log(VarC[20:]),1)[0])
			gammaLog.append(np.polyfit(Time_eul[20:],np.log(VarLogC[20:]),1)[0])
		gamma_all.append(gamma)
		gammaLog_all.append(gammaLog)
	Gamma_all+=np.array(gamma_all)/len(Seeds)
	GammaLog_all+=np.array(gammaLog_all)/len(Seeds)


Lyap=np.array([0.3,0.76,1.41,1.8])
Sigma2=np.array([0.3,0.67,1.0,1.14])

D=(Sigma/n)**2./2
Pe=a/D

plt.xlabel('$t$')
plt.ylabel('Var$[\log |c-<c>|]$')
plt.legend()
#%%%  Run DSM with particle pairs to compute lyapunov
import h5py


lyapA=[]
sigma2A=[]

AA=np.linspace(0.3,1.8,15)
for A in AA:
	lyap=[]
	sigma2=[]
	Seeds=np.arange(10)
	
	T=0.5
	for s in Seeds:
		np.random.seed(seed=s)
		PhaseX=np.random.rand(1000)*2*np.pi
		PhaseY=np.random.rand(1000)*2*np.pi
		Angle=np.random.rand(1000)*2*np.pi
		
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
		
		keyword='sine'
		
		l0=0.3
		#% Advection parameters
		INTERPOLATE='LINEAR'
		CURVATURE='DIFF'
		
		PLOT=False
		dx=0.001 # Maximum distance between points, above which the line will be refined
		alpha=200*dx/np.pi
		dt=0.25 # dt needs to be small compare to the CFL condition
		npar=6 # Number of parallel processors
		#tmax=Tmax # Maximum advection time
		Lmax=5e7 # Maximum number of points (prevent blow up)
		
		#A=0.5
		#A=1/np.sqrt(2)
		#A=2.5
		
		# Initial segment position and distance
		n=100000+1
		x0=np.random.rand(n)
		t0=np.random.rand(n)*2*np.pi
		L=np.vstack((x0+dx/2*np.cos(t0),x0+dx/2*np.sin(t0))).T
		L[2::3]=np.nan
		
		L=np.array(L)
		weights=np.ones(L.shape[0]-1)
		weights=weights/np.sum(weights)
		#L=2*(np.random.rand(2)-0.5)+np.array([x,l0*np.sin(x)],dtype=np.float64).T
		
		dist_old=np.sum(np.diff(L,axis=0)**2,1)**0.5
		weights=np.ones(dist_old.shape)
		W=weights
		# Initial segment width
		S=np.ones(L.shape[0]-1,dtype=np.float64)
		# Initial Wrapped Time
		wrapped_time=np.zeros(L.shape[0]-1)
		
		# Initialization
		t=0
		ct=time.time()
		
		mRho,vRho,m1_Rho,v1_Rho,Time = [],[],[],[],[]
		
		# MAIN PARALLEL LOOP #######################
		while (np.nanmean(S)>1e-20):
		
		#	v=vel(L,t,A)
		#	v=vel_cubic(L,t,A)
		#	v=vel_bifreq(L,t,A)
		#	v=vel_standard(L,t,A)
		#	v=vel_half(L,t,A)
		#	v=vel_nophase(L,t,A)
		
			v=locals()['vel_'+keyword](L,t,A)
			L+=v*dt
			
			# Compute stretching rates and elongations
			dist_new=np.sum(np.diff(L,axis=0)**2,1)**0.5
			gamma=dist_new/dist_old # elongation
			#gamma=np.maximum(dist_new/dist_old,dist_old/dist_new)
			S=S/(gamma)
			#wrapped_time=wrapped_time+dt*(1./S)**2.
			# Force positive elongation
			#rho1=np.abs(1./S-1.)+1.
		#	rho1=np.maximum(1/S,S)
			#rho1=1/S
			wrapped_time=wrapped_time+dt*(1/S)**2.
			#Force periodicity
			#L[0,:]=L[-1,:]
		
		# =============================================================================
		# rELOCATE
		# =============================================================================
			ref=np.where(dist_new>dx)[0]
		#	Relocate at distance dx/2
			k=(L[ref+1]-L[ref])
			k=k.T/np.sqrt(np.sum(k**2.,axis=1))
			L[ref+1]=L[ref]+k.T*dx/2.
			dist_old=np.sum(np.diff(L,axis=0)**2,1)**0.5
		
		# =============================================================================
			mRho.append(np.nanmean(np.log(1/S)))
			vRho.append(np.nanvar(np.log(1/S)))
			m1_Rho.append(np.nanmean(S))
			v1_Rho.append(np.nanvar((S)))
			Time.append(t)
			# Update time
			t=t+dt
			print('Time:',t,' - Npts:',len(L), '- S:',np.nanmean(S))
		
		# End of MAIN LOOOP #######################
		print('Computation time:', time.time() -ct)
		#plt.ylim([0,6])
		#plt.xlim([0,20])
		Time=np.array(Time)
		#Fit to match log rho
		lyap.append(np.polyfit(Time,mRho,1)[0])
		sigma2.append(np.polyfit(Time,vRho,1)[0])
	
	lyapA.append(np.mean(lyap))
	sigma2A.append(np.mean(sigma2))


lyapA=np.array(lyapA)
sigma2A=np.array(sigma2A)

np.savetxt('Sine_Lyap.txt',np.vstack((AA,lyapA,sigma2A)).T,header='# Amplitude, Lyapunov, Variance')


plt.figure()
plt.plot(Time,mRho,'r-',label=r'$\langle \log \rho \rangle'+' = {:1.2f} t$'.format(lyap))
plt.plot(Time,vRho,'k-',label=r'$\sigma^2_{\log \rho} '+'= {:1.2f} t$'.format(sigma2))
plt.plot(Time,Time*lyap,'r--')
plt.plot(Time,Time*sigma2,'k--')
plt.legend()

#Fit to match 1_rho
plt.figure()
plt.title('$A={:1.2f}, \lambda={:1.2f}, \sigma^2={:1.2f} $'.format(A,lyap,sigma2))
plt.plot(Time,m1_Rho,'r-',label=r'$\langle 1/\rho \rangle$')
plt.plot(Time,v1_Rho,'k-',label=r'$\sigma^2_{1/\rho}$')
plt.plot(Time,np.exp(Time*(-lyap+sigma2/2.)),'r--',label=r'$-\lambda+\sigma^2/2$')
plt.plot(Time,np.exp(-Time*lyap**2./(2*sigma2)),'k--',label=r'$-\lambda^2/(2\sigma^2)$')
plt.plot(Time,np.exp(Time*(-2*lyap+2*sigma2)),'k-',label=r'$-2\lambda+2\sigma^2$')
plt.yscale('log')
plt.xlabel('$t$')
plt.savefig('/home/joris/Dropbox/Articles/AggregationBatchelor/Figures/Sine_1_rho_A{:1.2f}.pdf'.format(A))
plt.legend()
plt.ylim([1e-60,1e2])

tt=t
plt.figure()
h,x=np.histogram(np.log(S[np.isfinite(S)]),1000,density='True')
plt.plot(x[1:],h,'ko')
plt.plot(x,1/np.sqrt(2*np.pi*sigma2*tt)*np.exp(-(x+lyap*tt)**2/(2*sigma2*tt)))
plt.yscale('log')

plt.figure()
h,x=np.histogram(S[np.isfinite(S)],np.logspace(-80,0,1000),density='True')
plt.plot(x[1:],h,'ko')
plt.plot(x,1/np.sqrt(2*np.pi*sigma2*tt)/x*np.exp(-(np.log(x)+lyap*tt)**2/(2*sigma2*tt)))
plt.yscale('log')
plt.xscale('log')
#%%%  Run DSM with particle to compute dispersion
import h5py


DispA=[]

AA=np.linspace(0.3,1.8,15)
for A in AA:
	Disp=[]
	Seeds=np.arange(10)
	
	plt.figure()
	T=0.5
	for s in Seeds:
		np.random.seed(seed=s)
		PhaseX=np.random.rand(1000)*2*np.pi
		PhaseY=np.random.rand(1000)*2*np.pi
		Angle=np.random.rand(1000)*2*np.pi
		
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
		
		keyword='sine'
		
		l0=0.3
		#% Advection parameters
		INTERPOLATE='LINEAR'
		CURVATURE='DIFF'
		
		PLOT=False
		dx=0.001 # Maximum distance between points, above which the line will be refined
		alpha=200*dx/np.pi
		dt=0.25 # dt needs to be small compare to the CFL condition
		npar=6 # Number of parallel processors
		#tmax=Tmax # Maximum advection time
		Lmax=5e7 # Maximum number of points (prevent blow up)
		
		#A=0.5
		#A=1/np.sqrt(2)
		#A=2.5
		
		# Initial segment position and distance
		n=100000+1
		#x0=np.random.rand(n)
		#t0=np.random.rand(n)*2*np.pi
		#L=np.vstack((x0+dx/2*np.cos(t0),x0+dx/2*np.sin(t0))).T
		#L[2::3]=np.nan
		#L=np.array(L)
		L=np.random.rand(n,2)
		weights=np.ones(L.shape[0]-1)
		weights=weights/np.sum(weights)
		#L=2*(np.random.rand(2)-0.5)+np.array([x,l0*np.sin(x)],dtype=np.float64).T
		
		dist_old=np.sum(np.diff(L,axis=0)**2,1)**0.5
		weights=np.ones(dist_old.shape)
		W=weights
		# Initial segment width
		S=np.ones(L.shape[0]-1,dtype=np.float64)
		# Initial Wrapped Time
		wrapped_time=np.zeros(L.shape[0]-1)
		
		# Initialization
		t=0
		ct=time.time()
		
		vX,Time = [],[]
		
		# MAIN PARALLEL LOOP #######################
		while t<100:
		
		#	v=vel(L,t,A)
		#	v=vel_cubic(L,t,A)
		#	v=vel_bifreq(L,t,A)
		#	v=vel_standard(L,t,A)
		#	v=vel_half(L,t,A)
		#	v=vel_nophase(L,t,A)
		
			v=locals()['vel_'+keyword](L,t,A)
			L+=v*dt
			
		# ===================================================
			vX.append(np.nanvar(np.sqrt(np.sum(L**2,axis=1))))
			Time.append(t)
			# Update time
			t=t+dt
			print('Time:',t,' - A:',A, '- Seed:',s)
		
		# End of MAIN LOOOP #######################
		print('Computation time:', time.time() -ct)
		#plt.ylim([0,6])
		#plt.xlim([0,20])
		Time=np.array(Time)
		#Fit to match log rho
		Disp.append(np.polyfit(Time[-150:],vX[-150:],1)[0]/2)
		
		plt.plot(Time[:],vX[:],'r-',label=r'$\k_{eff}$')
		plt.yscale('log')
		plt.xscale('log')
	
	DispA.append(np.mean(Disp))
	
DispA=np.array(DispA)

np.savetxt('Sine_Disp.txt',np.vstack((AA,DispA)).T,header='# Amplitude, Dispersion')

#%%% Plot dispersion
Lyap=np.loadtxt('Sine_Lyap.txt')
DispA=np.loadtxt('Sine_Disp.txt')

plt.figure()
plt.plot(DispA[:,0],4*DispA[:,1]*np.pi**2,'-',label=r'$4 \kappa_{eff} \pi^2$')
#plt.plot(DispA[:,0],4*DispA[:,1]*np.pi**2,'-',label=r'$4 \kappa_{eff} \pi^2$')
plt.plot(Lyap[:,0],Lyap[:,1],'-',label=r'$\lambda$')
plt.plot(Lyap[:,0],Lyap[:,0]**2/8*np.pi**2,'-',label=r'$U^2 \pi^2 /8$')
plt.yscale('log')
plt.xscale('log')
plt.legend()


plt.figure()
plt.plot(DispA[:,0],DispA[:,1],'o',label=r'$\kappa_{eff}$')
plt.plot(DispA[:,0],DispA[:,0]**2/8,'-',label=r'$U^2 T / 8 $')
plt.plot(DispA[:,0],DispA[:,0]**2/(4*np.pi**2),'-',label=r'$U^2 T / (4 \pi^2) $')
plt.yscale('log')
plt.xscale('log')
plt.xlabel('A')
plt.legend()

#%%% Plot lyapunov and D1
Lyap=np.loadtxt('Sine_Lyap.txt')
D1=np.loadtxt('Sine_D1.txt')

from scipy.optimize import curve_fit
def func(x, a):
	return x**2*np.pi**2/8 + a/x


plt.plot(Lyap[:,0],Lyap[:,1])
plt.plot(Lyap[:,0],Lyap[:,2])
p1, pcov = curve_fit(func, Lyap[:,0],Lyap[:,0])
p2, pcov = curve_fit(func, Lyap[:,0],Lyap[:,1])
plt.plot(Lyap[:,0],func(Lyap[:,0],p1[0]))
plt.plot(Lyap[:,0],func(Lyap[:,0],p2[0]))

plt.figure()
plt.plot(func(Lyap[:,0],p2[0])/func(Lyap[:,0],p1[0]))


p=np.polyfit(D1[:,0],D1[:,1],1)
plt.figure()
plt.plot(D1[:,0],D1[:,1])
plt.plot(D1[:,0],D1[:,0]*p[0]+p[1])
plt.plot()
#%% GRAPHICAL comparisons
#%%%  Run DSM and save images
import h5py

keyword='sine'
#keyword='double'
#keyword='cubic'
#keyword='bifreq'
#keyword='standard'
#keyword='half'
#keyword='halfsmooth'
#keyword='single'
#keyword='single'
dir_out='./Compare_stretching_concentration/'+keyword+'/'
if not os.path.exists(dir_out):
	os.makedirs(dir_out)

l0=0.3
#% Advection parameters
INTERPOLATE='LINEAR'
CURVATURE='DIFF'

PLOT=False
dx=0.001 # Maximum distance between points, above which the line will be refined
alpha=200*dx/np.pi
dt=0.05 # dt needs to be small compare to the CFL condition
npar=6 # Number of parallel processors
#tmax=Tmax # Maximum advection time
Lmax=1e7 # Maximum number of points (prevent blow up)

A=1.2 # Ampluitude
#A=0.5
A=1/np.sqrt(2)

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
plt.figure(figsize=(5,5))
Lmod=np.hstack((np.mod(L[:,1],1).reshape(-1,1),np.mod(-L[:,0],1).reshape(-1,1)))
idmod=np.where(np.nansum(np.diff(Lmod,axis=0)**2.,axis=1)>0.01)[0]
Lmod[idmod,:]=np.nan
#plt.plot(np.mod(L[:,1],1),np.mod(-L[:,0],1),'k.',alpha=1.0,linewidth=0.2,markersize=0.1)
lplot=plt.plot(Lmod[:,0],Lmod[:,1],'k-',alpha=0.9,linewidth=0.5,markersize=0.1)
plt.xlim([0,1])
plt.ylim([0,1])
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

# MAIN PARALLEL LOOP #######################
while (len(L)<Lmax):
	
#	v=vel(L,t,A)
#	v=vel_cubic(L,t,A)
#	v=vel_bifreq(L,t,A)
#	v=vel_standard(L,t,A)
#	v=vel_half(L,t,A)
#	v=vel_nophase(L,t,A)

	v=locals()['vel_'+keyword](L,t,A)
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
	
	Lmod=np.hstack((np.mod(L[:,1],1).reshape(-1,1),np.mod(-L[:,0],1).reshape(-1,1)))
	idmod=np.where(np.nansum(np.diff(Lmod,axis=0)**2.,axis=1)>0.01)[0]
	Lmod[idmod,:]=np.nan
	#plt.plot(np.mod(L[:,1],1),np.mod(-L[:,0],1),'k.',alpha=1.0,linewidth=0.2,markersize=0.1)
	lplot[0].set_data(Lmod[:,0],Lmod[:,1])
	#plt.yticks([0.1,0.2,0.3,0.4])
	#plt.xticks([0.5,0.6,0.7,0.8])
#	plt.axis('off')
	plt.savefig('/home/joris/Figures/sine_{:1.0f}.pdf'.format(t/dt),dpi=300,bbox_inches='tight')

	# Update time
	t=t+dt
	print('Time:',t,' - Npts:',len(L), '- Cmax:',np.mean(1./np.sqrt(1.+wrapped_time/1000)),np.max(kappa_new))

# End of MAIN LOOOP #######################
print('Computation time:', time.time() -ct)
#plt.ylim([0,6])
#plt.xlim([0,20])
plt.ylim([0,20])
plt.xlim([-12,0])
plt.ylabel(r'log $\langle \rho \rangle_{s_B}$')

f.close()
#%%% Run DSM and plot at multiple scales

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

fig.savefig('/home/joris/Dropbox/Articles/AggregationBatchelor/Figures/sine_c_fractalsA={:1.1f}.jpg'.format(A),bbox_inches='tight',dpi=600)

#%%% Run DSM and plot Coarse grained N, Cmax

A=0.5
seed=3

L,S,wrapped_time,W,t=run_DSM(1e7, A, seed)

#DNS
D=(1/50)**2/2*0.1
C0,mC,vC=run_DNS(A,int(2**11),seed,int(t),D)

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

plt.savefig('/home/joris/Dropbox/Articles/AggregationBatchelor/Figures/N-LAG.pdf',bbox_inches='tight')


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

fig.savefig('/home/joris/Dropbox/Articles/AggregationBatchelor/Figures/sine_N_LAG.pdf',bbox_inches='tight')
#%%% Comparison DNS DSM

A=0.5
seed=3

L,S,wrapped_time,W,t=run_DSM(1e7, A, seed)

clim=[-8,-2]

D=(1/200)**2/2*0.1
C1,mC,vC=run_DNS(A,int(2**11),3,int(t),D)
D=(1/50)**2/2*0.1
C0,mC,vC=run_DNS(A,int(2**11),seed,int(t),D)

fig,ax=plt.subplots(2,2,figsize=(4,4))
dist_old=np.sum(np.diff(L,axis=0)**2,1)**0.5
sB=1/200
C=np.histogram2d(np.mod(L[1:,0],1),np.mod(L[1:,1],1),np.arange(0,1,sB),weights=S*dist_old/sB*np.sqrt(2))[0]
C[C==0]=np.exp(clim[0])
i0=ax[0,0].imshow(np.log(C),clim=clim)
ax[0,0].axis('off')
cc=fig.colorbar(i0,ax=ax[1,:],location='right',shrink=0.8)
cc.set_label(r'$\log c$', color='w')
# set colorbar tick color
cc.ax.yaxis.set_tick_params(color='w')
# set colorbar edgecolor 
cc.outline.set_edgecolor('w')

plt.setp(plt.getp(cc.ax.axes, 'yticklabels'), color='w')

sB=1/50
C=np.histogram2d(np.mod(L[1:,0],1),np.mod(L[1:,1],1),np.arange(0,1,sB),weights=S*dist_old/sB*np.sqrt(2))[0]
C[C==0]=np.exp(clim[0])
ax[0,1].imshow(np.log(C),clim=clim)
ax[0,1].axis('off')

ax[1,0].imshow(np.log(C1),clim=clim)
ax[1,0].axis('off')

ax[1,1].imshow(np.log(C0),clim=clim)
ax[1,1].axis('off')

ax[0,0].text(0.05,0.9,'a)',c='w',transform = ax[0,0].transAxes)
ax[0,1].text(0.05,0.9,'b)',c='w',transform = ax[0,1].transAxes)
ax[1,0].text(0.05,0.9,'c)',c='w',transform = ax[1,0].transAxes)
ax[1,1].text(0.05,0.9,'d)',c='w',transform = ax[1,1].transAxes)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.02, hspace=0.02)

plt.savefig('/home/joris/Dropbox/Articles/AggregationBatchelor/Figures/DNS-LAG.pdf',bbox_inches='tight')
#%%% Superposition of Lagrangian and Eulerian
keyword='cubic'
keyword='sine'
#keyword='half'
#keyword='double'
#keyword='halfsmoothnp

plt.style.use('~/.config/matplotlib/joris.mplstyle')
import h5py
#from skimage.morphology import disk
#from skimage.filters import median,gaussian
ms=['o','d','s','<','>','+','*','1','2','3']
ms=ms*10

Brownian=1e-3
#Brownian=2e-4
#Brownian=5e-4
radius=0.01
l0=0.3
A=1/np.sqrt(2)
#A=1.2

dir_img='./Compare_stretching_concentration/'+keyword+'/Fourier_A{:1.1f}_l{:1.1f}_rad{:1.2f}_{:1.0e}'.format(A,l0,radius,Brownian)
dir_out='./Compare_stretching_concentration/'+keyword+'/'

periodicity='periodic'
k=1
#Tmax=[6]
#Tmax_vec=np.array([12])
	#% visual comparison
	#plt.close('all')
I0=cv2.imread(dir_img+'/{:04d}.tif'.format(0),2)
I0=np.float32(I0)/2.**16.
I0med=I0
cmm=np.mean(I0)
#I0med = gaussian(I0, sigma=3)
Imax=np.max(I0med)
f=h5py.File(dir_out+'DSM'+keyword+'_A{:1.1f}_l{:1.1f}.hdf5'.format(A,l0), 'r')

Tmax_vec=np.array([2,4,6,8,10,12])
Tmax_vec=np.arange(8)
try:
	Tmax_vec=np.uint8(np.arange(int(f.attrs['tmax']*0.25),f.attrs['tmax']+1,
														 np.maximum(int(f.attrs['tmax']*0.1),1)))
#	Tmax_vec=[f.attrs['tmax']]
except:
	Tmax_vec=[11]
#Tmax=11

Tmax_vec=[8]

Cm_all,Cstd_all,Csquare_all,logCm_all,logCstd_all,H2m_all=[],[],[],[],[],[]

for t in Tmax_vec:
	L=f['L_{:04d}'.format(int(t*10))][:]
	wrapped_time=f['wrapped_time_{:04d}'.format(int(t*10))][:]
	S=f['S_{:04d}'.format(int(t*10))][:]
	W=f['Weights_{:04d}'.format(int(t*10))][:]
	I=cv2.imread(dir_img+'/{:04d}.tif'.format(int(t*10)),2)
	I=np.float32(I)/2.**16.
	Imed=I
	#Imed = gaussian(I, sigma=3)
#	I=Imed/Imax
	Nbin=I.shape[0]

	# COmpute n order moment
	# compare cmax from individual strips theory and cmax from imag
	Lmid=(L[1:,:]+L[:-1,:])/2.
	if periodicity=='periodic':
		ix=np.int32(np.mod(Lmid[:,0],1)*Nbin)
		iy=np.int32(np.mod(Lmid[:,1],1)*Nbin)
		Lin=np.where((ix>=0)&(iy>=0)&(ix<Nbin)&(iy<Nbin))
		IL=I[ix[Lin],iy[Lin]].flatten()
	else:
		Lin=np.where((np.abs(Lmid[:,1])<1)&(np.abs(Lmid[:,0])<1))
		IL=I[np.int32((Lmid[Lin,0]+1)*Nbin/2),np.int32((Lmid[Lin,1]+1)*Nbin/2)].flatten()

plt.style.use('~/.config/matplotlib/joris.mplstyle')
from matplotlib.colors import ListedColormap
Fire=np.loadtxt('/home/joris/.config/matplotlib/LUT_Fire.csv',delimiter=',',skiprows=1)
Fire=Fire/255.
cm_fire=ListedColormap(Fire[:,1:], name='Fire', N=None)

plt.figure(figsize=(1.5,1.5))
plt.imshow(np.log(I+I[I>0].min()),extent=[0,1,0,1],cmap=cm_fire)
Lmod=np.hstack((np.mod(L[:,1],1).reshape(-1,1),np.mod(-L[:,0],1).reshape(-1,1)))
idmod=np.where(np.nansum(np.diff(Lmod,axis=0)**2.,axis=1)>0.01)[0]
Lmod[idmod,:]=np.nan
#plt.plot(np.mod(L[:,1],1),np.mod(-L[:,0],1),'k.',alpha=1.0,linewidth=0.2,markersize=0.1)
plt.plot(Lmod[:,0],Lmod[:,1],'k-',alpha=0.8,linewidth=0.1,markersize=0.1)

D=(Brownian)**2./(2*dt)
print(D)
#% Plot both lines and direct simulations
plt.figure(figsize=(5,5))
plt.imshow(I,extent=[0,1,0,1],cmap=cm_fire)
Lmod=np.hstack((np.mod(L[:,1],1).reshape(-1,1),np.mod(-L[:,0],1).reshape(-1,1)))
idmod=np.where(np.nansum(np.diff(Lmod,axis=0)**2.,axis=1)>0.01)[0]
Lmod[idmod,:]=np.nan
#plt.plot(np.mod(L[:,1],1),np.mod(-L[:,0],1),'k.',alpha=1.0,linewidth=0.2,markersize=0.1)
plt.plot(Lmod[:,0],Lmod[:,1],'w-',alpha=0.4,linewidth=0.5,markersize=0.1)
plt.xlim([0.5,0.9])
plt.ylim([0.5,0.9])
#plt.yticks([0.1,0.2,0.3,0.4])
#plt.xticks([0.5,0.6,0.7,0.8])
plt.axis('off')
plt.savefig('/home/joris/Dropbox/Articles/AggregationBatchelor/Figures/Eul_Lag_sine_{:1.0e}.pdf'.format(Brownian),dpi=300,bbox_inches='tight')

#%%% Reconstruction with Lagrangian on a grid

keyword='sine'

dx=0.001
t=11
plt.style.use('~/.config/matplotlib/joris.mplstyle')
import h5py
from skimage.morphology import disk
from skimage.filters import median,gaussian
ms=['o','d','s','<','>','+','*']
Brownian=1e-3
radius=0.01
l0=0.3
A=1/np.sqrt(2)
#A=1.2

dir_out='./Compare_stretching_concentration/'+keyword+'/Fourier_A{:1.1f}_l{:1.1f}_rad{:1.2f}_{:1.0e}'.format(A,l0,radius,Brownian)
periodicity='periodic'
k=1
#Tmax_vec=np.array([12])
	#% visual comparison
	#plt.close('all')
I0=cv2.imread(dir_out+'/{:04d}.tif'.format(0),2)
I0=np.float32(I0)/2.**16.
I0med=I0
#I0med = gaussian(I0, sigma=3)
Imax=np.max(I0med)
f=h5py.File('./Compare_stretching_concentration/'+keyword+'/DSM'+keyword+'_A{:1.1f}_l{:1.1f}.hdf5'.format(A,l0), 'r')

L=f['L_{:04d}'.format(int(t*10))][:]
wrapped_time=f['wrapped_time_{:04d}'.format(int(t*10))][:]
S=f['S_{:04d}'.format(int(t*10))][:]
W=f['Weights_{:04d}'.format(int(t*10))][:]
I=cv2.imread(dir_out+'/{:04d}.tif'.format(int(t*10)),2)
I=np.float32(I)/2.**16.
Imed=I
#Imed = gaussian(I, sigma=3)
I=Imed/Imax
Nbin=I.shape[0]

# COmpute n order moment

# compare cmax from individual strips theory and cmax from imag
Lmid=(L[1:,:]+L[:-1,:])/2.
if periodicity=='periodic':
	ix=np.int32(np.mod(Lmid[:,0],1)*Nbin)
	iy=np.int32(np.mod(Lmid[:,1],1)*Nbin)
	Lin=np.where((ix>=0)&(iy>=0)&(ix<Nbin)&(iy<Nbin))
	IL=I[ix[Lin],iy[Lin]].flatten()
else:
	Lin=np.where((np.abs(Lmid[:,1])<1)&(np.abs(Lmid[:,0])<1))
	IL=I[np.int32((Lmid[Lin,0]+1)*Nbin/2),np.int32((Lmid[Lin,1]+1)*Nbin/2)].flatten()

# Find min stretching rates 
argsort_id=np.argsort(1./S)
n=10000
#plt.plot((Lmid[argsort_id[:n],1]+1)*Nbin/2,(Lmid[argsort_id[:n],0]+1)*Nbin/2,'ro',alpha=1.0)

dt=0.25
D=(Brownian)**2./(2*dt)#+1e-08 # numerical diffusion ?
s0=radius#*np.sqrt(2)
#Pe=1e7
Tau=D/s0**2*wrapped_time[Lin]
Cmax=1./np.sqrt(1.+4*D/s0**2*wrapped_time[Lin])
Sc=s0*np.sqrt(1.+4*D/s0**2*wrapped_time[Lin])*S
Cmax2=1./np.sqrt(4*D/s0**2*wrapped_time[Lin])

Rho=1./S[Lin]

if keyword=='sine':
	lyap=0.65
	sigma=lyap*0.65
	
if keyword=='half':
	lyap=0.25
	sigma=0.26

sB=np.sqrt(D/lyap)
sB=np.mean(Sc)
# Taking a unique sB might not be the good approach
# We need about 2*sB to get a correct sum
sBx=sB*1.96
Lgrid=np.uint16(np.mod(L,1)/sBx)



dist_old=np.sum(np.diff(L,axis=0)**2,1)**0.5
Lmod=np.mod(L,1)
C=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.linspace(0,1,int(1/sBx))
									 ,weights=Cmax*dist_old/sBx*np.sqrt(2),density=False)[0]
C2=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.linspace(0,1,int(1/sBx))
									 ,weights=Cmax*Sc*dist_old/sBx*np.sqrt(2),density=False)[0]
C1=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.linspace(0,1,int(1/sBx))
									 ,weights=Cmax,density=False)[0]
N=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.linspace(0,1,int(1/sBx))
									 ,density=False)[0]
SC=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.linspace(0,1,int(1/sBx))
									 ,weights=Sc*dist_old/sBx*np.sqrt(2),density=False)[0]

SC=SC/(N)

Cm=C/(N)

#plt.hexbin(Lmod[1:,0],Lmod[1:,1],bins=np.linspace(0,1,int(1/sBx))
#									 ,C=Cmax*dist_old/sBx*np.sqrt(2))
# Check for which sB we have a minimum of error
#from skimage.transform import rescale
#R=np.linspace(1.8,2.1,20)
#Error=[]
#for r in R:
#	C=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.linspace(0,1,int(1/(sB*r)))
#									 ,weights=Cmax*dist_old/sBx*np.sqrt(2),density=False)[0]
#	Ir=rescale(I,C.shape[0]/I.shape[0],anti_aliasing=True)
#	Error.append(np.sum(np.abs(Ir-C)*(C>0)))
#plt.figure()
#plt.plot(R,Error)

plt.figure(figsize=(2,2))
plt.imshow(N,clim=[0,200]);plt.axis('off');plt.colorbar()

plt.figure(figsize=(2,2))
plt.imshow(SC,clim=[sB,s0]);plt.axis('off');plt.colorbar()

plt.figure(figsize=(2,2))
plt.imshow(np.log(C+1e-7),clim=[-7,0]);plt.axis('off');plt.colorbar()

plt.figure(figsize=(2,2))
plt.imshow(np.log(I+1e-7),clim=[-7,0]);plt.axis('off');plt.colorbar()



Ir=rescale(I,C.shape[0]/I.shape[0],anti_aliasing=False)
plt.figure()
h,x=np.histogram(np.log(C[C>0]).flatten(),100,density=True)
plt.plot(x[1:],h,'o')
h,x=np.histogram(np.log(Ir[C>0]).flatten(),100,density=True)
h,x=np.histogram(np.log(I[I>0]).flatten(),100,density=True)
plt.plot(x[1:],h,'+')
plt.yscale('log')

plt.figure(figsize=(2,2))

from skimage.transform import rescale, resize
Is=resize(I,(C.shape[0],C.shape[1]))
logcb=np.linspace(-12,0,100)
logsumc=bin_operation(np.log(C[C>0].flatten()),np.log(Is[C>0].flatten()),logcb,np.nanmean)
logsumc2=bin_operation(np.log(Is[C>0].flatten()),np.log(C[C>0].flatten()),logcb,np.nanmean)
logcmax=bin_operation(np.log(Cm[Cm>0].flatten()),np.log(Is[Cm>0].flatten()),logcb,np.nanmean)
logcmax2=bin_operation(np.log(Is[Cm>0].flatten()),np.log(Cm[Cm>0].flatten()),logcb,np.nanmean)
logcmax1=bin_operation(np.log(C1[C1>0].flatten()),np.log(Is[C1>0].flatten()),logcb,np.nanmean)
#plt.plot(np.log(C.flatten()),np.log(Is.flatten()),'.',alpha=0.01)
plt.plot(logcb[1:],logsumc,'o')
plt.plot(logsumc2,logcb[1:],'ko')
plt.plot(logcb[1:],logcmax,'+')
plt.plot(logcmax2,logcb[1:],'k+')
#plt.plot(logcb[1:],logcmax1,'o')
plt.plot([-10,0],[-10,0],'k--')
plt.ylabel(r'$\log c$')
plt.xlabel(r'$c_\mathrm{max}, \log \sum c_\mathrm{max}$')


logcb=np.linspace(-6,0,40)
plt.figure()
h2=np.histogram2d(np.log10(C).flatten(),np.log10(Is).flatten(),logcb)[0]
X,Y=np.meshgrid(logcb[1:],logcb[1:])
plt.contourf(X,Y,np.log(h2))
plt.plot([-5,0],[-5,0],'w--')
plt.xlabel('$\log c$')
plt.ylabel('$\log \sum c_\mathrm{max}$')

logcb=np.linspace(-6,0,40)
plt.figure()
h2=np.histogram2d(np.log10(C2/sBx*np.sqrt(2*np.pi)).flatten(),np.log10(Is).flatten(),logcb)[0]
X,Y=np.meshgrid(logcb[1:],logcb[1:])
plt.contourf(X,Y,np.log(h2))
plt.plot([-5,0],[-5,0],'w--')
plt.xlabel('$\log c$')
plt.ylabel('$\log [s_{s_B}^{-1} \sum \sqrt{2\pi }s c_\mathrm{max}]$')

plt.figure()
h2=np.histogram2d(np.log10(Cm).flatten(),np.log10(Is).flatten(),logcb)[0]
X,Y=np.meshgrid(logcb[1:],logcb[1:])
plt.contourf(X,Y,np.log(h2))
plt.plot([-5,0],[-5,0],'w--')
plt.xlabel('$\log  c$')
plt.ylabel('$\log  c_\mathrm{max}$')


plt.figure()
plt.imshow(np.log(I),clim=[-6,0])
plt.imshow(np.log(Is),clim=[-6,0])
plt.imshow(np.log(np.abs(C-Is))*(C>0)+1e-6,clim=[-6,0])


plt.figure()
plt.hist(np.log(I[I>0]),1000)
plt.hist(np.log(C[C>0]),1000)
plt.yscale('log')

#%%% Large concentrations compared to strip

plt.style.use('~/.config/matplotlib/joris.mplstyle')

Brownian=1e-3
periodicity='periodic'
Nbin=4096
#periodicity=''
dir_out='./Compare_stretching_concentration/{:1.0e}'.format(Brownian)+periodicity

Tmax_vec=[10]

radius =0.01
l0=0.3
for Tmax in Tmax_vec:
	#from colorline_toolbox import *
	# Inline or windows plots
	#%matplotlib auto
	#%matplotlibTrue inline
	PLOT=True
	PAR=False

	
	#% Advection parameters
	INTERPOLATE='LINEAR'
	CURVATURE='DIFF'
	
	PLOT=False
	dx=0.001 # Maximum distance between points, above which the line will be refined
	alpha=200*dx/np.pi
	dt=0.25 # dt needs to be small compare to the CFL condition
	npar=6 # Number of parallel processors
	tmax=Tmax # Maximum advection time
	tsave=0.1 # Time period between saving steps
	Lmax=5e7 # Maximum number of points (prevent blow up)
	th_pinch=100 # Curvature minimum (in nb of dx) to find a peak
	Pe=1e2 # Peclet
	#l0=0.1
	
	# Initial segment position and distance
	x=np.linspace(0,2*np.pi,int(1.8e3))
	L=np.array([radius*np.cos(x),radius*np.sin(x)],dtype=np.float64).T
	L[0,:]=L[-1,:]
	n=int(1.8e3)
	L=np.array([np.linspace(-l0/2,l0/2,n),np.zeros(n)],dtype=np.float64).T
	#	L=[]
	#	for i in range(100):
	#		x,y=np.random.rand(),np.random.rand()
	#		L.append([x-dx,y-dx])
	#		L.append([x+dx,y+dx])
	#		L.append([np.nan,np.nan])
	
	L=np.array(L)
	weights=np.ones(L.shape[0]-1)
	weights=weights/np.sum(weights)
	#L=2*(np.random.rand(2)-0.5)+np.array([x,l0*np.sin(x)],dtype=np.float64).T
	
	dist_old=np.sum(np.diff(L,axis=0)**2,1)**0.5
	weights=np.ones(dist_old.shape)
	# Initial segment width
	S=np.ones(L.shape[0]-1)
	# Initial Wrapped Time
	wrapped_time=np.zeros(L.shape[0]-1)
	
	# Initialization of saved variables
	Lsum=[]
	Ssum=np.sum(dist_old*S)
	# Initialization of saved variables
	Lsum=[]
	Ssum=np.sum(dist_old*S)
	Smean=[]
	Rhomean=[]
	logRhomean=[]
	logRhovar=[]
	logKappaMean=[]
	logKappaSum=[]
	logKappaVar=[]
	KappaMean=[]
	logdKappaMean=[]
	logdKappaVar=[]
	logSmean=[]
	logSvar=[]
	Emean=[]
	Evar=[]
	Npinches=[]
	Cmax=[]
	Gamma_PDF_x=np.linspace(-10,10,1000)
	Gamma_PDF=np.zeros(Gamma_PDF_x.shape[0]-1,dtype=np.uint64)
	Lp_x=np.linspace(0,10,100)
	Lp_PDF=np.zeros(Lp_x.shape[0]-1,dtype=np.uint64)
	Lp_var=[]
	CmaxM,IM,IML=[],[],[]
	if PLOT:
		plt.close('all')
		%matplotlib auto
		plt.close('all')
		plt.ion()
		fig, ax = plt.subplots(figsize=(10,10))
	#	ax.axis([-5,5,-5,5])
	#	line, = ax.plot(L[:,0], L[:,1],'-',alpha=1.,linewidth=0.5)
		ax.axis([-20,20,-3,3])
		ax.set_xlabel(r'$d\ln \kappa /dt$')
		ax.set_ylabel(r'$d\ln \rho /dt$')
		line, = ax.plot(0,0,'.',alpha=0.05,linewidth=0.5,markersize=0.8)
		ax2 = plt.axes([0.65, 0.65, 0.2, 0.2])
		ax2.axis([-np.pi/2.,np.pi/2.,-np.pi/2.,np.pi/2.])
		line2, = ax2.plot(L[:,0], L[:,1],'r-',alpha=1.,linewidth=0.5)
	
	# Initialization
	t=0
	Tv=np.arange(0,tsave+dt,dt) # Vector of time step between saving steps
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
	
	plt.figure()
	# MAIN PARALLEL LOOP #######################
	while (t<tmax)&(len(L)<Lmax):
		
		v=vel(L,t)
		L+=v*dt
		
		# Compute stretching rates and elongations
		dist_new=np.sum(np.diff(L,axis=0)**2,1)**0.5
		gamma=dist_new/dist_old # elongation
		S=S/gamma
		#wrapped_time=wrapped_time+dt*(1./S)**2.
		# Force positive elongation
		rho1=np.abs(1./S-1.)+1.
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
		
		# Remove strip if outside the domain
		Id_out=np.where((np.abs(L[:,0])>2)|(np.abs(L[:,1])>2))[0]
		L[Id_out,:]=np.nan
		# Filter consecutive nans to save memory space
		nnan=np.uint8(np.isnan(L[:,0]))
		nnan2=nnan[1:]+nnan[:-1]
		iddel=np.where(nnan2==2)[0]
		L=np.delete(L,iddel,axis=0)
		dist_old=np.delete(dist_old,iddel,axis=0)
		dist_new=np.delete(dist_new,iddel,axis=0)
		S=np.delete(S,iddel,axis=0)
		gamma=np.delete(gamma,iddel,axis=0)
		kappa_old=np.delete(kappa_old,iddel,axis=0)
		W=np.delete(W,iddel)
		weights=np.delete(weights,iddel)
		wrapped_time=np.delete(wrapped_time,iddel,axis=0)
		dkappa=np.delete(dkappa,iddel)
		
		#Save variables
		Lsum.append(np.sum(dist_old)) # Total length
		Rhomean.append(np.average(1./S,weights=W)) # Mean width
		logRhomean.append(np.average(np.log(1./S),weights=W)) # Mean width
		logRhovar.append(np.average((np.log(1./S)-logRhomean[-1])**2./(1.-np.sum(W**2.)),weights=W)) # Var Log Rho
	
		Smean.append(np.average(S,weights=dist_old)) # Mean width
		logSmean.append(np.average(np.log(S),weights=W)) # Mean width
		logSvar.append(np.average((np.log(S)-logSmean[-1])**2./(1.-np.sum(W**2.)),weights=W)) # Var width
	
		KappaMean.append(np.average(kappa_old,weights=W)) # !!! do we take weighted average or normal average ?
		logKappaMean.append(np.average(np.log(kappa_old),weights=W))
		logKappaSum.append(np.nansum(np.log(kappa_old)))
		logKappaVar.append(np.average((np.log(kappa_old)-logKappaMean[-1])**2./(1.-np.sum(W**2.)),weights=W)) # Var Log Rho
		logdKappaMean.append(dlKMean)
		logdKappaVar.append(dlKVar) # Variance of Elongation

		Emean.append(np.average(gamma,weights=W)) # Mean Elongation
		Evar.append(np.average((gamma-Emean[-1])**2./(1.-np.sum(W**2.)),weights=W)) # Variance of Elongation
		Cmax.append(np.average(1./np.sqrt(1.+wrapped_time/Pe),weights=dist_old)) # Variance of Elongation

		N=10
		if t==9:
			D=(Brownian)**2./(2*dt)
			s0=radius*0.95
			Cmaxv=1./np.sqrt(1.+4*D/s0**2*wrapped_time)
			CmaxM.append([np.nanmean(Cmaxv**n) for n in range(N)])
			pcmax,cmax=np.histogram(np.log(Cmaxv[np.isfinite(Cmaxv)]),100,density=True)
#			pcmax,cmax=np.histogram((1./np.sqrt(1.+wrapped_time[np.isfinite(wrapped_time)]/Pe)),50,density=True)
	#		plt.plot(rhob[1:],prhob,'.')
			plt.plot(cmax[1:],pcmax,'.',color=plt.cm.jet(t/tmax),label='{:1.0f} t'.format(t))
			# Load Image file and compare pdf
			# compare cmax from individual strips theory and cmax from imag
			I=cv2.imread(dir_out+'/{:04d}.tif'.format(int((t+dt)*10)),2)
			I=np.float32(I)/2.**16.
			# Non periodic
#			plt.imshow(np.log(I),extent=[-1,1,-1,1])
			#plt.plot(L[:,1],-L[:,0],'k-',alpha=1.0,linewidth=0.2)
			#Periàdic
			plt.imshow(np.log(I),extent=[0,1,0,1])
			plt.plot(np.mod(L[:,1],1),np.mod(-L[:,0],1),'k.',alpha=1.0,linewidth=0.2,markersize=0.1)
			break
			Lmid=(L[1:,:]+L[:-1,:])/2.
			if periodicity=='periodic':
				ix=np.int32(np.mod(Lmid[:,0],1)*Nbin)
				iy=np.int32(np.mod(Lmid[:,1],1)*Nbin)
				Lin=np.where((ix>=0)&(iy>=0)&(ix<Nbin)&(iy<Nbin))
				IL=I[ix[Lin],iy[Lin]].flatten()
			else:
				Lin=np.where((np.abs(Lmid[:,1])<1)&(np.abs(Lmid[:,0])<1))
				IL=I[np.int32((Lmid[Lin,0]+1)*Nbin/2),np.int32((Lmid[Lin,1]+1)*Nbin/2)].flatten()
			#pc,cb=np.histogram(np.log(IL),50,density=True)
			IM.append([np.nanmean(I**n) for n in range(N)])
			IML.append([np.nanmean(IL**n) for n in range(N)])
			pci,cbi=np.histogram(np.log(I[I>1e-5]),50,density=True)
			pcil,cbil=np.histogram(np.log(IL[IL>1e-5]),50,density=True)
			plt.plot(cbi[1:],pci,'s',color=plt.cm.jet(t/tmax),label='')
			plt.plot(cbil[1:],pcil,'*',color=plt.cm.jet(t/tmax),label='')
			plt.yscale('log')
			plt.xlabel(r'$\log c_{max}$')
			plt.ylabel(r'$P_{\log c_{max}}$')
			plt.xlim([-5,0.1])
		t=t+dt
		print('Time:',t,' - Npts:',len(L), '- Cmax:',np.mean(1./np.sqrt(1.+wrapped_time/Pe)),np.max(kappa_new))

#%
eps=0.05
plt.imshow(I,extent=[0,1,0,1])
idlarge=np.where(I.flatten()>eps*np.max(I))[0]
idlargeDSM=np.where(Cmaxv>eps*np.nanmax(Cmaxv))[0]
idx,idy=np.int16(idlarge/I.shape[0])/Nbin,np.mod(idlarge,I.shape[0])/Nbin

Lper=np.vstack((np.mod(L[:,1],1),np.mod(-L[:,0],1))).T
dLper=np.sqrt(np.sum(np.diff(Lper,axis=0)**2.,axis=1))
idper=np.where(dLper>0.01)[0]
Lper[idper]=np.nan

plt.plot(Lper[:,0],Lper[:,1],'k-',alpha=1,linewidth=0.2,markersize=0.1)
plt.xlim([0,1])
plt.ylim([0,1])
plt.gca().set_aspect('equal')
plt.savefig('./DSM_sineflow.png',bbox_inches='tight',dpi=600)

#plt.plot(Lper[idlargeDSM,0],Lper[idlargeDSM,1],'m*')
#plt.imshow(I>eps*np.max(I),extent=[0,1,0,1])

#%%  DNS - DSM local comparisons
#%%% * < c | cmax >

keyword='single'
keyword='cubic'
keyword='sine'
#keyword='half'
#keyword='double'
#keyword='halfsmooth'

plt.style.use('~/.config/matplotlib/joris.mplstyle')
import h5py
from skimage.morphology import disk
from skimage.filters import median,gaussian
ms=['o','d','s','<','>','+','*','1','2','3']
ms=ms*10

Brownian=2e-4
#Brownian=2e-4
#Brownian=5e-3
radius=0.01
l0=0.3
A=1/np.sqrt(2)
#A=1.2

dir_img='./Compare_stretching_concentration/'+keyword+'/Fourier_A{:1.1f}_l{:1.1f}_rad{:1.2f}_{:1.0e}'.format(A,l0,radius,Brownian)
dir_out='./Compare_stretching_concentration/'+keyword+'/'

periodicity='periodic'
k=1
#Tmax=[6]
#Tmax_vec=np.array([12])
	#% visual comparison
	#plt.close('all')
I0=cv2.imread(dir_img+'/{:04d}.tif'.format(0),2)
I0=np.float32(I0)/2.**16.
I0med=I0
cmm=np.mean(I0)
#I0med = gaussian(I0, sigma=3)
Imax=np.max(I0med)
f=h5py.File(dir_out+'DSM'+keyword+'_A{:1.1f}_l{:1.1f}.hdf5'.format(A,l0), 'r')

Tmax_vec=np.array([2,4,6,8,10,12])
Tmax_vec=np.arange(8)
try:
	Tmax_vec=np.uint8(np.arange(int(f.attrs['tmax']*0.25),f.attrs['tmax']+1,
														 np.maximum(int(f.attrs['tmax']*0.1),1)))
#	Tmax_vec=[f.attrs['tmax']]
except:
	Tmax_vec=[11]
#Tmax=11

#Tmax_vec=[1]

Cm_all,Cstd_all,Csquare_all,logCm_all,logCstd_all,H2m_all=[],[],[],[],[],[]

for t in Tmax_vec:
	L=f['L_{:04d}'.format(int(t*10))][:]
	wrapped_time=f['wrapped_time_{:04d}'.format(int(t*10))][:]
	S=f['S_{:04d}'.format(int(t*10))][:]
	W=f['Weights_{:04d}'.format(int(t*10))][:]
	I=cv2.imread(dir_img+'/{:04d}.tif'.format(int(t*10)),2)
	I=np.float32(I)/2.**16.
	Imed=I
	#Imed = gaussian(I, sigma=3)
#	I=Imed/Imax
	Nbin=I.shape[0]

	# COmpute n order moment
	# compare cmax from individual strips theory and cmax from imag
	Lmid=(L[1:,:]+L[:-1,:])/2.
	if periodicity=='periodic':
		ix=np.int32(np.mod(Lmid[:,0],1)*Nbin)
		iy=np.int32(np.mod(Lmid[:,1],1)*Nbin)
		Lin=np.where((ix>=0)&(iy>=0)&(ix<Nbin)&(iy<Nbin))
		IL=I[ix[Lin],iy[Lin]].flatten()
	else:
		Lin=np.where((np.abs(Lmid[:,1])<1)&(np.abs(Lmid[:,0])<1))
		IL=I[np.int32((Lmid[Lin,0]+1)*Nbin/2),np.int32((Lmid[Lin,1]+1)*Nbin/2)].flatten()
	
	# Find min stretching rates 
	argsort_id=np.argsort(1./S)
	n=10000
	#plt.plot((Lmid[argsort_id[:n],1]+1)*Nbin/2,(Lmid[argsort_id[:n],0]+1)*Nbin/2,'ro',alpha=1.0)

	dt=0.25
	D=(Brownian)**2./(2*dt)#+1e-08 # numerical diffusion ?
	s0=radius#*np.sqrt(2)
	#Pe=1e7
	Tau=D/s0**2*wrapped_time[Lin]
	Cmax=1./np.sqrt(1.+4*D/s0**2*wrapped_time[Lin]) #Gaussian
#	Cmax=1./(1.+4*D/s0**2*wrapped_time[Lin]) #Wave
	Sc=s0*np.sqrt(1.+4*D/s0**2*wrapped_time[Lin])*S
	Cmax2=1./np.sqrt(4*D/s0**2*wrapped_time[Lin])
	Rho=1./S[Lin]
	#plt.plot(IL,Rho,'.',alpha=0.1)
	xc=np.logspace(-5,0,30)
	#rhoc=bin_operation(IL,Rho,xc,np.mean)
	#xc=np.linspace(0,1,100)
#	Cm=bin_operation(IL,Cmax,xc,np.nanmean)
#	Cmstd=bin_operation(IL,Cmax,xc,np.nanstd)
#	CmN=bin_operation(IL,Cmax,xc,len)
	Cm=bin_operation(Cmax,IL,xc,np.nanmean)
	Cmstd=bin_operation(Cmax,IL,xc,np.nanstd)
	square = lambda x: np.mean(x**2.)
	Cmsquare=bin_operation(Cmax,IL,xc,square)
	CmN=bin_operation(Cmax,IL,xc,len)
	
	logCm=bin_operation(Cmax,np.log(IL),xc,np.nanmean)
	logCmstd=bin_operation(Cmax,np.log(IL),xc,np.nanstd)
	#plt.plot(xc[1:],rhoc,'k-')
	#plt.figure()
	Cm_all.append(Cm)
	Cstd_all.append(Cmstd)
	Csquare_all.append(Cmsquare)
	logCm_all.append(logCm)
	logCstd_all.append(logCmstd)
	
	
	#plt.errorbar(xc[1:],Cm,yerr=Cmstd,color=plt.cm.jet(t/13.))
	H2m_all.append(np.histogram2d(Cmax,IL,xc)[0])
f.close()

x=np.array([1e-4,1])
#cmm=cmm*0
plt.figure(figsize=(2,2))
for t in range(len(Tmax_vec)):
	plt.plot(xc[1:]/I.mean(),Cm_all[t]/I.mean(),'o-',
					color=plt.cm.cool((Tmax_vec[t]-Tmax_vec.min())/(Tmax_vec.max()-Tmax_vec.min())),
					)
#plt.plot(x,x,'k--',label='1')
#plt.plot(x,x**0.5,'k:',label='0.5')
plt.ylabel(r'$c/\langle c \rangle$ ')
plt.xlabel(r'$\theta / \langle c \rangle$')
plt.xlim([1e-3,2e2])
plt.ylim([1e-1,200])
plt.plot(xc/np.mean(I),(xc/I.mean()+xc*(1/xc-1)),'k-')
plt.yscale('log')
plt.xscale('log')
plt.legend(fontsize=6)
plt.savefig('/home/joris/Dropbox/Articles/AggregationBatchelor/Figures/ccmax_sine.pdf',bbox_inches='tight')

plt.figure(figsize=(2,2))
for t in range(len(Tmax_vec)):
	plt.plot(xc[1:]/I.mean(),Cstd_all[t]/Cm_all[t],'o-',
					color=plt.cm.cool((Tmax_vec[t]-Tmax_vec.min())/(Tmax_vec.max()-Tmax_vec.min())),
					)
# 	plt.plot(xc[1:]/I.mean(),np.sqrt(Csquare_all[t]-I.mean()**2.)/I.mean(),'o-',
# 					color=plt.cm.cool((Tmax_vec[t]-Tmax_vec.min())/(Tmax_vec.max()-Tmax_vec.min())),
# 					)

#plt.plot(x,x,'k--',label='1')
#plt.plot(x,x**0.5,'k:',label='0.5')
plt.plot(xc/I.mean(),np.zeros(xc.shape)+0.8,'k-')
plt.ylabel(r'$\sigma_{c|\theta}/\langle c|\theta\rangle$ ')
plt.xlabel(r'$\theta / \langle c \rangle$')
plt.xlim([1e-3,2e2])
plt.ylim([1e-2,10])
plt.yscale('log')
plt.xscale('log')
plt.legend(fontsize=6)
plt.savefig('/home/joris/Dropbox/Articles/AggregationBatchelor/Figures/sigmaccmax_sine.pdf',bbox_inches='tight')

#plt.savefig(dir_out+'Compare_c_cmax_'+periodicity+'_{:1.0e}.pdf'.format(Brownian),bbox_inches='tight')
#%%%  < c-<c> | cmax >

keyword='single'
keyword='cubic'
keyword='sine'
#keyword='half'
#keyword='double'
#keyword='halfsmooth'

plt.style.use('~/.config/matplotlib/joris.mplstyle')
import h5py
from skimage.morphology import disk
from skimage.filters import median,gaussian
ms=['o','d','s','<','>','+','*','1','2','3']
ms=ms*10
Brownian=2e-4
#Brownian=2e-4
#Brownian=5e-3
radius=0.01
l0=0.3
A=1/np.sqrt(2)
#A=1.2

dir_img='./Compare_stretching_concentration/'+keyword+'/Fourier_A{:1.1f}_l{:1.1f}_rad{:1.2f}_{:1.0e}'.format(A,l0,radius,Brownian)
dir_out='./Compare_stretching_concentration/'+keyword+'/'

periodicity='periodic'
k=1
#Tmax=[6]
#Tmax_vec=np.array([12])
	#% visual comparison
	#plt.close('all')
I0=cv2.imread(dir_img+'/{:04d}.tif'.format(0),2)
I0=np.float32(I0)/2.**16.
I0med=I0
cmm=np.mean(I0)
#I0med = gaussian(I0, sigma=3)
Imax=np.max(I0med)
f=h5py.File(dir_out+'DSM'+keyword+'_A{:1.1f}_l{:1.1f}.hdf5'.format(A,l0), 'r')

Tmax_vec=np.array([2,4,6,8,10,12])
Tmax_vec=np.arange(8)
try:
	Tmax_vec=np.uint8(np.arange(int(f.attrs['tmax']*0.25),f.attrs['tmax']+1,
														 np.maximum(int(f.attrs['tmax']*0.1),1)))
#	Tmax_vec=[f.attrs['tmax']]
except:
	Tmax_vec=[11]
#Tmax=11

#Tmax_vec=[1]

Cm_all,Cstd_all,logCm_all,logCstd_all,H2m_all=[],[],[],[],[]

for t in Tmax_vec:
	L=f['L_{:04d}'.format(int(t*10))][:]
	wrapped_time=f['wrapped_time_{:04d}'.format(int(t*10))][:]
	S=f['S_{:04d}'.format(int(t*10))][:]
	W=f['Weights_{:04d}'.format(int(t*10))][:]
	I=cv2.imread(dir_img+'/{:04d}.tif'.format(int(t*10)),2)
	I=np.float32(I)/2.**16.
	Imed=I
	#Imed = gaussian(I, sigma=3)
#	I=Imed/Imax
	Nbin=I.shape[0]

	# COmpute n order moment
	# compare cmax from individual strips theory and cmax from imag
	Lmid=(L[1:,:]+L[:-1,:])/2.
	if periodicity=='periodic':
		ix=np.int32(np.mod(Lmid[:,0],1)*Nbin)
		iy=np.int32(np.mod(Lmid[:,1],1)*Nbin)
		Lin=np.where((ix>=0)&(iy>=0)&(ix<Nbin)&(iy<Nbin))
		IL=I[ix[Lin],iy[Lin]].flatten()
	else:
		Lin=np.where((np.abs(Lmid[:,1])<1)&(np.abs(Lmid[:,0])<1))
		IL=I[np.int32((Lmid[Lin,0]+1)*Nbin/2),np.int32((Lmid[Lin,1]+1)*Nbin/2)].flatten()
	
	# Find min stretching rates 
	argsort_id=np.argsort(1./S)
	n=10000
	#plt.plot((Lmid[argsort_id[:n],1]+1)*Nbin/2,(Lmid[argsort_id[:n],0]+1)*Nbin/2,'ro',alpha=1.0)

	dt=0.25
	D=(Brownian)**2./(2*dt)#+1e-08 # numerical diffusion ?
	s0=radius#*np.sqrt(2)
	#Pe=1e7
	Tau=D/s0**2*wrapped_time[Lin]
	Cmax=1./np.sqrt(1.+4*D/s0**2*wrapped_time[Lin]) #Gaussian
#	Cmax=1./(1.+4*D/s0**2*wrapped_time[Lin]) #Wave
	Sc=s0*np.sqrt(1.+4*D/s0**2*wrapped_time[Lin])*S
	Cmax2=1./np.sqrt(4*D/s0**2*wrapped_time[Lin])
	Rho=1./S[Lin]
	#plt.plot(IL,Rho,'.',alpha=0.1)
	xc=np.logspace(-5,0,50)
	#rhoc=bin_operation(IL,Rho,xc,np.mean)
	#xc=np.linspace(0,1,100)
#	Cm=bin_operation(IL,Cmax,xc,np.nanmean)
#	Cmstd=bin_operation(IL,Cmax,xc,np.nanstd)
#	CmN=bin_operation(IL,Cmax,xc,len)
	cmm=I.mean()
	Cm=bin_operation(Cmax,np.abs(IL-cmm),xc,np.nanmean)
	Cmstd=bin_operation(Cmax,IL,xc,np.nanstd)
	CmN=bin_operation(Cmax,IL,xc,len)
	
	logCm=bin_operation(Cmax,np.log(IL),xc,np.nanmean)
	logCmstd=bin_operation(Cmax,np.log(IL),xc,np.nanstd)
	#plt.plot(xc[1:],rhoc,'k-')
	#plt.figure()
	Cm_all.append(Cm)
	Cstd_all.append(Cmstd)
	logCm_all.append(logCm)
	logCstd_all.append(logCmstd)
	
	#plt.errorbar(xc[1:],Cm,yerr=Cmstd,color=plt.cm.jet(t/13.))
	H2m_all.append(np.histogram2d(Cmax,IL,xc)[0])
f.close()

x=np.array([1e-4,1])
#cmm=cmm*0
for t in range(len(Tmax_vec)):
	plt.plot(xc[1:],Cm_all[t],ms[t]+'-',
					color=plt.cm.viridis((Tmax_vec[t]-Tmax_vec.min())/(Tmax_vec.max()-Tmax_vec.min())),
					label='$t={:1.0f}$'.format(Tmax_vec[t]))
plt.plot(x,x,'k--',label='1')
plt.plot(x,x**0.5,'k:',label='0.5')
plt.ylabel(r'$c-\langle c \rangle$ (DNS)')
plt.xlabel(r'$c_\mathrm{max}$ (LS)')
plt.xlim([1e-5,1.2])
plt.ylim([1e-5,1.2])
plt.yscale('log')
plt.xscale('log')
plt.legend(fontsize=6)
plt.savefig(dir_out+'Compare_c_cmax_'+periodicity+'_{:1.0e}.pdf'.format(Brownian),bbox_inches='tight')

#%%%  <c-<c> |  cmax>

keyword='single'
keyword='cubic'
keyword='sine'
#keyword='half'
#keyword='double'
#keyword='halfsmooth'

plt.style.use('~/.config/matplotlib/joris.mplstyle')
import h5py
from skimage.morphology import disk
from skimage.filters import median,gaussian
ms=['o','d','s','<','>','+','*','1','2','3']
ms=ms*10
Brownian=1e-3
#Brownian=2e-4
#Brownian=5e-3
radius=0.01
l0=0.3
A=1/np.sqrt(2)
#A=1.2

dir_img='./Compare_stretching_concentration/'+keyword+'/Fourier_A{:1.1f}_l{:1.1f}_rad{:1.2f}_{:1.0e}'.format(A,l0,radius,Brownian)
dir_out='./Compare_stretching_concentration/'+keyword+'/'

periodicity='periodic'
k=1
#Tmax=[6]
#Tmax_vec=np.array([12])
	#% visual comparison
	#plt.close('all')
I0=cv2.imread(dir_img+'/{:04d}.tif'.format(0),2)
I0=np.float32(I0)/2.**16.
cmm=np.mean(I0)
Nbin=I0.shape[0]
#I0med = gaussian(I0, sigma=3)
f=h5py.File(dir_out+'DSM'+keyword+'_A{:1.1f}_l{:1.1f}.hdf5'.format(A,l0), 'r')

Tmax_vec=np.array([8])
#Tmax_vec=np.arange(8)
#try:
#	Tmax_vec=np.uint8(np.arange(int(f.attrs['tmax']*0.25),f.attrs['tmax']+1,
#														 np.maximum(int(f.attrs['tmax']*0.1),1)))
##	Tmax_vec=[f.attrs['tmax']]
#except:
#	Tmax_vec=[11]
##Tmax=11

#Tmax_vec=[1]

Cm_all,Cm2_all,Cstd_all,logCm_all,logCstd_all,H2m_all=[],[],[],[],[],[]

for t in Tmax_vec:
	L=f['L_{:04d}'.format(int(t*10))][:]
	wrapped_time=f['wrapped_time_{:04d}'.format(int(t*10))][:]
	S=f['S_{:04d}'.format(int(t*10))][:]
	W=f['Weights_{:04d}'.format(int(t*10))][:]
	I=cv2.imread(dir_img+'/{:04d}.tif'.format(int(t*10)),2)
	I=np.float32(I)/2.**16.
	I=I-I.mean()
	dt=0.25
	D=(Brownian)**2./(2*dt)#+1e-08 # numerical diffusion ?
	s0=radius#*np.sqrt(2)
	#Pe=1e7
	Tau=D/s0**2*wrapped_time
	Cmax=1./np.sqrt(1.+4*D/s0**2*wrapped_time) #Gaussian
	Cmax=1./(1.+4*D/s0**2*wrapped_time) #Wave
	Sc=s0*np.sqrt(1.+4*D/s0**2*wrapped_time)*S
	Cmax2=1./np.sqrt(4*D/s0**2*wrapped_time)
	Rho=1./S
	#plt.plot(IL,Rho,'.',alpha=0.1)
	xc=np.logspace(-10,0,50)
	# COmpute n order moment
	# compare cmax from individual strips theory and cmax from imag
	Lmid=np.mod((L[1:,:]+L[:-1,:])/2.,1)
	Nbin=int(1/Sc.mean())
	ix=np.int32(np.mod(Lmid[:,0],1)*Nbin)
	iy=np.int32(np.mod(Lmid[:,1],1)*Nbin)
	
	nsamples=1000
	ii=np.uint32(np.random.rand(nsamples)*(ix.shape[0]-1))
	Cmax_lag,Cmax_lagmean=[],[]
	Cmax_eul=[]
	sB=Sc.mean()*2
	print(sB)
	from scipy import stats,ndimage
	n=1
	Cmax_lag = stats.binned_statistic_2d(np.mod(L[1::n,0],1), np.mod(L[1::n,1],1), Cmax[::n], statistic='max', bins=int(1/sB))[0]
	Cmax_lag_mean = stats.binned_statistic_2d(np.mod(L[1::n,0],1), np.mod(L[1::n,1],1), Cmax[::n], statistic='mean', bins=int(1/sB))[0]
	Iscaled=ndimage.maximum_filter(I, size=int(I.shape[0]*sB))
	Cmax_eul=ndimage.zoom(Iscaled,int(1/sB)/I.shape[0])
#	for i in ii:
#		idin=(ix==ix[i])&(iy==iy[i])
#		print(np.sum(idin))
#		Cmax_lag.append(Cmax[idin].max())
#		Cmax_lagmean.append(Cmax[idin].mean())
#		Cmax_eul.append(I[ix[i],iy[i]])
	#rhoc=bin_operation(IL,Rho,xc,np.mean)
	#xc=np.linspace(0,1,100)
#	Cm=bin_operation(IL,Cmax,xc,np.nanmean)
#	Cmstd=bin_operation(IL,Cmax,xc,np.nanstd)
#	CmN=bin_operation(IL,Cmax,xc,len)
	cmm=I.mean()
	Cm=bin_operation(Cmax_lag.flatten(),np.abs(Cmax_eul).flatten(),xc,np.nanmean)
	Cm2=bin_operation(Cmax_lag_mean.flatten(),np.abs(Cmax_eul).flatten(),xc,np.nanmean)
	Cm_all.append(Cm)
	Cm2_all.append(Cm2)
	
f.close()

x=np.array([1e-4,1])
#cmm=cmm*0
for t in range(len(Tmax_vec)):
	plt.plot(xc[1:],Cm_all[t],ms[t]+'-',
					color=plt.cm.viridis((Tmax_vec[t]-Tmax_vec.min())/(Tmax_vec.max()-Tmax_vec.min())),
					label='$t={:1.0f}$'.format(Tmax_vec[t]))
	plt.plot(xc[1:],Cm2_all[t],ms[t]+'--',
					color=plt.cm.viridis((Tmax_vec[t]-Tmax_vec.min())/(Tmax_vec.max()-Tmax_vec.min())),
					label='$t={:1.0f}$'.format(Tmax_vec[t]))
plt.plot(x,x,'k--',label=r'${\tau}$')
plt.plot(x,x**0.5,'k:',label=r'$\sqrt{\tau}$')
plt.ylabel(r'$c-\langle c \rangle$ (DNS)')
plt.xlabel(r'$1/\tau$ (LS)')
plt.xlim([1e-8,1.0])
plt.ylim([1e-8,1.0])
plt.yscale('log')
plt.xscale('log')
plt.legend(fontsize=6)
#plt.savefig(dir_out+'Compare_c_cmax_'+periodicity+'_{:1.0e}.pdf'.format(Brownian),bbox_inches='tight')



#%% Eulerian Statistics of Bundles

#%%% ** Grid based scalings

plt.style.use('~/.config/matplotlib/joris.mplstyle')

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
plt.savefig('/home/joris/Dropbox/Articles/AggregationBatchelor/Figures/agg1_sine.pdf',bbox_inches='tight')

#%%% <1/rho^2 n>_B
keyword='single'
#keyword='cubic'
keyword='sine'
#keyword='half'
#keyword='double'
#keyword='halfsmooth'

plt.style.use('~/.config/matplotlib/joris.mplstyle')
import h5py
from skimage.morphology import disk
from skimage.filters import median,gaussian
ms=['o','d','s','<','>','+','*','1','2','3']
ms=ms*10
Brownian=5e-4
Brownian=1e-3
#Brownian=2e-4
Brownian=1e-3
radius=0.01
l0=0.3
A=1/np.sqrt(2)
A=1.2

dir_img='./Compare_stretching_concentration/'+keyword+'/Fourier_A{:1.1f}_l{:1.1f}_rad{:1.2f}_{:1.0e}'.format(A,l0,radius,Brownian)
dir_out='./Compare_stretching_concentration/'+keyword+'/'

periodicity='periodic'
k=1
#Tmax=[6]
#Tmax_vec=np.array([12])
	#% visual comparison
	#plt.close('all')
f=h5py.File(dir_out+'DSM'+keyword+'_A{:1.1f}_l{:1.1f}.hdf5'.format(A,l0), 'r')

try:
	t=f.attrs['tmax']
except:
	t=11

T=np.arange(7,t+1)
T=np.array([t])
dt=0.25
D=(Brownian)**2./(2*dt)#+1e-08 # numerical diffusion ?
s0=radius#*np.sqrt(2)

VM=[]
fig,ax=plt.subplots(3,1,figsize=(2,6),sharex=True)
for t in T:
	L=f['L_{:04d}'.format(int(t*10))][:]
	wrapped_time=f['wrapped_time_{:04d}'.format(int(t*10))][:]
	S=f['S_{:04d}'.format(int(t*10))][:]
	W=f['Weights_{:04d}'.format(int(t*10))][:]
	s0=radius#*np.sqrt(2)
	Tau=D/s0**2*wrapped_time
	Cmax=1./np.sqrt(1.+4*D/s0**2*wrapped_time)
	Sc=s0*np.sqrt(1.+4*D/s0**2*wrapped_time)*S
	sB=4*np.mean(Sc)
	
	Lmod=np.mod(L,1)
	dist_old=np.sum(np.diff(L,axis=0)**2,1)**0.5
	N=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1,sB)
										 ,weights=dist_old/sB,density=False)[0]
	S1=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1,sB)
										 ,weights=S*dist_old/sB,density=False)[0]
	C1=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1,sB)
										 ,weights=Cmax*dist_old/sB,density=False)[0]
	S2=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1,sB)
										 ,weights=S**2.*dist_old/sB,density=False)[0]
	S4=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1,sB)
										 ,weights=S**4.*dist_old/sB,density=False)[0]
	C2=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1,sB)
										 ,weights=Cmax**2.*dist_old/sB,density=False)[0]
#	VM.append(np.mean(S2))
	V=S2/N-(S1/N)**2.
	Vr=V**0.5/(S1/N)
	#plt.plot((N[N>1]).flatten(),(S2[N>1]/N[N>1]).flatten(),'.',alpha=0.01,color=plt.cm.jet(t/T.max()))
	nbin=np.logspace(0,3,30)
	s2bin=bin_operation((N[N>1]).flatten(),(V[N>1]).flatten(),nbin,np.mean)
	ax[1].plot(np.log(N[N>1].flatten()),np.log(V[N>1].flatten()),'.',alpha=0.05,color=plt.cm.cool(t/T.max()),label='$t={:1.0f}$'.format(t))
	s2bin=bin_operation((N[N>1]).flatten(),(C1[N>1]/N[N>1]).flatten(),nbin,np.mean)
	ax[0].plot(np.log(N[N>1].flatten()),np.log((C1[N>1]/N[N>1]).flatten()),'.',alpha=0.05,color=plt.cm.cool(t/T.max()),label='$t={:1.0f}$'.format(t))
	s2bin=bin_operation((N[N>1]).flatten(),(Vr[N>1]).flatten(),nbin,np.mean)
	ax[2].plot(np.log(N[N>1].flatten()),np.log(Vr[N>1].flatten()),'.',alpha=0.05,color=plt.cm.cool(t/T.max()),label='$t={:1.0f}$'.format(t))
	VM.append(np.nanmean(V[N>1]*N[N>1]))
# [axx.set_yscale('log') for axx in ax]
# [axx.set_xscale('log') for axx in ax]
x=np.linspace(1,100,100)
ax[0].plot(x,1e-2/x,'k--',label='-1')
ax[1].plot(x,1e-5/x**2,'k--',label='-1')
ax[2].plot(x,1e-1*x**0.5,'k--',label='0.5')
ax[0].set_ylabel(r'$\mu_{\theta,\mathcal{B}}$')
ax[1].set_ylabel(r'$\sigma^2_{\theta,\mathcal{B}}$')
ax[2].set_ylabel(r'$\sigma_{\theta,\mathcal{B}}/\mu_{\theta,\mathcal{B}}$')
#plt.plot(x,1e-4*x**-2,'k-',label='-2')
plt.legend(ncol=2,fontsize=6)
plt.xlabel('$n$')
plt.figure()
plt.plot(T,np.log(VM),'o-')
P=np.polyfit(T,np.log(VM),1)
plt.plot(T,P[0]*T+P[1],'k--',label='{:1.2f}'.format(P[0]))
plt.xlabel(r'$t$')
plt.ylabel(r'$\delta^2(t) = \langle \langle \theta^2\rangle _B  n \rangle $')
plt.legend()
#plt.yscale('log')

I=cv2.imread(dir_img+'/{:04d}.tif'.format(int(t*10)),2)
I=np.float32(I)/2.**16.
Imed=I
Imax=I.max()
cm=np.mean(I)

plt.figure()
plt.imshow(I,clim=[0,np.max(I)])
plt.axis('off')
plt.figure()
plt.imshow((C1),clim=[0,np.max(I)])
plt.axis('off')

#FIXME
#TODO: Check why we do not find < theta ^2>_B ~ 1/n in eulerian
# It seems that sigma_theta / mean_theta follows the write scaling N^0.5
#%%% <1/rho^2 n>_B Parrallel
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
#%%% Plot

Q=np.loadtxt('Sine_scaling_N_sB1_75.txt')
D=np.loadtxt('Sine_D1.txt')
L=np.loadtxt('Sine_Lyap.txt')

lyap=L[:,1]
sigma2=L[:,2]
d1=np.linspace(1.0,2,100)
ratio=(lyap+sigma2/2)/(lyap+sigma2)
plt.plot(Q[:,0],Q[:,3],'ko',fillstyle='full')
plt.plot(ratio+1,Q[:,3],'ro',fillstyle='full')
plt.plot(d1,1/(d1-1),'k-',fillstyle='full')
plt.ylim([0,4])

plt.figure()
plt.plot(Q[:,0],1/ratio,'ro',fillstyle='full')
plt.plot(Q[:,0],Q[:,3],'ko',fillstyle='full')
plt.plot(d1,1/(d1-1),'k-',fillstyle='full')
plt.ylim([0,4])

plt.figure()
plt.plot(D[:,0],D[:,1])
plt.plot(D[:,0],ratio+1,'ro',fillstyle='full')

plt.figure()
ratio=(lyap+sigma2/2)/(lyap+sigma2)
plt.plot(ratio,1/Q[:,3])
plt.plot((Q[:,0]-1),1/Q[:,3])
plt.plot(ratio,ratio,'k-')

plt.figure()
plt.plot(Q[:,0]-1,1/Q[:,3],'*')
plt.plot(Q[:,0]-1,Q[:,0]-1,'-')
plt.plot(Q[:,0]-1,ratio,'o')



#%%% <c^2 | n>_B Parrallel
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

#%%% Why rho ^-1 | n do not predict c | n ?

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
		plt.savefig(dir_out+'/Sine_c2_n_a{:1.1f}_s{:1.0f}.png'.format(a,seed))
		
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

#%%% Plot
A=np.array([]).reshape(-1,5)
Seeds=np.arange(5)
for s in Seeds:
	A=np.vstack((A,np.loadtxt('./Sine_scaling_with_n_seed{:1.0f}.txt'.format(s))))

A=np.array(A)
plt.plot(A[:,1],A[:,3],'*')
plt.plot(A[:,1],A[:,4],'o')

plt.ylim([-1,0])

R=[]
for a in np.unique(A[:,1]):
	ida=np.where(A[:,1]==a)[0]
	R.append([a,A[ida,3].mean(),A[ida,4].mean()])

R=np.array(R)
plt.figure()
plt.plot(R[:,0],R[:,1])
plt.plot(R[:,0],R[:,2])

np.savetxt('Sine_scaling_with_n.txt',R,header='A,n \gamma_\rho^{-1}, \gamma_c')
#%%% compare c^2|n and 1/rho^2 |n 
sB=1/50
A=np.loadtxt('Sine_scaling_C|N_sB1_{:1.0f}.txt'.format(1/sB))

B=np.loadtxt('Sine_scaling_N_sB1_{:1.0f}.txt'.format(1/sB))

plt.figure()
plt.plot(A[:,0],A[:,1],'rd',label='$\gamma_{2,c|n}$')
plt.plot(B[:,0],B[:,2]-1,'ko',label=r'$\gamma_{2,\rho^{-1}|n}-1$')
plt.legend()
plt.xlabel('$D_1$')


plt.figure()
plt.plot(A[:,0],A[:,2],'rd',label='$\omega_{2,c|n}$')
plt.plot(B[:,0],B[:,-1],'ko',label=r'$\omega_{2,\rho^{-1}|n}-1$')
plt.legend()
plt.xlabel('$D_1$')
plt.ylim([-20,0])

#%%% (1/rho)_B , N funcion of sB
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
	
SB=1/np.logspace(1,3,20)
#AA=np.linspace(0.3,1.8,15)
a=1.2
Pall=[]
plt.figure()
Pallseeds=[]
def parrallel(seed):
	Pall=[]
	print('Seed=',seed,'A=',a)
	L,S,wrapped_time,W,t=run_DSM(1e7, a, seed)
	Lmod=np.mod(L,1)
	dist_old=np.sum(np.diff(L,axis=0)**2,1)**0.5
	for sB in SB:
		s0=0.01
		D=sB**2*0.5/2
		Tau=D/s0**2*wrapped_time
		Cmax=1./np.sqrt(1.+4*D/s0**2*wrapped_time)
		Sc=s0*np.sqrt(1.+4*D/s0**2*wrapped_time)*S
		print(np.mean(Sc))
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

Seeds=np.arange(10)

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
plt.plot(1/SB,-Pm[:,1,0],'rd')
plt.plot(1/SB,-Pm[:,0,0],'kd')
plt.xlabel("$1/s_B$")
plt.ylabel("$\gamma_1,\gamma_2$")
plt.savefig('Sine_gamma_sb.pdf',bbox_inches='tight')

plt.figure()
plt.plot(np.log(1/SB),Pm[:,1,0],'bo')
plt.plot(np.log(1/SB),Pm[:,1,1],'rd')
plt.plot(np.log(1/SB),-2*np.log(1/(SB*l0)),'r-')
plt.xlabel("$1/s_B$")
plt.ylabel("$\omega_2$")
plt.savefig('Sine_omega_sb.pdf',bbox_inches='tight')
plt.ylim([-20,0])


plt.figure()
plt.plot(1/SB,Pm[:,2,0],'kd')
plt.plot(1/SB,Pm[:,3,0],'rd')
plt.xlabel(r"$1/s_B$")
plt.ylabel(r"$\alpha,\beta$")
plt.savefig('Sine_alpha_beta.pdf',bbox_inches='tight')

np.savetxt('Sine_scaling_N_A_{:1.1f}.txt'.format(a),np.vstack((SB,-Pm[:,0,0],-Pm[:,1,0],Pm[:,2,0],Pm[:,3,0],Pm[:,1,1])).T,header='#sB,\gamma_1,\gamma_2,\alpha,\beta,\omega_2')
#%%% var(rho)_sb , N funcion of Pe
# =============================================================================
# Exponent is stable with Péclet'
# =============================================================================
A=1.3
Pall=[]
L,S,wrapped_time,W=run_DSM(1e7, A)
plt.figure()
DD=np.logspace(-4,-7,8)
for D in DD:
	
	s0=0.01#*np.sqrt(2)
	Tau=D/s0**2*wrapped_time
	Cmax=1./np.sqrt(1.+4*D/s0**2*wrapped_time)
	Sc=s0*np.sqrt(1.+4*D/s0**2*wrapped_time)*S
	sB=4*np.mean(Sc)
	Lmod=np.mod(L,1)
	dist_old=np.sum(np.diff(L,axis=0)**2,1)**0.5
	N=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1,sB)
										 ,weights=dist_old/sB,density=False)[0]
	S1=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1,sB)
										 ,weights=S*dist_old/sB,density=False)[0]
	C1=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1,sB)
										 ,weights=Cmax*dist_old/sB,density=False)[0]
	S2=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1,sB)
										 ,weights=S**2.*dist_old/sB,density=False)[0]
	S4=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1,sB)
										 ,weights=S**4.*dist_old/sB,density=False)[0]
	C2=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1,sB)
										 ,weights=Cmax**2.*dist_old/sB,density=False)[0]
	#	VM.append(np.mean(S2))
	V=S2/N-(S1/N)**2.
	Vr=V**0.5/(S1/N)
	nbin=np.logspace(0,3,30)
	n=15
	s2bin=bin_operation((N[N>n]).flatten(),(V[N>1]).flatten(),nbin,np.mean)
	plt.plot(np.log(N[N>n].flatten()),np.log(V[N>n].flatten()),'.',alpha=0.1)
	P=np.polyfit(np.log(N[N>n].flatten()),np.log(V[N>n].flatten()),1)
	P1=lin_reg(np.log(N[N>n].flatten()),np.log(V[N>n].flatten()))
	plt.plot(np.log(nbin[1:]),np.log(nbin[1:])*P1[1]+P1[0],label=r'${:1.2f} \pm {:1.2f} (r^2={:1.2f})$ '.format(P1[1],P1[3],P1[4]))
	P2 = linregress(np.log(N[N>n].flatten()),np.log(V[N>n].flatten()))
	Pall.append(P2)
plt.plot(np.log(nbin[1:]),np.log(nbin[1:])*(-2)+P1[0],'k--')
plt.plot(np.log(nbin[1:]),np.log(nbin[1:])*(-1)+P1[0],'k--')
plt.legend()

Pe=1/DD
Pall=np.array(Pall)
plt.figure()
plt.plot(Pe,-Pall[:,0],'ko',label='Sine FLow ($A=1.3$)')
plt.xscale('log')
plt.ylim([0.5,2])
plt.legend()

np.savetxt('gamma2_Sine_{:1.1f}_Peclet.txt'.format(A),np.vstack((Pe,-Pall[:,0])).T)
#%%% <1/rho^2 n>_B / (N<1/rho>_L)
#HERE
keyword='single'
#keyword='cubic'
keyword='sine'
#keyword='half'
#keyword='double'
#keyword='halfsmooth'

plt.style.use('~/.config/matplotlib/joris.mplstyle')
import h5py
from skimage.morphology import disk
from skimage.filters import median,gaussian
ms=['o','d','s','<','>','+','*','1','2','3']
ms=ms*10
Brownian=5e-4
Brownian=1e-3
#Brownian=2e-4
#Brownian=1e-2
radius=0.01
l0=0.3
A=1/np.sqrt(2)
#A=1.2

dir_img='./Compare_stretching_concentration/'+keyword+'/Fourier_A{:1.1f}_l{:1.1f}_rad{:1.2f}_{:1.0e}'.format(A,l0,radius,Brownian)
dir_out='./Compare_stretching_concentration/'+keyword+'/'

periodicity='periodic'
k=1
#Tmax=[6]
#Tmax_vec=np.array([12])
	#% visual comparison
	#plt.close('all')
f=h5py.File(dir_out+'DSM'+keyword+'_A{:1.1f}_l{:1.1f}.hdf5'.format(A,l0), 'r')

try:
	t=f.attrs['tmax']
except:
	t=12

T=np.arange(7,t+1)

dt=0.25
D=(Brownian)**2./(2*dt)#+1e-08 # numerical diffusion ?
s0=radius#*np.sqrt(2)

VM=[]
fig,ax=plt.subplots(3,1,figsize=(2,6),sharex=True)
for t in T:
	L=f['L_{:04d}'.format(int(t*10))][:]
	wrapped_time=f['wrapped_time_{:04d}'.format(int(t*10))][:]
	S=f['S_{:04d}'.format(int(t*10))][:]
	W=f['Weights_{:04d}'.format(int(t*10))][:]
	s0=radius#*np.sqrt(2)
	Tau=D/s0**2*wrapped_time
	Cmax=1./np.sqrt(1.+4*D/s0**2*wrapped_time)
	Sc=s0*np.sqrt(1.+4*D/s0**2*wrapped_time)*S
	sB=np.mean(Sc)
	
	Lmod=np.mod(L,1)
	dist_old=np.sum(np.diff(L,axis=0)**2,1)**0.5
	N=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1,sB)
										 ,weights=dist_old/sB,density=False)[0]
	S1=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1,sB)
										 ,weights=S*dist_old/sB,density=False)[0]
	C1=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1,sB)
										 ,weights=Cmax*dist_old/sB,density=False)[0]
	S2=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1,sB)
										 ,weights=S**2.*dist_old/sB,density=False)[0]
	S4=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1,sB)
										 ,weights=S**4.*dist_old/sB,density=False)[0]
	C2=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1,sB)
										 ,weights=Cmax**2.*dist_old/sB,density=False)[0]
#	VM.append(np.mean(S2))
	V=S2/N-(S1/N)**2.
	Vr=V**0.5/(S1/N)
	#plt.plot((N[N>1]).flatten(),(S2[N>1]/N[N>1]).flatten(),'.',alpha=0.01,color=plt.cm.jet(t/T.max()))
	nbin=np.logspace(0,3,30)
	s2bin=bin_operation((N[N>1]).flatten(),(V[N>1]).flatten(),nbin,np.mean)
	ax[1].plot(nbin[1:],s2bin,'ko-',color=plt.cm.cool(t/T.max()),label='$t={:1.0f}$'.format(t))
	s2bin=bin_operation((N[N>1]).flatten(),(C1[N>1]/N[N>1]).flatten(),nbin,np.mean)
	ax[0].plot(nbin[1:],s2bin,'ks-',color=plt.cm.cool(t/T.max()),label='$t={:1.0f}$'.format(t))
	s2bin=bin_operation((N[N>1]).flatten(),(Vr[N>1]).flatten(),nbin,np.mean)
	ax[2].plot(nbin[1:],s2bin,'ks-',color=plt.cm.cool(t/T.max()),label='$t={:1.0f}$'.format(t))
	
	varS=(np.average(S**2,weights=S)-np.average(S,weights=S)**2.)
	
	VM.append(np.nanmean(V*N)/(np.nanmean(N)*np.var(S)))
	#VM.append(np.nanmean(V*N)/(np.nanmean(N)*varS))
	
[axx.set_yscale('log') for axx in ax]
[axx.set_xscale('log') for axx in ax]
x=np.linspace(1,100,100)
ax[0].plot(x,1e-2/x,'k--',label='-1')
ax[1].plot(x,1e-5/x,'k--',label='-1')
ax[2].plot(x,1e-1*x**0.5,'k--',label='0.5')
ax[0].set_ylabel(r'$\mu_{\theta,\mathcal{B}}$')
ax[1].set_ylabel(r'$\sigma^2_{\theta,\mathcal{B}}$')
ax[2].set_ylabel(r'$\sigma_{\theta,\mathcal{B}}/\mu_{\theta,\mathcal{B}}$')
#plt.plot(x,1e-4*x**-2,'k-',label='-2')
plt.legend(ncol=2,fontsize=6)
plt.xlabel('$n$')

plt.figure()
plt.plot(T,(VM),'o-')
plt.xlabel(r'$t$')
plt.ylabel(r'$ \langle \langle n \theta^2 \rangle_\mathcal{B} \rangle_A / \langle n \rangle_A \langle \theta^2 \rangle_L $')
plt.ylim([0,1])
plt.legend()
#plt.yscale('log')

I=cv2.imread(dir_img+'/{:04d}.tif'.format(int(t*10)),2)
I=np.float32(I)/2.**16.
Imed=I
Imax=I.max()
cm=np.mean(I)

plt.figure()
plt.imshow(I,clim=[0,np.max(I)])
plt.axis('off')
plt.figure()
plt.imshow((C1),clim=[0,np.max(I)])
plt.axis('off')

#FIXME
#TODO: Check why we do not find < theta ^2>_B ~ 1/n in eulerian
# It seems that sigma_theta / mean_theta follows the write scaling N^0.5

#%%% Pdf of 1/rho in bundles
s=3
A=0.8

L,S,wrapped_time,W,t=run_DSM(1e7,A,s,STOP_ON_LMAX=True)

bins=np.logspace(-9,0,20)
dlogb=np.diff(np.log(bins))[0]

Lmod=np.mod(L,1)
dist_old=np.sum(np.diff(L,axis=0)**2,1)**0.5
sB=1/100
Nb=[]
N=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1+sB,sB)
									 ,weights=dist_old/sB,density=False)[0]

for i in range(len(bins)-1):	
	print(bins[i])
	Nb.append(np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1+sB,sB)
									 ,weights=dist_old/sB*(S>bins[i])*(S<=bins[i+1]),density=False)[0])

# Save images
dir_save='/data/Simulations/SineFlow/'
for i in range(len(Nb)):
	plt.figure()
	plt.imshow(Nb[i])
	plt.title(r'${:1.1e}<1/\rho<{:1.1e}$'.format(bins[i],bins[i+1]))
	plt.axis('off')
	plt.savefig(dir_save+'pdf_in_bundles/{:04d}.png'.format(i),bbox_inches='tight')
	


# Normalize by density
B=np.tile(bins[:-1].reshape(-1,1,1),(N.shape[0],N.shape[1]))
Nb_norm=np.array(Nb)/(np.sum(Nb,axis=0)/B*dlogb)
dir_save='/data/Simulations/SineFlow/'
for i in range(len(Nb)):
	plt.figure()
	plt.imshow(Nb_norm[i,:,:])
	plt.title(r'${:1.1e}<1/\rho<{:1.1e}$'.format(bins[i],bins[i+1]))
	plt.axis('off')
	plt.savefig(dir_save+'pdf_in_bundles/{:04d}.png'.format(i),bbox_inches='tight')

plt.figure()
for i in range(Nb_norm.shape[1]):
	print(i)
	for j in range(Nb_norm.shape[2]):
		plt.plot(B[:,i,j],Nb_norm[:,i,j],'.',alpha=0.1,color=plt.cm.jet(np.log(N[i,j])/5))
plt.yscale('log');plt.xscale('log');
 
# PDF Scaled by local n
Bn=B*N # new bins
# interpolate on comon bins
bins_s=np.logspace(-8,3,19)
Bs=np.tile(bins_s.reshape(-1,1,1),(N.shape[0],N.shape[1]))
Nb_norm[~np.isfinite(Nb_norm)]=0
Nb_scaled=np.zeros(Nb_norm.shape)
plt.figure()
for i in range(Nb_norm.shape[1]):
	print(i)
	for j in range(Nb_norm.shape[2]):
		Nb_scaled[:,i,j]=np.interp(bins_s,Bn[:,i,j],Nb_norm[:,i,j])
#		plt.plot(Bn[:,i,j],Nb_norm[:,i,j],'.',alpha=0.1,color=plt.cm.jet(np.log(N[i,j])/5))
		plt.plot(bins_s,Nb_scaled[:,i,j],'.',alpha=0.1,color=plt.cm.jet(np.log(N[i,j])/5))
plt.yscale('log');plt.xscale('log');
hs,xs=np.histogram(S,np.logspace(-9,0,100),density=True)
plt.plot(xs[1:],hs,'k--');plt.yscale('log');plt.xscale('log');
# Normalize by density
#Nb_norm=np.array(Nb)/(np.sum(Nb,axis=0)/Bn*dlogb)
dir_save='/data/Simulations/SineFlow/'
for i in range(len(Nb)):
	plt.figure()
	plt.imshow(Nb_scaled[i,:,:])
	plt.title(r'${:1.1e}<1/\rho<{:1.1e}$'.format(bins_s[i],bins_s[i+1]))
	plt.axis('off')
	plt.savefig(dir_save+'pdf_in_bundles/{:04d}.png'.format(i),bbox_inches='tight')

#%% Lagrangian Statistics in Bundles
#%%% Initiation
keyword='single'
#keyword='cubic'
keyword='sine'
#keyword='half'
#keyword='double'
#keyword='halfsmooth'

plt.style.use('~/.config/matplotlib/joris.mplstyle')
import h5py
from skimage.morphology import disk
from skimage.filters import median,gaussian
ms=['o','d','s','<','>','+','*','1','2','3']
ms=ms*10
Brownian=1e-2
#Brownian=2e-4
#Brownian=5e-3
radius=0.01
l0=0.3
A=1/np.sqrt(2)
#A=1.2

dir_img='./Compare_stretching_concentration/'+keyword+'/Fourier_A{:1.1f}_l{:1.1f}_rad{:1.2f}_{:1.0e}'.format(A,l0,radius,Brownian)
dir_out='./Compare_stretching_concentration/'+keyword+'/'

periodicity='periodic'
k=1
#Tmax=[6]
#Tmax_vec=np.array([12])
	#% visual comparison
	#plt.close('all')
I0=cv2.imread(dir_img+'/{:04d}.tif'.format(0),2)
I0=np.float32(I0)/2.**16.
I0med=I0
Imax=I0.max()
cm=np.mean(I0)
f=h5py.File(dir_out+'DSM'+keyword+'_A{:1.1f}_l{:1.1f}.hdf5'.format(A,l0), 'r')

try:
	t=f.attrs['tmax']
except:
	t=11

t=11

L=f['L_{:04d}'.format(int(t*10))][:]
wrapped_time=f['wrapped_time_{:04d}'.format(int(t*10))][:]
S=f['S_{:04d}'.format(int(t*10))][:]
W=f['Weights_{:04d}'.format(int(t*10))][:]
I=cv2.imread(dir_img+'/{:04d}.tif'.format(int(t*10)),2)
I=np.float32(I)/2.**16.
Imed=I
#Imed = gaussian(I, sigma=3)
I=Imed/Imax
Nbin=I.shape[0]

# COmpute n order moment
# compare cmax from individual strips theory and cmax from imag
Lmid=(L[1:,:]+L[:-1,:])/2.
if periodicity=='periodic':
	ix=np.int32(np.mod(Lmid[:,0],1)*Nbin)
	iy=np.int32(np.mod(Lmid[:,1],1)*Nbin)
	Lin=np.where((ix>=0)&(iy>=0)&(ix<Nbin)&(iy<Nbin))
	IL=I[ix[Lin],iy[Lin]].flatten()
else:
	Lin=np.where((np.abs(Lmid[:,1])<1)&(np.abs(Lmid[:,0])<1))
	IL=I[np.int32((Lmid[Lin,0]+1)*Nbin/2),np.int32((Lmid[Lin,1]+1)*Nbin/2)].flatten()

# Find min stretching rates 
argsort_id=np.argsort(1./S)
n=10000
#plt.plot((Lmid[argsort_id[:n],1]+1)*Nbin/2,(Lmid[argsort_id[:n],0]+1)*Nbin/2,'ro',alpha=1.0)

dt=0.25
D=(Brownian)**2./(2*dt)#+1e-08 # numerical diffusion ?
s0=radius#*np.sqrt(2)
#Pe=1e7
Tau=D/s0**2*wrapped_time[Lin]
Cmax=1./np.sqrt(1.+4*D/s0**2*wrapped_time[Lin])
Sc=s0*np.sqrt(1.+4*D/s0**2*wrapped_time[Lin])*S
Cmax2=1./np.sqrt(4*D/s0**2*wrapped_time[Lin])
Rho=1./S[Lin]
#plt.plot(IL,Rho,'.',alpha=0.1)
xc=np.logspace(-5,0,50)
#rhoc=bin_operation(IL,Rho,xc,np.mean)
#xc=np.linspace(0,1,100)
#	Cm=bin_operation(IL,Cmax,xc,np.nanmean)
#	Cmstd=bin_operation(IL,Cmax,xc,np.nanstd)
#	CmN=bin_operation(IL,Cmax,xc,len)
Cm=bin_operation(Cmax,IL,xc,np.nanmean)
Cmstd=bin_operation(Cmax,IL,xc,np.nanstd)
CmN=bin_operation(Cmax,IL,xc,len)

logCm=bin_operation(Cmax,np.log(IL),xc,np.nanmean)
logCmstd=bin_operation(Cmax,np.log(IL),xc,np.nanstd)
	#plt.plot(xc[1:],rhoc,'k-')
	#plt.figure()
# Choose a bundle size, typically of the order Batchelor scale
sB=np.sqrt(D/0.5) # theoretical
sB=np.mean(Sc)/2. # disk of diameter sB Lagrangian 
#sB=1.96*np.mean(S) # Lagrangian 

h,sh=np.histogram(Sc,np.logspace(-4,1,100),density=True)
plt.plot(sh[1:],h,'k*',label=r'$p(s)$')
plt.plot(sh[1:],1e-9*sh[1:]**(-4),'k--',label='$s^{-4}$')
plt.yscale('log')
plt.xscale('log')
plt.xlabel(r'$s=s_0\sqrt{1+4\tau}/\rho$')
plt.legend()
plt.savefig(dir_out+'p(s).pdf',bbox_inches='tight')
#
##% Plots Compare C C/cmax
#Cm_all=np.array(Cm_all)
#Cstd_all=np.array(Cstd_all)
#logCm_all=np.array(logCm_all)
#logCstd_all=np.array(logCstd_all)
#
#
#plt.figure(figsize=(2,2))
#[plt.plot(xc[1:],Cm_all[t,:],color=plt.cm.viridis(t/Cm_all.shape[0])) for t in range(Cm_all.shape[0])]
##[plt.errorbar(xc[1:],Cm_all[t,:],yerr=Cstd_all[t,:],color=plt.cm.viridis(t/5.),alpha=0.5) for t in range(Cm_all.shape[0])]
#
#plt.yscale('log')
#plt.xscale('log')
#plt.xlabel('$c_\mathrm{max}$ (LS)')
#plt.ylabel('$<c | c_\mathrm{max}>$ (DNS)')
#cm=np.mean(I)
##plt.plot(xc,np.sqrt(1+cm**2.*(1/xc**2.-1))*xc,'k--',label='$<c>+c_{max}(1-<c>)$')
#plt.plot(xc,(1+cm*(1/xc-1))*xc,'k-',label='$c_\mathrm{max}(1+<c>(1/c_\mathrm{max}-1/c_0)c_\mathrm{max}$')
#plt.plot(xc,xc,'k--',label='$c_\mathrm{max}$')
##plt.legend(fontsize=6),
#plt.ylim([0.5*np.mean(I),1.1])
#
#plt.savefig(dir_out+'mean_l{:1.1f}_rad{:1.2f}_{:1.0e}.pdf'.format(l0,radius,Brownian),
#						bbox_inches='tight')
#
#plt.figure(figsize=(2,2))
#[plt.plot(xc[1:],Cstd_all[t,:]/Cm_all[t,:],color=plt.cm.viridis(t/Cm_all.shape[0])) for t in range(Cm_all.shape[0])]
#
##[plt.plot(xc[1:],(Cstd_all[t,:]/xc[1:])**2,color=plt.cm.viridis(t/Cm_all.shape[0])) for t in range(Cm_all.shape[0])]
#
##[plt.plot(xc[1:],(Cstd_all[t,:]/xc[1:])**2*xc[1:],color=plt.cm.viridis(t/Cm_all.shape[0])) for t in range(Cm_all.shape[0])]
##plt.plot(xc[1:],xc[1:]**(-1),'k--')
##plt.hlines(y=0,xmin=1e-6,xmax=1e-1)
##plt.vlines(x=cm,ymin=1e-6,ymax=1e5)
#plt.yscale('log')
#plt.xscale('log')
#plt.xlabel('$c_\mathrm{max}$ (LS)')
#plt.ylabel('STD$[c|c_\mathrm{max}]/c$')
#
#plt.savefig(dir_out+'std_l{:1.1f}_rad{:1.2f}_{:1.0e}.pdf'.format(l0,radius,Brownian),
#						bbox_inches='tight')
#
#plt.figure(figsize=(2,2))
#[plt.plot(xc[1:],logCstd_all[t,:],color=plt.cm.viridis(t/Cm_all.shape[0])) for t in range(Cm_all.shape[0])]
##plt.yscale('log')
#plt.xscale('log')
#plt.xlabel('$c_\mathrm{max}$ (LS)')
#plt.ylabel('STD$[\log c|c_\mathrm{max}]$')
#
#plt.savefig(dir_out+'stdlogc_l{:1.1f}_rad{:1.2f}_{:1.0e}.pdf'.format(l0,radius,Brownian),
#						bbox_inches='tight')
#
## Comparaison log c / c in case of lognormal
#plt.figure(figsize=(2,2))
#[plt.plot(xc[1:],np.sqrt(np.log(1+(Cstd_all[t,:]/Cm_all[t,:])**2.)),'--',color=plt.cm.viridis(t/Cm_all.shape[0])) for t in range(Cm_all.shape[0])]
#[plt.plot(xc[1:],logCstd_all[t,:],color=plt.cm.viridis(t/Cm_all.shape[0])) for t in range(Cm_all.shape[0])]
#plt.yscale('log')
#plt.xscale('log')
#plt.xlabel('$c_\mathrm{max}$ (LS)')
#plt.ylabel(r'$\sigma_{\log c}$ (--),$\sqrt{\log(\sigma_c/c+1)}$ (- -)')
#plt.ylim([1e-1,1e0])
#
#
## Comparaison log c / c in case of gamma
#plt.figure(figsize=(2,2))
#[plt.plot(xc[1:],np.sqrt(np.log(1+(Cstd_all[t,:]/Cm_all[t,:])**2.)),'--',color=plt.cm.viridis(t/Cm_all.shape[0])) for t in range(Cm_all.shape[0])]
#[plt.plot(xc[1:],logCstd_all[t,:],color=plt.cm.viridis(t/Cm_all.shape[0])) for t in range(Cm_all.shape[0])]
#plt.yscale('log')
#plt.xscale('log')
#plt.xlabel('$c_\mathrm{max}$ (LS)')
#plt.ylabel(r'$\sigma_{\log c}$ (--),$\sqrt{\log(\sigma_c/c+1)}$ (- -)')
#plt.ylim([1e-1,1e0])


#% Compare pdf of cmax
plt.figure(figsize=(2,2))
cmaxb=np.logspace(-5,0,50)
pcmax,cmaxb=np.histogram(Cmax,cmaxb,density=True)
cmaxb=np.logspace(-5,0,50)
pc,cb=np.histogram(IL.flatten(),cmaxb,density=True)
pcI,cbI=np.histogram(I.flatten(),cmaxb,density=True)
#pci,cbi=np.histogram(np.log(I[I>1e-5]),50,density=True)
# Rescaled PDF
#plt.plot()
plt.plot((cmaxb[1:]/(1-np.mean(I))+np.mean(I)),pcmax/((1-np.mean(I))),'b.-',label='Cmax LS modif')
plt.plot(cmaxb[1:],pcmax,'g.-',label='$p(c_\mathrm{max})$ (LS)')
plt.plot(cb[1:],pc,'r.-',label='$p(c|c_\mathrm{max})$')
#plt.plot(cbi[1:],pci,'g--',label='Cmax direct')
plt.yscale('log')
plt.xscale('log')
plt.legend(fontsize=6)
plt.savefig(dir_out+'pdf_l{:1.1f}_rad{:1.2f}_{:1.0e}.pdf'.format(l0,radius,Brownian),
						bbox_inches='tight')


c=np.logspace(-4,0,100)
pc2=[p_c(cc,pcmax,cmaxb[1:],np.mean(I),1) for cc in c]
plt.figure(figsize=(2,2))
plt.plot(c,pc2,label='$\int p(c|c_\mathrm{max}) p(c_\mathrm{max}) \mathrm{d}c_\mathrm{max}$')
plt.plot(cmaxb[1:],pcmax,'g.-',label='$p(c_\mathrm{max})$ (LS)')
plt.plot(cb[1:],pc,'r.-',label='$p(c_\mathrm{max})$ (DNS)')
plt.plot(cbI[1:],pcI,label='$p(c) (DNS)$')
plt.yscale('log')
plt.xscale('log')
plt.ylim([1e-5,1e3])
plt.legend(fontsize=6)

# Cuttoff
#Cm_max=[]
#for t in range(Cm_all.shape[0]):
#	idgood=np.where(np.isfinite(Cm_all[t,:]))[0]
#	Cm_max.append(np.nanmean(Cm_all[t,idgood[-4:]]))
#Cm_max=np.array(Cm_max)
##Cm_max[0]=1.
#Tv=np.arange(Cm_max.shape[0])
#np.savetxt(dir_out+'/c_cutoff.txt',np.vstack((Tv,Cm_max)).T)

#plt.plot(Tv,np.exp(-Tv*0.25))
#%

plt.figure(figsize=(1.5,1.5))
plt.imshow(np.log(I+I[I>0].min()),extent=[0,1,0,1],cmap=cm_fire)
plot_per(L)
# Lmod=np.hstack((np.mod(L[:,1],1).reshape(-1,1),np.mod(-L[:,0],1).reshape(-1,1)))
# idmod=np.where(np.nansum(np.diff(Lmod,axis=0)**2.,axis=1)>0.01)[0]
# Lmod[idmod,:]=np.nan
# #plot_per(L)
# plt.plot(np.mod(L[:,1],1),np.mod(-L[:,0],1),'k.',alpha=1.0,linewidth=0.2,markersize=0.1)
# # plt.plot(Lmod[:,0],Lmod[:,1],'k-',alpha=0.8,linewidth=0.1,markersize=0.1)
# plt.plot(np.mod(L[:,0],1),np.mod(L[:,1],1),'k.',alpha=0.1,linewidth=0.1,markersize=0.1)
plt.xlim([0.1,0.25])
plt.ylim([0.1,0.25])
plt.axis('off')
plt.savefig(dir_out+'concentrations_A{:1.1f}_l{:1.1f}_rad{:1.2f}_{:1.0e}.pdf'.format(A,l0,radius,Brownian),
						bbox_inches='tight')
#plt.figure(figsize=(2,2))
#%
#t=10
H2m=np.histogram2d(Cmax,IL,xc)
X,Y=np.meshgrid(H2m[1][1:],H2m[1][1:])
fig,ax=plt.subplots(1,1,figsize=(2,2))
#H2=H2m_all[t]
H2=H2m[0]
ax.contourf(np.log10(X),np.log10(Y),np.log(H2))
plt.xlabel('log $c$ (DNS)')
plt.ylabel('log $c_\mathrm{max}$ (LS)')
cm=np.mean(I)
plt.plot(np.log10((1+cm*(1/xc-1))*xc),np.log10(xc),'k--',label=r'$(1+\langle c \rangle(1/c_\mathrm{max}-1/c_0)c_\mathrm{max}$')
plt.plot(np.log10(xc),np.log10(xc),'k:',label=r'1:1')
plt.legend(fontsize=6)
plt.savefig(dir_out+'2DHist_l{:1.1f}_rad{:1.2f}_{:1.0e}.pdf'.format(l0,radius,Brownian),
						bbox_inches='tight')

plt.figure(figsize=(1.5,1.5))
n_all=[]
iCmax=np.arange(0,H2.shape[0],1)
for n in iCmax:
	prob=H2[n,:]/np.sum(H2[n,:]*np.diff(xc))
	imax=np.argmax(prob)
	plt.plot(X[0,:],prob,'.',color=plt.cm.jet(n/50.))
	pfit=np.where(np.isfinite(np.log10(prob[:imax])))[0]
	if len(pfit)>0:
		p=np.polyfit(np.log10(X[0,:imax][pfit]),np.log10(prob[:imax][pfit]),1)
		n_all.append(p[0])
		#plt.plot(X[0,:],10**p[1]*X[0,:]**p[0],'--',color=plt.cm.jet(n/50.),alpha=0.5)
	else:
		n_all.append(np.nan)
	
plt.yscale('log')
plt.xscale('log')
plt.title('$t={:1.1f}$'.format(t))
plt.xlabel('$c$')
plt.ylabel('$p(c|c_\mathrm{max})$')
plt.xlim([1e-4,1e-0])
plt.ylim([1e-4,2e2])

# p(c|cmax)
plt.figure(figsize=(1.5,1.5))
n_all=[]
iCmax=np.arange(0,H2.shape[0],1)
for n in iCmax:
	prob=H2[n,:]/np.sum(H2[n,:]*np.diff(xc))
	imax=np.argmax(prob)
	plt.plot(X[0,:]/H2m[1][n],prob*H2m[1][n],'-',color=plt.cm.jet(n/50.))
	pfit=np.where(np.isfinite(np.log10(prob[:imax])))[0]
	if len(pfit)>0:
		p=np.polyfit(np.log10(X[0,:imax][pfit]),np.log10(prob[:imax][pfit]),1)
		n_all.append(p[0])
		#plt.plot(X[0,:],10**p[1]*X[0,:]**p[0],'--',color=plt.cm.jet(n/50.),alpha=0.5)
	else:
		n_all.append(np.nan)
plt.yscale('log')
plt.xscale('log')
plt.title('$t={:1.1f}$'.format(t))
plt.xlabel('$c/c_\mathrm{max}$')
plt.ylabel('$p(c/c_\mathrm{max}|c_\mathrm{max})$')
plt.xlim([1e-1,1e3])
plt.ylim([1e-5,2e1])

plt.figure(figsize=(1.5,1.5))
plt.plot(Y[iCmax,0],n_all,'o-')
plt.xscale('log')
plt.xlabel('$c_\mathrm{max}$')
plt.ylabel('$n$')

plt.figure(figsize=(1.5,1.5))
n=30 
[plt.plot((X[0,:]),(H2m_all[t][n,:])/np.sum(H2m_all[t][n,:]*np.diff(xc)),'.-',color=plt.cm.jet(t/12.)) for t in range(len(H2m_all))]
plt.yscale('log')
plt.xscale('log')
plt.xlabel('$c$')
plt.ylabel('$p(c|c_\mathrm{max}'+'={:1.1e})$'.format(Y[n,0]))
plt.xlim([1e-3,2e-1])
plt.ylim([1e-1,2e2])

#%%% Neighboors finding
#sB=0.0005
#dx=0.001 # Maximum distance between points, above which the line will be refined
dx=0.005

tree=spatial.cKDTree(np.mod(L,1))
#nsamples=int(sB/dx*10)

# Take samples in within cmax
nsamples=10000
idsamples=np.uint32(np.linspace(0,L.shape[0]-10,nsamples))

# Take idsample in a grid to respect an equi porportion in the space
# ng=100
# idshuffle=np.arange(len(L)-2)
# np.random.shuffle(idshuffle)
# Lmod=np.mod(L,1)
# Lmodint=np.uint32(Lmod*ng)
# idu=np.unique(np.copy(Lmodint[idshuffle,:]), return_index=True,axis=0)
# idsamples=idshuffle[idu[1]]

neighboors=tree.query_ball_point(np.mod(L[idsamples,:],1), sB/2) # ball radius = sb/2 !! 
neighboors_uniq=[[] for k in range(len(idsamples))]
neighboors_all=[[] for k in range(len(idsamples))]
dist_old=np.sqrt(np.sum(np.diff(L,axis=0)**2.,axis=1))
dist_all=np.hstack((0,np.cumsum(dist_old)))
for ii,ns in enumerate(idsamples):		
	nt=np.sort(neighboors[ii])
	# First possibility
	kk_all=np.concatenate((nt,np.array([ns])))
	neighboors_all[ii].append(kk_all)
	idgood=np.where(np.diff(nt)>3*sB/dx)[0]
	NT=[]
	for idg in idgood:
		if (np.abs(nt[idg+1]-ns)>3*sB/dx)&(np.abs(nt[idg]-ns)<3*sB/dx): # depending on the direction of the filament
			NT.append(nt[idg+1])
		else:
			NT.append(nt[idg])
	nt=np.array(NT,dtype=np.uint32)
	kk=np.concatenate((nt,np.array([ns])))
	neighboors_uniq[ii].append(kk)
	# Other possibility (do not work properly !!!)
#	idgood=np.where(np.diff(dist_all[nt])>5*sB)[0]
#	kk_all=np.concatenate((nt[idgood],np.array([ns])))
#	# remove extra lamella
#	kk_all=np.delete(kk_all,np.where(((dist_all[kk_all[:-1]]-dist_all[kk_all[-1]])<5*sB))[0])
#	neighboors_uniq[ii].append(kk_all)
#	
nagg=np.array([len(n[0]) for n in neighboors_uniq])

def bundle(variable,operator,binarize,nbin):
	# Function that compute statistics on bundles and binarize results
	v=np.array([operator(variable[n[0]]) for n in neighboors_uniq])
	if binarize=='N':
		nagg=np.array([len(n[0]) for n in neighboors_uniq])
		nagg_bin=np.logspace(0,np.max(np.log10(nagg)),nbin)
		return bin_operation(nagg,v,nagg_bin,np.nanmean),nagg_bin
	if binarize=='Rho':
		rhoagg_mean=np.array([np.nanmean((1/S[n[0]])) for n in neighboors_uniq])
		rho_bin=np.logspace(np.log10(rhoagg_mean.min()),np.log10(rhoagg_mean.max()),nbin)
		return bin_operation(rhoagg_mean,v,rho_bin,np.nanmean),rho_bin
	if binarize=='Cmax':
		c_mean=np.array([np.nanmean((Cmax[n[0]])) for n in neighboors_uniq])
		c_bin=np.logspace(np.log10(c_mean.min()),np.log10(c_mean.max()),nbin)
		return bin_operation(c_mean,v,c_bin,np.nanmean),c_bin


# Check if all aggregates are found
#plt.figure()
#plt.imshow(I,extent=[0,1,0,1])
#Lmod2=np.hstack((np.mod(L[:,1],1).reshape(-1,1),np.mod(-L[:,0],1).reshape(-1,1)))
#idmod=np.where(np.nansum(np.diff(Lmod2,axis=0)**2.,axis=1)>0.01)[0]
#Lmod2[idmod,:]=np.nan
##plt.plot(np.mod(L[:,1],1),np.mod(-L[:,0],1),'k.',alpha=1.0,linewidth=0.2,markersize=0.1)
#plt.plot(Lmod2[:,0],Lmod2[:,1],'w-',alpha=0.8,linewidth=0.1,markersize=0.2)
#for ne in neighboors_uniq[::40]:
#	plt.plot(Lmod[ne[0],0],Lmod[ne[0],1],'+')

# Cmax
cmaxagg=np.array([np.nansum(Cmax[n[0]]) for n in neighboors_uniq])
cmaxagg_mean=np.array([np.nanmean(Cmax[n[0]]) for n in neighboors_uniq])
cmaxagg_std=np.array([np.nanstd(Cmax[n[0]]) for n in neighboors_uniq])
cmaxagg_var=np.array([np.nanvar(Cmax[n[0]]) for n in neighboors_uniq])
cmaxagg2_var=np.array([np.nanvar(Cmax2[n[0]]) for n in neighboors_uniq])
cmaxagg2_mean=np.array([np.nanmean(Cmax2[n[0]]) for n in neighboors_uniq])
cmaxagg_std3=np.array([np.nanstd(Cmax2[n[0]]) for n in neighboors_uniq])
logcmaxagg_mean=np.array([np.nanmean(np.log(Cmax[n[0]])) for n in neighboors_uniq])
logcmaxagg_std=np.array([np.nanstd(np.log(Cmax[n[0]])) for n in neighboors_uniq])
logcmaxagg_var=np.array([np.nanvar(np.log(Cmax[n[0]])) for n in neighboors_uniq])

# Tau
tauagg_var=np.array([np.nanvar(Tau[n[0]]) for n in neighboors_uniq])
tauagg_std=np.array([np.nanstd(Tau[n[0]]) for n in neighboors_uniq])
tauagg_mean=np.array([np.nanmean(Tau[n[0]]) for n in neighboors_uniq])
sqrttauagg_var=np.array([np.nanvar(np.sqrt(Tau[n[0]])) for n in neighboors_uniq])
sqrttauagg_std=np.array([np.nanstd(np.sqrt(Tau[n[0]])) for n in neighboors_uniq])
sqrttauagg_mean=np.array([np.nanmean(np.sqrt(Tau[n[0]])) for n in neighboors_uniq])


pdf_rho=np.logspace(np.log10(1/S.max()),np.log10(1/S.min()),40)
#logRHo
logrhoagg_mean=np.array([np.nanmean(np.log(1/S[n[0]])) for n in neighboors_uniq])
logSagg_mean=np.array([np.nanmean(np.log(S[n[0]])) for n in neighboors_uniq])
logrhoagg_var=np.array([np.nanvar(np.log(1/S[n[0]])) for n in neighboors_uniq])
logrhoagg_std=np.array([np.nanstd(np.log(1/S[n[0]])) for n in neighboors_uniq])
logrho_pdf=np.array([np.histogram(np.log(1/S[n[0]]),np.log(pdf_rho),density=True)[0] for n in neighboors_uniq])

#rho

rhoagg_mean=np.array([np.nanmean((1/S[n[0]])) for n in neighboors_uniq])
Sagg_mean=np.array([np.nanmean((S[n[0]])) for n in neighboors_uniq])
rhoagg_var=np.array([np.nanvar((1/S[n[0]])) for n in neighboors_uniq])
rhoagg_std=np.array([np.nanstd((1/S[n[0]])) for n in neighboors_uniq])

pdf_c=np.logspace(-8,0,50)
cmax_pdf=np.array([np.histogram(Cmax[n[0]],pdf_c,density=True)[0] for n in neighboors_uniq])
cmax_pdf_all=np.histogram(Cmax,pdf_c,density=True)[0]

pdf_S=np.logspace(np.log10(S.min()),np.log10(S.max()),40)
rho_pdf=np.array([np.histogram(1/S[n[0]],pdf_rho,density=True)[0] for n in neighboors_uniq])
S_pdf=np.array([np.histogram(S[n[0]],pdf_S,density=True)[0] for n in neighboors_uniq])
#

#cmax2agg=np.array([np.sum((Cmax[neighboors_uniq[n]]-cmaxagg[n])**2.) for n in range(len(neighboors_uniq))])

# Binning as a function of N
nagg_bin=np.logspace(0,np.max(np.log10(nagg)),30)

logrho_bin=bin_operation(nagg,np.log(1./S[idsamples]),nagg_bin,np.nanmean)
logrho_bin_std=bin_operation(nagg,np.log(1./S[idsamples]),nagg_bin,np.nanstd)
rho_bin=bin_operation(nagg,(1./S[idsamples]),nagg_bin,np.nanmean)
rho_bin_std=bin_operation(nagg,(1./S[idsamples]),nagg_bin,np.nanstd)
logrhomean_naggbin=bin_operation(np.log(nagg),logrhoagg_mean,np.log(nagg_bin),np.nanmean)
logSmean_naggbin=bin_operation(np.log(nagg),logSagg_mean,np.log(nagg_bin),np.nanmean)
logrhovar_naggbin=bin_operation(np.log(nagg),logrhoagg_var,np.log(nagg_bin),np.nanmean)
logrhostd_naggbin=bin_operation(np.log(nagg),logrhoagg_std,np.log(nagg_bin),np.nanmean)
rhomean_naggbin=bin_operation(nagg,rhoagg_mean,nagg_bin,np.nanmean)
Smean_naggbin=bin_operation(nagg,Sagg_mean,nagg_bin,np.nanmean)
rhovar_naggbin=bin_operation(nagg,rhoagg_var,nagg_bin,np.nanmean)
rhostd_naggbin=bin_operation(nagg,rhoagg_std,nagg_bin,np.nanmean)

Tauvar_bin=bin_operation(nagg,tauagg_var,nagg_bin,np.nanmean)
Taustd_bin=bin_operation(nagg,tauagg_std,nagg_bin,np.nanmean)
Taumean_bin=bin_operation(nagg,tauagg_mean,nagg_bin,np.nanmean)
sqrtTauvar_bin=bin_operation(nagg,sqrttauagg_var,nagg_bin,np.nanmean)
sqrtTaustd_bin=bin_operation(nagg,sqrttauagg_std,nagg_bin,np.nanmean)
sqrtTaumean_bin=bin_operation(nagg,sqrttauagg_mean,nagg_bin,np.nanmean)

cmax_naggbin=bin_operation(nagg,cmaxagg,nagg_bin,np.nanmean)
cmax_naggbin_std=bin_operation(nagg,cmaxagg,nagg_bin,np.nanstd)
logcmax_naggbin=bin_operation(nagg,np.log(cmaxagg),nagg_bin,np.nanmean)
logcmax_naggbin_var=bin_operation(nagg,np.log(cmaxagg),nagg_bin,np.nanvar)

cmaxmean_naggbin=bin_operation(nagg,cmaxagg_mean,nagg_bin,np.nanmean)
cmaxstd_naggbin=bin_operation(nagg,cmaxagg_std,nagg_bin,np.nanmean)
cmaxvar_naggbin=bin_operation(nagg,cmaxagg_var,nagg_bin,np.nanmean)

cmaxmean2_naggbin=bin_operation(nagg,cmaxagg2_mean,nagg_bin,np.nanmean)
cmaxvar2_naggbin=bin_operation(nagg,cmaxagg2_var,nagg_bin,np.nanmean)

logcmaxmean_naggbin=bin_operation(nagg,logcmaxagg_mean,nagg_bin,np.nanmean)
logcmaxstd_naggbin=bin_operation(nagg,logcmaxagg_std,nagg_bin,np.nanmean)
logcmaxvar_naggbin=bin_operation(nagg,logcmaxagg_var,nagg_bin,np.nanmean)

cmaxstd_bin=bin_operation(nagg,cmaxagg_std/cmaxagg_mean,nagg_bin,np.nanmean)
cmaxstd_bin2=bin_operation(nagg,cmaxagg_std,nagg_bin,np.nanmean)
cmaxstd_bin3=bin_operation(nagg,cmaxagg_std3,nagg_bin,np.nanmean)
cmax_pdf_bin=bin_operation2(nagg,cmax_pdf,nagg_bin,np.nanmean)
#cmax2_bin=bin_operation(nagg,cmax2agg,nagg_bin,np.nanmean)

logrho_pdf_naggbin=bin_operation2(nagg,logrho_pdf,nagg_bin,np.nanmean)

S_pdf_naggbin=bin_operation2(nagg,S_pdf,nagg_bin,np.nanmean)

# Binning as a function of rho
rho_bin=np.logspace(0,np.max(np.log10(1./S[idsamples])),20)
rho=rhoagg_mean
inv_rho=Sagg_mean

lognagg_rhobin=bin_operation(rho,np.log(nagg),rho_bin,np.nanmean)
logrho_bin=bin_operation(rho,np.log(1./S[idsamples]),rho_bin,np.nanmean)
logrho_bin_std=bin_operation(rho,np.log(1./S[idsamples]),rho_bin,np.nanstd)
rho_bin=bin_operation(rho,(1./S[idsamples]),rho_bin,np.nanmean)
S_bin=bin_operation(rho,(S[idsamples]),rho_bin,np.nanmean)
rho_bin_std=bin_operation(rho,(1./S[idsamples]),rho_bin,np.nanstd)
logrhomean_rhobin=bin_operation(np.log(rho),logrhoagg_mean,np.log(rho_bin),np.nanmean)
logSmean_rhobin=bin_operation(np.log(rho),logSagg_mean,np.log(rho_bin),np.nanmean)
logrhovar_rhobin=bin_operation(np.log(rho),logrhoagg_var,np.log(rho_bin),np.nanmean)
logrhostd_rhobin=bin_operation(np.log(rho),logrhoagg_std,np.log(rho_bin),np.nanmean)
rhomean_rhobin=bin_operation(rho,rhoagg_mean,rho_bin,np.nanmean)
rhovar_rhobin=bin_operation(rho,rhoagg_var,rho_bin,np.nanmean)
rhostd_rhobin=bin_operation(rho,rhoagg_std,rho_bin,np.nanmean)

Tauvar_rhobin=bin_operation(rho,tauagg_var,rho_bin,np.nanmean)
Taustd_rhobin=bin_operation(rho,tauagg_std,rho_bin,np.nanmean)
Taumean_rhobin=bin_operation(rho,tauagg_mean,rho_bin,np.nanmean)
sqrtTauvar_rhobin=bin_operation(rho,sqrttauagg_var,rho_bin,np.nanmean)
sqrtTaustd_rhobin=bin_operation(rho,sqrttauagg_std,rho_bin,np.nanmean)
sqrtTaumean_rhobin=bin_operation(rho,sqrttauagg_mean,rho_bin,np.nanmean)

cmax_rhobin=bin_operation(rho,cmaxagg,rho_bin,np.nanmean)
cmax_rhobin_std=bin_operation(rho,cmaxagg,rho_bin,np.nanstd)
logcmax_rhobin=bin_operation(rho,np.log(cmaxagg),rho_bin,np.nanmean)
logcmax_rhobin_var=bin_operation(rho,np.log(cmaxagg),rho_bin,np.nanvar)

cmaxmean_rhobin=bin_operation(rho,cmaxagg_mean,rho_bin,np.nanmean)
cmaxstd_rhobin=bin_operation(rho,cmaxagg_std,rho_bin,np.nanmean)
cmaxvar_rhobin=bin_operation(rho,cmaxagg_var,rho_bin,np.nanmean)
logcmaxmean_rhobin=bin_operation(rho,logcmaxagg_mean,rho_bin,np.nanmean)
logcmaxstd_rhobin=bin_operation(rho,logcmaxagg_std,rho_bin,np.nanmean)
logcmaxvar_rhobin=bin_operation(rho,logcmaxagg_var,rho_bin,np.nanmean)

#cmaxstd_rhobin=bin_operation(rho,cmaxagg_std/cmaxagg_mean,rho_bin,np.nanmean)
cmaxstd_rhobin2=bin_operation(rho,cmaxagg_std,rho_bin,np.nanmean)
cmaxstd_rhobin3=bin_operation(rho,cmaxagg_std3,rho_bin,np.nanmean)
cmax_pdf_rhobin=bin_operation2(rho,cmax_pdf,rho_bin,np.nanmean)
rho_pdf_rhobin=bin_operation2(rho,rho_pdf,rho_bin,np.nanmean)
S_pdf_rhobin=bin_operation2(rho,S_pdf,rho_bin,np.nanmean)
#cmax2_bin=bin_operation(nagg,cmax2agg,nagg_bin,np.nanmean)

# Binning as a function of Cmax reference
cm_bin=np.logspace(-10,0,40)
cmax_bin=bin_operation(Cmax[idsamples],cmaxagg,cm_bin,np.nanmean)
cmax_bin_std=bin_operation(Cmax[idsamples],cmaxagg,cm_bin,np.nanstd)
#cmax_bin=bin_operation(cmaxagg_mean,cmaxagg,cm_bin,np.nanmean)
#cmax_bin_std=bin_operation(cmaxagg_mean,cmaxagg,cm_bin,np.nanstd)

naggmean_cmaxbin=bin_operation(cmaxagg_mean,nagg,cm_bin,np.nanmean)
lognaggmean_cmaxbin=bin_operation(cmaxagg_mean,np.log(nagg),cm_bin,np.nanmean)

# All lamella
cmax_std=np.nanstd(Cmax)
# Weighted
meanlogrho=np.average(np.log(1/S),weights=W)
varlogrho=np.average((np.log(1/S)-meanlogrho)**2.,weights=W)
meanrho=np.average(1/S,weights=W)
varrho=np.average((1/S-meanlogrho)**2.,weights=W)
# Non Weighted
meanlogrho=np.average(np.log(1/S))
varlogrho=np.average((np.log(1/S)-meanlogrho)**2.)
meanrho=np.average(1/S)
varrho=np.average((1/S-meanlogrho)**2.)


# Plot p(N|cmax)
# Plot p(N|c)

plt.figure(figsize=(1.5,1.5))
plt.yscale('log')
plt.xlabel('$N$')
plt.ylabel(r'$p(N|\rho)$')
rho_b=np.logspace(np.min(np.log10(rho)),np.max(np.log10(rho)),10)
for i in range(1,len(rho_b)):
	idN=np.where((rho<rho_b[i])&(rho>rho_b[i-1]))
	Nh,Nx=np.histogram(nagg[idN],50,density=True)
	plt.plot(Nx[1:],Nh,'.',color=plt.cm.jet(i/len(rho_b)))
	
Nh,Nx=np.histogram(nagg,50,density=True)
plt.plot(Nx[1:],Nh,'k-',label='$p(N)$')
plt.legend()

dist_old=np.sum(np.diff(L,axis=0)**2,1)**0.5
# p(rho)/<rho>) in a bundle
plt.figure(figsize=(1.5,1.5))
plt.yscale('log')
plt.xscale('log')
plt.xlabel(r'$\rho/\langle \rho \rangle $')
plt.ylabel(r'$p_{s_B}(\rho/\langle \rho \rangle)$')
for i in range(len(rho_bin)-1):
#	plt.plot(pdf_rho[1:]/rho_bin[i],rho_pdf_rhobin[i,:]*rho_bin[i],'.',color=plt.cm.jet(i/len(rho_bin)))
	plt.plot(pdf_rho[1:],rho_pdf_rhobin[i,:],'.',color=plt.cm.jet(i/len(rho_bin)))
plt.xlim([0.1,1e10])
#plt.ylim([1e-3,1e1])
h,x=np.histogram(1/S,pdf_rho,density=True,weights=dist_old*W)
plt.plot(x[1:],h,'k*')
logrho=np.linspace(-2,20,100)
#h,x=np.histogram(1/S,pdf_rho,density=True,weights=dist_old*W)
#plt.plot(x[1:],h,'k*')
#plt.yscale('log')
#plt.xscale('log')
#h,x=np.histogram(np.log(1/S),logrho,density=True,weights=dist_old*W)
#plt.plot(x[1:],h,'ko')
#plt.yscale('log')
#h,x=np.histogram(np.log(1/S),logrho,density=True,weights=dist_old)
#plt.plot(x[1:],h,'k*')
#h,x=np.histogram(1/S,np.exp(logrho),density=True,weights=dist_old)
##plt.plot(np.log(x[1:]),h*x[1:],'r*')
#plt.plot((x[1:]),h,'r*')
plt.yscale('log')
plt.xscale('log')
#plt.plot(x[1:],h,'k--')
x=np.logspace(-3,3,100)
x=np.logspace(0,7,100)
mu=0
sig=mu/4
#plt.plot(x,1/x/np.sqrt(2*np.pi*sig)*np.exp(-(np.log(x)-(mu))**2/(2*sig)),'k-')

# p(rho) in a bundle
plt.figure(figsize=(1.5,1.5))
plt.yscale('log')
plt.xscale('log')
plt.xlabel(r'$\rho $')
plt.ylabel(r'$p_{s_B}(\rho)$')
x=np.logspace(0,7,100)
for i in range(len(rho_bin)-1):
#	plt.plot(pdf_rho[1:]/rho_bin[i],rho_pdf_rhobin[i,:]*rho_bin[i],'.',color=plt.cm.jet(i/len(rho_bin)))
	plt.plot(pdf_rho[1:],rho_pdf_rhobin[i,:],'.',color=plt.cm.jet(i/len(rho_bin)))
	mu=np.log(rho_bin[i])
	sig=mu/6
	plt.plot(x,1/x/np.sqrt(2*np.pi*sig)*np.exp(-(np.log(x)-(mu))**2/(2*sig)),'-',color=plt.cm.jet(i/len(rho_bin)))
plt.ylim([1e-18,1e-1])

# p(rho) in a bundle
plt.figure(figsize=(1.5,1.5))
plt.yscale('log')
plt.xscale('log')
plt.xlabel(r'$\langle \rho \rangle /\rho $')
plt.ylabel(r'$p_{s_B}(\langle \rho \rangle / \rho)$')
for i in range(len(rho_bin)-1):
	plt.plot(pdf_S[1:]/S_bin[i],S_pdf_rhobin[i,:]*S_bin[i],'.',color=plt.cm.jet(i/len(rho_bin)))
plt.xlim([1e-3,1e4])
plt.ylim([1e-8,1e1])
h,x=np.histogram(S,pdf_S,density=True,weights=dist_old)
plt.plot(x[1:]/np.nanmean(S),h*np.nanmean(S),'k--')
# LogNormal approx
x=np.logspace(-3,3,100)
sig=1
plt.plot(x,1/x/np.sqrt(2*np.pi*sig)*np.exp(-np.log(x)**2/(2*sig)),'k-')

#%%% Compare <cmax> to cmax

cmaxagg_mean=np.array([np.nanmean(Cmax[n[0]]) for n in neighboors_uniq])
bin_c=np.logspace(np.log10(cmaxagg_mean.min()),np.log10(cmaxagg_mean.max()),50)
bc=bin_operation(cmaxagg_mean,Cmax[idsamples],bin_c,np.mean)
plt.plot(cmaxagg_mean,Cmax[idsamples],'k.',alpha=0.1)
plt.plot(bin_c[1:],bc,'r.',alpha=1,
				 label=r'$\langle \theta | \langle \theta \rangle_{s_B} \rangle$')
plt.yscale('log')
plt.xscale('log')
plt.plot(c_bin[1:],c_bin[1:],'k--')
plt.xlabel(r'$\langle \theta \rangle_{s_B}$')
plt.ylabel(r'$ \theta $')
plt.legend()
#%%% Compare cmax to sum(cmax)
masse_tot=l0*s0*1*np.sqrt(2*np.pi)
cm_lagrangian=masse_tot/(1*1)
cmax_lag=l0*s0*1.0
#plt.plot(cm_bin[1:],cmax_bin-cm/np.sqrt(2*np.pi),'ko')
plt.plot(cm_bin[1:],cmax_bin-cmax_lag,'ko',label=r'$\sum_{s_{s_B}} \theta - \langle c \rangle/\sqrt{2\pi}$')
plt.plot(cm_bin[1:],cmax_bin,'rd',label=r'$\sum_{s_{s_B}} \theta $')

plt.plot(cm_bin[1:],cm_bin[1:],'k--',label='1:1')
plt.plot(cm_bin[1:],np.zeros(len(cm_bin)-1)+cmax_lag,'r-',label=r'$\langle c \rangle/\sqrt{2\pi}$')
plt.yscale('log')
plt.xscale('log')
plt.ylim([1e-4,1])
plt.xlim([1e-8,1])
plt.xlabel(r'$\theta$')
plt.ylabel(r'')
plt.legend()
plt.savefig(dir_out+'cmax_sumcmax.pdf')
#%%% p(log rho) in a bundle
nn=4
fig,ax=plt.subplots(1,2,figsize=(4,2))
ax[0].set_yscale('log')
#plt.xscale('log')
ax[0].set_xlabel(r'$\log \rho $')
ax[0].set_ylabel(r'$p_{s_B}(\log \rho)$')
for i in np.arange(0,len(nagg_bin)-1,nn):
	ax[0].plot(np.log(pdf_rho[1:]),logrho_pdf_naggbin[i,:],'o',color=plt.cm.jet(i/len(nagg_bin)))
	mu=logrhomean_naggbin[i]
	#sig=np.log(nagg_bin[i])*1.6+1
	sig=logrhovar_naggbin[i]
	x=np.linspace(np.log(pdf_rho[1:]).min()-2,np.log(pdf_rho[1:]).max()+2,100)
	ax[0].plot(x,1/np.sqrt(2*np.pi*sig)*np.exp(-(x-(mu))**2/(2*sig)),'--',color=plt.cm.jet(i/len(nagg_bin)))
#	plt.plot(x,1/np.sqrt(2*np.pi*sig)*np.exp(-(x-(mu))**2/(2*sig)),'-',color=plt.cm.jet(i/len(nagg_bin)))
ax[0].set_ylim([0.1*np.nanmin(logrho_pdf_naggbin[logrho_pdf_naggbin>0]),10*np.nanmax(logrho_pdf_naggbin)])
ax[0].set_xlim([np.log(pdf_rho[1:]).min()-2,np.log(pdf_rho[1:]).max()+2])
h,x=np.histogram(np.log(1/S),np.log(pdf_rho),density=True)
#ax[0].plot(x[1:],h,'k-')
#plt.xscale('log')

# 1/ ryho 
ax[1].set_yscale('log')
ax[1].set_xscale('log')
ax[1].set_xlabel(r'$1/\rho $')
ax[1].set_ylabel(r'$p_{s_B}(1/\rho)$')
for i in np.arange(0,len(nagg_bin)-1,nn):
	ax[1].plot(pdf_S[1:],S_pdf_naggbin[i,:],'o',color=plt.cm.jet(i/len(nagg_bin)))
	mu=logrhomean_naggbin[i]
	#sig=np.log(nagg_bin[i])*1.6+1
	sig=logrhovar_naggbin[i]
	x=np.logspace(np.log10(0.1*pdf_S[1:].min()),np.log10(10*pdf_S[1:].max()),100)
	ax[1].plot(x,1/x/np.sqrt(2*np.pi*sig)*np.exp(-(np.log(x)+(mu))**2/(2*sig)),'--',color=plt.cm.jet(i/len(nagg_bin)))

ax[1].set_ylim([0.1*np.nanmin(S_pdf_naggbin[S_pdf_naggbin>0]),10*np.nanmax(S_pdf_naggbin)])
ax[1].set_xlim([0.1*pdf_S[1:].min(),10*pdf_S[1:].max()])
ax[1].legend()
fig.subplots_adjust(wspace=0.3)
ax2 = fig.add_axes([0.75, 0.9, 0.2, 0.05])
import matplotlib as mpl
#norm = mpl.colors.Normalize(vmin=T_all[0]*u_pore/d, vmax=T_all[-1]*u_pore/d)
norm = mpl.colors.Normalize(vmin=0, vmax=np.log10(nagg_bin[-1]))
cb1 = mpl.colorbar.ColorbarBase(ax2, 
																cmap=plt.cm.jet,norm=norm,
																orientation='horizontal')
cb1.set_label(r'$\log_{10} N$', color='k',size=8,labelpad=-7)
plt.savefig(dir_out+'pB_logrho.pdf',bbox_inches='tight')
#tickpos=np.array([0.,0.5,1.])
#cb.ax.xaxis.set_ticks(tickpos)
#cb.ax.xaxis.set_ticklabels(['{:1.1f}','$15$','$30$'])
#%%% p(1/rho) in a bundle by rho
plt.figure(figsize=(1.5,1.5))
plt.yscale('log')
plt.xscale('log')
plt.xlabel(r'$1/\rho $')
plt.ylabel(r'$p_{s_B}(1/\rho)$')
for i in range(len(rho_bin)-1):
	plt.plot(pdf_S[1:],S_pdf_rhobin[i,:],'.',color=plt.cm.jet(i/len(rho_bin)))

h,x=np.histogram(S,pdf_S,density=True)
plt.plot(x[1:],h,'k--')
# LogNormal approx
x=np.logspace(-6,-3,100)
plt.plot(x,1e-8*x**(-2),'k-')
sig=1
plt.plot(x,1/x/np.sqrt(2*np.pi*sig)*np.exp(-np.log(x)**2/(2*sig)),'k-')



plt.figure(figsize=(1.5,1.5))
plt.yscale('log')
plt.xlabel(r'$\theta$')
plt.ylabel(r'$N$')
plt.plot(cm_bin[1:],naggmean_cmaxbin,'r+')
plt.plot(cm_bin[1:],np.exp(lognaggmean_cmaxbin),'ro')
plt.plot(cmaxmean_naggbin,nagg_bin[1:],'k+')
plt.plot(np.exp(logcmaxmean_naggbin),nagg_bin[1:],'ko')
plt.xscale('log')


#%%% * SCALINGS log N with of cmax, rho , 1/rho
keyword='sine'

if keyword=='half': fractal=1.50
if keyword=='sine': fractal=1.7
if keyword=='single': fractal=1.7

nb=40
cm_bin=np.logspace(-10,0,nb)

import matplotlib as mpl
fig,ax=plt.subplots(1,1,figsize=(2,2))
nbin=np.log(np.unique(np.uint32(np.logspace(0,3.5,nb)))-0.5)
cbin=np.log(np.logspace(np.log10(cmaxagg_mean.min()),0,nb))
ax.hist2d(np.log(cmaxagg_mean),np.log(nagg),[cbin,nbin],norm=mpl.colors.LogNorm(),cmap=plt.cm.Greys)
plt.ylabel('log $N$')
ax.patch.set_facecolor(plt.cm.Greys(0))
ax.set_xlabel(r'log $\langle \theta \rangle_{s_B}$')
plt.plot(np.log(cm_bin[1:]),np.log(1+cm*(1/cm_bin[1:]**1-1)),'r-',label=r'$1+\langle c \rangle  (\theta ^{-1} -1)$')
#plt.plot(np.log(cm_bin[1:]),np.log(1+cm*(1/cm_bin[1:]**0.67-1)),'w-',label=r'$1+\langle c \rangle  (\theta ^{-(D-1)} -1)$')
legend = plt.legend(frameon=False,fontsize=6)
plt.setp(legend.get_texts(), color='k')
plt.savefig(dir_out+'N-theta.pdf',bbox_inches='tight')


fig,ax=plt.subplots(1,1,figsize=(2,2))
nbin=np.log(np.unique(np.uint32(np.logspace(0,3.5,nb)))-0.5)
rhobin=np.log(np.logspace(0,np.log10(rho.max()),nb))
ax.hist2d(np.log(rho),np.log(nagg),[rhobin,nbin],norm=mpl.colors.LogNorm(),cmap=plt.cm.Greys)
plt.ylabel('log $N$')
# plt.ylim([0,6])
# plt.xlim([0,20])
plt.xlabel(r'log $\langle \rho \rangle_{s_B}$')
ax.patch.set_facecolor(plt.cm.Greys(0))
rx=np.linspace(0,25,3)
ax.plot(rx,rx*(fractal-1)-5.,'r--',label=r'$D_1-1$')
#plt.plot((rhobin),np.log(1+cm*(np.exp(rhobin)**(fractal-1)-1)),'w--',label=r'$1+\langle c \rangle (\rho^{D_f-1}-1)$')
legend = plt.legend(frameon=False,fontsize=6)
plt.setp(legend.get_texts(), color='k')
plt.savefig(dir_out+'N-rho2.pdf',bbox_inches='tight')

fractal=1.7
fig,ax=plt.subplots(1,2,figsize=(3,1.5),sharey=True)
nbin=np.log(np.unique(np.uint32(np.logspace(0,3.5,40)))-0.5)
rhobin=np.log(np.logspace(0,np.log10(rho.max()),50))
ax[1].hist2d(logrhoagg_mean,np.log(nagg),[rhobin,nbin],cmap=plt.cm.Greys)
Plogrho = linregress(logrhoagg_mean,np.log(nagg))
Plogrho = linregress(np.log(nagg),logrhoagg_mean)
#ax[1].hist2d(np.log(rho),np.log(nagg),[rhobin,nbin],cmap=plt.cm.jet,alpha=0.1)
#ax[1].ylabel('log $N$')
# plt.ylim([0,6])
# plt.xlim([0,20])
ax[1].set_xlabel(r'$\langle \log  \rho(\textbf{x}) \rangle_{s_B}$')
ax[1].patch.set_facecolor(plt.cm.Greys(0))
rx=np.linspace(7,15,3)
ax[1].plot(rx,rx-7,'r--',label=r'$1$')
ax[1].plot(rx,rx*(fractal-1)-6.5,'r-',label=r'$D_1-1$')
#plt.plot((rhobin),np.log(1+cm*(np.exp(rhobin)**(fractal-1)-1)),'w--',label=r'$1+\langle c \rangle (\rho^{D_f-1}-1)$')
legend = ax[1].legend(frameon=False,fontsize=6)
#plt.setp(legend.get_texts(), color='k')
#plt.savefig(dir_out+'N-rho.pdf',bbox_inches='tight')


#fig,ax=plt.subplots(1,1,figsize=(2,2))
nbin=np.log(np.unique(np.uint32(np.logspace(0,3,40)))-0.5)
invrhobin=np.log(np.logspace(np.log10(inv_rho.min()),0,50))
ax[0].hist2d(np.log(inv_rho),np.log(nagg),[invrhobin,nbin],cmap=plt.cm.Greys)
ax[0].set_ylabel(r'$\log n(\textbf{x})$')
#plt.ylim([0,7])_{s_B}
#plt.xlim([-15,-5])
ax[0].set_xlabel(r'$\log\langle \rho^{-1}(\textbf{x}) \rangle_{s_B}$')
ax[0].patch.set_facecolor(plt.cm.Greys(0))
rx=np.linspace(-15,-5,3)
ax[0].plot(rx,-rx-9,'r--',label=r'$-1$')
legend = ax[1].legend(frameon=False,fontsize=6)

plt.savefig('/home/joris/Dropbox/Articles/AggregationBatchelor/Figures/agg1_sine.pdf',bbox_inches='tight')


#%%% Scaling of <1/rho**n>_B shows saturation of moments
N=20
binarize='N'

plt.figure()
s1,n=bundle((S),np.mean,binarize,N)
s2,n=bundle((S)**2.,np.mean,binarize,N)
v2,n=bundle(S,np.var,binarize,N)jheyman
s3,n=bundle((S)**3.,np.mean,binarize,N)
s4,n=bundle((S)**4.,np.mean,binarize,N)

K=np.arange(10)
sn=[bundle((S)**k,np.mean,binarize,N)[0] for k in K]
plt.plot(n[1:],s1,'o-',label=r'$\langle 1/\rho \rangle_B$')
plt.plot(n[1:],s2,'o-',label=r'$\langle 1/\rho^2 \rangle_B$')
plt.plot(n[1:],v2,'+-',label=r'$\langle 1/\rho^2 \rangle_B$')
plt.plot(n[1:],s3,'o-',label=r'$\langle 1/\rho^3 \rangle_B$')
plt.plot(n[1:],s4,'o-',label=r'$\langle 1/\rho^4 \rangle_B$')
plt.plot(n,1/n*1e-5,'k--',label=r'$-1$')
plt.yscale('log')
plt.xscale('log')
plt.xlabel(binarize)
plt.legend()
plt.savefig(dir_out+'1_rho**n.pdf',bbox_inches='tight')

a=[]

for k in K:
	idgood=np.where(np.isfinite(sn[k])&(n[1:]>10))[0]
	a.append(np.polyfit(np.log(n[idgood]),np.log(sn[k][idgood]),1)[0])
	

# mu ~ 1.2 log(N)
# sigma2 ~ 0.5 log (N)
# cmax^n ~ exp(-1.2 log(N) * n + 0.5 log (N) * n^ 2 /2)

mu=1/(fractal(keyword)-1)
sigma2=2*(2-fractal(keyword))/((fractal(keyword)-1))

plt.ylabel(r'$\log \delta^2(t) /t$')
plt.figure()
plt.plot(K,-np.array(a),'*-',label='Observed')
plt.plot(K,-(-K*mu+K**2*sigma2/2.),'--',label='Theorical')
plt.xlabel(r'$n$')
plt.ylabel(r'$\gamma, \langle (1/\rho)^n \rangle_{s_B} \propto N^{- \gamma }$')
plt.ylim([-2,6])
plt.legend()
plt.savefig(dir_out+'1_rho**n_gamma.pdf',bbox_inches='tight')
#%%% Scaling of <cmax**n>_B


plt.figure()
s1,n=bundle((Cmax),np.mean,'N',N)
s2,n=bundle((Cmax)**2.,np.mean,'N',N)
s3,n=bundle((Cmax)**3.,np.mean,'N',N)
s4,n=bundle((Cmax)**4.,np.mean,'N',N)

K=np.arange(10)
sn=[bundle((Cmax)**k,np.mean,binarize,N)[0] for k in K]
plt.plot(n[1:],s1,'o-',label=r'$\langle \theta \rangle_B$')
plt.plot(n[1:],s2,'o-',label=r'$\langle \theta^2 \rangle_B$')
plt.plot(n[1:],s3,'o-',label=r'$\langle \theta^3 \rangle_B$')
plt.plot(n[1:],s4,'o-',label=r'$\langle \theta^4 \rangle_B$')
plt.plot(n,1/n*1e-5,'k--',label=r'-1')
plt.yscale('log')
plt.xscale('log')
plt.xlabel('$N$')
plt.legend()
plt.savefig(dir_out+'cmax**n.pdf',bbox_inches='tight')

a=[]

for k in K:
	idgood=np.where(np.isfinite(sn[k])&(n[1:]>10))[0]
	a.append(np.polyfit(np.log(n[idgood]),np.log(sn[k][idgood]),1)[0])
	

# mu ~ 1.2 log(N)
# sigma2 ~ 0.5 log (N)
# cmax^n ~ exp(-1.2 log(N) * n + 0.5 log (N) * n^ 2 /2)

mu=1/(fractal(keyword)-1)
sigma2=2*(2-fractal(keyword))/((fractal(keyword)-1))

plt.figure()
plt.plot(K,-np.array(a),'*-',label='Observed')
plt.plot(K,-(-K*mu+K**2*sigma2/2.),'--',label='Theorical')
plt.xlabel(r'$n$')
plt.ylabel(r'$\gamma, \langle c^n \rangle_{s_B} \propto N^{- \gamma }$')
plt.ylim([-2,6])
plt.legend()
plt.savefig(dir_out+'cmax**n_gamma.pdf',bbox_inches='tight')


#%%% Binning By N
#keyword='sine'
if keyword=='sine':
	lyap=0.65
	sigma=lyap*0.65
	
if keyword=='half':
	lyap=0.25
	sigma=0.26

fig, axs = plt.subplots(6, 4,figsize=(10,9))
plt.subplots_adjust(wspace=0.3,hspace=0.3)
#mean log rho
#plt.figure(figsize=(1.5,1.5))
plt.sca(axs[0,0])
#plt.plot(nagg,(1./S[idsamples]),'k.',markersize=0.5,c='0.5')
#plt.yscale('log')
#plt.xscale('log')
plt.plot(np.log(nagg_bin[:-1]),logrhomean_naggbin,'ko')
#plt.plot(np.log(nagg_bin[:-1]),-logSmean_naggbin,'ro')
plt.plot(np.log(nagg_bin[:-1]),np.zeros(nagg_bin[:-1].shape)+meanlogrho,'k-',label='All')
plt.plot(np.log(nagg_bin[:-1]),np.log(nagg_bin[:-1])+7.5,'k--',label='$\log N$')
plt.xlabel(r'$\log N$')
plt.ylabel(r'$\langle \log \rho \rangle_{s_B}$')
mu=np.polyfit(np.log(nagg_bin[:-1])[np.isfinite(logrhomean_naggbin)],logrhomean_naggbin[np.isfinite(logrhomean_naggbin)],1)

Mu=np.log(nagg_bin[:-1])*mu[0]+mu[1]
plt.plot(np.log(nagg_bin[:-1]),Mu,'r--',label='${:1.1f}\log N+{:1.1f}$'.format(mu[0],mu[1]))
plt.legend(fontsize=6)
#Var log rho
#plt.figure(figsize=(1.5,1.5))
plt.sca(axs[0,1])
#plt.plot(nagg,(1./S[idsamples]),'k.',markersize=0.5,c='0.5')
#plt.yscale('log')
#plt.xscale('log')
#plt.plot(np.log(nagg_bin[:-1]),logrhomean_naggbin,'ko')
plt.plot(np.log(nagg_bin[:-1]),logrhovar_naggbin,'ko')
plt.plot(np.log(nagg_bin[:-1]),np.log(nagg_bin[:-1])*0.5,'k--',label='$0.5\log N$')
plt.plot(np.log(nagg_bin[:-1]),np.zeros(nagg_bin[:-1].shape)+varlogrho,'k-',label='All')
plt.xlabel(r'$\log N$')
plt.ylabel(r'${\mathrm{Var}_{s_B}[\log \rho]}$')

sig2=np.polyfit(np.log(nagg_bin[:-1])[np.isfinite(logrhovar_naggbin)],logrhovar_naggbin[np.isfinite(logrhovar_naggbin)],1)

Si=np.log(nagg_bin[:-1])*sig2[0]+sig2[1]
plt.plot(np.log(nagg_bin[:-1]),Si,'r--',label='${:1.1f}\log N+{:1.1f}$'.format(sig2[0],sig2[1]))
plt.legend(fontsize=6)


#Std log rho
#plt.figure(figsize=(1.5,1.5))
plt.sca(axs[0,2])
#plt.plot(nagg,(1./S[idsamples]),'k.',markersize=0.5,c='0.5')
#plt.yscale('log')
#plt.xscale('log')
#plt.plot(np.log(nagg_bin[:-1]),logrhomean_naggbin,'ko')
plt.plot(np.log(nagg_bin[:-1]),logrhostd_naggbin,'ko')
plt.plot(np.log(nagg_bin[:-1]),np.sqrt(0.5*np.log(nagg_bin[:-1])),'k--',label='$\sqrt{0.5\log N}$')
#plt.plot(np.log(nagg_bin[:-1]),0.5*np.log(nagg_bin[:-1]),'k--',label='$\sqrt{0.5\log N}$')

plt.xlabel(r'$\log N$')
plt.ylabel(r'${\mathrm{Std}_{s_B}[\log \rho]}$')
plt.legend(fontsize=6)

#Std log rho
#plt.figure(figsize=(1.5,1.5))
plt.sca(axs[0,3])
#plt.plot(nagg,(1./S[idsamples]),'k.',markersize=0.5,c='0.5')
#plt.yscale('log')
#plt.xscale('log')
#plt.plot(np.log(nagg_bin[:-1]),logrhomean_naggbin,'ko')
plt.plot(np.log(nagg_bin[:-1]),logrhostd_naggbin/logrhomean_naggbin,'ko')
#plt.plot(np.log(nagg_bin[:-1]),1/np.sqrt(0.5*np.log(nagg_bin[:-1])),'k--',label='$\sqrt{0.5\log N}$')
plt.plot(np.log(nagg_bin[:-1]),np.sqrt(Si)/Mu,'r--',label='$\sigma/\mu$')

plt.xlabel(r'$\log N$')
plt.ylabel(r'${\mathrm{Std}_{s_B}[\log \rho]}/\langle \log \rho \rangle_{s_B}$')
plt.legend(fontsize=6)#plt.ylim([0,2])

# Mean rho

#plt.figure(figsize=(1.5,1.5))
plt.sca(axs[1,0])
#plt.plot(nagg,(1./S[idsamples]),'k.',markersize=0.5,c='0.5')
plt.yscale('log')
plt.xscale('log')
plt.plot((nagg_bin[:-1]),rhomean_naggbin,'ko')
plt.plot((nagg_bin[:-1]),1/Smean_naggbin,'ro')
plt.plot((nagg_bin[:-1]),(nagg_bin[:-1])*5e3,'k--',label='$ N$')
plt.plot((nagg_bin[:-1]),np.zeros(nagg_bin[:-1].shape)+meanrho,'k-',label='All')
plt.plot(nagg_bin[:-1],np.exp(Mu),'r--',label='$\mu+\sigma^2/2$')

plt.xlabel(r'$ N$')
plt.ylabel(r'$\langle  \rho \rangle_{s_B}$')
plt.legend(fontsize=6)
#plt.xlim([0.5,500])
#Var rho

#plt.figure(figsize=(1.5,1.5))
plt.sca(axs[1,1])
#plt.plot(nagg,(1./S[idsamples]),'k.',markersize=0.5,c='0.5')
plt.yscale('log')
plt.xscale('log')
#plt.plot(np.log(nagg_bin[:-1]),logrhomean_naggbin,'ko')
plt.plot((nagg_bin[:-1]),rhovar_naggbin,'ko')
plt.plot((nagg_bin[:-1]),3e7*(nagg_bin[:-1])**3.,'k--',label='$N^3$')
plt.plot((nagg_bin[:-1]),3e7*(nagg_bin[:-1])**2.,'k-',label='$N^2$')
plt.plot((nagg_bin[:-1]),np.zeros(nagg_bin[:-1].shape)+varrho,'k-',label='All')
plt.plot(nagg_bin[:-1],np.exp(2*Mu+2*Si),'r--',label='$2\mu+2\sigma^2$')


plt.xlabel(r'$N$')
plt.ylabel(r'${\mathrm{Var}_{s_B}[ \rho]}$')
plt.legend(fontsize=6)
plt.xlim([0.5,500])
#Std rho

#plt.figure(figsize=(1.5,1.5))
plt.sca(axs[1,2])
#plt.plot(nagg,(1./S[idsamples]),'k.',markersize=0.5,c='0.5')
plt.yscale('log')
plt.xscale('log')
#plt.plot(np.log(nagg_bin[:-1]),logrhomean_naggbin,'ko')
plt.plot((nagg_bin[:-1]),rhostd_naggbin,'ko')
plt.plot((nagg_bin[:-1]),(nagg_bin[:-1])*4e3,'k--',label='${ N}$')
plt.xlabel(r'$N$')
plt.ylabel(r'${\mathrm{Std}_{s_B}[ \rho]}$')
plt.legend(fontsize=6)
plt.xlim([0.5,500])
#Std rho

#plt.figure(figsize=(1.5,1.5))
plt.sca(axs[1,3])
#plt.plot(nagg,(1./S[idsamples]),'k.',markersize=0.5,c='0.5')
plt.yscale('log')
plt.xscale('log')
#plt.plot(np.log(nagg_bin[:-1]),logrhomean_naggbin,'ko')
plt.plot((nagg_bin[:-1]),rhostd_naggbin/rhomean_naggbin,'ko')
plt.plot(nagg_bin[:-1],np.exp(Mu+Si)/np.exp(Mu+Si/2),'r--')
#plt.plot((nagg_bin[:-1]),(nagg_bin[:-1])*4e3,'k--',label='${ N}$')
plt.xlabel(r'$N$')
plt.ylabel(r'${\mathrm{Std}_{s_B}[ \rho]}/\langle  \rho \rangle_{s_B}$')
plt.ylim([0.1,10])
plt.legend(fontsize=6)
plt.xlim([0.5,500])



#plt.figure(figsize=(1.5,1.5))
plt.sca(axs[3,2])
#plt.plot(nagg,(1./S[idsamples]),'k.',markersize=0.5,c='0.5')
plt.yscale('log')
plt.xscale('log')
plt.plot(nagg_bin[:-1],cmaxstd_naggbin,'ko')
plt.plot(nagg_bin[:-1],1e-3/np.sqrt(nagg_bin[:-1]),'k--',label='$1/\sqrt{N}$')
#plt.plot(nagg_bin[:-1],(nagg_bin[:-1]),'k--')
#plt.plot(nagg_bin[:-1],np.exp(rho_bin[0])*nagg_bin[1:],'k--',label=r'$N\sim\rho$')
plt.xlabel(r'$N$')
plt.ylabel(r'STD $(c_\mathrm{max})$')
plt.xlim([0.5,500])
plt.legend(fontsize=6)

#plt.figure(figsize=(1.5,1.5))
plt.sca(axs[3,1])
#plt.plot(nagg,(1./S[idsamples]),'k.',markersize=0.5,c='0.5')
plt.yscale('log')
plt.xscale('log')
plt.plot(nagg_bin[:-1],cmaxvar_naggbin,'ko')
plt.plot(nagg_bin[:-1],cmaxvar2_naggbin,'b+')
plt.plot(nagg_bin[:-1],1e-5/nagg_bin[:-1],'k--',label='$1/N$')
plt.plot(nagg_bin[:-1],(s0**2*lyap/4/D)*(np.exp(-2*Mu+Si)*(np.exp(Si)-1)),'r-',label=r'$e^{-2\mu+\sigma^2}(e^{\sigma^2} -1)$')
plt.plot(nagg_bin[:-1],(s0**2*lyap/4/D)*(np.exp(-2*Mu+2*Si)),'r--',label=r'$e^{-2\mu+2\sigma^2}$')

#plt.plot(nagg_bin[:-1],(nagg_bin[:-1]),'k--')
#plt.plot(nagg_bin[:-1],np.exp(rho_bin[0])*nagg_bin[1:],'k--',label=r'$N\sim\rho$')
plt.xlabel(r'$N$')
plt.ylabel(r'Var $(c_\mathrm{max})$')
plt.xlim([0.5,500])
plt.legend(fontsize=6)

#plt.figure(figsize=(1.5,1.5))
plt.sca(axs[3,0])
#plt.plot(nagg,(1./S[idsamples]),'k.',markersize=0.5,c='0.5')
plt.yscale('log')
plt.xscale('log')
plt.plot(nagg_bin[:-1],cmaxmean_naggbin,'ko')
plt.plot(nagg_bin[:-1],cmaxmean2_naggbin,'b+')
plt.plot(nagg_bin[:-1],3e-3/(nagg_bin[:-1]),'k--',label='$1/{N}$')
plt.plot(nagg_bin[:-1],np.sqrt(s0**2*lyap/4/D)*np.exp(-Mu+Si/2),'r--',label=r'$e^{-\mu+\sigma^2/2}$')


#plt.plot(nagg_bin[:-1],(nagg_bin[:-1]),'k--')
#plt.plot(nagg_bin[:-1],np.exp(rho_bin[0])*nagg_bin[1:],'k--',label=r'$N\sim\rho$')
plt.xlabel(r'$N$')
plt.ylabel(r'$\langle c_\mathrm{max} \rangle$')
plt.xlim([0.5,500])
plt.legend(fontsize=6)
#


#plt.figure(figsize=(1.5,1.5))
plt.sca(axs[3,3])
#plt.plot(nagg,(1./S[idsamples]),'k.',markersize=0.5,c='0.5')
plt.yscale('log')
plt.xscale('log')
plt.plot(nagg_bin[:-1],cmaxstd_naggbin/cmaxmean_naggbin,'ko')
plt.plot(nagg_bin[:-1],np.exp(-Mu+Si)/np.exp(-Mu+Si/2),'r--')
plt.plot(nagg_bin[:-1],0.5*(nagg_bin[:-1])**0.5,'k--',label=r'$\sqrt{N}$')
#plt.plot(nagg_bin[:-1],(nagg_bin[:-1]),'k--')
#plt.plot(nagg_bin[:-1],np.exp(rho_bin[0])*nagg_bin[1:],'k--',label=r'$N\sim\rho$')
plt.xlabel(r'$N$')
plt.ylabel(r'STD $(c_\mathrm{max})/ \langle c_\mathrm{max} \rangle$')
plt.legend(fontsize=6)
plt.xlim([0.5,500])


#plt.figure(figsize=(1.5,1.5))
plt.sca(axs[4,2])
#plt.plot(nagg,(1./S[idsamples]),'k.',markersize=0.5,c='0.5')
#plt.yscale('log')
plt.xscale('log')
plt.plot(nagg_bin[:-1],logcmaxstd_naggbin,'ko')
#plt.plot(nagg_bin[:-1],1e-3/np.sqrt(nagg_bin[:-1]),'k--',label='$1/\sqrt{N}$')
#plt.plot(nagg_bin[:-1],(nagg_bin[:-1]),'k--')
#plt.plot(nagg_bin[:-1],np.exp(rho_bin[0])*nagg_bin[1:],'k--',label=r'$N\sim\rho$')
plt.xlabel(r'$N$')
plt.ylabel(r'STD $(\log c_\mathrm{max})$')
plt.legend(fontsize=6)

#plt.figure(figsize=(1.5,1.5))
plt.sca(axs[4,1])
#plt.plot(nagg,(1./S[idsamples]),'k.',markersize=0.5,c='0.5')
#plt.yscale('log')
plt.xscale('log')
plt.plot(nagg_bin[:-1],logcmaxvar_naggbin,'ko')
plt.plot(nagg_bin[:-1],Si,'r--',label='$\sigma^2$')
#plt.plot(nagg_bin[:-1],1e-5/nagg_bin[:-1],'k--',label='$1/N$')
#plt.plot(nagg_bin[:-1],(nagg_bin[:-1]),'k--')
#plt.plot(nagg_bin[:-1],np.exp(rho_bin[0])*nagg_bin[1:],'k--',label=r'$N\sim\rho$')
plt.xlabel(r'$N$')
plt.ylabel(r'Var $(\log  c_\mathrm{max})$')
plt.legend(fontsize=6)

#plt.figure(figsize=(1.5,1.5))
plt.sca(axs[4,0])
#plt.plot(nagg,(1./S[idsamples]),'k.',markersize=0.5,c='0.5')
#plt.yscale('log')
plt.xscale('log')
plt.plot(nagg_bin[:-1],logcmaxmean_naggbin,'ko')
plt.plot(nagg_bin[:-1],-Mu,'r--',label='$-\mu$')
#plt.plot(nagg_bin[:-1],3e-3/(nagg_bin[:-1]),'k--',label='$1/{N}$')
#plt.plot(nagg_bin[:-1],(nagg_bin[:-1]),'k--')
#plt.plot(nagg_bin[:-1],np.exp(rho_bin[0])*nagg_bin[1:],'k--',label=r'$N\sim\rho$')
plt.xlabel(r'$N$')
plt.ylabel(r'$\langle \log  c_\mathrm{max} \rangle$')
plt.legend(fontsize=6)#


#plt.figure(figsize=(1.5,1.5))
plt.sca(axs[4,3])
#plt.plot(nagg,(1./S[idsamples]),'k.',markersize=0.5,c='0.5')
#plt.yscale('log')
plt.xscale('log')
plt.plot(nagg_bin[:-1],logcmaxstd_naggbin/logcmaxmean_naggbin,'ko')
plt.plot(nagg_bin[:-1],-np.sqrt(Si)/Mu,'r--',label='-$\sigma/\mu$')
#plt.plot(nagg_bin[:-1],0.5*(nagg_bin[:-1])**0.5,'k--',label=r'$\sqrt{N}$')
#plt.plot(nagg_bin[:-1],(nagg_bin[:-1]),'k--')
#plt.plot(nagg_bin[:-1],np.exp(rho_bin[0])*nagg_bin[1:],'k--',label=r'$N\sim\rho$')
plt.xlabel(r'$N$')
plt.ylabel(r'STD $(\log  c_\mathrm{max})/ \langle \log  c_\mathrm{max} \rangle$')
plt.legend(fontsize=6)


#plt.figure(figsize=(1.5,1.5))
plt.sca(axs[5,0])
#plt.plot(rho,(1./S[idsamples]),'k.',markersize=0.5,c='0.5')
plt.yscale('log')
plt.xscale('log')
plt.plot(nagg_bin[:-1],cmax_naggbin,'ko')
#plt.plot(rho_bin[:-1],-Mu,'r--',label='$-\mu$')
#plt.plot(rho_bin[:-1],3e-3/(rho_bin[:-1]),'k--',label='$1/{\rho}$')
#plt.plot(rho_bin[:-1],(rho_bin[:-1]),'k--')
#plt.plot(rho_bin[:-1],np.exp(rho_bin[0])*rho_bin[1:],'k--',label=r'$\rho\sim\rho$')
plt.xlabel(r'$N$')
plt.ylabel(r'$\langle \sum c_\mathrm{max} \rangle$')
plt.legend(fontsize=6)#

plt.sca(axs[5,1])
#plt.plot(rho,(1./S[idsamples]),'k.',markersize=0.5,c='0.5')
#plt.yscale('log')
plt.xscale('log')
plt.plot(nagg_bin[:-1],logcmax_naggbin_var,'ko')
#plt.plot(rho_bin[:-1],-Mu,'r--',label='$-\mu$')
#plt.plot(rho_bin[:-1],3e-3/(rho_bin[:-1]),'k--',label='$1/{\rho}$')
#plt.plot(rho_bin[:-1],(rho_bin[:-1]),'k--')
#plt.plot(rho_bin[:-1],np.exp(rho_bin[0])*rho_bin[1:],'k--',label=r'$\rho\sim\rho$')
plt.xlabel(r'$N$')
plt.ylabel(r'Var $ \log \sum c_\mathrm{max}$')
plt.ylim([0,2])
plt.legend(fontsize=6)#



plt.sca(axs[5,2])
#plt.plot(rho,(1./S[idsamples]),'k.',markersize=0.5,c='0.5')
#plt.yscale('log')
plt.xscale('log')
plt.plot(nagg_bin[:-1],cmax_naggbin_std,'ko')
#plt.plot(rho_bin[:-1],-Mu,'r--',label='$-\mu$')
#plt.plot(rho_bin[:-1],3e-3/(rho_bin[:-1]),'k--',label='$1/{\rho}$')
#plt.plot(rho_bin[:-1],(rho_bin[:-1]),'k--')
#plt.plot(rho_bin[:-1],np.exp(rho_bin[0])*rho_bin[1:],'k--',label=r'$\rho\sim\rho$')
plt.xlabel(r'$N$')
plt.ylabel(r'Std $\sum c_\mathrm{max} $')
#plt.ylim([0,2])
plt.legend(fontsize=6)#

plt.sca(axs[5,3])
#plt.plot(rho,(1./S[idsamples]),'k.',markersize=0.5,c='0.5')
#plt.yscale('log')
plt.xscale('log')
plt.plot(nagg_bin[:-1],cmax_naggbin_std/cmax_naggbin,'ko')
#plt.plot(rho_bin[:-1],-Mu,'r--',label='$-\mu$')
#plt.plot(rho_bin[:-1],3e-3/(rho_bin[:-1]),'k--',label='$1/{\rho}$')
#plt.plot(rho_bin[:-1],(rho_bin[:-1]),'k--')
#plt.plot(rho_bin[:-1],np.exp(rho_bin[0])*rho_bin[1:],'k--',label=r'$\rho\sim\rho$')
plt.xlabel(r'$N$')
plt.ylabel(r'Std $\sum c_\mathrm{max} / \langle \sum c_\mathrm{max} \rangle$')
#plt.ylim([0,2])
plt.legend(fontsize=6)#
##plt.plot(nagg,(1./S[idsamples]),'k.',markersize=0.5,c='0.5')
#plt.yscale('log')
#plt.xscale('log')
#plt.plot(nagg_bin[:-1],cmaxstd_bin2,'ko')
#plt.plot(nagg_bin[:-1],cmaxstd_bin3,'bo')
#plt.plot(nagg_bin[:-1],cmaxmean_bin2,'ro')
#plt.plot(nagg_bin[:-1],Taustd_bin,'go')
#plt.plot(nagg_bin[:-1],np.sqrt(nagg_bin[:-1])**(-1),'k--',label='$1/\sqrt{N}$')
#
#plt.plot(nagg_bin[:-1],(nagg_bin[:-1]),'k--',label='$1/\sqrt{N}$')
#plt.plot(nagg_bin[:-1],(nagg_bin[:-1])**2,'k--',label='$1/\sqrt{N}$')
##plt.plot(nagg_bin[:-1],(nagg_bin[:-1]),'k--')
##plt.plot(nagg_bin[:-1],np.exp(rho_bin[0])*nagg_bin[1:],'k--',label=r'$N\sim\rho$')
#plt.xlabel(r'$N$')
#plt.ylabel(r'STD $(c_\mathrm{max})$')
##plt.ylim([1e-1,1e1])
##plt.xlim([1e0,1e3])
#plt.legend()


#
##plt.figure(figsize=(1.5,1.5))
#plt.sca(axs[2,1])
##plt.plot(nagg,(1./S[idsamples]),'k.',markersize=0.5,c='0.5')
#plt.yscale('log')
#plt.xscale('log')
#plt.plot(nagg_bin[:-1],Taustd_bin,'ko')
#plt.plot(nagg_bin[:-1],1e8*(nagg_bin[:-1])**2,'k--',label='$N^2$')
#plt.xlabel(r'$N$')
#plt.ylabel(r'STD $\tau$')
#plt.legend()
#
#plt.figure(figsize=(1.5,1.5))
##plt.plot(nagg,(1./S[idsamples]),'k.',markersize=0.5,c='0.5')
#plt.yscale('log')
#plt.xscale('log')
#plt.plot(nagg_bin[:-1],Tauvar_bin,'ko')
#plt.plot(nagg_bin[:-1],1e19*(nagg_bin[:-1])**4,'k--',label='$N^4$')
#plt.xlabel(r'$N$')
#plt.ylabel(r'Var $\tau$')
#plt.legend()




#plt.figure(figsize=(1.5,1.5))
# =============================================================================
# #plt.sca(axs[2,2])
# ##plt.plot(nagg,(1./S[idsamples]),'k.',markersize=0.5,c='0.5')
# #plt.yscale('log')
# #plt.xscale('log')
# #plt.plot(nagg_bin[:-1],Taustd_bin/Taumean_bin,'ko')
# ##plt.plot(nagg_bin[:-1],1e8*(nagg_bin[:-1])**2,'k--',label='$N^2$')
# #plt.xlabel(r'$N$')
# #plt.ylim([1e-1,1e1])
# #plt.ylabel(r'STD $\tau / \langle \tau \rangle$')
# #plt.legend()
# =============================================================================



#plt.figure(figsize=(1.5,1.5))
plt.sca(axs[2,0])
#plt.plot(nagg,(1./S[idsamples]),'k.',markersize=0.5,c='0.5')
plt.yscale('log')
plt.xscale('log')
plt.plot(nagg_bin[:-1],sqrtTaumean_bin,'ko')
plt.plot(nagg_bin[:-1],1e4*(nagg_bin[:-1]),'k--',label='$N$')

plt.plot(nagg_bin[:-1],np.sqrt(D/s0**2/lyap)*np.exp((Mu+Si/2)),'r--',label='$\mu+\sigma^2/2$')
plt.xlabel(r'$N$')
plt.ylabel(r'$\langle \sqrt{\tau} \rangle$')
plt.legend(fontsize=6)
plt.xlim([0.5,500])

#plt.figure(figsize=(1.5,1.5))
plt.sca(axs[2,2])
#plt.plot(nagg,(1./S[idsamples]),'k.',markersize=0.5,c='0.5')
plt.yscale('log')
plt.xscale('log')
plt.plot(nagg_bin[:-1],sqrtTaustd_bin,'ko')
plt.plot(nagg_bin[:-1],1e4*(nagg_bin[:-1]),'k--',label='$N$')
plt.xlabel(r'$N$')
plt.ylabel(r'STD $\sqrt{\tau}$')
plt.legend(fontsize=6)

plt.xlim([0.5,500])
#plt.figure(figsize=(1.5,1.5))
plt.sca(axs[2,1])
#plt.plot(nagg,(1./S[idsamples]),'k.',markersize=0.5,c='0.5')
plt.yscale('log')
plt.xscale('log')
plt.plot(nagg_bin[:-1],sqrtTauvar_bin,'ko')
plt.plot(nagg_bin[:-1],1e8*(nagg_bin[:-1])**2,'k--',label='$N^2$')
plt.plot(nagg_bin[:-1],np.exp(2*Mu+2*Si),'r--',label='$2\mu+2\sigma^2$')
plt.xlabel(r'$N$')
plt.ylabel(r'Var $\sqrt{\tau}$')
plt.legend(fontsize=6)

plt.xlim([0.5,500])

#plt.figure(figsize=(1.5,1.5))
plt.sca(axs[2,3])
#plt.plot(nagg,(1./S[idsamples]),'k.',markersize=0.5,c='0.5')
plt.yscale('log')
plt.xscale('log')
plt.plot(nagg_bin[:-1],sqrtTaustd_bin/sqrtTaumean_bin,'ko')
plt.plot(nagg_bin[:-1],np.exp(Mu+Si)/np.exp(Mu+Si/2),'r--')

#plt.plot(nagg_bin[:-1],(nagg_bin[:-1])**0.25,'k--',label='$N^{1/4}$')
#plt.plot(nagg_bin[:-1],1e8*(nagg_bin[:-1])**2,'k--',label='$N^2$')
plt.xlabel(r'$N$')
plt.ylim([1e-1,1e1])
plt.ylabel(r'STD $\sqrt{\tau} / \langle \sqrt{\tau} \rangle$')
plt.legend(fontsize=6)
plt.xlim([0.5,500])


plt.savefig(dir_out+'AllResults_{:1.1f}_D{:1.0e}.pdf'.format(A,Brownian),bbox_inches='tight')

#%%% Bining by Rho

fig, axs = plt.subplots(6, 4,figsize=(10,9))
plt.subplots_adjust(wspace=0.5,hspace=0.5)
#mean log rho
#plt.figure(figsize=(1.5,1.5))
plt.sca(axs[0,0])
#plt.plot(nagg,(1./S[idsamples]),'k.',markersize=0.5,c='0.5')
#plt.yscale('log')
#plt.xscale('log')
plt.plot(np.log(rho_bin[:-1]),logrhomean_rhobin,'ko')
plt.plot(np.log(rho_bin[:-1]),np.zeros(rho_bin[:-1].shape)+meanlogrho,'k-',label='All')
plt.plot(np.log(rho_bin[:-1]),np.log(rho_bin[:-1])+7.5,'k--',label='$\log \rho$')
plt.xlabel(r'$\log \rho$')
plt.ylabel(r'$\langle \log \rho \rangle_{s_B}$')
mu=np.polyfit(np.log(rho_bin[:-1])[np.isfinite(logrhomean_rhobin)],logrhomean_rhobin[np.isfinite(logrhomean_rhobin)],1)

Mu=np.log(rho_bin[:-1])*mu[0]+mu[1]
plt.plot(np.log(rho_bin[:-1]),Mu,'r--',label=r'${:1.2f}\log \rho+{:1.2f}$'.format(mu[0],mu[1]))
plt.legend(fontsize=6)
#Var log rho
#plt.figure(figsize=(1.5,1.5))
plt.sca(axs[0,1])
#plt.plot(rho,(1./S[idsamples]),'k.',markersize=0.5,c='0.5')
#plt.yscale('log')
#plt.xscale('log')
#plt.plot(np.log(rho_bin[:-1]),logrhomean_rhobin,'ko')
plt.plot(np.log(rho_bin[:-1]),logrhovar_rhobin,'ko')
plt.plot(np.log(rho_bin[:-1]),np.log(rho_bin[:-1])*0.5,'k--',label=r'$0.5\log \rho$')
plt.plot(np.log(rho_bin[:-1]),np.zeros(rho_bin[:-1].shape)+varlogrho,'k-',label='All')
plt.xlabel(r'$\log \rho$')
plt.ylabel(r'${\mathrm{Var}_{s_B}[\log \rho]}$')

sig2=np.polyfit(np.log(rho_bin[:-1])[np.isfinite(logrhovar_rhobin)],logrhovar_rhobin[np.isfinite(logrhovar_rhobin)],1)

Si=np.log(rho_bin[:-1])*sig2[0]+sig2[1]
plt.plot(np.log(rho_bin[:-1]),Si,'r--',label=r'${:1.2f}\log \rho+{:1.2f}$'.format(sig2[0],sig2[1]))
plt.legend(fontsize=6)


#Std log rho
#plt.figure(figsize=(1.5,1.5))
plt.sca(axs[0,2])
#plt.plot(rho,(1./S[idsamples]),'k.',markersize=0.5,c='0.5')
#plt.yscale('log')
#plt.xscale('log')
#plt.plot(np.log(rho_bin[:-1]),logrhomean_rhobin,'ko')
plt.plot(np.log(rho_bin[:-1]),logrhostd_rhobin,'ko')
plt.plot(np.log(rho_bin[:-1]),np.sqrt(0.5*np.log(rho_bin[:-1])),'k--',label=r'$\sqrt{0.5\log \rho}$')
#plt.plot(np.log(rho_bin[:-1]),0.5*np.log(rho_bin[:-1]),'k--',label='$\sqrt{0.5\log \rho}$')

plt.xlabel(r'$\log \rho$')
plt.ylabel(r'${\mathrm{Std}_{s_B}[\log \rho]}$')
plt.legend(fontsize=6)

#Std log rho
#plt.figure(figsize=(1.5,1.5))
plt.sca(axs[0,3])
#plt.plot(rho,(1./S[idsamples]),'k.',markersize=0.5,c='0.5')
#plt.yscale('log')
#plt.xscale('log')
#plt.plot(np.log(rho_bin[:-1]),logrhomean_rhobin,'ko')
plt.plot(np.log(rho_bin[:-1]),logrhostd_rhobin/logrhomean_rhobin,'ko')
#plt.plot(np.log(rho_bin[:-1]),1/np.sqrt(0.5*np.log(rho_bin[:-1])),'k--',label='$\sqrt{0.5\log \rho}$')
plt.plot(np.log(rho_bin[:-1]),np.sqrt(Si)/Mu,'r--',label='$\sigma/\mu$')

plt.xlabel(r'$\log \rho$')
plt.ylabel(r'${\mathrm{Std}_{s_B}[\log \rho]}/\langle \log \rho \rangle_{s_B}$')
plt.legend(fontsize=6)#plt.ylim([0,2])

# Mean rho

#plt.figure(figsize=(1.5,1.5))
plt.sca(axs[1,0])
#plt.plot(rho,(1./S[idsamples]),'k.',markersize=0.5,c='0.5')
plt.yscale('log')
plt.xscale('log')
plt.plot((rho_bin[:-1]),rhomean_rhobin,'ko')
plt.plot((rho_bin[:-1]),(rho_bin[:-1])*5e3,'k--',label=r'$ \rho$')
plt.plot((rho_bin[:-1]),np.zeros(rho_bin[:-1].shape)+meanrho,'k-',label='All')
plt.plot(rho_bin[:-1],np.exp(Mu+0.5*Si),'r--',label='$\mu+\sigma^2/2$')

plt.xlabel(r'$ \rho$')
plt.ylabel(r'$\langle  \rho \rangle_{s_B}$')
plt.legend(fontsize=6)
#plt.xlim([0.5,500])
#Var rho

#plt.figure(figsize=(1.5,1.5))
plt.sca(axs[1,1])
#plt.plot(rho,(1./S[idsamples]),'k.',markersize=0.5,c='0.5')
plt.yscale('log')
plt.xscale('log')
#plt.plot(np.log(rho_bin[:-1]),logrhomean_rhobin,'ko')
plt.plot((rho_bin[:-1]),rhovar_rhobin,'ko')
plt.plot((rho_bin[:-1]),3e7*(rho_bin[:-1])**3.,'k--',label=r'$\rho^3$')
plt.plot((rho_bin[:-1]),3e7*(rho_bin[:-1])**2.,'k-',label=r'$\rho^2$')
plt.plot((rho_bin[:-1]),np.zeros(rho_bin[:-1].shape)+varrho,'k-',label='All')
plt.plot(rho_bin[:-1],np.exp(2*Mu+2*Si),'r--',label='$2\mu+2\sigma^2$')


plt.xlabel(r'$\langle \rho \rangle$')
plt.ylabel(r'${\mathrm{Var}_{s_B}[ \rho]}$')
plt.legend(fontsize=6)
#plt.xlim([0.5,500])
#Std rho

#plt.figure(figsize=(1.5,1.5))
plt.sca(axs[1,2])
#plt.plot(rho,(1./S[idsamples]),'k.',markersize=0.5,c='0.5')
plt.yscale('log')
plt.xscale('log')
#plt.plot(np.log(rho_bin[:-1]),logrhomean_rhobin,'ko')
plt.plot((rho_bin[:-1]),rhostd_rhobin,'ko')
plt.plot((rho_bin[:-1]),(rho_bin[:-1])*4e3,'k--',label=r'${ \rho}$')
plt.xlabel(r'$\langle \rho \rangle$')
plt.ylabel(r'${\mathrm{Std}_{s_B}[ \rho]}$')
plt.legend(fontsize=6)
#plt.xlim([0.5,500])
#Std rho

#plt.figure(figsize=(1.5,1.5))
plt.sca(axs[1,3])
#plt.plot(rho,(1./S[idsamples]),'k.',markersize=0.5,c='0.5')
plt.yscale('log')
plt.xscale('log')
#plt.plot(np.log(rho_bin[:-1]),logrhomean_rhobin,'ko')
plt.plot((rho_bin[:-1]),rhostd_rhobin/rhomean_rhobin,'ko')
plt.plot(rho_bin[:-1],np.exp(Mu+Si)/np.exp(Mu+Si/2),'r--')
#plt.plot((rho_bin[:-1]),(rho_bin[:-1])*4e3,'k--',label='${ \rho}$')
plt.xlabel(r'$\langle \rho \rangle$')
plt.ylabel(r'${\mathrm{Std}_{s_B}[ \rho]}/\langle  \rho \rangle_{s_B}$')
plt.ylim([0.1,10])
plt.legend(fontsize=6)
#plt.xlim([0.5,500])



#plt.figure(figsize=(1.5,1.5))
plt.sca(axs[3,2])
#plt.plot(rho,(1./S[idsamples]),'k.',markersize=0.5,c='0.5')
plt.yscale('log')
plt.xscale('log')
plt.plot(rho_bin[:-1],cmaxstd_rhobin,'ko')
plt.plot(rho_bin[:-1],1e-3/np.sqrt(rho_bin[:-1]),'k--',label=r'$1/\sqrt{\rho}$')
#plt.plot(rho_bin[:-1],(rho_bin[:-1]),'k--')
#plt.plot(rho_bin[:-1],np.exp(rho_bin[0])*rho_bin[1:],'k--',label=r'$\rho\sim\rho$')
plt.xlabel(r'$\langle \rho \rangle$')
plt.ylabel(r'STD $(c_\mathrm{max})$')
#plt.xlim([0.5,500])
plt.legend(fontsize=6)

#plt.figure(figsize=(1.5,1.5))
plt.sca(axs[3,1])
#plt.plot(rho,(1./S[idsamples]),'k.',markersize=0.5,c='0.5')
plt.yscale('log')
plt.xscale('log')
plt.plot(rho_bin[:-1],cmaxvar_rhobin,'ko')
plt.plot(rho_bin[:-1],1e-5/rho_bin[:-1],'k--',label=r'$1/\rho$')
plt.plot(rho_bin[:-1],np.exp(-2*Mu+2*Si),'r--',label='$-2\mu+2\sigma^2$')

#plt.plot(rho_bin[:-1],(rho_bin[:-1]),'k--')
#plt.plot(rho_bin[:-1],np.exp(rho_bin[0])*rho_bin[1:],'k--',label=r'$\rho\sim\rho$')
plt.xlabel(r'$\langle \rho \rangle$')
plt.ylabel(r'Var $(c_\mathrm{max})$')
#plt.xlim([0.5,500])
plt.legend(fontsize=6)

#plt.figure(figsize=(1.5,1.5))
plt.sca(axs[3,0])
#plt.plot(rho,(1./S[idsamples]),'k.',markersize=0.5,c='0.5')
plt.yscale('log')
plt.xscale('log')
plt.plot(rho_bin[:-1],cmaxmean_rhobin,'ko')
plt.plot(rho_bin[:-1],3e-3/(rho_bin[:-1]),'k--',label=r'$1/{\rho}$')
plt.plot(rho_bin[:-1],np.exp(-Mu+Si/2),'r--',label='$-\mu+\sigma^2/2$')


#plt.plot(rho_bin[:-1],(rho_bin[:-1]),'k--')
#plt.plot(rho_bin[:-1],np.exp(rho_bin[0])*rho_bin[1:],'k--',label=r'$\rho\sim\rho$')
plt.xlabel(r'$\langle \rho \rangle$')
plt.ylabel(r'$\langle c_\mathrm{max} \rangle$')
#plt.xlim([0.5,500])
plt.legend(fontsize=6)
#


#plt.figure(figsize=(1.5,1.5))
plt.sca(axs[3,3])
#plt.plot(rho,(1./S[idsamples]),'k.',markersize=0.5,c='0.5')
plt.yscale('log')
plt.xscale('log')
plt.plot(rho_bin[:-1],cmaxstd_rhobin/cmaxmean_rhobin,'ko')
plt.plot(rho_bin[:-1],np.exp(-Mu+Si)/np.exp(-Mu+Si/2),'r--')
plt.plot(rho_bin[:-1],0.5*(rho_bin[:-1])**0.5,'k--',label=r'$\sqrt{\rho}$')
#plt.plot(rho_bin[:-1],(rho_bin[:-1]),'k--')
#plt.plot(rho_bin[:-1],np.exp(rho_bin[0])*rho_bin[1:],'k--',label=r'$\rho\sim\rho$')
plt.xlabel(r'$\langle \rho \rangle$')
plt.ylabel(r'STD $(c_\mathrm{max})/ \langle c_\mathrm{max} \rangle$')
plt.legend(fontsize=6)
#plt.xlim([0.5,500])


#plt.figure(figsize=(1.5,1.5))
plt.sca(axs[4,2])
#plt.plot(rho,(1./S[idsamples]),'k.',markersize=0.5,c='0.5')
#plt.yscale('log')
plt.xscale('log')
plt.plot(rho_bin[:-1],logcmaxstd_rhobin,'ko')
#plt.plot(rho_bin[:-1],1e-3/np.sqrt(rho_bin[:-1]),'k--',label='$1/\sqrt{\rho}$')
#plt.plot(rho_bin[:-1],(rho_bin[:-1]),'k--')
#plt.plot(rho_bin[:-1],np.exp(rho_bin[0])*rho_bin[1:],'k--',label=r'$\rho\sim\rho$')
plt.xlabel(r'$\langle \rho \rangle$')
plt.ylabel(r'STD $(\log c_\mathrm{max})$')
plt.legend(fontsize=6)

#plt.figure(figsize=(1.5,1.5))
plt.sca(axs[4,1])
#plt.plot(rho,(1./S[idsamples]),'k.',markersize=0.5,c='0.5')
#plt.yscale('log')
plt.xscale('log')
plt.plot(rho_bin[:-1],logcmaxvar_rhobin,'ko')
plt.plot(rho_bin[:-1],Si,'r--',label='$\sigma^2$')
#plt.plot(rho_bin[:-1],1e-5/rho_bin[:-1],'k--',label='$1/\rho$')
#plt.plot(rho_bin[:-1],(rho_bin[:-1]),'k--')
#plt.plot(rho_bin[:-1],np.exp(rho_bin[0])*rho_bin[1:],'k--',label=r'$\rho\sim\rho$')
plt.xlabel(r'$\langle \rho \rangle$')
plt.ylabel(r'Var $(\log  c_\mathrm{max})$')
plt.legend(fontsize=6)

#plt.figure(figsize=(1.5,1.5))
plt.sca(axs[4,0])
#plt.plot(rho,(1./S[idsamples]),'k.',markersize=0.5,c='0.5')
#plt.yscale('log')
plt.xscale('log')
plt.plot(rho_bin[:-1],logcmaxmean_rhobin,'ko')
plt.plot(rho_bin[:-1],-Mu,'r--',label='$-\mu$')
#plt.plot(rho_bin[:-1],3e-3/(rho_bin[:-1]),'k--',label='$1/{\rho}$')
#plt.plot(rho_bin[:-1],(rho_bin[:-1]),'k--')
#plt.plot(rho_bin[:-1],np.exp(rho_bin[0])*rho_bin[1:],'k--',label=r'$\rho\sim\rho$')
plt.xlabel(r'$\langle \rho \rangle$')
plt.ylabel(r'$\langle \log  c_\mathrm{max} \rangle$')
plt.legend(fontsize=6)#


#plt.figure(figsize=(1.5,1.5))
plt.sca(axs[4,3])
#plt.plot(rho,(1./S[idsamples]),'k.',markersize=0.5,c='0.5')
#plt.yscale('log')
plt.xscale('log')
plt.plot(rho_bin[:-1],logcmaxstd_rhobin/logcmaxmean_rhobin,'ko')
plt.plot(rho_bin[:-1],-np.sqrt(Si)/Mu,'r--',label='-$\sigma/\mu$')
#plt.plot(rho_bin[:-1],0.5*(rho_bin[:-1])**0.5,'k--',label=r'$\sqrt{\rho}$')
#plt.plot(rho_bin[:-1],(rho_bin[:-1]),'k--')
#plt.plot(rho_bin[:-1],np.exp(rho_bin[0])*rho_bin[1:],'k--',label=r'$\rho\sim\rho$')
plt.xlabel(r'$\langle \rho \rangle$')
plt.ylabel(r'STD $(\log  c_\mathrm{max})/ \langle \log  c_\mathrm{max} \rangle$')
plt.legend(fontsize=6)


#plt.figure(figsize=(1.5,1.5))
plt.sca(axs[5,0])
#plt.plot(rho,(1./S[idsamples]),'k.',markersize=0.5,c='0.5')
plt.yscale('log')
plt.xscale('log')
plt.plot(rho_bin[:-1],cmax_rhobin,'ko')
#plt.plot(rho_bin[:-1],-Mu,'r--',label='$-\mu$')
#plt.plot(rho_bin[:-1],3e-3/(rho_bin[:-1]),'k--',label='$1/{\rho}$')
#plt.plot(rho_bin[:-1],(rho_bin[:-1]),'k--')
#plt.plot(rho_bin[:-1],np.exp(rho_bin[0])*rho_bin[1:],'k--',label=r'$\rho\sim\rho$')
plt.xlabel(r'$\langle \rho \rangle$')
plt.ylabel(r'$\langle \sum c_\mathrm{max} \rangle$')
plt.legend(fontsize=6)#

plt.sca(axs[5,1])
#plt.plot(rho,(1./S[idsamples]),'k.',markersize=0.5,c='0.5')
#plt.yscale('log')
plt.xscale('log')
plt.plot(rho_bin[:-1],logcmax_rhobin_var,'ko')
#plt.plot(rho_bin[:-1],-Mu,'r--',label='$-\mu$')
#plt.plot(rho_bin[:-1],3e-3/(rho_bin[:-1]),'k--',label='$1/{\rho}$')
#plt.plot(rho_bin[:-1],(rho_bin[:-1]),'k--')
#plt.plot(rho_bin[:-1],np.exp(rho_bin[0])*rho_bin[1:],'k--',label=r'$\rho\sim\rho$')
plt.xlabel(r'$\langle \rho \rangle$')
plt.ylabel(r'Var $ \log \sum c_\mathrm{max}$')
plt.ylim([0,2])
plt.legend(fontsize=6)#



plt.sca(axs[5,2])
#plt.plot(rho,(1./S[idsamples]),'k.',markersize=0.5,c='0.5')
#plt.yscale('log')
plt.xscale('log')
plt.plot(rho_bin[:-1],cmax_rhobin_std,'ko')
#plt.plot(rho_bin[:-1],-Mu,'r--',label='$-\mu$')
#plt.plot(rho_bin[:-1],3e-3/(rho_bin[:-1]),'k--',label='$1/{\rho}$')
#plt.plot(rho_bin[:-1],(rho_bin[:-1]),'k--')
#plt.plot(rho_bin[:-1],np.exp(rho_bin[0])*rho_bin[1:],'k--',label=r'$\rho\sim\rho$')
plt.xlabel(r'$\langle \rho \rangle$')
plt.ylabel(r'Std $\sum c_\mathrm{max} $')
#plt.ylim([0,2])
plt.legend(fontsize=6)#

plt.sca(axs[5,3])
#plt.plot(rho,(1./S[idsamples]),'k.',markersize=0.5,c='0.5')
#plt.yscale('log')
plt.xscale('log')
plt.plot(rho_bin[:-1],cmax_rhobin_std/cmax_rhobin,'ko')
#plt.plot(rho_bin[:-1],-Mu,'r--',label='$-\mu$')
#plt.plot(rho_bin[:-1],3e-3/(rho_bin[:-1]),'k--',label='$1/{\rho}$')
#plt.plot(rho_bin[:-1],(rho_bin[:-1]),'k--')
#plt.plot(rho_bin[:-1],np.exp(rho_bin[0])*rho_bin[1:],'k--',label=r'$\rho\sim\rho$')
plt.xlabel(r'$\langle \rho \rangle$')
plt.ylabel(r'Std $\sum c_\mathrm{max} / \langle \sum c_\mathrm{max} \rangle$')
plt.ylim([0,2])
plt.legend(fontsize=6)#
##plt.plot(rho,(1./S[idsamples]),'k.',markersize=0.5,c='0.5')
#plt.yscale('log')
#plt.xscale('log')
#plt.plot(rho_bin[:-1],cmaxstd_bin2,'ko')
#plt.plot(rho_bin[:-1],cmaxstd_bin3,'bo')
#plt.plot(rho_bin[:-1],cmaxmean_bin2,'ro')
#plt.plot(rho_bin[:-1],Taustd_bin,'go')
#plt.plot(rho_bin[:-1],np.sqrt(rho_bin[:-1])**(-1),'k--',label='$1/\sqrt{\rho}$')
#
#plt.plot(rho_bin[:-1],(rho_bin[:-1]),'k--',label='$1/\sqrt{\rho}$')
#plt.plot(rho_bin[:-1],(rho_bin[:-1])**2,'k--',label='$1/\sqrt{\rho}$')
##plt.plot(rho_bin[:-1],(rho_bin[:-1]),'k--')
##plt.plot(rho_bin[:-1],np.exp(rho_bin[0])*rho_bin[1:],'k--',label=r'$\rho\sim\rho$')
#plt.xlabel(r'$\langle \rho \rangle$')
#plt.ylabel(r'STD $(c_\mathrm{max})$')
##plt.ylim([1e-1,1e1])
###plt.xlim([1e0,1e3])
#plt.legend()


#
##plt.figure(figsize=(1.5,1.5))
#plt.sca(axs[2,1])
##plt.plot(rho,(1./S[idsamples]),'k.',markersize=0.5,c='0.5')
#plt.yscale('log')
#plt.xscale('log')
#plt.plot(rho_bin[:-1],Taustd_bin,'ko')
#plt.plot(rho_bin[:-1],1e8*(rho_bin[:-1])**2,'k--',label='$\rho^2$')
#plt.xlabel(r'$\langle \rho \rangle$')
#plt.ylabel(r'STD $\tau$')
#plt.legend()
#
#plt.figure(figsize=(1.5,1.5))
##plt.plot(rho,(1./S[idsamples]),'k.',markersize=0.5,c='0.5')
#plt.yscale('log')
#plt.xscale('log')
#plt.plot(rho_bin[:-1],Tauvar_bin,'ko')
#plt.plot(rho_bin[:-1],1e19*(rho_bin[:-1])**4,'k--',label='$\rho^4$')
#plt.xlabel(r'$\langle \rho \rangle$')
#plt.ylabel(r'Var $\tau$')
#plt.legend()




#plt.figure(figsize=(1.5,1.5))
# =============================================================================
# #plt.sca(axs[2,2])
# ##plt.plot(rho,(1./S[idsamples]),'k.',markersize=0.5,c='0.5')
# #plt.yscale('log')
# #plt.xscale('log')
# #plt.plot(rho_bin[:-1],Taustd_bin/Taumean_bin,'ko')
# ##plt.plot(rho_bin[:-1],1e8*(rho_bin[:-1])**2,'k--',label='$\rho^2$')
# #plt.xlabel(r'$\langle \rho \rangle$')
# #plt.ylim([1e-1,1e1])
# #plt.ylabel(r'STD $\tau / \langle \tau \rangle$')
# #plt.legend()
# =============================================================================



#plt.figure(figsize=(1.5,1.5))
plt.sca(axs[2,0])
#plt.plot(rho,(1./S[idsamples]),'k.',markersize=0.5,c='0.5')
plt.yscale('log')
plt.xscale('log')
plt.plot(rho_bin[:-1],sqrtTaumean_rhobin,'ko')
plt.plot(rho_bin[:-1],1e4*(rho_bin[:-1]),'k--',label='$\langle \rho \rangle$')

plt.plot(rho_bin[:-1],np.exp(Mu+Si/2),'r--',label='$\mu+\sigma^2/2$')
plt.xlabel(r'$\langle \rho \rangle$')
plt.ylabel(r'$\langle \sqrt{\tau} \rangle$')
plt.legend(fontsize=6)
#plt.xlim([0.5,500])

#plt.figure(figsize=(1.5,1.5))
plt.sca(axs[2,2])
#plt.plot(rho,(1./S[idsamples]),'k.',markersize=0.5,c='0.5')
plt.yscale('log')
plt.xscale('log')
plt.plot(rho_bin[:-1],sqrtTaustd_rhobin,'ko')
plt.plot(rho_bin[:-1],1e4*(rho_bin[:-1]),'k--',label='$\langle \rho \rangle$')
plt.xlabel(r'$\langle \rho \rangle$')
plt.ylabel(r'STD $\sqrt{\tau}$')
plt.legend(fontsize=6)

#plt.xlim([0.5,500])
#plt.figure(figsize=(1.5,1.5))
plt.sca(axs[2,1])
#plt.plot(rho,(1./S[idsamples]),'k.',markersize=0.5,c='0.5')
plt.yscale('log')
plt.xscale('log')
plt.plot(rho_bin[:-1],sqrtTauvar_rhobin,'ko')
plt.plot(rho_bin[:-1],1e8*(rho_bin[:-1])**2,'k--',label='r$\rho^2$')
plt.plot(rho_bin[:-1],np.exp(2*Mu+2*Si),'r--',label='$2\mu+2\sigma^2$')
plt.xlabel(r'$\langle \rho \rangle$')
plt.ylabel(r'Var $\sqrt{\tau}$')
plt.legend(fontsize=6)

#plt.xlim([0.5,500])

#plt.figure(figsize=(1.5,1.5))
plt.sca(axs[2,3])
#plt.plot(rho,(1./S[idsamples]),'k.',markersize=0.5,c='0.5')
plt.yscale('log')
plt.xscale('log')
plt.plot(rho_bin[:-1],sqrtTaustd_rhobin/sqrtTaumean_rhobin,'ko')
plt.plot(rho_bin[:-1],np.exp(Mu+Si)/np.exp(Mu+Si/2),'r--')

#plt.plot(rho_bin[:-1],(rho_bin[:-1])**0.25,'k--',label='$\rho^{1/4}$')
#plt.plot(rho_bin[:-1],1e8*(rho_bin[:-1])**2,'k--',label='$\rho^2$')
plt.xlabel(r'$\langle \rho \rangle$')
plt.ylim([1e-1,1e1])
plt.ylabel(r'STD $\sqrt{\tau} / \langle \sqrt{\tau} \rangle$')
plt.legend(fontsize=6)
#plt.xlim([0.5,500])


plt.savefig(dir_out+'AllResults_rho_{:1.1f}_D{:1.0e}.pdf'.format(A,Brownian),bbox_inches='tight')

#%%% Other plots
plt.figure(figsize=(1.5,1.5))

#plt.plot(rho,(1./S[idsamples]),'k.',markersize=0.5,c='0.5')
plt.yscale('log')
plt.xscale('log')
plt.plot(cm_bin[:-1],cmax_bin_std/cmax_bin,'ko')
#plt.plot(cm_bin[:-1],np.zeros(len(cm_bin)-1)+cm,'k--',label='$\langle c \rangle$')

#plt.plot(nagg_bin[:-1],np.exp(rho_bin[0])*nagg_bin[1:],'k--',label=r'$N\sim\rho$')
plt.xlabel(r'$c_\mathrm{max}$')
plt.ylabel(r'STD $(\sum_N c_\mathrm{max}) / \langle \sum_N c_\mathrm{max} \rangle$')
plt.ylim([1e-1,1e1])
plt.xlim([1e-6,1e0])

#rho
fractal=1.7
plt.figure(figsize=(2,2))
#plt.plot(nagg,(1./S[idsamples]),'k.',markersize=0.5,c='0.5')
plt.yscale('log')
plt.xscale('log')
plt.plot(nagg_bin[:-1],np.exp(logrhomean_naggbin),'ko')
#plt.plot(nagg_bin[:-1],np.exp(logrho_bin[0])*nagg_bin[1:],'k--',label=r'$N\sim\rho$')
plt.plot(nagg_bin[:-1],1.5e3*nagg_bin[1:]**(1/(fractal-1)),'k--',label=r'$\rho \sim N^{1/(D_f-1)}$')
plt.xlabel(r'$N$')
plt.ylabel(r'$\langle \rho \rangle$')
#plt.ylim([1e2,1e7])
#plt.xlim([1e-1,1e3])
#plt.xticks([1,10,100])
plt.legend()
plt.savefig(dir_out+'./Compare_stretching_concentration/rhomean_Nbin.pdf',
						bbox_inches='tight')


plt.figure(figsize=(1.5,1.5))
#plt.plot(nagg,(1./S[idsamples]),'k.',markersize=0.5,c='0.5')
plt.yscale('log')
plt.xscale('log')
plt.plot(nagg_bin[:-1],cmaxstd_bin,'ko')
plt.plot(nagg_bin[:-1],np.sqrt(nagg_bin[:-1]),'k--',label='$\sqrt{N}$')
#plt.plot(nagg_bin[:-1],(nagg_bin[:-1]),'k--')
#plt.plot(nagg_bin[:-1],np.exp(rho_bin[0])*nagg_bin[1:],'k--',label=r'$N\sim\rho$')
plt.xlabel(r'$N$')
plt.ylabel(r'STD $(c_\mathrm{max}) / \langle c_\mathrm{max} \rangle$')
plt.ylim([1e-1,1e1])
plt.xlim([1e0,1e3])
plt.legend()

plt.savefig(dir_out+'./Compare_stretching_concentration/lamellaSTD_l{:1.1f}_rad{:1.2f}_{:1.0e}.pdf'.format(l0,radius,Brownian),
						bbox_inches='tight')


plt.figure(figsize=(1.5,1.5))
#plt.plot(nagg,(1./S[idsamples]),'k.',markersize=0.5,c='0.5')
plt.yscale('log')
plt.xscale('log')
plt.plot(nagg_bin[:-1],cmax_naggbin,'ko',label='$\langle c \rangle$')
plt.plot(nagg_bin[:-1],np.zeros(len(cmax_naggbin))+cm,'k--',label='$\langle c \rangle$')
#plt.plot(nagg_bin[:-1],np.exp(rho_bin[0])*nagg_bin[1:],'k--',label=r'$N\sim\rho$')
plt.xlabel(r'$N$')
plt.ylabel(r'$\langle \sum_N c_\mathrm{max} \rangle$')
plt.ylim([1e-3,1e-1])
plt.savefig(dir_out+'./Compare_stretching_concentration/lamellameanN_l{:1.1f}_rad{:1.2f}_{:1.0e}.pdf'.format(l0,radius,Brownian),
						bbox_inches='tight')
#plt.xticks([1,10,100])


plt.figure(figsize=(2,1.5))
[plt.plot(pdf_c[1:],cmax_pdf_bin[n,:],'.',color=plt.cm.jet(n/cmax_pdf_bin.shape[0]),label=r'$N={:1.1e}$'.format(nagg_bin[n])) for n in range(0,cmax_pdf_bin.shape[0],2)]
plt.plot(pdf_c[1:],cmax_pdf_all,'k-')
plt.yscale('log')
plt.xscale('log')
plt.xlabel('$c_\mathrm{max}$')
plt.ylabel('pdf')

plt.savefig(dir_out+'./Compare_stretching_concentration/lamellamPDF_l{:1.1f}_rad{:1.2f}_{:1.0e}.pdf'.format(l0,radius,Brownian),
						bbox_inches='tight')
#plt.legend(fontsize=5)

plt.figure(figsize=(1.5,1.5))
#plt.plot(nagg,(1./S[idsamples]),'k.',markersize=0.5,c='0.5')
plt.yscale('log')
plt.xscale('log')
plt.plot(nagg_bin[:-1],cmax_naggbin_std/cmax_naggbin,'ko')
#plt.plot(nagg_bin[:-1],np.exp(rho_bin[0])*nagg_bin[1:],'k--',label=r'$N\sim\rho$')
plt.xlabel(r'$N$')
plt.ylabel(r'STD $(\sum_N c_\mathrm{max}) / \langle \sum_N c_\mathrm{max} \rangle$')
plt.ylim([1e-1,1e1])
plt.xlim([1e0,1e2])
plt.savefig(dir_out+'./Compare_stretching_concentration/lamellaSTDN_l{:1.1f}_rad{:1.2f}_{:1.0e}.pdf'.format(l0,radius,Brownian),
						bbox_inches='tight')
#plt.xticks([1,10,100])

plt.figure(figsize=(1.5,1.5))
#plt.plot(nagg,(1./S[idsamples]),'k.',markersize=0.5,c='0.5')
plt.yscale('log')
plt.xscale('log')
plt.plot(cm_bin[:-1],cmax_bin,'ko')
plt.plot(cm_bin[:-1],np.zeros(len(cm_bin)-1)+cm,'k--',label='$\langle c \rangle$')

#plt.plot(nagg_bin[:-1],np.exp(rho_bin[0])*nagg_bin[1:],'k--',label=r'$N\sim\rho$')
plt.xlabel(r'$c_\mathrm{max}$')
plt.ylabel(r'$\langle \sum_N c_\mathrm{max} \rangle$')
plt.ylim([1e-3,1e0])
plt.xlim([1e-6,1e0])

plt.savefig(dir_out+'./Compare_stretching_concentration/lamellamean_l{:1.1f}_rad{:1.2f}_{:1.0e}.pdf'.format(l0,radius,Brownian),
						bbox_inches='tight')


plt.figure(figsize=(1.5,1.5))
#plt.plot(nagg,(1./S[idsamples]),'k.',markersize=0.5,c='0.5')
plt.yscale('log')
plt.xscale('log')
plt.plot(nagg_bin[:-1],(rho_bin),'ko')
plt.plot(nagg_bin[:-1],(rho_bin[0])*nagg_bin[1:],'k--',label=r'$N\sim\rho$')
plt.plot(nagg_bin[:-1],(rho_bin[0])*nagg_bin[1:]**(5/4),'k--',label=r'$N\sim\rho$')
plt.xlabel(r'$N$')
plt.ylabel(r'$\rho$')
plt.ylim([1e2,1e7])
plt.xlim([1e-1,1e3])
#plt.xticks([1,10,100])
plt.legend()

#%%% Effect of sB on bundle statistics

A=1/np.sqrt(2)
l0=0.3
dt=0.25
t=12
f=h5py.File('./Compare_stretching_concentration/DSM_A{:1.1f}_l{:1.1f}.hdf5'.format(A,l0), 'r')
L=f['L_{:04d}'.format(int(t*10))][:]
wrapped_time=f['wrapped_time_{:04d}'.format(int(t*10))][:]
S=f['S_{:04d}'.format(int(t*10))][:]
W=f['Weights_{:04d}'.format(int(t*10))][:]

Brownian=1e-3
D=(Brownian)**2./(2*dt)#+1e-08 # numerical diffusion ?
#*np.sqrt(2)
sB=np.sqrt(D/0.5)
#sB=0.0005
dx=0.001 # Maximum distance between points, above which the line will be refined

sBv=[0.01,0.001, 0.0001]
Nagg_bin,Rho_bin=[],[]
TNagg_bin,TRho_bin=[],[]
for sB in sBv:
	tree=spatial.cKDTree(np.mod(L,1))
	nsamples=int(sB/dx*10)
	nsamples=10000
	idsamples=np.uint32(np.linspace(0,L.shape[0]-2,nsamples))
	neighboors=tree.query_ball_point(np.mod(L[idsamples,:],1), sB)
	neighboors_uniq=[[] for k in range(len(idsamples))]
	neighboors_all=[[] for k in range(len(idsamples))]
	for ii,ns in enumerate(idsamples):		
		nt=np.sort(neighboors[ii])
		kk_all=np.concatenate((nt,np.array([ns])))
		neighboors_all[ii].append(kk_all)
		idgood=np.where(np.diff(nt)>3*sB/dx)[0]
		NT=[]
		for idg in idgood:
			if (np.abs(nt[idg+1]-ns)>3*sB/dx)&(np.abs(nt[idg]-ns)<3*sB/dx): # depending on the direction of the filament
				NT.append(nt[idg+1])
			else:
				NT.append(nt[idg])
		nt=np.array(NT,dtype=np.uint32)
		kk=np.concatenate((nt,np.array([ns])))
		neighboors_uniq[ii].append(kk)
	# Binning as a function of N
	nagg=np.array([len(n[0]) for n in neighboors_uniq])
	nagg_bin=np.logspace(0,np.max(np.log10(nagg)),40)
	rho_bin=bin_operation(nagg,(1./S[idsamples]),nagg_bin,np.nanmean)
	Nagg_bin.append(nagg_bin)
	Rho_bin.append(rho_bin)

sB=0.001
tv=[10,11,12]
for t in tv:
	L=f['L_{:04d}'.format(int(t*10))][:]
	wrapped_time=f['wrapped_time_{:04d}'.format(int(t*10))][:]
	S=f['S_{:04d}'.format(int(t*10))][:]
	W=f['Weights_{:04d}'.format(int(t*10))][:]
	tree=spatial.cKDTree(np.mod(L,1))
	nsamples=int(sB/dx*10)
	nsamples=10000
	idsamples=np.uint32(np.linspace(0,L.shape[0]-2,nsamples))
	neighboors=tree.query_ball_point(np.mod(L[idsamples,:],1), sB)
	neighboors_uniq=[[] for k in range(len(idsamples))]
	neighboors_all=[[] for k in range(len(idsamples))]
	for ii,ns in enumerate(idsamples):		
		nt=np.sort(neighboors[ii])
		kk_all=np.concatenate((nt,np.array([ns])))
		neighboors_all[ii].append(kk_all)
		idgood=np.where(np.diff(nt)>3*sB/dx)[0]
		NT=[]
		for idg in idgood:
			if (np.abs(nt[idg+1]-ns)>3*sB/dx)&(np.abs(nt[idg]-ns)<3*sB/dx): # depending on the direction of the filament
				NT.append(nt[idg+1])
			else:
				NT.append(nt[idg])
		nt=np.array(NT,dtype=np.uint32)
		kk=np.concatenate((nt,np.array([ns])))
		neighboors_uniq[ii].append(kk)
	# Binning as a function of N
	nagg=np.array([len(n[0]) for n in neighboors_uniq])
	nagg_bin=np.logspace(0,np.max(np.log10(nagg)),40)
	rho_bin=bin_operation(nagg,(1./S[idsamples]),nagg_bin,np.nanmean)
	TNagg_bin.append(nagg_bin)
	TRho_bin.append(rho_bin)


plt.figure(figsize=(1.5,1.5))
[plt.plot(Nagg_bin[i][1:],Rho_bin[i],'o',label='$s_{s_B}={:1.0e}$'.format(sBv[i])) for i in range(len(sBv))]
[plt.plot(TNagg_bin[i][1:],TRho_bin[i],'+',label='$t={:1.0f}$'.format(tv[i])) for i in range(len(tv))]
plt.plot(np.array([0,1000]),np.array([0,1000])*1e4,'k--',label=r'$N\sim \rho$')
plt.xscale('log')
plt.yscale('log')
plt.xlim([1,10000])
plt.xlabel(r'$N$')
plt.ylabel(r'$\rho$')
plt.legend(fontsize=4)


#%% GLOBAL PDF 
#%%% PDF of aggregated 1/s

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
#%%%% various a

M=['d','o','s','*']
ms='none'
sB=1/50

fig=plt.figure(figsize=(3,2))
nb=150
factor=[1e2,2e2,5e2]
for i in range(3):
	L1=L[i]
	S1=S[i]
	Lmod=np.mod(L1,1)
	dist_old=np.sum(np.diff(L1,axis=0)**2,1)**0.5
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
plt.xlabel(r'$c / (\theta_0 s_0 / s_B)$')
plt.legend()
plt.xlim([1e-4,5e-1])
plt.ylim([1e-2,1e3])
subscript(plt.gca(),0)
plt.savefig(figdir+'Sine_pdf_A.pdf',bbox_inches='tight')



#%%%% various sB
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
	plt.plot(x[1:],h,M[i],color=plt.cm.viridis(i/3),label=r'$s_B=1/{:1.0f}$'.format(1/sB),fillstyle=ms)

if THEORY:
	from scipy import special
	Lyap=np.loadtxt('Sine_Lyap.txt')
	D1=np.loadtxt('Sine_D1.txt')
	for i in range(0,3):
		sB=SB[i]
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
		Ksi=np.loadtxt('Sine_scaling_C|N_sB1_{:1.0f}.txt'.format(50))
		ksi=np.interp(d1,Ksi[:,0],Ksi[:,1])
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

plt.xlim([1e-5,1e-1])
plt.ylim([1e-5,1e4])
plt.xlabel(r'$c / (\theta_0 s_0 / s_B)$')
plt.legend(loc=3)

subscript(plt.gca(),1)
plt.savefig(figdir+'Sine_pdf_sB.pdf',bbox_inches='tight')


#%%%% various times
theta0=1
s0=1
A=0.9
s=3

Lmax=[1e6,1e7,1e8]
L1,S1,wrapped_time,W1,t1=run_DSM(Lmax[0],A,s,STOP_ON_LMAX=True)
L2,S2,wrapped_time,W2,t2=run_DSM(Lmax[1],A,s,STOP_ON_LMAX=True)
L3,S3,wrapped_time,W3,t3=run_DSM(Lmax[2],A,s,STOP_ON_LMAX=True)


L=[L1,L2,L3]
S=[S1,S2,S3]
W=[W1,W2,W3]
t=[t1,t2,t3]
#%%% Theory various time
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
plt.plot([],[],'k--',label='Fully Random')
plt.plot([],[],'k-',label='Correlated')
plt.xlim([1e-5,5e-1])
plt.ylim([1e-2,1e4])
plt.xlabel(r'$c / (\theta_0 s_0 / s_B)$')
plt.legend(loc=3,fontsize=8,frameon=False)

#subscript(plt.gca(),1)
plt.savefig(figdir+'Sine_pdf_time.pdf',bbox_inches='tight')
#%%% various theory
theta0=1
s0=0.01
A=0.9
s=3

L,S,wrapped_time,W,t=run_DSM(1e8,A,s,STOP_ON_LMAX=True)

#%%%% Plot
nb=100 
l0=0.3
ms='none'
factor=[1,1,1]
THEORY=True
THEORY_RANDOM=True
THEORY_CORREL=True

M=['d','o','s','*']
SB=[1/50,1/100,1/500]
fig,ax=plt.subplots(1,3,figsize=(7,2))
sB=1/50
Time=[2]
Lyap=np.loadtxt('Sine_Lyap.txt')
lyap=np.interp(A,Lyap[:,0],Lyap[:,1])
sigma2=np.interp(A,Lyap[:,0],Lyap[:,2])
for i in Time:
	Lmod=np.hstack((np.mod(L[:,1],1).reshape(-1,1),np.mod(-L[:,0],1).reshape(-1,1)))
	idmod=np.where(np.nansum(np.diff(Lmod,axis=0)**2.,axis=1)>0.01)[0]
	Lmod[idmod,:]=np.nan
	ax[0].plot(Lmod[:,0],Lmod[:,1],'k-',linewidth=0.01)
	ax[0].axis('off')
	ax[0].set_aspect('equal')
	ax[0].set_xlim([0,1])
	ax[0].set_ylim([0,1])
	Lmod=np.mod(L,1)
	dist_old=np.sum(np.diff(L,axis=0)**2,1)**0.5
	C=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1,sB)
										 ,weights=S*dist_old/sB,density=False)[0]
	h,x=np.histogram(C,np.logspace(-6,0,nb),density=True)
	ax[1].imshow(C,clim=[0.002,0.010],cmap=plt.cm.Greys)
	#plt.colorbar()
	ax[1].axis('off')
	ax[2].plot(x[1:],h,'o',color=plt.cm.cool(2/3),label=r'Simulation',fillstyle=ms)
	#h,x=np.histogram(S,np.logspace(-9,0,100),density=True)
#	plt.plot(x[1:],factor[i]*h,'--',color=plt.cm.cool(i/3))

ax2 = fig.add_axes([0.5, 0.1, 0.1, 0.05])
import matplotlib as mpl
#norm = mpl.colors.Normalize(vmin=T_all[0]*u_pore/d, vmax=T_all[-1]*u_pore/d)
norm = mpl.colors.Normalize(vmin=0.2, vmax=1)
cb1 = mpl.colorbar.ColorbarBase(ax2,cmap=plt.cm.Greys,norm=norm,
								orientation='horizontal')
cb1.set_label(r'$c (10^{-2})$', color='k',labelpad=0)
cb1.set_ticks([0.2,0.6,1.0])

if THEORY_RANDOM:
	from scipy import special
	Lyap=np.loadtxt('Sine_Lyap.txt')
	D1=np.loadtxt('Sine_D1.txt')
	for i in Time:
		Ltot=l0*np.exp((lyap+sigma2/2)*t)
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
			ax[2].plot(C,pdf_random,'--',color='indianred',label='Fully Random')


if THEORY_CORREL:
	from scipy import special
	Lyap=np.loadtxt('Sine_Lyap.txt')
	D1=np.loadtxt('Sine_D1.txt')
	s0=0.01
	for i in Time:
		C=np.logspace(-5,0,200)
		h=1/C/np.sqrt(2*np.pi*sigma2*t)*np.exp(-(np.log(C)+lyap*t)**2/(2*sigma2*t))
		#h,x=np.histogram(s0/sB*S,C,weights=W,density=True)
		h=h/np.sum(h[1:]*np.diff(C+l0*sB/1))
		ax[2].plot(C+l0*sB/1,h,':',color='darkorange', label='Fully correlated')

if THEORY:
	from scipy import special
	Lyap=np.loadtxt('Sine_Lyap.txt')
	D1=np.loadtxt('Sine_D1.txt')
	for i in Time:
		L1=L[i]
		S1=S[i]
		Lmod=np.mod(L1,1)
#			sB=SB[i]
		n=np.logspace(0.1,6,1000)
		lyap=np.interp(A,Lyap[:,0],Lyap[:,1])
		sigma2=np.interp(A,Lyap[:,0],Lyap[:,2])
		d1=np.interp(A,D1[:,0],D1[:,1])
		# Theory
		#dist_old=np.sum(np.diff(L,axis=0)**2,1)**0.5
		#Ltot=np.sum(dist_old)
		Ltot=l0*np.exp((lyap+sigma2/2)*t)
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
		ksi=-np.interp(d1,Ksi[:,0],Ksi[:,1])
		mu_c_n=l0*sB/1
		#
		#tc=1/lyap*np.log(1/l0/sB)
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
		
		C=np.logspace(-5,-1,200)
		pdf_c=np.array([convolution_lognormal(c) for c in C])
		pdf_c=pdf_c/np.sum(pdf_c*np.diff(np.log(C))[0]*C)
		ax[2].plot(C,pdf_c,'-',color='blueviolet',label='Correlated')
		ax[2].set_aspect(0.45)
		#print(pdf_c)

ax[2].set_yscale('log')
ax[2].set_xscale('log')
ax[2].set_xlim([1e-3,5e-1])
ax[2].set_ylim([1e-2,1e4])
ax[2].set_xlabel(r'$c$')
ax[2].legend(loc=1,fontsize=6,frameon=False)

subscript(ax[0],0,color='k')
subscript(ax[1],1)
subscript(ax[2],2)
plt.savefig(figdir+'Sine_pdf_models.png',bbox_inches='tight',dpi=300)

#%%

C=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1,sB)
									 ,weights=S1*dist_old/sB,density=False)[0]
h,x=np.histogram(C,np.logspace(-6,0,nb),density=True)

plt.imshow(C,clim=[0.002,0.010],cmap=plt.cm.Greys)
plt.axis('off')
plt.colorbar()
plt.savefig(figdir+'Sine_pdf_concentration.pdf',bbox_inches='tight')

#%%% Negative moments of truncated Gamma distrib with increasing scale

from scipy import special
n=np.logspace(0,10,1000)

D2=1.7
sB=1/50

k_n=1/(sB**(D2-2)-1)
Xi=np.linspace(-2,0,20)
print(k_n)

T=np.linspace(0,20,20)
moment=[]
for t in T:
	theta_n=np.exp(t)*(sB**(D2-2)-1)
	pdf_n=1/special.gamma(k_n)/theta_n**k_n*n**(k_n-1)*np.exp(-n/theta_n)
	pdf_n=pdf_n/np.sum(pdf_n*np.diff(np.log(n))[0]*n)
	moment.append([np.sum(pdf_n*n**i*np.diff(np.log(n))[0]*n) for i in Xi])
	plt.plot(n,pdf_n); plt.yscale('log'); plt.xscale('log')
	plt.ylim([1e-20,1e5])

moment=np.array(moment)
plt.figure()
plt.plot(T,moment)
plt.plot(T,np.exp(T*-k_n),'k--')
[plt.plot(T,np.exp(T*(xi)),'--') for xi in Xi[Xi>-k_n]]
plt.yscale('log')

#%%% Theoretical pdf, various t
from scipy import special
sB=1/100
Lyap=np.loadtxt('Sine_Lyap.txt')
D1=np.loadtxt('Sine_D1.txt')
A=1.2
l0=0.3
TT=np.arange(5,10)
for i,t in enumerate(TT):
	lyap=np.interp(A,Lyap[:,0],Lyap[:,1])
	sigma2=np.interp(A,Lyap[:,0],Lyap[:,2])
	n=np.logspace(0,10,1000)
	d1=np.interp(A,D1[:,0],D1[:,1])
	# Theory
	Ltot=l0*np.exp((lyap+sigma2/2)*t)
	mu_n=sB*Ltot
	sigma2_n=mu_n**2.*(2*np.log(1/sB))*(2-d1)/(d1-1)
	# gamma
#	k=mu_n**2./sigma2_n
	k=(d1-1)/(2*np.log(1/sB)*(2-d1))
#	theta=sigma2_n/mu_n
	theta=mu_n*np.log(1/sB)*2*(2-d1)/(d1-1)
	pdf_n=1/special.gamma(k)/theta**k*n**(k-1)*np.exp(-n/theta)
	# Normalize
	dn=np.diff(np.log(n))[0]*n
	pdf_n=pdf_n/np.sum(pdf_n*dn)
#	plt.plot(n,pdf_n)
	l0=0.3
	epsilon=epsilon_d1(d1)
	ksi=-1+epsilon
#	ksi=-0.7
	# true ksi
	#		Ksi=np.loadtxt('Sine_scaling_C|N_sB1_{:1.0f}.txt'.format(50))
	#		ksi=-np.interp(d1,Ksi[:,0],Ksi[:,1])
	mu_c_n=l0*sB/1
	#sigma2_c_n=(l0*sB/1)**2.*(n**(ksi)-n**-1)
	sigma2_c_n=(l0*sB/1)**2.*(n**(ksi)) 
	print(np.mean(sigma2_c_n*pdf_n*dn))
	# pdf_c_n is gamma
	k_c_n=mu_c_n**2./sigma2_c_n
	theta_c_n=sigma2_c_n/mu_c_n
#	k_c_n=mu_c_n**2./sigma2_c_n
#	theta_c_n=sigma2_c_n/mu_c_n
	def convolution_gamma(c):
		import scipy.integrate
		pdf_c_n=1/special.gamma(k_c_n)/theta_c_n**k_c_n*c**(k_c_n-1)*np.exp(-c/theta_c_n)
		idgood=np.isfinite(pdf_c_n)
		return np.nansum(pdf_c_n[idgood]*pdf_n[idgood]*dn[idgood])
	# pdf_c_n is lognormal
	mu_logc_n=np.log(mu_c_n/np.sqrt(sigma2_c_n/mu_c_n**2.+1))
	sigma2_logc_n=np.log(sigma2_c_n/mu_c_n**2.+1)
	def convolution_lognormal(c):
		import scipy.integrate
		pdf_c_n=1/np.sqrt(2*np.pi*sigma2_logc_n)/c*np.exp(-(np.log(c)-mu_logc_n)**2./2/sigma2_logc_n)
		return np.nansum(pdf_c_n*pdf_n*dn)
	C=np.logspace(-10,0,1000)
	dc=np.diff(np.log(C))[0]*C
	pdf_c=np.array([convolution_lognormal(c) for c in C])
	pdf_c=pdf_c/np.sum(pdf_c*np.diff(np.log(C))[0]*C)
	plt.plot(C,pdf_c,'-',color=plt.cm.cool(i/len(T)))
	print(np.mean(C**2*pdf_c*dc))
plt.yscale('log')
plt.xscale('log')

plt.xlim([1e-5,1e-1])
plt.ylim([1e-5,1e4])
plt.xlabel(r'$c / (\theta_0 s_0 / s_B)$')
plt.legend(loc=3)
#%%% Keff and Krand
from scipy import special
k_rand,k_rand2,k_eff,k_min=[],[],[],[]
Lyap=np.loadtxt('Sine_Lyap.txt')
D1=np.loadtxt('Sine_D1.txt')
DD1=np.linspace(1.7,1.9,100)
n=np.logspace(0,25,1000)
sB=1/20
for d1 in DD1:
	A=np.interp(d1,D1[:,1],D1[:,0])
	lyap=np.interp(A,Lyap[:,0],Lyap[:,1])
	sigma2=np.interp(A,Lyap[:,0],Lyap[:,2])
	k_rand.append(lyap+sigma2/2)
	k_rand2.append(lyap)
	k_min.append(lyap**2/(2*sigma2))
	# Effective k
	# Theory
	k_eff_t=[]
	TT=np.linspace(5,10,10)
	l0=0.3
	epsilon=epsilon_d1(d1)
	ksi=-1+epsilon
	Ksi=np.loadtxt('Sine_scaling_C|N_sB1_{:1.0f}.txt'.format(50))
#	ksi=-np.interp(d1,Ksi[:,0],Ksi[:,1])
	print(ksi)
	for t in TT:
		Ltot=l0*np.exp((lyap+sigma2/2)*t)
		mu_n=sB*Ltot
		sigma2_n=mu_n**2.*(2*np.log(1/sB))*(2-d1)/(d1-1)
		# gamma
	#	k=mu_n**2./sigma2_n
		k=(d1-1)/(2*np.log(1/sB)*(2-d1))
	#	theta=sigma2_n/mu_n
		theta=mu_n*np.log(1/sB)*2*(2-d1)/(d1-1)
		if k<10: #Gamma
			pdf_n=1/special.gamma(k)/theta**k*n**(k-1)*np.exp(-n/theta)
			
		else: # Gaussian
			pdf_n=1/(2*np.pi*sigma2_n)*np.exp(-(n-mu_n)**2./(2*sigma2))
		# Normalize
		dn=np.diff(np.log(n))[0]*n
		pdf_n=pdf_n/np.sum(pdf_n*dn)
	#	plt.plot(n,pdf_n)
	#	ksi=-0.7
		# true ksi
		#		Ksi=np.loadtxt('Sine_scaling_C|N_sB1_{:1.0f}.txt'.format(50))
		#		ksi=-np.interp(d1,Ksi[:,0],Ksi[:,1])
		mu_c_n=l0*sB/1
		sigma2_c_n=(l0*sB/1)**2.*(n**(ksi)-n**-1)
		k_eff_t.append(1/np.nansum((n**(ksi)-n**-1)*pdf_n*dn))
	#plt.plot(TT,np.log(k_eff_t))
	k_eff.append(np.polyfit(TT,np.log(k_eff_t),1)[0])

plt.figure()
epsilon=np.array([epsilon_d1(d1) for d1 in DD1])
ksi=-1+epsilon
G=np.loadtxt('Sine_scaling_N_sB1_{:1.0f}.txt'.format(1/50))
epsilon=np.interp(DD1,G[:,0],2-G[:,2])
ksi=-1+epsilon
Ksi=np.loadtxt('Sine_scaling_C|N_sB1_{:1.0f}.txt'.format(50))
ksi=-np.interp(DD1,Ksi[:,0],Ksi[:,1])
k=(DD1-1)/(2*np.log(1/sB)*(2-DD1))
#theta=mu_n*np.log(1/sB)*2*(2-DD1)/(DD1-1)
A=np.interp(DD1,D1[:,1],D1[:,0])
lyap=np.interp(A,Lyap[:,0],Lyap[:,1])
sigma2=np.interp(A,Lyap[:,0],Lyap[:,2])
plt.plot(DD1,(lyap+sigma2/2)*(k-ksi))
plt.plot(DD1,(lyap+sigma2/2)*(-ksi),'r--')
plt.plot(DD1,lyap+sigma2/2,'k-',label='Random aggregation')
plt.plot(DD1,lyap,'k--',label='Random aggregation (no variability)')
plt.plot(DD1,2*(lyap-sigma2),'g--',label='Fully Correlated')
#plt.plot(DD1,lyap**2./(2*sigma2),'g-',label='Fully Correlated (1)')
plt.plot(DD1,k_eff,'r-',label='Correlated aggregation')
#plt.plot(DD1,lyap**2/(2*sigma2),'b--',label='Isolated strip')
plt.plot(DD1,(lyap-sigma2/2),'b-',label='Isolated strip')
plt.legend(frameon=False)
plt.ylabel('$\gamma_k$')

#%%
d1=1.8
mu_n=10000
k=(d1-1)/(2*np.log(1/sB)*(2-d1))
theta=mu_n*np.log(1/sB)*2*(2-d1)/(d1-1)
n=np.logspace(0,20,1000)
pdf_n=1/special.gamma(k)/theta**k*n**(k-1)*np.exp(-n/theta)
dn=np.diff(np.log(n))[0]*n
print(np.nansum(n**(-1)*pdf_n*dn))
print(1/theta/(k-1)) # moment i fintegrable
plt.plot(n,pdf_n);plt.yscale('log');plt.xscale('log')
plt.plot(n,n**(-1)*pdf_n);plt.yscale('log');plt.xscale('log')

a=0.01
n=np.logspace(-10,10,1000)
plt.plot(n-1,n**(-1+a)-n**-1)
#[plt.plot(n,n**(-1+a)-n**-1) for a in np.logspace(-3,-1,10)]
plt.yscale('log');
plt.xscale('log');
plt.plot(n,1/n,'k--')
plt.plot(n,n,'k--')
plt.plot(n,n**(-1+a),'r--')
plt.plot(n,(a*n)**(-1+a),'k-')
#%%% Keff and Krand Baker
k_rand,k_rand2,k_eff,k_min,k_var=[],[],[],[],[]
A=np.linspace(0.1,0.45,100)
Lyap=-A*np.log(A)-(1-A)*np.log(1-A)
Sigma2=(A*np.log(A)**2.+(1-A)*np.log(1-A)**2.-(-A*np.log(A)-(1-A)*np.log(1-A))**2.)
DD1=1+2*np.log(2)/np.log(A**(-1)+(1-A)**(-1))
sB=1/50
for i,d1 in enumerate(DD1):
	lyap=Lyap[i]
	sigma2=Sigma2[i]
	k_rand.append(lyap+sigma2/2)
	k_rand2.append(lyap)
	k_min.append(-np.log(1-3*A[i]+3*A[i]**2.))
	k_var.append(-np.log(1-2*A[i]+2*A[i]**2.))
	# Effective k
	# Theory
	k_eff_t=[]
	k_eff_t2=[]
	TT=np.linspace(30,32,2)
	n=np.logspace(0,30,1000)
	dn=np.diff(np.log(n))[0]*n
	for t in TT:
		Ltot=2**t
		mu_n=sB*Ltot
		sigma2_n=mu_n**2.*(2*np.log(1/sB))*(2-d1)/(d1-1)
		# gamma
	#	k=mu_n**2./sigma2_n
		k=(d1-1)/(2*np.log(1/sB)*(2-d1))
	#	theta=sigma2_n/mu_n
		theta=mu_n*np.log(1/sB)*2*(2-d1)/(d1-1)
		if k<50: #Gamma
			pdf_n=1/special.gamma(k)/theta**k*n**(k-1)*np.exp(-n/theta)
		else: # Gaussian
			pdf_n=1/(2*np.pi*sigma2_n)*np.exp(-(n-mu_n)**2./(2*sigma2_n))
		# Normalize
		pdf_n=pdf_n/np.sum(pdf_n*dn)
	#	plt.plot(n,pdf_n)
		l0=0.3
		epsilon=epsilon_d1(d1)
#		epsilon=2-d1
		ksi=-1+epsilon
	#	ksi=-0.7
		# true ksi
		#		Ksi=np.loadtxt('Sine_scaling_C|N_sB1_{:1.0f}.txt'.format(50))
		#		ksi=-np.interp(d1,Ksi[:,0],Ksi[:,1])
		mu_c_n=l0*sB/1
		sigma2_c_n=(l0*sB/1)**2.*(n**(ksi)-n**-1)
		#.append(1/(np.sum((n**(ksi))*pdf_n*dn)-np.sum((n**(-1))*pdf_n*dn)))
		k_eff_t.append(1/(np.sum((n**(ksi)-n**(-1))*pdf_n*dn)))
		k_eff_t2.append(np.sum((n**(-1))*pdf_n*dn))
	plt.plot(TT,np.array(k_eff_t),'*-');plt.yscale('log')
	k_eff.append(np.polyfit(TT,np.log(k_eff_t),1)[0])
	#k_eff.append(np.log(k_eff_t)[-1]/(TT[-1]))
k=(DD1-1)/(2*np.log(1/sB)*(2-DD1))
epsilon=np.array([epsilon_d1(d1) for d1 in DD1])
ksi=-1+epsilon
#ksi=1-DD1

plt.figure()
id_int=k<-ksi
plt.plot(DD1[id_int],-np.log(2)*(k[id_int]+ksi[id_int]),'*')
plt.plot(DD1[~id_int],np.log(2)*(-ksi[~id_int]),'+')
plt.plot(DD1,np.zeros(DD1.shape)+np.log(2),'k-',label='Random aggregation')
plt.plot(DD1,Lyap,'k--',label='Random aggregation (no variability)')
plt.plot(DD1,k_eff,'r-',label='Correlated aggregation')
plt.plot(DD1,k_min,'b--',label='Isolated strip')
plt.plot(DD1,k_var,'b-',label='Isolated strip')

plt.legend(frameon=False)
plt.ylabel('$\gamma_k$')
plt.xlabel('$D_1$')
plt.ylim([0,1.5])
#%%% Theoretical PDF
A=np.linspace(0.3,1.8,20)
from scipy import special
Lyap=np.loadtxt('Sine_Lyap.txt')
D1=np.loadtxt('Sine_D1.txt')
t=15
l0=1.
SB=1/np.logspace(1,3,10)
SB=np.array([1/100])
Coeff=[]

for sB in SB:
	thetafit=[]
	kfit=[]
	epsfit=[]
	
	for i,a in enumerate(A):
		n=np.linspace(1,1000,1000)
		lyap=np.interp(A[i],Lyap[:,0],Lyap[:,1])
		sigma2=np.interp(A[i],Lyap[:,0],Lyap[:,2])
		d1=np.interp(A[i],D1[:,0],D1[:,1])
		# Theory
		#dist_old=np.sum(np.diff(L1,axis=0)**2,1)**0.5
		Ltot=l0*np.exp((lyap+sigma2/2)*t)
		mu_n=sB*Ltot
		sigma2_n=mu_n**2.*(-np.log(sB))*2*(2-d1)/(d1-1)
		# gamma
		k=mu_n**2./sigma2_n
		theta=sigma2_n/mu_n
		pdf_n=1/special.gamma(k)/theta**k*n**(k-1)*np.exp(-n/theta)
		# Normalize
		pdf_n=pdf_n/np.sum(pdf_n*np.diff(n)[0])
		l0=0.3
		epsilon=epsilon_d1(d1)
		mu_c_n=l0*sB/1
		sigma2_c_n=(l0*sB/1)**2.*(n**(-1+epsilon)-1/n)
		# pdf_c_n is gamma
		k_c_n=mu_c_n**2./sigma2_c_n
		theta_c_n=sigma2_c_n/mu_c_n
		def convolution(c):
			import scipy.integrate
			pdf_c_n=1/special.gamma(k_c_n)/theta**k_c_n*c**(k_c_n-1)*np.exp(-c/theta_c_n)
			return np.sum(pdf_c_n*pdf_n*np.diff(n)[0])
		C=np.logspace(-5,-1,100)
		pdf_c=np.array([convolution(c) for c in C])
		pdf_c=pdf_c/np.sum(pdf_c[1:]*np.diff(C))
		plt.plot(C,pdf_c,'-',color=plt.cm.cool(i/len(A)))
		
		cfit=30
		# get exponential decay
		thetafit.append([theta,-np.polyfit(C[-cfit:],np.log(pdf_c[-cfit:]),1)[0]])
	
		# get powerlaw 
		kfit.append([k,np.polyfit(np.log(C[:cfit]),np.log(pdf_c[:cfit]),1)[0]])
	
		epsfit.append(epsilon)
	#	plt.plot(C,np.exp(-C*theta))
	
	plt.yscale('log')
	#plt.xscale('log')
	plt.ylim([1e-10,1e3])
	
	fig,ax=plt.subplots(2,1,sharex=True)
	thetafit=np.array(thetafit)
	ax[0].plot(A,thetafit[:,0],'k-',label=r'$\theta_n$')
	ax[0].plot(A,thetafit[:,1],'r--',label=r'$\theta_c$')
	ax[0].set_yscale('log')
	ax[0].legend()
	kfit=np.array(kfit)
	epsfit=np.array(epsfit)
	d1=np.interp(A,D1[:,0],D1[:,1])
	ax[1].plot(A,kfit[:,0]**(epsfit+1)*np.log(1/sB)**1.5,'k-',label=r'$(\log s_B)^{3/2} (k_n)^{\varepsilon +1}$')
	ax[1].plot(A,kfit[:,1],'r--',label='$k_c$')
	ax[1].set_yscale('log')
	ax[1].legend()
	ax[1].set_xlabel('$A$')
	Coeff.append(np.nanmean(kfit[:,0]**(epsfit+1)/kfit[:,1]))

plt.savefig(figdir + 'knkc.pdf',bbox_inches='tight')


plt.figure()
plt.plot(SB,Coeff)
plt.plot(SB,1/(np.log(1/SB))**1.5)
plt.yscale('log')
plt.xscale('log')
#%%% p(log rho,t) p(1/rho ,t)
plt.style.use('~/.config/matplotlib/joris.mplstyle')


keyword='sine'

if keyword=='sine':
	lyap=0.6
	sigma=0.5
	
if keyword=='half':
	lyap=0.25
	sigma=0.26

#sigma=0.5
plt.style.use('~/.config/matplotlib/joris.mplstyle')


A=1/np.sqrt(2)
l0=0.3
dt=0.25

radius=0.01
s0=radius
f=h5py.File('./Compare_stretching_concentration/'+keyword+'/DSM'+keyword+'_A{:1.1f}_l{:1.1f}.hdf5'.format(A,l0), 'r')

Brownian=1e-3
D=(Brownian)**2./(2*dt)#+1e-08 # numerical diffusion ?
#*np.sqrt(2)
sB=np.sqrt(D/0.5)
#sB=0.0005
dx=0.001 # Maximum distance between points, above which the line will be refined

#sB=0.001
TNagg,TNagg2,TCmax,Nmean=[],[],[],[]
nx=np.linspace(0,300,100)
nx2=np.linspace(0,5,30)
nx2=np.logspace(-2,1,50)
xc=np.logspace(-6,0,100)
tv=[3,6,9,11]

pdf_rho=np.linspace(-2,20,50)
plt.figure(figsize=(1.5,1.5))
for t in tv:
	print(t)
	L=f['L_{:04d}'.format(int(t*10))][:]
	dist_old=np.sum(np.diff(L,axis=0)**2,1)**0.5
	#dist_old=np.append(dist_old,dist_old[-1])
	wrapped_time=f['wrapped_time_{:04d}'.format(int(t*10))][:]
	S=f['S_{:04d}'.format(int(t*10))][:]
	W=f['Weights_{:04d}'.format(int(t*10))][:]
	h,x=np.histogram(np.log(1/S),pdf_rho,density=True,weights=W)
	plt.plot(x[1:],h,'o',color=plt.cm.jet(t/12))
	#h,x=np.histogram(np.log(1/S),pdf_rho,density=True,weights=dist_old)
	#plt.plot(x[1:],h,'*',color=plt.cm.jet(t/12))
	plt.plot(x,1/np.sqrt(2*np.pi*sigma*t)*np.exp(-(x-lyap*t)**2/(2*sigma*t)),'-',color=plt.cm.jet(t/12))
	#plt.plot(x,1/np.sqrt(2*np.pi*sigma*t)*np.exp(-(x-(lyap+sigma)*t)**2/(2*sigma*t)),'--',color=plt.cm.jet(t/12))
	#h,x=np.histogram(np.log(1/S),pdf_rho,density=True)
	#plt.plot(x[1:],h,'--')
#plt.plot(x[1:]/np.nanmean(1/S),h*np.nanmean(1/S),'k--')
plt.yscale('log')
plt.xlabel(r'$\log \rho$')
plt.ylabel(r'$p(\log \rho)$')
plt.ylim([1e-4,1e1])

pdf_rho=np.linspace(-2,20,50)
plt.figure(figsize=(1.5,1.5))
for t in tv:
	print(t)
	L=f['L_{:04d}'.format(int(t*10))][:]
	dist_old=np.sum(np.diff(L,axis=0)**2,1)**0.5
	#dist_old=np.append(dist_old,dist_old[-1])
	wrapped_time=f['wrapped_time_{:04d}'.format(int(t*10))][:]
	S=f['S_{:04d}'.format(int(t*10))][:]
	W=f['Weights_{:04d}'.format(int(t*10))][:]
	#h,x=np.histogram(np.log(1/S),pdf_rho,density=True,weights=W)
	#plt.plot(x[1:],h,'o',color=plt.cm.jet(t/12))
	h,x=np.histogram(np.log(1/S),pdf_rho,density=True,weights=dist_old)
	plt.plot(x[1:],h,'*',color=plt.cm.jet(t/12))
	#plt.plot(x,1/np.sqrt(2*np.pi*sigma*t)*np.exp(-(x-lyap*t)**2/(2*sigma*t)),'-',color=plt.cm.jet(t/12))
	plt.plot(x,1/np.sqrt(2*np.pi*sigma*t)*np.exp(-(x-(lyap+sigma)*t)**2/(2*sigma*t)),'--',color=plt.cm.jet(t/12))
	#h,x=np.histogram(np.log(1/S),pdf_rho,density=True)
	#plt.plot(x[1:],h,'--')
	
#plt.plot(x[1:]/np.nanmean(1/S),h*np.nanmean(1/S),'k--')
plt.yscale('log')
plt.xlabel(r'$\log \rho$')
plt.ylabel(r'$p_L(\log \rho)$')
plt.ylim([1e-4,1e1])

pdf_1_rho=np.linspace(-20,2,50)
plt.figure(figsize=(1.5,1.5))
for t in tv:
	print(t)
	L=f['L_{:04d}'.format(int(t*10))][:]
	wrapped_time=f['wrapped_time_{:04d}'.format(int(t*10))][:]
	S=f['S_{:04d}'.format(int(t*10))][:]
	W=f['Weights_{:04d}'.format(int(t*10))][:]
	h,x=np.histogram(np.log(S),pdf_1_rho,density=True,weights=W)
	plt.plot(x[1:],h,'o',color=plt.cm.jet(t/12))
	plt.plot(x,1/np.sqrt(2*np.pi*sigma*t)*np.exp(-(x+lyap*t)**2/(2*sigma*t)),'-',color=plt.cm.jet(t/12))
	#h,x=np.histogram(np.log(1/S),pdf_rho,density=True)
	#plt.plot(x[1:],h,'--')
#plt.plot(x[1:]/np.nanmean(1/S),h*np.nanmean(1/S),'k--')
plt.yscale('log')
plt.xlabel(r'$\log 1/\rho$')
plt.ylabel(r'$p(\log 1/\rho)$')
plt.ylim([1e-4,1e1])


pdf_1_rho=np.linspace(-20,2,50)
plt.figure(figsize=(1.5,1.5))
for t in tv:
	print(t)
	L=f['L_{:04d}'.format(int(t*10))][:]
	dist_old=np.sum(np.diff(L,axis=0)**2,1)**0.5
	wrapped_time=f['wrapped_time_{:04d}'.format(int(t*10))][:]
	S=f['S_{:04d}'.format(int(t*10))][:]
	W=f['Weights_{:04d}'.format(int(t*10))][:]
	h,x=np.histogram(np.log(S),pdf_1_rho,density=True,weights=dist_old)
	plt.plot(x[1:],h,'*',color=plt.cm.jet(t/12))
	plt.plot(x,1/np.sqrt(2*np.pi*sigma*t)*np.exp(-(x+(lyap+sigma)*t)**2/(2*sigma*t)),'--',color=plt.cm.jet(t/12))
	#h,x=np.histogram(np.log(1/S),pdf_rho,density=True)
	#plt.plot(x[1:],h,'--')
#plt.plot(x[1:]/np.nanmean(1/S),h*np.nanmean(1/S),'k--')
plt.yscale('log')
plt.xlabel(r'$\log 1/\rho$')
plt.ylabel(r'$p_L(\log 1/\rho)$')
plt.ylim([1e-4,1e1])


#%%% p(cmax) p(c) and p(sum(cmax))

keyword='sine'
t=11

lyap=0.65
plt.style.use('~/.config/matplotlib/joris.mplstyle')
import h5py
from skimage.morphology import disk
from skimage.filters import median,gaussian
ms=['o','d','s','<','>','+','*']
Brownian=5e-4
radius=0.01
l0=0.3
A=1/np.sqrt(2)
#A=1.2

dir_out='./Compare_stretching_concentration/'+keyword+'/Fourier_A{:1.1f}_l{:1.1f}_rad{:1.2f}_{:1.0e}'.format(A,l0,radius,Brownian)
periodicity='periodic'
#Tmax=[6]
#Tmax_vec=np.array([12])
	#% visual comparison
	#plt.close('all')
I0=cv2.imread(dir_out+'/{:04d}.tif'.format(0),2)
I0=np.float32(I0)/2.**16.
I0med=I0
#I0med = gaussian(I0, sigma=3)
Imax=np.max(I0med)
f=h5py.File('./Compare_stretching_concentration/'+keyword+'/DSM'+keyword+'_A{:1.1f}_l{:1.1f}.hdf5'.format(A,l0), 'r')

Cm_all,Cstd_all,logCm_all,logCstd_all,H2m_all=[],[],[],[],[]


L=f['L_{:04d}'.format(int(t*10))][:]
wrapped_time=f['wrapped_time_{:04d}'.format(int(t*10))][:]
S=f['S_{:04d}'.format(int(t*10))][:]
W=f['Weights_{:04d}'.format(int(t*10))][:]
I=cv2.imread(dir_out+'/{:04d}.tif'.format(int(t*10)),2)
I=np.float32(I)/2.**16.
Imed=I
#Imed = gaussian(I, sigma=3)
I=Imed/Imax
Nbin=I.shape[0]

# Take idsample in a grid to respect an equi porportion in the space
ng=100
idshuffle=np.arange(len(L)-1)
np.random.shuffle(idshuffle)
Lmod=np.mod(L,1)
Lmodint=np.uint32(Lmod*ng)
idu=np.unique(np.copy(Lmodint[idshuffle,:]), return_index=True,axis=0)
idsamples=idshuffle[idu[1]]
#idsamples=idu[1]
#plt.plot(Lmod[idsamples,0],Lmod[idsamples,1],'*')
#plt.plot(Lmodint[idsamples,0]/ng,Lmodint[idsamples,1]/ng,'*')
# Take idsamples according to 1/rho

# compare cmax from individual strips theory and cmax from imag
Lmid=(L[1:,:]+L[:-1,:])/2.
Lmid=L[idsamples,:]
if periodicity=='periodic':
	ix=np.int32(np.mod(Lmid[:,0],1)*Nbin)
	iy=np.int32(np.mod(Lmid[:,1],1)*Nbin)
	Lin=np.where((ix>=0)&(iy>=0)&(ix<Nbin)&(iy<Nbin))
	IL=I[ix[Lin],iy[Lin]].flatten()
else:
	Lin=np.where((np.abs(Lmid[:,1])<1)&(np.abs(Lmid[:,0])<1))
	IL=I[np.int32((Lmid[Lin,0]+1)*Nbin/2),np.int32((Lmid[Lin,1]+1)*Nbin/2)].flatten()

# Find min stretching rates 
argsort_id=np.argsort(1./S)
n=10000
#plt.plot((Lmid[argsort_id[:n],1]+1)*Nbin/2,(Lmid[argsort_id[:n],0]+1)*Nbin/2,'ro',alpha=1.0)

dt=0.25
D=(Brownian)**2./(2*dt)#+1e-08 # numerical diffusion ?
s0=radius#*np.sqrt(2)
#Pe=1e7
Tau=D/s0**2*wrapped_time[Lin]
Cmax=1./np.sqrt(1.+4*D/s0**2*wrapped_time[idsamples])
Cmax_all=1./np.sqrt(1.+4*D/s0**2*wrapped_time)
Sc=s0*np.sqrt(1.+4*D/s0**2*wrapped_time[Lin])*S[Lin]
Cmax2=1./np.sqrt(4*D/s0**2*wrapped_time[Lin])
sB=np.sqrt(D/lyap)
sB=np.nanmean(S)
Rho=1./S[Lin]
#plt.plot(IL,Rho,'.',alpha=0.1)


from scipy import spatial
tree=spatial.cKDTree(np.mod(L,1))
#nsamples=int(sB/dx*10)
nsamples=10000
dist_old=np.sqrt(np.sum(np.diff(L,axis=0)**2.,axis=1))

#idsamples=np.uint32(np.linspace(0,L.shape[0]-2,nsamples))
neighboors=tree.query_ball_point(Lmod[idsamples,:], sB)
neighboors_uniq=[[] for k in range(len(idsamples))]
neighboors_all=[[] for k in range(len(idsamples))]
dist_all=np.hstack((0,np.cumsum(dist_old)))
for ii,ns in enumerate(idsamples):		
	nt=np.sort(neighboors[ii])
	kk_all=np.concatenate((nt,np.array([ns])))
	neighboors_all[ii].append(kk_all)
#	idgood=np.where(np.diff(nt)>3*sB/dx)[0]
#	NT=[]
#	for idg in idgood:
#		if (np.abs(nt[idg+1]-ns)>3*sB/dx)&(np.abs(nt[idg]-ns)<3*sB/dx): # depending on the direction of the filament
#			NT.append(nt[idg+1])
#		else:
#			NT.append(nt[idg])
#	nt=np.array(NT,dtype=np.uint32)
#	kk=np.concatenate((nt,np.array([ns])))
#	neighboors_uniq[ii].append(kk)
	idgood=np.where(np.diff(dist_all[nt])>2*sB)[0]
	kk_all=np.concatenate((nt[idgood],np.array([ns])))
	# remove extra lamella
	kk_all=np.delete(kk_all,np.where(((dist_all[kk_all[:-1]]-dist_all[kk_all[-1]])<5*sB))[0])
	neighboors_uniq[ii].append(kk_all)
	
	
nagg=np.array([len(n[0]) for n in neighboors_uniq])
cmaxagg=np.array([np.sum(Cmax_all[n[0]]) for n in neighboors_uniq])

#plt.figure()
#plt.imshow(I,extent=[0,1,0,1])
#Lmod2=np.hstack((np.mod(L[:,1],1).reshape(-1,1),np.mod(-L[:,0],1).reshape(-1,1)))
#idmod=np.where(np.nansum(np.diff(Lmod2,axis=0)**2.,axis=1)>0.01)[0]
#Lmod2[idmod,:]=np.nan
##plt.plot(np.mod(L[:,1],1),np.mod(-L[:,0],1),'k.',alpha=1.0,linewidth=0.2,markersize=0.1)
#plt.plot(Lmod2[:,0],Lmod2[:,1],'w-',alpha=0.8,linewidth=0.1,markersize=0.2)
#for ne in neighboors_uniq[::20]:
#	plt.plot(Lmod[ne[0],0],Lmod[ne[0],1],'o')

import scipy.signal
plt.figure(figsize=(2,2))
cmaxb=np.logspace(-7,0,70)
pcmax,cmaxb=np.histogram(Cmax,cmaxb,density=True)
pcmax2,cmaxb=np.histogram(Cmax_all,cmaxb,density=True)
pcmaxagg,cmaxb=np.histogram(cmaxagg,cmaxb,density=True)
pc,cb=np.histogram(IL.flatten(),cmaxb,density=True)
pcI,cbI=np.histogram(I.flatten(),cmaxb,density=True)
#pci,cbi=np.histogram(np.log(I[I>1e-5]),50,density=True)
# Rescaled PDF
#plt.plot()

#%%%  Compare PDF of  DSM and DNS

keyword='single'
keyword='cubic'
keyword='sine'
#keyword='half'
#keyword='double'
#keyword='halfsmooth'

keyword2='0mean'
plt.style.use('~/.config/matplotlib/joris.mplstyle')
import h5py
from skimage.morphology import disk
from skimage.filters import median,gaussian
ms=['o','d','s','<','>','+','*','1','2','3']
ms=ms*10
Brownian=1e-3
#Brownian=2e-4
#Brownian=5e-3
radius=0.01
l0=0.3
A=1/np.sqrt(2)
#A=1.2

dir_img='./Compare_stretching_concentration/'+keyword+'/Fourier_'+keyword2+'A{:1.1f}_l{:1.1f}_rad{:1.2f}_{:1.0e}'.format(A,l0,radius,Brownian)
dir_out='./Compare_stretching_concentration/'+keyword+'/'

periodicity='periodic'
k=1
#Tmax=[6]
#Tmax_vec=np.array([12])
	#% visual comparison
	#plt.close('all')
I0=cv2.imread(dir_img+'/{:04d}.tif'.format(0),2)
I0=np.float32(I0)/2.**16.
cmm=np.mean(I0)
Nbin=I0.shape[0]
#I0med = gaussian(I0, sigma=3)
f=h5py.File(dir_out+'DSM'+keyword+'_A{:1.1f}_l{:1.1f}.hdf5'.format(A,l0), 'r')

Tmax_vec=np.array([4,6,8,10,12])
#Tmax_vec=np.arange(8)
#try:
#	Tmax_vec=np.uint8(np.arange(int(f.attrs['tmax']*0.25),f.attrs['tmax']+1,
#														 np.maximum(int(f.attrs['tmax']*0.1),1)))
##	Tmax_vec=[f.attrs['tmax']]
#except:
#	Tmax_vec=[11]
##Tmax=11

#Tmax_vec=[1]

Cm_all,Cm2_all,Cstd_all,logCm_all,logCstd_all,H2m_all=[],[],[],[],[],[]

PDFC=[]
PDFrho=[]

PDFCmax=[]
for t in Tmax_vec:
	L=f['L_{:04d}'.format(int(t*10))][:]
	wrapped_time=f['wrapped_time_{:04d}'.format(int(t*10))][:]
	S=f['S_{:04d}'.format(int(t*10))][:]
	W=f['Weights_{:04d}'.format(int(t*10))][:]
	I=cv2.imread(dir_img+'/{:04d}.tif'.format(int(t*10)),2)
	I=np.float32(I)/2.**16.
	#I=I-I.mean()
	dt=0.25
	D=(Brownian)**2./(2*dt)#+1e-08 # numerical diffusion ?
	s0=radius#*np.sqrt(2)
	#Pe=1e7
	Tau=D/s0**2*wrapped_time
#	Cmax=1./np.sqrt(1.+4*D/s0**2*wrapped_time) #Gaussian
	Cmax=1./(1.+4*D/s0**2*wrapped_time) #Wave
	Sc=s0*np.sqrt(1.+4*D/s0**2*wrapped_time)*S
	sB=Sc.mean()
	Cmax2=1./np.sqrt(4*D/s0**2*wrapped_time)
	Rho=1./S
	#plt.plot(IL,Rho,'.',alpha=0.1)
	binC=np.logspace(-15,0,200)
	binrho=np.logspace(0,15,200)
	binC_lin=np.linspace(-1,1,1000)
	PDFrho.append((np.histogram(1/S,binrho,density=True)[0]))
	PDFC.append((np.histogram(I,binC,density=True)[0]))
	PDFC_lin=(np.histogram((I),binC_lin,density=True)[0])
	PDFCmax.append((np.histogram(Cmax,binC,density=True)[0]))



plt.style.use('~/.config/matplotlib/joris.mplstyle')


c=np.logspace(-10,0,100)
cm=0.1
C=np.linspace(cm,1,10000)



plt.yscale('log')
plt.xscale('log')

lyap=0.65
sigma2=0.45


plt.figure(figsize=(3,2))
rho=np.logspace(0,8,100)
[plt.plot(rho, 1/(rho*np.sqrt(2*np.pi*sigma2*Tmax_vec[t]))*np.exp(-(np.log(rho)-(lyap+sigma2)*Tmax_vec[t])**2./(2*sigma2*Tmax_vec[t])),'-',color=plt.cm.jet(t/len(PDFC))) for t,pdf in enumerate(PDFrho)]
[plt.plot(binrho[1:],pdf,'+-',color=plt.cm.jet(t/len(PDFC)),label=r'$A={:1.1f}$'.format(a)) for t,pdf in enumerate(PDFrho)]
plt.yscale('log')
plt.xscale('log')
plt.xlim([1,1e10])
plt.ylim([1e-12,1e-1])
plt.xlabel('$\rho$')
plt.savefig('/home/joris/Dropbox/Articles/AggregationBatchelor/pdfrho_sine.pdf',bbox_inches='tight')

plt.figure(figsize=(3,2))
C=np.linspace(cm,1,10000)
cm=0
tm=10
t=tmax-tm
c=np.logspace(-20,1,100)
Cmax=1/(c*np.sqrt(2*np.pi*sigma2*t))*np.exp(-(np.log(c)/2+(lyap+sigma2)*t)**2./(2*sigma2*t))
factor=100
#plt.plot(c*(s0/sB)**2.*4,Cmax*factor/(s0/sB)**2.*4)
alpha=-(-1-(lyap+sigma2)/(2*sigma2))
#alpha=-(-1-(lyap+sigma2)/(sigma2))
[plt.plot(c, 1/(c*np.sqrt(2*np.pi*sigma2*Tmax_vec[t]))*np.exp(-(np.log(c)/2+(lyap+sigma2)*Tmax_vec[t])**2./(2*sigma2*Tmax_vec[t])),'-',color=plt.cm.jet(t/len(PDFC))) for t,pdf in enumerate(PDFrho)]
[plt.plot(binC[1:],pdf,'+',color=plt.cm.jet(t/len(PDFC)),label=r'$A={:1.1f}$'.format(a)) for t,pdf in enumerate(PDFCmax)]
[plt.plot(binC[1:],pdf,'o-',color=plt.cm.jet(t/len(PDFC)),label=r'$A={:1.1f}$'.format(a)) for t,pdf in enumerate(PDFC)]
plt.plot(binC[30:-10],1e-3*binC[30:-10]**(-alpha),'k--',label=r'$\alpha={:1.2f}$'.format(alpha))
plt.yscale('log')
plt.xscale('log')
plt.xlim([1e-5,1])
plt.ylim([1e-8,1e6])

plt.savefig('/home/joris/Dropbox/Articles/AggregationBatchelor/pdfC_sine.pdf',bbox_inches='tight')





plt.figure()
plt.plot(binC_lin[1:],PDFC_lin,'+-',color=plt.cm.jet(t/len(PDFC)),label=r'$A={:1.1f}$'.format(a))
plt.yscale('log')
plt.xlim([-0.1,0.1])


#%%% Convolution of p(cmax)
keyword='sine'
t=11

lyap=0.65
plt.style.use('~/.config/matplotlib/joris.mplstyle')
import h5py
from skimage.morphology import disk
from skimage.filters import median,gaussian
ms=['o','d','s','<','>','+','*']
Brownian=1e-3
radius=0.01
l0=0.3
A=1/np.sqrt(2)
#A=1.2

dir_out='./Compare_stretching_concentration/'+keyword+'/Fourier_A{:1.1f}_l{:1.1f}_rad{:1.2f}_{:1.0e}'.format(A,l0,radius,Brownian)
periodicity='periodic'
#Tmax=[6]
#Tmax_vec=np.array([12])
	#% visual comparison
	#plt.close('all')
I0=cv2.imread(dir_out+'/{:04d}.tif'.format(0),2)
I0=np.float32(I0)/2.**16.
I0med=I0
#I0med = gaussian(I0, sigma=3)
Imax=np.max(I0med)
f=h5py.File('./Compare_stretching_concentration/'+keyword+'/DSM'+keyword+'_A{:1.1f}_l{:1.1f}.hdf5'.format(A,l0), 'r')

Cm_all,Cstd_all,logCm_all,logCstd_all,H2m_all=[],[],[],[],[]


L=f['L_{:04d}'.format(int(t*10))][:]
wrapped_time=f['wrapped_time_{:04d}'.format(int(t*10))][:]
S=f['S_{:04d}'.format(int(t*10))][:]
W=f['Weights_{:04d}'.format(int(t*10))][:]
I=cv2.imread(dir_out+'/{:04d}.tif'.format(int(t*10)),2)
I=np.float32(I)/2.**16.
Imed=I
#Imed = gaussian(I, sigma=3)
I=Imed/Imax
Nbin=I.shape[0]

# Take idsample in a grid to respect an equi porportion in the space
ng=10000
idshuffle=np.arange(len(L)-1)
np.random.shuffle(idshuffle)
Lmod=np.mod(L,1)
Lmodint=np.uint32(Lmod*ng)
idu=np.unique(np.copy(Lmodint[idshuffle,:]), return_index=True,axis=0)
idsamples=idshuffle[idu[1]]
#idsamples=idu[1]
#plt.plot(Lmod[idsamples,0],Lmod[idsamples,1],'*')
#plt.plot(Lmodint[idsamples,0]/ng,Lmodint[idsamples,1]/ng,'*')
# Take idsamples according to 1/rho

# all samples
#ng=100000
#idsamples=np.uint64(np.linspace(0,len(L)-2,ng))
#idsamples=np.arange(len(L)-1)

# compare cmax from individual strips theory and cmax from imag
Lmid=(L[1:,:]+L[:-1,:])/2.
Lmid=L[idsamples,:]
if periodicity=='periodic':
	ix=np.int32(np.mod(Lmid[:,0],1)*Nbin)
	iy=np.int32(np.mod(Lmid[:,1],1)*Nbin)
	Lin=np.where((ix>=0)&(iy>=0)&(ix<Nbin)&(iy<Nbin))
	IL=I[ix[Lin],iy[Lin]].flatten()
else:
	Lin=np.where((np.abs(Lmid[:,1])<1)&(np.abs(Lmid[:,0])<1))
	IL=I[np.int32((Lmid[Lin,0]+1)*Nbin/2),np.int32((Lmid[Lin,1]+1)*Nbin/2)].flatten()

# Find min stretching rates 
argsort_id=np.argsort(1./S)
n=10000
#plt.plot((Lmid[argsort_id[:n],1]+1)*Nbin/2,(Lmid[argsort_id[:n],0]+1)*Nbin/2,'ro',alpha=1.0)

dt=0.25
D=(Brownian)**2./(2*dt)#+1e-08 # numerical diffusion ?
s0=radius#*np.sqrt(2)
#Pe=1e7
Tau=D/s0**2*wrapped_time[Lin]
Cmax=1./np.sqrt(1.+4*D/s0**2*wrapped_time[idsamples])
Cmax_all=1./np.sqrt(1.+4*D/s0**2*wrapped_time)
Sc=s0*np.sqrt(1.+4*D/s0**2*wrapped_time[Lin])*S[Lin]
Cmax2=1./np.sqrt(4*D/s0**2*wrapped_time[Lin])
sB=np.sqrt(D/lyap)
sB=np.nanmean(S)
Rho=1./S[Lin]
#plt.plot(IL,Rho,'.',alpha=0.1)


if keyword=='sine':
	lyap=0.55
	sigma=0.45
	
if keyword=='half':
	lyap=0.25
	sigma=0.26

ttttt=np.array([0,1,2,3,4,5,6,7,8,9,10,11,12])
if keyword=='sine':  nconv=[0,0,0,0,0,0,1,1,1,3,6,10,16]
if keyword=='half':
	ttttt=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
	nconv=[0,0,0,0,0,0,0,0,0,0,0 ,0 ,0 ,0 ,0 ,0 ,2 ,3 ,3 ,3 ,3 ,5 ,5, 6 ]

tc=8
nconv=np.uint16(np.exp((lyap+sigma/2)*(ttttt-tc)))
nconvt=nconv[t]
nconvt=np.int(1e-2/np.mean(Cmax))

import scipy.signal
plt.figure(figsize=(2,2))
cmaxb=np.logspace(-7,0,70)
pcmax,cmaxb=np.histogram(Cmax,cmaxb,density=True)
pcmax2,cmaxb=np.histogram(Cmax_all,cmaxb,density=True)
pcmaxagg,cmaxb=np.histogram(cmaxagg,cmaxb,density=True)

pc,cb=np.histogram(IL.flatten(),cmaxb,density=True)
pcI,cbI=np.histogram(I.flatten(),cmaxb,density=True)

# Pcmax
cmaxb_lin=np.linspace(0,1,2**16)
pcmax_lin,cmaxb_lin=np.histogram(Cmax,cmaxb_lin,density=True)
# If t> tmax, cmax is taken from the lognormal

#cmaxb_lin=np.linspace(0,1,2**16)
#pcmax_lin=1/cmaxb_lin[1:]/np.sqrt(2*np.pi*sigma*t)*np.exp(-(np.log(cmaxb_lin[1:])+(lyap)*t)**2./(2*sigma*t))
pc_conv=pdf_selfconvolve(pcmax_lin,cmaxb_lin[1:],nconvt)
pc_conv2=bin_operation(cmaxb_lin[1:],pc_conv,cmaxb,np.nanmean)


def p_c_LN(c,p_cmax,cmax,cm,c0):
	log_cm= np.log(cmax*(1+cm*(1/cmax-1/c0)))
	cv= 0.5
	p_c_cmax=1/c/np.sqrt(2*np.pi*cv**2)*np.exp(-(np.log(c)-log_cm)**2/(2*cv**2))
	return  np.trapz(p_cmax*p_c_cmax,cmax)

def p_c_Gam(c,p_cmax,cmax,cm,c0):
	k = 4 # 1/0.5**2.
	t= cmax*(1+cm*(1/cmax-1/c0))/k
	p_c_cmax=1/(gamma(k)*t**k)*c**(k-1)*np.exp(-c/t)
	return  np.trapz(p_cmax*p_c_cmax,cmax)

cm=np.mean(I)
c=np.logspace(-4,0,100)
pc2=[p_c_LN(cc,pcmax,cmaxb[1:],np.mean(I),1) for cc in c]
pc3=[p_c_Gam(cc,pcmax,cmaxb[1:],np.mean(I),1) for cc in c]
plt.figure(figsize=(2,2))
plt.plot(cbI[1:],pcI,'dr',label='$p(c)$ (DNS)')
plt.plot(cb[1:],pc,'sr',label=r'$p(c(x_{max}))$ (DNS)')
plt.plot(cmaxb[1:],pcmax,'g.',label=r'$p(\theta)$ (LS)')
#plt.plot(cmaxb[1:],pcmax2,'g:',label='$p(\theta)$ (LS)')
#plt.plot(cmaxb_lin[1:],pcmax_lin,'c-',label='$p(c_\mathrm{max})$ (LS)')
#plt.plot(cmaxb[1:],pcmax2,'g-',label='$p(c_\mathrm{max})$ All (LS)')
plt.plot(cmaxb[1:],pc_conv2,'b-.',label=r'$p(\theta)^{\times \langle c \rangle / \langle \theta \rangle}$')
plt.yscale('log')
plt.xscale('log')
#plt.plot(cmaxb[1:],pcmaxagg,'b--',label='$p(\sum c_\mathrm{max})$ (LS)')
#plt.plot(cm-cmaxb[1:]*(cm/1-1.),pcmax/(1-cm/1),'b:',label='$p(c|c_\mathrm{max})$ Dirac')
#plt.plot(c,pc2,'b-',label='$p(c|c_\mathrm{max})$ LogNormal with $\sigma^2=0.5$)')
#plt.plot(c,pc3,'b-',label='$p(c|c_\mathrm{max})$ Gamma with $k=4$)',linewidth=1.2)
#pc_conv2=pdf_selfconvolve(pc2,c,4)
#plt.plot(c,pc_conv2,'b-',label='convolution $p(c_\mathrm{max})^{\star n}$')
plt.yscale('log')
#plt.xscale('log')
plt.ylim([1e-6,1e5])
plt.xlim([1e-8,2])
plt.text(1e-4,1e3,'$T={:1.0f}$'.format(t))
plt.yscale('log')
plt.xlabel(r'$c_\mathrm{max},c$')
plt.legend(fontsize=6,facecolor='white',framealpha=1)
#plt.savefig(dir_out+'./Compare_stretching_concentration/'+keyword+'/pdf_l{:1.1f}_rad{:1.2f}_{:1.0e}_T{:1.0f}.pdf'.format(l0,radius,Brownian,t),
#						bbox_inches='tight')


# plt.figure()
# plt.plot(cmaxagg,IL,'.',alpha=0.1)
# plt.plot(Cmax,IL,'.',alpha=0.1)
# plt.xlabel(r'$\sum c_\mathrm{max}$')
# plt.ylabel(r'$c$')
# plt.yscale('log')
# plt.xscale('log')
# plt.plot([1e-5,1],[1e-5,1],'k--')

#%%% Study of cutoff in the PDF at low stretching
plt.style.use('~/.config/matplotlib/joris.mplstyle')
Brownians=[1e-2,5e-3,1e-3,5e-4,2e-4]
C=[]
radius=0.01
l0=0.3
A=1/np.sqrt(2)
dt=0.25

cm=0.007447596

for Brownian in Brownians:
	dir_out='./Compare_stretching_concentration/Fourier_A{:1.1f}_l{:1.1f}_rad{:1.2f}_{:1.0e}'.format(A,l0,radius,Brownian)
	Ctemp=np.loadtxt(dir_out+'/c_cutoff.txt')
	C.append(Ctemp[:,1])
C=np.array(C)
Tv=Ctemp[:,0]
plt.figure(figsize=(2,2))
for t in range(C.shape[0]):
	D=(Brownians[t])**2./(2*dt)#+1e-08 # numerical diffusion ?
	plt.plot(Tv,C[t,:]-cm,'.-',color=plt.cm.jet((5-t)/5),label='D={:1.1e}'.format(D))
	s0=radius#*np.sqrt(2)
#	plt.plot(Tv,1/(4*np.pi*D*Tv),color=plt.cm.jet((5-t)/5))
#%%% Study of PDF Self-similarity

Brownians=[1e-2,5e-3,1e-3,5e-4,2e-4]
C=[]
radius=0.01
l0=0.3
A=1/np.sqrt(2)
dt=0.25
Nmoments=10
cm=0.007447596
P=[]
for t in range(len(Brownians)):
	print(Brownians[t])
	D=(Brownians[t])**2./(2*dt)#+1e-08 # numerical diffusion ?
	s0=radius#*np.sqrt(2)
	Brownian=Brownians[t]
	radius=0.01
	l0=0.3
	A=1/np.sqrt(2)
	dir_out='./Compare_stretching_concentration/Fourier_A{:1.1f}_l{:1.1f}_rad{:1.2f}_{:1.0e}_half'.format(A,l0,radius,Brownian)
	periodicity='periodic'
	k=1
	Tmax_vec=np.arange(30)
	I0=cv2.imread(dir_out+'/{:04d}.tif'.format(0),2)
	I0=np.float32(I0)/2.**16.
	I0med=I0
	Imax=np.max(I0med)
	CM,CV,Cmoments=[],[],[]
	
	for i in Tmax_vec:
		I=cv2.imread(dir_out+'/{:04d}.tif'.format(int(i*10)),2)
		I=np.float32(I)/2.**16.
		Imed=I
		Cmoments.append([np.mean(np.abs(I-np.mean(I))**n) for n in range(Nmoments)])
	Cmoments=np.array(Cmoments)
	Cmoments=Cmoments/Cmoments[0,:]
	p=[]
	for n in range(Nmoments):
		idfit=np.where((np.isfinite(Cmoments[:,n]))&(Tmax_vec>10))[0]
		p.append(-np.polyfit(Tmax_vec[idfit],np.log(Cmoments[idfit,n]),1)[0])
	P.append(np.array(p))
P=np.array(P)

Plam=[]
for t in range(len(Brownians)):
	print(Brownians[t])
	D=(Brownians[t])**2./(2*dt)#+1e-08 # numerical diffusion ?
	s0=radius#*np.sqrt(2)
	Brownian=Brownians[t]
	radius=0.01
	l0=0.3
	A=1/np.sqrt(2)
	dir_out='./Compare_stretching_concentration/Fourier_A{:1.1f}_l{:1.1f}_rad{:1.2f}_{:1.0e}'.format(A,l0,radius,Brownian)
	periodicity='periodic'
	k=1
	Tmax_vec=np.arange(30)
	I0=cv2.imread(dir_out+'/{:04d}.tif'.format(0),2)
	I0=np.float32(I0)/2.**16.
	I0med=I0
	Imax=np.max(I0med)
	CM,CV,Cmoments=[],[],[]
	
	for i in Tmax_vec:
		I=cv2.imread(dir_out+'/{:04d}.tif'.format(int(i*10)),2)
		I=np.float32(I)/2.**16.
		Imed=I
		Cmoments.append([np.mean(np.abs(I-np.mean(I))**n) for n in range(Nmoments)])
	Cmoments=np.array(Cmoments)
	Cmoments=Cmoments/Cmoments[0,:]
	p=[]
	for n in range(Nmoments):
		idfit=np.where((np.isfinite(Cmoments[:,n]))&(Tmax_vec>10))[0]
		p.append(-np.polyfit(Tmax_vec[idfit],np.log(Cmoments[idfit,n]),1)[0])
	Plam.append(np.array(p))
Plam=np.array(Plam)
#plt.plot(np.arange(Nmoments),P.T/P[:,2].T)

[plt.plot(np.arange(Nmoments),P[i,:],color=plt.cm.jet(float(i)/P.shape[0])) for i in range(P.shape[0])]
[plt.plot(np.arange(Nmoments),Plam[i,:],'--',color=plt.cm.jet(float(i)/Plam.shape[0])) for i in range(Plam.shape[0])]


#plt.xscale('log')
plt.xlabel('$n$')
plt.ylabel(r'$\gamma_n$')
#plt.plot(Tv,1/((1+Tv)),'k--',label=r'$ 1/(1+t)$')
plt.legend(fontsize=5)
plt.savefig(dir_out+'./Compare_stretching_concentration/selfsimilarity.pdf',
						bbox_inches='tight')

#%%% Selfsimilarity : Check High order moments C DNS  / Cmax LS
plt.style.use('~/.config/matplotlib/joris.mplstyle')

Brownian=1e-3
periodicity=''
Nbin=4096
#periodicity=''
dir_out='./Compare_stretching_concentration/{:1.0e}'.format(Brownian)+periodicity

Tmax_vec=[10]

radius =0.01
l0=0.3
for Tmax in Tmax_vec:
	#from colorline_toolbox import *
	# Inline or windows plots
	#%matplotlib auto
	#%matplotlibTrue inline
	PLOT=True
	PAR=False

	
	#% Advection parameters
	INTERPOLATE='LINEAR'
	CURVATURE='DIFF'
	
	PLOT=False
	dx=0.001 # Maximum distance between points, above which the line will be refined
	alpha=200*dx/np.pi
	dt=0.25 # dt needs to be small compare to the CFL condition
	npar=6 # Number of parallel processors
	tmax=Tmax # Maximum advection time
	tsave=0.1 # Time period between saving steps
	Lmax=5e7 # Maximum number of points (prevent blow up)
	th_pinch=100 # Curvature minimum (in nb of dx) to find a peak
	Pe=1e2 # Peclet
	#l0=0.1
	
	# Initial segment position and distance
	x=np.linspace(0,2*np.pi,int(1.8e3))
	L=np.array([radius*np.cos(x),radius*np.sin(x)],dtype=np.float64).T
	L[0,:]=L[-1,:]
	n=int(1.8e3)
	L=np.array([np.linspace(-l0/2,l0/2,n),np.zeros(n)],dtype=np.float64).T
	#	L=[]
	#	for i in range(100):
	#		x,y=np.random.rand(),np.random.rand()
	#		L.append([x-dx,y-dx])
	#		L.append([x+dx,y+dx])
	#		L.append([np.nan,np.nan])
	
	L=np.array(L)
	weights=np.ones(L.shape[0]-1)
	weights=weights/np.sum(weights)
	#L=2*(np.random.rand(2)-0.5)+np.array([x,l0*np.sin(x)],dtype=np.float64).T
	
	dist_old=np.sum(np.diff(L,axis=0)**2,1)**0.5
	weights=np.ones(dist_old.shape)
	# Initial segment width
	S=np.ones(L.shape[0]-1)
	# Initial Wrapped Time
	wrapped_time=np.zeros(L.shape[0]-1)
	
	# Initialization of saved variables
	Lsum=[]
	Ssum=np.sum(dist_old*S)
	Smean=[]
	Rhomean=[]
	logRhomean=[]
	logRhovar=[]
	logKappaMean=[]
	logKappaSum=[]
	logKappaVar=[]
	KappaMean=[]
	logdKappaMean=[]
	logdKappaVar=[]
	logSmean=[]
	logSvar=[]
	Emean=[]
	Evar=[]
	Npinches=[]
	Cmax=[]
	Gamma_PDF_x=np.linspace(-10,10,1000)
	Gamma_PDF=np.zeros(Gamma_PDF_x.shape[0]-1,dtype=np.uint64)
	Lp_x=np.linspace(0,10,100)
	Lp_PDF=np.zeros(Lp_x.shape[0]-1,dtype=np.uint64)
	Lp_var=[]
	CmaxM,IM,IML,CmaxMkappa=[],[],[],[]
	if PLOT:
		plt.close('all')
		%matplotlib auto
		plt.close('all')
		plt.ion()
		fig, ax = plt.subplots(figsize=(10,10))
	#	ax.axis([-5,5,-5,5])
	#	line, = ax.plot(L[:,0], L[:,1],'-',alpha=1.,linewidth=0.5)
		ax.axis([-20,20,-3,3])
		ax.set_xlabel(r'$d\ln \kappa /dt$')
		ax.set_ylabel(r'$d\ln \rho /dt$')
		line, = ax.plot(0,0,'.',alpha=0.05,linewidth=0.5,markersize=0.8)
		ax2 = plt.axes([0.65, 0.65, 0.2, 0.2])
		ax2.axis([-np.pi/2.,np.pi/2.,-np.pi/2.,np.pi/2.])
		line2, = ax2.plot(L[:,0], L[:,1],'r-',alpha=1.,linewidth=0.5)
	
	# Initialization
	t=0
	Tv=np.arange(0,tsave+dt,dt) # Vector of time step between saving steps
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
	
	plt.figure()
	# MAIN PARALLEL LOOP #######################
	while (t<tmax)&(len(L)<Lmax):
		
		v=vel(L,t)
		L+=v*dt
		
		# Compute stretching rates and elongations
		dist_new=np.sum(np.diff(L,axis=0)**2,1)**0.5
		gamma=dist_new/dist_old # elongation
		S=S/gamma
		rho1=1/S
		# Force positive elongation
		#rho1=np.abs(1./S-1.)+1.
		#rho1=np.maximum(1/S,S)
		
		wrapped_time=wrapped_time+dt*rho1**2.		
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
		
		# Remove strip if outside the domain
		Id_out=np.where((np.abs(L[:,0])>2)|(np.abs(L[:,1])>2))[0]
#		L[Id_out,:]=np.nan
#		# Filter consecutive nans to save memory space
#		nnan=np.uint8(np.isnan(L[:,0]))
#		nnan2=nnan[1:]+nnan[:-1]
#		iddel=np.where(nnan2==2)[0]
#		L=np.delete(L,iddel,axis=0)
#		dist_old=np.delete(dist_old,iddel,axis=0)
#		dist_new=np.delete(dist_new,iddel,axis=0)
#		S=np.delete(S,iddel,axis=0)
#		gamma=np.delete(gamma,iddel,axis=0)
#		kappa_old=np.delete(kappa_old,iddel,axis=0)
#		W=np.delete(W,iddel)
#		weights=np.delete(weights,iddel)
#		wrapped_time=np.delete(wrapped_time,iddel,axis=0)
#		dkappa=np.delete(dkappa,iddel)
		
		#Save variables
		Lsum.append(np.sum(dist_old)) # Total length
		Rhomean.append(np.average(1./S,weights=W)) # Mean width
		logRhomean.append(np.average(np.log(1./S),weights=W)) # Mean width
		logRhovar.append(np.average((np.log(1./S)-logRhomean[-1])**2./(1.-np.sum(W**2.)),weights=W)) # Var Log Rho
	
		Smean.append(np.average(S,weights=dist_old)) # Mean width
		logSmean.append(np.average(np.log(S),weights=W)) # Mean width
		logSvar.append(np.average((np.log(S)-logSmean[-1])**2./(1.-np.sum(W**2.)),weights=W)) # Var width
	
		KappaMean.append(np.average(kappa_old,weights=W)) # !!! do we take weighted average or normal average ?
		logKappaMean.append(np.average(np.log(kappa_old),weights=W))
		logKappaSum.append(np.nansum(np.log(kappa_old)))
		logKappaVar.append(np.average((np.log(kappa_old)-logKappaMean[-1])**2./(1.-np.sum(W**2.)),weights=W)) # Var Log Rho
		logdKappaMean.append(dlKMean)
		logdKappaVar.append(dlKVar) # Variance of Elongation
	#	pinches=find_pinch(L,dist_old,th_pinch)
	#	Npinches.append(len(pinches)) # Pinches
	
		Emean.append(np.average(gamma,weights=W)) # Mean Elongation
		Evar.append(np.average((gamma-Emean[-1])**2./(1.-np.sum(W**2.)),weights=W)) # Variance of Elongation
	
		Cmax.append(np.average(1./np.sqrt(1.+wrapped_time/Pe),weights=dist_old)) # Variance of Elongation
	#	h,hx=np.histogram(gamma,Gamma_PDF_x,weights=W) # PDF of Gamma
	#	Gamma_PDF=Gamma_PDF+h
	#	if len(pinches)>2:
	#		pinches_bnd=np.hstack((0,pinches,L.shape[0])) # PDF of pinches length
	#		L_pinches=[np.sum(dist_old[pinches_bnd[j]:pinches_bnd[j+1]-1]) for j in range(len(pinches_bnd)-1)]
	#		h,hx=np.histogram(L_pinches,Lp_x)
	#		Lp_PDF=Lp_PDF+h
	#		Lp_var.append(np.var(L_pinches))
#		prhob,rhob=np.histogram(np.log(1./S[np.isfinite(S)]),100,density=True)
	
#		if np.mod(t,1)==0:
#			prhob_w,rhob_w=np.histogram(np.log(1./S[np.isfinite(S)]),100,density=True,weights=1./S[np.isfinite(S)])
#	#		plt.plot(rhob[1:],prhob,'.')
#			plt.plot(rhob_w[1:],prhob_w,'.',color=plt.cm.jet(t/tmax),label='{:1.0f} t'.format(t))
#			plt.yscale('log')
#			plt.xlabel(r'$\log \rho$')
#			plt.ylabel(r'$P_{\log \rho}$')
		N=10
		if np.mod(t,2)==0:
			D=(Brownian)**2./(2*dt)
			s0=radius*0.95
			Cmaxv=1./np.sqrt(1.+4*D/s0**2*wrapped_time)
			c0=np.nanmean(Cmaxv)
			CmaxM.append([np.nanmean(np.abs(Cmaxv-c0)**n) for n in range(N)])
			pcmax,cmax=np.histogram(np.log(Cmaxv[np.isfinite(Cmaxv)]),100,density=True)
#			pcmax,cmax=np.histogram((1./np.sqrt(1.+wrapped_time[np.isfinite(wrapped_time)]/Pe)),50,density=True)
	#		plt.plot(rhob[1:],prhob,'.')
			plt.plot(cmax[1:],pcmax,'.',color=plt.cm.jet(t/tmax),label=''.format(t))
			# Cmax without sharp folds
			Cmaxv_kappa=Cmaxv[kappa_old<3] 
			c0=np.nanmean(Cmaxv_kappa)
			CmaxMkappa.append([np.nanmean(np.abs(Cmaxv_kappa-c0)**n) for n in range(N)])
			pcmaxk,cmaxk=np.histogram(np.log(Cmaxv_kappa[np.isfinite(Cmaxv_kappa)]),100,density=True)
			plt.plot(cmaxk[1:],pcmaxk,'+',color=plt.cm.jet(t/tmax),label=''.format(t))

			# Load Image file and compare pdf
			# compare cmax from individual strips theory and cmax from imag
			I=cv2.imread(dir_out+'/{:04d}.tif'.format(int((t+dt)*10)),2)
			I=np.float32(I)/2.**16.
			# Non periodic
#			plt.imshow(np.log(I),extent=[-1,1,-1,1])
			#plt.plot(L[:,1],-L[:,0],'k-',alpha=1.0,linewidth=0.2)
			#Periàdic
#			plt.imshow(np.log(I),extent=[0,1,0,1])
#			plt.plot(np.mod(L[:,1],1),np.mod(-L[:,0],1),'k.',alpha=1.0,linewidth=0.2,markersize=0.1)
			Lmid=(L[1:,:]+L[:-1,:])/2.
			if periodicity=='periodic':
				ix=np.int32(np.mod(Lmid[:,0],1)*Nbin)
				iy=np.int32(np.mod(Lmid[:,1],1)*Nbin)
				Lin=np.where((ix>=0)&(iy>=0)&(ix<Nbin)&(iy<Nbin))
				IL=I[ix[Lin],iy[Lin]].flatten()
			else:
				Lin=np.where((np.abs(Lmid[:,1])<1)&(np.abs(Lmid[:,0])<1))
				IL=I[np.int32((Lmid[Lin,0]+1)*Nbin/2),np.int32((Lmid[Lin,1]+1)*Nbin/2)].flatten()
			#pc,cb=np.histogram(np.log(IL),50,density=True)
			c0=np.nanmean(I)
			IM.append([np.nanmean(np.abs(I-c0)**n) for n in range(N)])
			c0=np.nanmean(IL)
			IML.append([np.nanmean(np.abs(IL-c0)**n) for n in range(N)])
			pci,cbi=np.histogram(np.log(I[I>1e-5]),50,density=True)
			pcil,cbil=np.histogram(np.log(IL[IL>1e-5]),50,density=True)
			plt.plot(cbi[1:],pci,'s',color=plt.cm.jet(t/tmax),label='')
			plt.plot(cbil[1:],pcil,'*',color=plt.cm.jet(t/tmax),label='')
			plt.yscale('log')
			plt.xlabel(r'$\log c_{max}$')
			plt.ylabel(r'$P_{\log c_{max}}$')
			plt.xlim([-5,0.1])
		if PLOT:
	#		line.set_xdata(L[:,0])
	#		line.set_ydata(L[:,1])
			line.set_xdata(dkappa/dt)
			line.set_ydata(np.log(gamma)/dt)
			line2.set_xdata(L[:,0])
			line2.set_ydata(L[:,1])
			plt.draw()
			#plt.savefig('{:03d}.png'.format(int(t*100)))
			plt.pause(0.01)
			
			
		# Analysis of curvature production
		
	#	plt.figure()
	#	plt.
		# Update time
		t=t+dt
		print('Time:',t,' - Npts:',len(L), '- Cmax:',np.mean(1./np.sqrt(1.+wrapped_time/Pe)),np.max(kappa_new))
	plt.legend()
	mu_rho=np.average(np.log(1./S[np.isfinite(S)]),weights=1/S[np.isfinite(S)])
	s_rho=np.average((np.log(1./S[np.isfinite(S)])-mu_rho)**2.,weights=1/S[np.isfinite(S)])
	#plt.plot(rhob_w,1./np.sqrt(2*np.pi*s_rho)*np.exp(-(rhob_w-mu_rho)**2./(2*s_rho)),'k-')
	plt.savefig('pdf_cmax.pdf')
	#plt.savefig('pdf_rho.pdf')
	# End of MAIN LOOOP #######################
	print('Computation time:', time.time() -ct)
	if PAR:
		mpool.close()
	#%matplotlib inline
	#%matplotlib auto
	
	#np.savetxt(INTERPOLATE+'_{:d}PTS.txt'.format(L.shape[0]),np.vstack((KappaMean,logKappaMean,logKappaVar,Rhomean,logRhomean,logRhovar)).T)
	
	logRhomean=np.array(logRhomean)
	logRhovar=np.array(logRhovar)
	Lsum=np.array(Lsum)
	plt.figure()
	plt.plot(np.mod(L[:,0],1),np.mod(L[:,1],1),'.',markersize=0.01)
	plt.figure()
	prhob,rhob=np.histogram(np.log(1./S[np.isfinite(S)]),100,density=True)
	prhob_w,rhob_w=np.histogram(np.log(1./S[np.isfinite(S)]),100,density=True,weights=1./S[np.isfinite(S)])
	plt.plot(rhob[1:],prhob,'.')
	plt.plot(rhob_w[1:],prhob_w,'.')
	plt.yscale('log')

# Plot moments of order n
IM=np.array(IM)
IML=np.array(IML)
CmaxM=np.array(CmaxM)
CmaxMkappa=np.array(CmaxMkappa)
plt.figure()
plt.plot((IM/IM[0,:]),'g-')
plt.plot((IML/IML[0,:]),'b-')
plt.plot((CmaxM/CmaxM[0,:]),'r-')
plt.yscale('log')
# Compute exponential decay exponent
Ti=np.linspace(0,Tmax,IM.shape[0])
aIM,aIML,aCmax,aCmaxk=[],[],[],[]
ii=int(len(Ti)/2)
for i in range(IM.shape[1]):
	a=np.polyfit(Ti[ii:],np.log(IM[ii:,i]),1)
	aIM.append(-a[0])
	a=np.polyfit(Ti[ii:],np.log(IML[ii:,i]),1)
	aIML.append(-a[0])
	a=np.polyfit(Ti[ii:],np.log(CmaxM[ii:,i]),1)
	aCmax.append(-a[0])
	a=np.polyfit(Ti[ii:],np.log(CmaxMkappa[ii:,i]),1)
	aCmaxk.append(-a[0])
n=np.arange(IM.shape[1])
aIM=np.array(aIM)
aIML=np.array(aIML)
aCmax=np.array(aCmax)
aCmaxk=np.array(aCmaxk)
plt.figure()
#plt.plot(n,aIM,'g-*',label='$c$')
plt.plot(n,aIML/aIML[2],'b-o',label='$c_{max}$')
plt.plot(n,aCmax/aCmax[2],'r-+',label='$c_{max}$ DSM')
plt.plot(n,aCmaxk/aCmaxk[2],'r--',label='$c_{max}$ DSM without curves')
plt.legend()
plt.xlabel('Moment order $n$')
plt.ylabel('Time decay exponent $\gamma_n/\gamma_2$')
plt.savefig('SineFlow_selfsimilarity_'+periodicity+'.pdf')
#np.savetxt('SineFlow_selfsimilarity_'+periodicity+'_rho1.txt',np.vstack((n,np.array(aIM),np.array(aIML),np.array(aCmax))).T)
np.savetxt('SineFlow_selfsimilarity_'+periodicity+'_rho1.txt',np.vstack((n,np.array(aIM),np.array(aIML),np.array(aCmax),np.array(aCmaxk))).T)

#%%% Selfsimilarity plots
plt.figure(figsize=(5,4))
periodicity='periodic'
n,aIM,aIML,aCmax=np.loadtxt('SineFlow_selfsimilarity_'+periodicity+'.txt').T
plt.plot(n,aIM/aIM[2],'g-*',label='$c$ (periodic)')
plt.plot(n,aIML/aIML[2],'b-*',label='$c_\mathbf{max}$ (periodic)')
plt.plot(n,aCmax/aCmax[2],'r-+',label='$c_\mathbf{max}$ (DSM)')
periodicity=''
n,aIM,aIML,aCmax=np.loadtxt('SineFlow_selfsimilarity_'+periodicity+'.txt').T
plt.plot(n,aIM/aIM[2],'g-s',label='$c$  (non periodic)')
plt.plot(n,aIML/aIML[2],'b-s',label='$c_\mathbf{max}$ (non periodic)')
periodicity='periodic'
n,aIM,aIML,aCmax=np.loadtxt('SineFlow_selfsimilarity_'+periodicity+'_rho1.txt').T
plt.plot(n,aCmax/aCmax[2],'r-s',label=r'$c_\mathbf{max}$ (DSM with $\rho>1$)')
plt.plot(n,n/2.,'k-')
plt.legend()
plt.xlabel('Moment order $n$')
plt.ylabel('Time decay exponent $\gamma_n/\gamma_2$')
plt.savefig('SineFlow_selfsimilarity_all_normalized.pdf')


plt.figure(figsize=(5,4))
periodicity='periodic'
n,aIM,aIML,aCmax=np.loadtxt('SineFlow_selfsimilarity_'+periodicity+'.txt').T
plt.plot(n,aIM,'g-*',label='$c$ (periodic)')
plt.plot(n,aIML,'b-*',label='$c_\mathbf{max}$ (periodic)')
plt.plot(n,aCmax,'r-+',label='$c_\mathbf{max}$ (DSM)')
periodicity=''
n,aIM,aIML,aCmax=np.loadtxt('SineFlow_selfsimilarity_'+periodicity+'.txt').T
plt.plot(n,aIM,'g-s',label='$c$  (non periodic)')
plt.plot(n,aIML,'b-s',label='$c_\mathbf{max}$ (non periodic)')
periodicity='periodic'
n,aIM,aIML,aCmax=np.loadtxt('SineFlow_selfsimilarity_'+periodicity+'_rho1.txt').T
plt.plot(n,aCmax,'r-s',label=r'$c_\mathbf{max}$ (DSM with $\rho>1$)')
#plt.plot(n,n/2.,'k-')
plt.legend()
plt.xlabel('Moment order $n$')
plt.ylabel('Time decay exponent $\gamma_n$')
plt.savefig('SineFlow_selfsimilarity_all.pdf')
#%%% PDF of s

L0,S0,wrapped_time0,W0,t0=run_DSM(1e5, 0.8, 2)

L1,S1,wrapped_time1,W1,t1=run_DSM(1e6, 0.8, 2)

L2,S2,wrapped_time2,W2,t2=run_DSM(1e7, 0.8, 2)

Brownian=1e-3
dt=0.25
D=(Brownian)**2./(2*dt)#+1e-08 # numerical diffusion ?
s0=0.02#*np.sqrt(2)
#Pe=1e7
Tau=D/s0**2*wrapped_time0
Sc=s0*np.sqrt(1.+4*D/s0**2*wrapped_time0)*S0
h,sh=np.histogram(Sc,50,density=True,weights=W0)
plt.plot(sh[1:],h,'*',label=r'$t={:1.0f}$'.format(t0))

Tau=D/s0**2*wrapped_time1
Sc=s0*np.sqrt(1.+4*D/s0**2*wrapped_time1)*S1
h,sh=np.histogram(Sc,50,density=True,weights=W1)
plt.plot(sh[1:],h,'d',label=r'$t={:1.0f}$'.format(t1))


Tau=D/s0**2*wrapped_time2
Sc=s0*np.sqrt(1.+4*D/s0**2*wrapped_time2)*S2
h,sh=np.histogram(Sc,50,density=True,weights=W2)
plt.plot(sh[1:],h,'s',label=r'$t={:1.0f}$'.format(t2))


plt.plot(sh[1:],1e-9*sh[1:]**(-4),'k--',label='$s^{-4}$')
plt.yscale('log')
plt.xscale('log')
plt.xlabel(r'$s=s_0\sqrt{1+4\tau}/\rho$')
plt.legend()
#plt.savefig(dir_out+'p(s).pdf',bbox_inches='tight')

#%%% PDF of Distance between lamellae


#keyword='cubic'
#keyword=''
#keyword='bifreq'
#keyword='half'
#keyword='single'
#keyword='cubic'
keyword='sine'

dx=0.001
plt.style.use('~/.config/matplotlib/joris.mplstyle')
import h5py
from skimage.morphology import disk
from skimage.filters import median,gaussian
ms=['o','d','s','<','>','+','*']
Brownian=1e-3
radius=0.01
l0=0.3
A=1/np.sqrt(2)
#A=1.2


f=h5py.File('./Compare_stretching_concentration/'+keyword+'/DSM'+keyword+'_A{:1.1f}_l{:1.1f}.hdf5'.format(A,l0), 'r')

t=f.attrs['tmax']
t=12
L=f['L_{:04d}'.format(int(t*10))][:]
wrapped_time=f['wrapped_time_{:04d}'.format(int(t*10))][:]
S=f['S_{:04d}'.format(int(t*10))][:]
W=f['Weights_{:04d}'.format(int(t*10))][:]
DD=[]
SD=[]
Lmod=np.mod(L,1)
#plot_per(L)
for nsample in range(200):
	# Equation of line
	idL=int(np.random.rand(1)*L.shape[0])
	n=L[idL,:]-L[idL-1,:]
	n=n/np.sum(n**2.)**0.5
	c=-np.dot(n,Lmod[idL,:])
	a=n[0]
	b=n[1]
 	x=np.linspace(0,1,100)
 	plot_per2(L)
 	plt.plot(Lmod[idL,0],Lmod[idL,1],'ro')
 	plt.plot(x,(-a*x-c)/b,'r-')
 	plt.xlim([0,1])
 	plt.ylim([0,1])
	
	
	# Distance of points to line (looks like curvature)
	D=np.abs(a*Lmod[:,0]+b*Lmod[:,1]+c)
	#plt.plot(D)
	#plt.yscale('log')
	
	# Find local minimums
	idmin=np.where((D[2:]>D[1:-1])*(D[:-2]>D[1:-1])*(D[1:-1]<1e-3))[0]
	#plt.plot(Lmod[idmin,0],Lmod[idmin,1],'go')
	
	# Take only those minimum that are close to the point (to avoid non-perpendcular)
	sB=0.1
	idgood=np.where(np.sum((Lmod[idmin,:]-Lmod[idL,:])**2.,axis=1)<sB**2.)[0]
	#plt.plot(Lmod[idmin[idgood],0],Lmod[idmin[idgood],1],'ro')
	
	v=Lmod[idmin[idgood],:]-Lmod[idL,:]
	tr=np.array([1,-b/a])
	tr=tr/np.sum(tr**2)**0.5
	dist=np.dot(v,tr)
	idsortdist=np.argsort(dist)
	dist_sort=dist[idsortdist]
		
	
	# Inter lamella distance
	diffdist=np.diff(dist_sort)
	# Compression
	rhodist=S[idmin[idgood[idsortdist]]]
	wrappeddist=wrapped_time[idmin[idgood[idsortdist]]]
	DD.extend(diffdist)
	SD.extend((rhodist[:-1]+rhodist[1:])/2.)
	
	plt.plot(dist_sort)
	plt.plot(np.cumsum(rhodist),'.')
	plt.xlabel('i-th lamella')
	plt.ylabel('distance to central lamella')

SD=np.array(SD)
DD=np.array(DD)


fig,ax=plt.subplots(1,1,figsize=(2,2))
nbin=np.linspace(-7,-1,50)
ax.hist2d(np.log10(DD),np.log10(SD),[nbin,nbin],norm=mpl.colors.LogNorm())
plt.plot(nbin,nbin,'w--')
ax.patch.set_facecolor(plt.cm.viridis(0))
plt.xlabel(r'$\log r$')
plt.ylabel(r'$\log 1/\rho$')
plt.text(-6.5,-2,'t={:1.0f}'.format(t),color='w')

#%%% Curvature PDF
#keyword='cubic'
#keyword='bifreq'
keyword=''
keyword='half'
keyword='single'
dx=0.001
t=12
plt.style.use('~/.config/matplotlib/joris.mplstyle')
import h5py
from skimage.morphology import disk
from skimage.filters import median,gaussian
ms=['o','d','s','<','>','+','*']
Brownian=1e-3
radius=0.01
l0=0.3
A=1/np.sqrt(2)
#A=1.2


f=h5py.File('./Compare_stretching_concentration/DSM'+keyword+'_A{:1.1f}_l{:1.1f}.hdf5'.format(A,l0), 'r')

L=f['L_{:04d}'.format(int(t*10))][:]
wrapped_time=f['wrapped_time_{:04d}'.format(int(t*10))][:]
S=f['S_{:04d}'.format(int(t*10))][:]
W=f['Weights_{:04d}'.format(int(t*10))][:]

dist_new=np.sum(np.diff(L,axis=0)**2,1)**0.5
kappa1=curvature(L,dist_new)

k,x=np.histogram(np.log(kappa),200)

plt.figure(figsize=(1.5,1.5))
plt.plot(x[1:],np.log(k),'ko')
plt.plot(x,-x)
plt.plot(x,x)
#%% GLOBAL MOMENTS
#%%% Moments of cmax on L
import matplotlib as mpl

# Set the default color cycle
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color='cmyk') 
keyword='sine'
t=11

lyap=0.65
plt.style.use('~/.config/matplotlib/joris.mplstyle')
import h5py
from skimage.morphology import disk
from skimage.filters import median,gaussian
ms=['o','d','s','<','>','+','*']
Brownian=5e-4
radius=0.01
l0=0.3
A=1/np.sqrt(2)
#A=1.2

dir_out='./Compare_stretching_concentration/'+keyword+'/Fourier_A{:1.1f}_l{:1.1f}_rad{:1.2f}_{:1.0e}'.format(A,l0,radius,Brownian)
periodicity='periodic'
#Tmax=[6]
#Tmax_vec=np.array([12])
f=h5py.File('./Compare_stretching_concentration/'+keyword+'/DSM'+keyword+'_A{:1.1f}_l{:1.1f}.hdf5'.format(A,l0), 'r')

MCmax=[]
Lmean=[]
VC=[]
VCe=[]
M=np.arange(1,4)
Tv=np.arange(12)
for t in Tv:
	L=f['L_{:04d}'.format(int(t*10))][:]
	dist=np.sum(np.sqrt(np.sum(np.diff(L,axis=0)**2.,axis=1)))
	Lmean.append(dist)
	wrapped_time=f['wrapped_time_{:04d}'.format(int(t*10))][:]
	S=f['S_{:04d}'.format(int(t*10))][:]
	W=f['Weights_{:04d}'.format(int(t*10))][:]
	Cmax=1./np.sqrt(1.+4*D/s0**2*wrapped_time)
	MCmax.append([np.mean(Cmax**n) for n in M])
	VC.append(np.sum(Cmax**2.*S))
	I=cv2.imread(dir_out+'/{:04d}.tif'.format(int(t*10)),2)
	VCe.append(np.var(I))
#	I=np.float32(I)/2.**16.
#	Imed=I
	#Imed = gaussian(I, sigma=3)

MCmax=np.array(MCmax)
VC=np.array(VC)
VCe=np.array(VCe)

[plt.plot(Tv,MCmax[:,q],label=r'$\langle \theta^{:1.0f} \rangle_L$'.format(M[q]),linewidth=1.5) for q in range(MCmax.shape[1])]
plt.plot(Tv,1/np.array(Lmean),'c--',label=r'$1/L$',linewidth=1.5)
plt.plot(Tv,VC/VC[0],'r-',label=r'$\int_L s c^2 = L \langle \theta^2 \rangle_L$',linewidth=1.5)
plt.plot(Tv,VCe/VCe[0],'r--',label=r'$\sigma^2_c$',linewidth=1.5)
lyap=0.6
plt.plot(Tv,0.5*np.exp(-lyap/2*Tv),'k-',label=r'$\lambda/2$')
plt.plot(Tv,5*np.exp(-3*lyap/2*Tv),'k--',label=r'$-3/2 \lambda$')
plt.plot(Tv,2*np.exp(-2*lyap*Tv),'k:',label=r'$-2 \lambda$')
plt.yscale('log')
plt.legend()
plt.xlabel('Time')

#%%% Plot Var(cmax), VAr(c)

dir_out='./Compare_stretching_concentration/'
lyap=0.65
nconv=np.array([0,0,0,0,0,0,1,1,3,5,8,16,23])
#nconv=np.array([0,0,0,0,0,0,1,1,2,3,6,10,16])
plt.style.use('~/.config/matplotlib/joris.mplstyle')
plt.figure(figsize=(2,2))
T=np.arange(len(nconv))
plt.plot(nconv,'-o')
plt.plot(T,np.exp(T*lyap/2),'-',label=r'$\bar{\lambda}$')
plt.yscale('log')
plt.xlabel('$T$')
plt.ylabel('$k$ (fit $p(c)$ to gamma)')
plt.legend()
plt.savefig(dir_out+'./Compare_stretching_concentration/aggregationK.pdf',bbox_inches='tight')

plt.style.use('~/.config/matplotlib/joris.mplstyle')
import h5py
from skimage.morphology import disk
from skimage.filters import median,gaussian
ms=['o','d','s','<','>','+','*']
Brownian=2e-4
radius=0.01
l0=0.3
A=1/np.sqrt(2)
#A=1.2

dir_out='./Compare_stretching_concentration/Fourier_A{:1.1f}_l{:1.1f}_rad{:1.2f}_{:1.0e}'.format(A,l0,radius,Brownian)
periodicity='periodic'
k=1
Tmax_vec=np.arange(12)

#Tmax=[6]
#Tmax_vec=np.array([12])
	#% visual comparison
	#plt.close('all')
I0=cv2.imread(dir_out+'/{:04d}.tif'.format(0),2)
I0=np.float32(I0)/2.**16.
I0med=I0
#I0med = gaussian(I0, sigma=3)
Imax=np.max(I0med)
f=h5py.File('./Compare_stretching_concentration/DSM_A{:1.1f}_l{:1.1f}.hdf5'.format(A,l0), 'r')


Mean_Theta=[]
Var_Theta=[]
Sum_Theta=[]
M_Theta=[]
Var_I=[]

for t in Tmax_vec:
	print(t)
	L=f['L_{:04d}'.format(int(t*10))][:]
	wrapped_time=f['wrapped_time_{:04d}'.format(int(t*10))][:]
	S=f['S_{:04d}'.format(int(t*10))][:]
	W=f['Weights_{:04d}'.format(int(t*10))][:]
	I=cv2.imread(dir_out+'/{:04d}.tif'.format(int(t*10)),2)
	I=np.float32(I)/2.**16.
	Imed=I
	#Imed = gaussian(I, sigma=3)
	I=Imed/Imax
	Nbin=I.shape[0]

	# COmpute n order moment
	# compare cmax from individual strips theory and cmax from imag
	Lmid=(L[1:,:]+L[:-1,:])/2.
	if periodicity=='periodic':
		ix=np.int32(np.mod(Lmid[:,0],1)*Nbin)
		iy=np.int32(np.mod(Lmid[:,1],1)*Nbin)
		Lin=np.where((ix>=0)&(iy>=0)&(ix<Nbin)&(iy<Nbin))
		IL=I[ix[Lin],iy[Lin]].flatten()
	else:
		Lin=np.where((np.abs(Lmid[:,1])<1)&(np.abs(Lmid[:,0])<1))
		IL=I[np.int32((Lmid[Lin,0]+1)*Nbin/2),np.int32((Lmid[Lin,1]+1)*Nbin/2)].flatten()
	
	# Find min stretching rates 
	argsort_id=np.argsort(1./S)
	n=10000
	#plt.plot((Lmid[argsort_id[:n],1]+1)*Nbin/2,(Lmid[argsort_id[:n],0]+1)*Nbin/2,'ro',alpha=1.0)

	dt=0.25
	D=(Brownian)**2./(2*dt)#+1e-08 # numerical diffusion ?
	s0=radius#*np.sqrt(2)
	#Pe=1e7
	Tau=D/s0**2*wrapped_time[Lin]
	Cmax=1./np.sqrt(1.+4*D/s0**2*wrapped_time)
	Sc=s0*np.sqrt(1.+4*D/s0**2*wrapped_time[Lin])*S
	Cmax2=1./np.sqrt(4*D/s0**2*wrapped_time[Lin])
	sB=np.sqrt(D/0.5)
	Rho=1./S[Lin]
	Mean_Theta.append(np.mean(Cmax))
	Var_Theta.append(np.mean(Cmax**2))
	Sum_Theta.append(np.sum(Sc*Cmax**2))
	M_Theta.append([np.mean(Cmax**n) for n in range(1,10)]) 
	#Mean_Theta.append(np.average(Cmax,weigths))
	#Var_Theta.append(np.var(Cmax))
	Var_I.append(np.var(I))

Mean_Theta=np.array(Mean_Theta)
Var_Theta=np.array(Var_Theta)
Sum_Theta=np.array(Sum_Theta)
Var_I=np.array(Var_I)

plt.figure()
#plt.plot(Tmax_vec,Mean_Theta,'.-')
plt.plot(Tmax_vec,Var_I/Var_I[0],'3-',label=r'Var$(c)$')
plt.plot(Tmax_vec,Sum_Theta/Sum_Theta[0],'2-',label=r'$A^{-1}\int_L s \theta^2$')
plt.plot(Tmax_vec,Var_Theta/Var_Theta[0],'1-',label=r'$\langle \theta^2 \rangle$')
plt.plot(Tmax_vec,Mean_Theta/Mean_Theta[0],'1--',label=r'$\langle \theta \rangle$')

plt.plot(Tmax_vec,np.array(Var_Theta)*np.array(nconv)[:-1],'4-',label=r'$k \langle \theta^2 \rangle$')
#plt.plot(Tmax_vec,M_Theta,'-')#plt.plot(Tmax_vec,np.array(Var_Theta)*np.exp(lyap*Tmax_vec),'.-')
plt.yscale('log')
plt.plot(Tmax_vec,np.exp(-(lyap/2)*Tmax_vec),'k--',label=r'$e^{- \bar{ \lambda}/2t}$')
plt.plot(Tmax_vec,np.exp(-(lyap)*Tmax_vec),'k:',label=r'$e^{- \bar{ \lambda} t}$')
plt.plot(Tmax_vec,np.exp(-(3*lyap/2)*Tmax_vec),'k-.',label=r'$e^{- 3/2 \bar{ \lambda} t}$')
plt.plot(Tmax_vec,np.exp(-(2*lyap)*Tmax_vec),'k-',label=r'$e^{-2 \bar{ \lambda} t}$')

plt.ylim([1e-6,10])
plt.legend(ncol=2,fontsize=8)
plt.xlabel('$t$')
plt.savefig(dir_out+'./Compare_stretching_concentration/scaling_variance.pdf',bbox_inches='tight')
# plot of n theta ^2
# plot of n theta


#%%% P(c) DNS versus P(cmax) 
plt.figure(figsize=(2,2))
cmaxb=np.logspace(-5,0,50)
pcmax,cmaxb=np.histogram(Cmax,cmaxb,density=True)
pcmaxagg,cmaxb=np.histogram(cmaxagg,cmaxb,density=True)
pcmaxagg2,cmaxb=np.histogram(C[C>0].flatten(),cmaxb,density=True,weights=SC[C>0].flatten())
pcmaxagg2,cmaxb=np.histogram(C[C>0].flatten(),cmaxb,density=True)
cmaxb=np.logspace(-5,0,50)
pc,cb=np.histogram(IL.flatten(),cmaxb,density=True)
pcI,cbI=np.histogram(I.flatten(),cmaxb,density=True)
#pci,cbi=np.histogram(np.log(I[I>1e-5]),50,density=True)
# Rescaled PDF
#plt.plot()


c=np.logspace(-4,0,100)
pc2=[p_c(cc,pcmax,cmaxb[1:],np.mean(I),1) for cc in c]
plt.figure(figsize=(3,3))
plt.plot(c,pc2,label='$\int p(c|c_\mathrm{max}) p(c_\mathrm{max}) \mathrm{d}c_\mathrm{max}$')
plt.plot(cmaxb[1:],pcmax,'g.-',label='$p(c_\mathrm{max})$ (LS)')
plt.plot(cb[1:],pc,'r.-',label='$p(c_\mathrm{max})$ (DNS)')
plt.plot(cbI[1:],pcI,label='$p(c) (DNS)$')
plt.yscale('log')
plt.xscale('log')
plt.ylim([1e-5,1e4])
plt.legend(fontsize=6)

plt.plot(cmaxb[1:],pcmaxagg,'k--',label='$p(\sum c_\mathrm{max})$ (LS, neighboor finding)')

plt.plot(cmaxb[1:],pcmaxagg2,'k-',label='$p(\sum c_\mathrm{max})$ (LS, cell reconstruction)')

plt.yscale('log')
plt.xscale('log')
plt.xlabel(r'$c_\mathrm{max},c$')
plt.legend(fontsize=6)
plt.savefig(dir_out+'./Compare_stretching_concentration/pdf_l{:1.1f}_rad{:1.2f}_{:1.0e}.pdf'.format(l0,radius,Brownian),
						bbox_inches='tight')
#%%% Scalar decay of gridbased DSM parrallel
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

#%%%
for t in range(C.shape[0]):
	plt.hist(C[t,:,:].flatten(),100,alpha=0.2,color=plt.cm.jet(t/C.shape[0]))
	plt.yscale('log')
	plt.xlim([0,0.1])

for t in range(C.shape[0]):
	plt.imshow(np.log(C[t,:,:]),clim=[-7,-4])
	plt.savefig('Sine_decayrate_a{:1.1f}_sB1_{:1.0f}_seed{:1.0f}_t{:1.02f}.jpg'.format(A,1/sB,seed,t))
#%%%Plot
plt.style.use('~/.config/matplotlib/joris.mplstyle')
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
#[plt.plot([],[],M[i],color=plt.cm.cool(i/3),label='$s_B=1/{:1.0f}$'.format(1/sB)) for i,sB in enumerate(SB)]
plt.legend()
d1=AA*p[0]+p[1]


p=np.polyfit(D1[:,0],D1[:,2],1)
d2=AA*p[0]+p[1]
kn=1/(sB**(d2-2)-1)


# plt.figure()
# plt.plot(P[:,0],-P[:,1],'k'+M[i],label='$s_B=1/{:1.0f}$'.format(1/sB))
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
plt.plot([],[],'ko',label=r'Data')
plt.plot(d1,lyap-sigma2/2,'-',color='seagreen',label=r'Isolated strip',linewidth=1.2)
plt.plot([],[],'-',color='w',label=r'\textbf{Aggregation models:}',linewidth=1.2)


#plt.plot(d1,-isol,'-',color='seagreen',label=r'Isolated strip',linewidth=1.2))
fully_cor=2*(lyap-sigma2)
fully_cor[lyap<2*sigma2]=lyap[lyap<2*sigma2]**2/(2*sigma2[lyap<2*sigma2])
plt.plot(d1,fully_cor,'--',color='darkorange',label=r'Fully correlated',linewidth=1.2)
plt.plot(d1,l,':',color='indianred',label=r'Fully random',linewidth=1.2)

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
plt.plot(d1,(l)*ksi,'-',color='blueviolet',label=r'Correlated',linewidth=1.2) #kn>ksi
plt.legend(fontsize=6)

plt.savefig('/home/joris/Dropbox/Articles/AggregationBatchelor/Figures/gamma2_sine.pdf',bbox_inches='tight')



#%%% Scalar decay of gridbased DSM

plt.style.use('~/.config/matplotlib/joris.mplstyle')
Lmax=1e7
A=1.2
sB=1/50
seed=3
AA=[0.4,0.8,1.2,1.8]
SS=np.arange(3)
SB=[1/20,1/100,1/500]

M=['d','o','s','*']
P=[]
for i,sB in enumerate(SB):
	for A in AA:
		p=[]
		for seed in SS:
			print(sB,A,seed)
			Time,C=run_DSM_grid(sB,Lmax,A,seed)
			varC=np.var(C.reshape(C.shape[0],-1),axis=1)
			p.append(np.polyfit(Time[-10:],np.log(varC[-10:]),1)[0])
			plt.plot(Time,varC,color=plt.cm.cool(A/2))
		P.append([A,sB,np.mean(p)])

P=np.array(P)


plt.figure()
plt.plot(P[:,0],-P[:,1],'k'+M[i],label='$s_B=1/{:1.0f}$'.format(1/sB))
Lyap=np.loadtxt('Sine_Lyap.txt')
D1=np.loadtxt('Sine_D1.txt')
A=P[:,0]
lyap=np.interp(A,Lyap[:,0],Lyap[:,1])
sigma2=np.interp(A,Lyap[:,0],Lyap[:,2])
d1=np.interp(A,D1[:,0],D1[:,1])
plt.plot(A,lyap**2/(2*sigma2),'r-',label='$\mu^2/(2\sigma^2)$')
plt.plot(A,2*lyap-2*sigma2,'r--',label='$2\mu-2\sigma^2$')
plt.xlabel(r'$A$')
plt.ylabel(r'$\gamma_2$')
plt.legend()
#%%% Variance and cmax of DSM

plt.style.use('~/.config/matplotlib/joris.mplstyle')
import h5py
ms=['o','d','s','<','>','+','*']
#Brownian=5e-04
Brownian=Brownians[t]
radius=0.01
l0=0.3
A=1/np.sqrt(2)
#A=1.2

dir_out='./Compare_stretching_concentration/Fourier_A{:1.1f}_l{:1.1f}_rad{:1.2f}_{:1.0e}'.format(A,l0,radius,Brownian)
periodicity='periodic'
k=1
Tmax_vec=np.array([2,4,6,8,10,12])
Tmax_vec=np.arange(30)
#Tmax_vec=[12]
#Tmax=[6]
#Tmax_vec=np.array([12])
	#% visual comparison
	#plt.close('all')
I0=cv2.imread(dir_out+'/{:04d}.tif'.format(0),2)
I0=np.float32(I0)/2.**16.
I0med=I0
#I0med = gaussian(I0, sigma=3)
Imax=np.max(I0med)
CM,CV,Cmoments=[],[],[]

for i in Tmax_vec:
	I=cv2.imread(dir_out+'/{:04d}.tif'.format(int(i*10)),2)
	I=np.float32(I)/2.**16.
	Imed=I
	CM.append(np.percentile(I,99.9))
	CV.append(np.var(I))
	Cmoments.append([np.mean(np.abs(I-np.mean(I))**n) for n in range(10)])

CM=np.array(CM)
CV=np.array(CV)
Cmoments=np.array(Cmoments)

plt.plot(Tmax_vec,CM-np.mean(I0),color=plt.cm.jet((5-t)/5))
#plt.plot(Tmax_vec,CV/CV[0])
plt.yscale('log')

plt.yscale('log')
#plt.xscale('log')
plt.xlabel('Time')
plt.ylabel(r'$c$ cutoff  $-\langle c \rangle$')
#plt.plot(Tv,1/((1+Tv)),'k--',label=r'$ 1/(1+t)$')
plt.plot(Tv,np.exp(-(0.64/2)*Tv),'k--',label=r'$ \exp(-\mu_{\lambda}/2 t)$')
plt.legend(fontsize=5)
plt.savefig(dir_out+'./Compare_stretching_concentration/cutoff.pdf',
						bbox_inches='tight')

plt.figure()
mean_c=np.mean(I)
sig=1e-3
def p_c_cmax(c,cmax):
	cb=mean_c+cmax*(1-mean_c)
	return np.exp(-(c-cb)**2./(2*sig**2))
C=cmaxb
P_c=np.array([np.sum([p_c_cmax(c,cmaxb[i])*pcmax[i]*np.diff(cmaxb)[i] for i in range(len(pcmax))]) for c in C])
plt.plot(C,P_c)
plt.plot(C[1:],pcmax)
plt.yscale('log')
plt.xscale('log')
plt.ylim([1e-5,1e2])
#%% N(t)
#%%% P(N,t)
keyword='sine'


#keyword='sine'
import h5py

periodicity='periodic'

if keyword=='sine':
	lyap=0.55
	sigma=0.45
	nu=1.7
	
if keyword=='half':
	lyap=0.25
	sigma=0.26
	nu=1.5


dir_out='./Compare_stretching_concentration/'+keyword+'/'

plt.style.use('~/.config/matplotlib/joris.mplstyle')
#% Check cmax of bundles

A=1/np.sqrt(2)
l0=0.3
dt=0.25

radius=0.01
s0=radius
f=h5py.File('./Compare_stretching_concentration/'+keyword+'/DSM'+keyword+'_A{:1.1f}_l{:1.1f}.hdf5'.format(A,l0), 'r')

Brownian=1e-3
D=(Brownian)**2./(2*dt)#+1e-08 # numerical diffusion ?
#*np.sqrt(2)
#sB=np.sqrt(D/0.5)
#sB=0.0005
dx=0.001 # Maximum distance between points, above which the line will be refined
dir_img='./Compare_stretching_concentration/'+keyword+'/Fourier_A{:1.1f}_l{:1.1f}_rad{:1.2f}_{:1.0e}'.format(A,l0,radius,Brownian)

#sB=0.001
Lmean,kI,kIL=[],[],[]
TNagg,TNagg2,TCmax,Nmean,TNagg_eul,Nmean_eul,Nmean_eul_all,Cmaxmean,Cmaxvar=[],[],[],[],[],[],[],[],[]
nx=np.unique(np.uint16(np.logspace(-0.5,3,100)))
nx2=np.logspace(-2,2,40)
xc=np.logspace(-6,0,100)
#tv=[1,2,3,4,5,6,7,8,9,10,11,12]
try:
	tv=np.arange(1,f.attrs['tmax'],1)
except:
	tv=[1,2,3,4,5,6,7,8,9,10,11,12]

tv=np.arange(20)
if keyword=='sine':
	tv=np.array([6,8,10,12])
if keyword=='half':
	tv=np.array([10,15,20,25])

tv=np.arange(12)
for t in tv:
	print(t)
	if t<=f.attrs['tmax']:
		L=f['L_{:04d}'.format(int(t*10))][:]
		wrapped_time=f['wrapped_time_{:04d}'.format(int(t*10))][:]
		S=f['S_{:04d}'.format(int(t*10))][:]
		W=f['Weights_{:04d}'.format(int(t*10))][:]
		Cmax=1./np.sqrt(1.+4*D/s0**2*wrapped_time)
		s=s0*np.sqrt(1.+4*D/s0**2*wrapped_time)*S
		sB=np.mean(s)
		tree=spatial.cKDTree(np.mod(L,1))
		nsamples=int(sB/dx*10)
		nsamples=10000
		idsamples=np.uint32(np.linspace(0,L.shape[0]-2,nsamples))
# =============================================================================
# 		#LAGRANGIAN MEASURE OF N
# 		# Take idsample in a grid to respect an equi porportion in the space
# =============================================================================
		ng=int(1/sB)
		ng=50
		idshuffle=np.arange(len(L)-3)
		np.random.shuffle(idshuffle)
		Lmod=np.mod(L[:-2],1)
		Lmodint=np.uint32(Lmod*ng)
		idu=np.unique(np.copy(Lmodint[idshuffle,:]), return_index=True,axis=0)
		idsamples=idshuffle[idu[1]]
		neighboors=tree.query_ball_point(np.mod(L[idsamples,:],1), sB/2.)
		neighboors_uniq=[[] for k in range(len(idsamples))]
		neighboors_all=[[] for k in range(len(idsamples))]
		for ii,ns in enumerate(idsamples):		
			nt=np.sort(neighboors[ii])
			kk_all=np.concatenate((nt,np.array([ns])))
			neighboors_all[ii].append(kk_all)
			idgood=np.where(np.diff(nt)>3*sB/dx)[0]
			NT=[]
			for idg in idgood:
				if (np.abs(nt[idg+1]-ns)>3*sB/dx)&(np.abs(nt[idg]-ns)<3*sB/dx): # depending on the direction of the filament
					NT.append(nt[idg+1])
				else:
					NT.append(nt[idg])
			nt=np.array(NT,dtype=np.uint32)
			kk=np.concatenate((nt,np.array([ns])))
			neighboors_uniq[ii].append(kk)
		nagg=np.array([len(n[0]) for n in neighboors_uniq])
		nh,xh=np.histogram(nagg,nx,density=True)
		
# =============================================================================
# 		Eulerian measure of N
# 		dist_old=np.sum(np.diff(L,axis=0)**2,1)**0.5
# =============================================================================
		Lmod=np.mod(L,1)
		dist_old=np.sum(np.diff(L,axis=0)**2,1)**0.5
		N=np.uint16((np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1,sB),
										weights=dist_old/sB,density=False)[0]))
		nagg_eul=np.maximum(N[N>0],1)
		nh_eul,xh=np.histogram(nagg_eul,nx,density=True)
		
		TNagg.append(nh)
		TNagg_eul.append(nh_eul)
		
		dist=np.sum(np.sqrt(np.sum(np.diff(L,axis=0)**2.,axis=1)))
		Lmean.append(dist)
		Nmean.append(np.mean(nagg))
		Nmean_eul.append(np.mean(nagg_eul))
		Nmean_eul_all.append(np.mean(N))
		# Binning as a function of N/<N>
		nh2,xh2=np.histogram(nagg/np.mean(nagg),nx2,density=True)
		TNagg2.append(nh2)
		
		Cmaxmean.append(np.mean(Cmax))
		Cmaxvar.append(np.var(Cmax))
		
		nc,xc=np.histogram(Cmax,xc,density=True)
		TCmax.append(nc)
	I=cv2.imread(dir_img+'/{:04d}.tif'.format(int(t*10)),2)
	I=np.float32(I)/2.**16.
	kI.append(np.mean(I)**2/np.var(I))
	# Measure only on peaks
	# all samples
	ng=100000
	idsamples=np.uint64(np.linspace(0,len(L)-2,ng))
	#idsamples=np.arange(len(L)-1)
	
	# compare cmax from individual strips theory and cmax from imag
	Lmid=(L[1:,:]+L[:-1,:])/2.
	Lmid=L[idsamples,:]
	Nbin=I.shape[0]
	if periodicity=='periodic':
		ix=np.int32(np.mod(Lmid[:,0],1)*Nbin)
		iy=np.int32(np.mod(Lmid[:,1],1)*Nbin)
		Lin=np.where((ix>=0)&(iy>=0)&(ix<Nbin)&(iy<Nbin))
		IL=I[ix[Lin],iy[Lin]].flatten()
	else:
		Lin=np.where((np.abs(Lmid[:,1])<1)&(np.abs(Lmid[:,0])<1))
		IL=I[np.int32((Lmid[Lin,0]+1)*Nbin/2),np.int32((Lmid[Lin,1]+1)*Nbin/2)].flatten()
	kIL.append(np.mean(IL)**2/np.var(IL))
	
TNagg=np.array(TNagg)
TNagg_eul=np.array(TNagg_eul)
TNagg2=np.array(TNagg2)
TCmax=np.array(TCmax)


plt.rcParams['axes.prop_cycle'] = plt.cycler(color=[plt.cm.jet(k) for k in np.linspace(0,1,5)]) 

plt.figure(figsize=(1.5,1.5))
plt.plot(nx[:-1],TNagg.T)
plt.yscale('log')
#plt.xscale('log')
plt.xlabel('$N$')
plt.ylabel('$p(N,t)$')

#P(N/<N>,t)
plt.figure(figsize=(2,2))
plt.plot(nx2[:-1],TNagg2.T,'o')
plt.yscale('log')
plt.xscale('log')
plt.xlabel(r'$N/\langle N \rangle$')
plt.ylabel(r'$p(N/\langle N \rangle,t)$ for $t=6,8,10,12$')
from scipy.special import gamma
k=2.0
tt=0.5
texp=1
plt.plot(nx2,1/gamma(k)/tt**k*nx2**(k-1)*np.exp(-nx2/tt),'k-',label='Gamma({:1.1f},{:1.1f})'.format(k,t))
plt.plot(nx2,texp*np.exp(-nx2/texp),'k--',label='Exp({:1.1f})'.format(texp))
plt.legend(fontsize=6)
plt.ylim([1e-3,10])
plt.savefig(dir_out+'p(N_N,t).pdf',
						bbox_inches='tight')


plt.figure(figsize=(1.5,1.5))
plt.plot(nx[:-1],TNagg.T)
plt.plot(1+1e-3*(1/xc[1:]-1),TCmax.T)
plt.yscale('log')
plt.xscale('log')
plt.xlabel('$N$')
plt.ylabel('$p(N,t)$')

#TODO add theoretical PDF
plt.figure()
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=[plt.cm.jet(k) for k in np.linspace(0,1,len(tv))]) 
#plt.plot(nx[:-1],TNagg.T,'o-',label='$P_l(N)$')
plt.plot(nx[:-1],TNagg_eul.T,'+-',label='')
plt.yscale('log')
plt.xscale('log')
plt.xlabel(r'$N$')
plt.legend(title='Sine Flow')
for i,t in enumerate(tv):
	logrhoc=np.log(1/sB)
	l=(nu-1)*(lyap*t-logrhoc)
	s=sigma*(nu-1)**2.*t
	plt.plot(nx,1/np.sqrt(2*np.pi*s)/nx*np.exp(-(np.log(nx)-l)**2./(2*s)),'--',label=r'',
					color=plt.cm.jet(i/(len(tv)-1)))
	#!!! no dependence on fractal dim ?ension !?
	l=(lyap*t-logrhoc)
	s=sigma*t
	plt.plot(nx,1/np.sqrt(2*np.pi*s)/nx*np.exp(-(np.log(nx)-l)**2./(2*s)),':',label=r'',
					color=plt.cm.jet(i/(len(tv)-1)))
plt.ylim([1e-6,2])

logrho=np.array([np.mean(np.log(1/S[n])) for n in neighboors_uniq])
Nh=(np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1,sB),density=False)[0])
logrho_eul=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1,sB),density=False,weights=np.log(1/S))[0]/Nh
nn=np.array([1,10])
logrhoc=6
plt.figure()
plt.plot(np.log(nagg).flatten(),logrho.flatten(),'r.',alpha=0.1,label='')
plt.plot(np.log(N).flatten(),logrho_eul.flatten(),'k.',alpha=0.01,label='')
plt.plot(nn,1/(nu-1)*nn+logrhoc,'k--',label=r'$1/(\nu-1)$')
plt.plot(nn,nn+logrhoc,'k-',label='$1$')
plt.legend()
plt.xlabel(r'$\log N$')
plt.ylabel(r'$  \langle \log\rho \rangle$')

#%%% P(n,t) ...
a=0.8
s=2

L1,S1,wrapped_time,W1,t1=run_DSM(1e7,a,s)
L2,S2,wrapped_time,W2,t2=run_DSM(1e6,a,s)
L3,S3,wrapped_time,W3,t3=run_DSM(1e5,a,s)
#%%% ... Eulerian

sB=1/50
l0=0.3
rhoc=1/(sB*l0)
rhoc=20

#rhoc=50
#l0 * rho *sB  = A *n

# Scaling 1 : n ~ rho
# n = l0 rho sB / A

# Scaling 2 : log n ~ (D1-1) log rho
# 1/n ~ 1/rho


Lyap=np.loadtxt('Sine_Lyap.txt')
lyap=np.interp(a,Lyap[:,0],Lyap[:,1])
sigma2=np.interp(a,Lyap[:,0],Lyap[:,2])
D=np.loadtxt('Sine_D1.txt')
D1=np.interp(a,D[:,0],D[:,1])
plt.figure()
#D1=1.77
#D1=2
print("D1=",D1)


plt.figure()
Lmod=np.mod(L1,1)
dist_old=np.sum(np.diff(L1,axis=0)**2,1)**0.5
N1=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1,sB)
									 ,weights=dist_old/sB,density=False)[0]
h,x=np.histogram(np.log(N1[N1>0]),50,density=True)
plt.plot(x[1:],h,'o',c=plt.cm.cool(t1/t1))
hs,xs=np.histogram((D1-1)*np.log(1/rhoc/S1),50,weights=W1,density=True)
plt.plot(xs[1:],hs,'-',c=plt.cm.cool(t1/t1))
#plt.plot(xs,1/np.sqrt(2*np.pi*sigma2*t1)*np.exp(-(xs-(lyap*t1-np.log(rhoc)))**2/(2*sigma2*t1)),'--',c=plt.cm.cool(t1/t1))

Lmod=np.mod(L2,1)
dist_old=np.sum(np.diff(L2,axis=0)**2,1)**0.5
N2=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1,sB)
									 ,weights=dist_old/sB,density=False)[0]
h,x=np.histogram(np.log(N2[N2>0]),50,density=True)
plt.plot(x[1:],h,'s',c=plt.cm.cool(t2/t1))
hs,xs=np.histogram((D1-1)*np.log(1/rhoc/S2),50,weights=W2,density=True)
plt.plot(xs[1:],hs,'-',c=plt.cm.cool(t2/t1))
#plt.plot(xs,1/np.sqrt(2*np.pi*sigma2*t2)*np.exp(-(xs-(lyap*t2-np.log(rhoc)))**2/(2*sigma2*t2)),'--',c=plt.cm.cool(t2/t1))


Lmod=np.mod(L3,1)
dist_old=np.sum(np.diff(L3,axis=0)**2,1)**0.5
N3=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1,sB)
									 ,weights=dist_old/sB,density=False)[0]
h,x=np.histogram(np.log(N3[N3>0]),50,density=True)
plt.plot(x[1:],h,'d',c=plt.cm.cool(t3/t1))
hs,xs=np.histogram((D1-1)*np.log(1/rhoc/S3),50,weights=W3,density=True)
plt.plot(xs[1:],hs,'-',c=plt.cm.cool(t3/t1))
#plt.plot(xs,1/np.sqrt(2*np.pi*sigma2*t3)*np.exp(-(xs-(lyap*t3-np.log(rhoc)))**2/(2*sigma2*t3)),'--',c=plt.cm.cool(t3/t1))


plt.xlim([-2,10])
plt.ylim([1e-3,1e0])
plt.xlabel('$\log n$')
plt.yscale('log')


plt.figure()
Lmod=np.mod(L1,1)
dist_old=np.sum(np.diff(L1,axis=0)**2,1)**0.5
N1=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1,sB)
									 ,weights=dist_old/sB,density=False)[0]
mlogrho=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1,sB),weights=dist_old/sB*np.log(1/S1),density=False)[0]
mlogrho=mlogrho/N1
h,x=np.histogram(mlogrho[N1>2],50,density=True)
plt.plot(x[1:],h,'o',c=plt.cm.cool(t1/t1))
hs,xs=np.histogram(np.log(1/S1),50,weights=W1,density=True)
plt.plot(xs[1:],hs,'-',c=plt.cm.cool(t1/t1))
#plt.plot(xs,1/np.sqrt(2*np.pi*sigma2*t1)*np.exp(-(xs-(lyap*t1-np.log(rhoc)))**2/(2*sigma2*t1)),'--',c=plt.cm.cool(t1/t1))

Lmod=np.mod(L2,1)
dist_old=np.sum(np.diff(L2,axis=0)**2,1)**0.5
N2=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1,sB)
									 ,weights=dist_old/sB,density=False)[0]

mlogrho=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1,sB),weights=dist_old/sB*np.log(1/S2),density=False)[0]
mlogrho=mlogrho/N2
h,x=np.histogram(mlogrho[N2>2],50,density=True)
plt.plot(x[1:],h,'s',c=plt.cm.cool(t2/t1))
hs,xs=np.histogram(np.log(1/S2),50,weights=W2,density=True)
plt.plot(xs[1:],hs,'-',c=plt.cm.cool(t2/t1))
#plt.plot(xs,1/np.sqrt(2*np.pi*sigma2*t2)*np.exp(-(xs-(lyap*t2-np.log(rhoc)))**2/(2*sigma2*t2)),'--',c=plt.cm.cool(t2/t1))


Lmod=np.mod(L3,1)
dist_old=np.sum(np.diff(L3,axis=0)**2,1)**0.5
N3=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1,sB)
									 ,weights=dist_old/sB,density=False)[0]

mlogrho=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1,sB),weights=dist_old/sB*np.log(1/S3),density=False)[0]
mlogrho=mlogrho/N3
h,x=np.histogram(mlogrho[N3>2],50,density=True)
plt.plot(x[1:],h,'d',c=plt.cm.cool(t3/t1),label=r'$P_{\langle \log \rho\rangle_B}$')
hs,xs=np.histogram(np.log(1/S3),50,weights=W3,density=True)
plt.plot(xs[1:],hs,'-',c=plt.cm.cool(t3/t1),label=r'$P_{\log \rho}$')
#plt.plot(xs,1/np.sqrt(2*np.pi*sigma2*t3)*np.exp(-(xs-(lyap*t3-np.log(rhoc)))**2/(2*sigma2*t3)),'--',c=plt.cm.cool(t3/t1))


#plt.xlim([-2,10])
plt.ylim([1e-3,1e0])
plt.xlabel(r'$\log \rho$')
plt.yscale('log')
plt.legend()

#%%% Various A
s=3
plt.figure()
for i,a in enumerate([0.5]):
	L1,S1,wrapped_time,W1,t1=run_DSM(1e7,a,s)
	
	
	l0=0.3
	sB=1/1000
	
	Lmod=np.mod(L1,1)
	dist_old=np.sum(np.diff(L1,axis=0)**2,1)**0.5
	for i,sB in enumerate([1/50]):
		N1=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1,sB)
											 ,weights=dist_old/sB,density=False)[0]
		h,x=np.histogram(1/(N1[N1>0]-1),np.logspace(-6,2,50),density=True)
		plt.plot(x[1:],h,'o',c=plt.cm.cool(i/3.),label=r'$P_{1/n}'+r', s_B=1/{:1.0f}, A={:1.1f}$'.format(1/sB,a))
	
		h,x=np.histogram(S1,np.logspace(-8,0,50),density=True,weights=S1)
		plt.plot(x[1:]/(l0*sB),h*(sB*l0),'-',c=plt.cm.cool(i/3.))#,label=r'$P_{1/\rho,0}$')
	plt.xscale('log')
	plt.yscale('log')
	plt.xlabel(r'$1/(n-1), 1/\rho/(l_0 s_B)$')
	plt.xlim([1e-6,1e2])
	plt.ylim([1e-4,1e3])
	#plt.vlines(sB*l0, 1e-3, 1e4,label='$s_B l_0/A$')
	plt.legend()

#%%% * Various Pe
s=3

a=0.9
plt.figure(figsize=(3,2))
M=['d','o','s','*']
for i,a in enumerate([0.8]):
	L1,S1,wrapped_time,W1,t1=run_DSM(1e7,a,s,STOP_ON_LMAX=True)
	
	
	l0=0.3
	sB=1/1000
	
	Lmod=np.mod(L1,1)
	dist_old=np.sum(np.diff(L1,axis=0)**2,1)**0.5
	for i,sB in enumerate([1/50,1/100,1/500]):
		N1=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1,sB)
											 ,weights=dist_old/sB,density=False)[0]
		h,x=np.histogram(N1.mean()/(N1[N1>=1]),np.logspace(-2,3,50),density=True)
		plt.plot(x[1:],h,M[i],c=plt.cm.cool(i/3.),label=r'$s_B=1/{:1.0f}$'.format(1/sB,a))
		sm=np.average(1/S1)
		h,x=np.histogram(S1*sm,np.logspace(-2,6,50),density=True)
		h[h==0]=np.nan
	plt.plot(x[1:],h,'k-',label=r'$P_{\langle \rho \rangle /\rho }$')
	plt.xscale('log')
	plt.yscale('log')
	plt.xlabel(r'$\langle n \rangle/n$')
	plt.xlim([1e-2,1e3])
	plt.ylim([1e-5,2e0])
	#plt.vlines(sB*l0, 1e-3, 1e4,label='$s_B l_0/A$')
	plt.legend()
plt.savefig('/home/joris/Dropbox/Articles/AggregationBatchelor/Figures/Sine_N_sb.pdf',bbox_inches='tight')

#%%% * Spatial mean of mean(log rho|n) and var(log rho|n)
s=3
A=0.4

Res=[]
Res2=[]

VarA=np.loadtxt('Sine_N_t.txt')
Lyap=np.loadtxt('Sine_Lyap.txt')
D1=np.loadtxt('Sine_D1.txt')
d1=np.interp(A,D1[:,0],D1[:,1])

for Lmax in np.logspace(3,7,20):
	L1,S1,wrapped_time,W1,t1=run_DSM(Lmax,A,s,STOP_ON_LMAX=True)
	
	l0=0.3
	sB=1/100
	
	i=1
	a=0.4
	Lmod=np.mod(L1,1)
	dist_old=np.sum(np.diff(L1,axis=0)**2,1)**0.5
	#	for i,sB in enumerate([1/100]):
	N1=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1,sB)
										 ,weights=dist_old/sB,density=False)[0]
	N0=np.round(N1)
	logrho=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1,sB)
						  ,weights=dist_old/sB*np.log(1/S1),density=False)[0]
	logrho2=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1,sB)
						  ,weights=dist_old/sB*np.log(1/S1)**2,density=False)[0]
	
	Mlogrho=logrho[N0>0]/N1[N0>0]
	Vlogrho=logrho2[N0>0]/N1[N0>0]-Mlogrho**2.
	
	# Difference
	diff=Vlogrho-2*(2-d1)*Mlogrho
	
	# compare with moments of the strip
	ML=np.average(np.log(1/S1),weights=W1)
	L2=np.average(np.log(1/S1)**2.,weights=W1)
	VL=L2-ML**2.
	
	MLL=np.mean(np.log(1/S1))
	VLL=np.var(np.log(1/S1))
	
	Res.append([t1,np.nanmean(Mlogrho),np.average(Mlogrho,weights=N1[N0>0]),ML,MLL,np.nanmean(Vlogrho),np.average(Vlogrho,weights=N1[N0>0]),VL,VLL])
	Res2.append([t1,np.nanmean(diff)])
	
Res=np.array(Res)

i=1
plt.plot(Res[:,0],Res[:,i],'b-',label=r'$\mu_B$'); i+=1
plt.plot(Res[:,0],Res[:,i],'bo',label=r'$\mu_{n,B}$'); i+=1
plt.plot(Res[:,0],Res[:,i],'b--',label=r'$\mu_0$'); i+=1
plt.plot(Res[:,0],Res[:,i],'b:',label=r'$\mu_L$'); i+=1

plt.xlabel('$t$')
plt.legend()

plt.title(r'Mean of $\log \rho \, (A={:1.1f})$'.format(A))

plt.figure()
plt.plot(Res[:,0],Res[:,i],'r-',label=r'$\sigma^2_B$'); i+=1
plt.plot(Res[:,0],Res[:,i],'ro',label=r'$\sigma^2_{n,B}$'); i+=1
plt.plot(Res[:,0],Res[:,i],'r--',label=r'$\sigma^2_0$'); i+=1
plt.plot(Res[:,0],Res[:,i],'r:',label=r'$\sigma^2_L$'); i+=1

plt.xlabel('$t$')
plt.legend()

plt.title(r'Variance of $\log \rho \, (A={:1.1f})$'.format(A))

Res2=np.array(Res2)
plt.figure()
plt.plot(Res2[:,0],Res2[:,1])
plt.ylabel('$\sigma^2_B-2(2-D_1)\mu_B$')
plt.xlabel('$t$')
#%%% * Various A
s=3


L1,S1,wrapped_time,W1,t1=run_DSM(1e7,0.4,s,STOP_ON_LMAX=True)
L2,S2,wrapped_time,W2,t2=run_DSM(1e7,0.9,s,STOP_ON_LMAX=True)
L3,S3,wrapped_time,W3,t3=run_DSM(1e7,1.8,s,STOP_ON_LMAX=True)

a=1.4
#%%%% * 1/rho ~ 1/n
plt.figure(figsize=(3,2)) 
M=['d','o','s','*']


l0=0.3
sB=1/100

i=1
a=0.4
Lmod=np.mod(L1,1)
dist_old=np.sum(np.diff(L1,axis=0)**2,1)**0.5
#	for i,sB in enumerate([1/100]):
N1=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1,sB)
									 ,weights=dist_old/sB,density=False)[0]
h,x=np.histogram(N1.mean()/(N1[N1>=1]),np.logspace(-2,3,50),density=True)
plt.plot(x[1:],h,M[i],c=plt.cm.cool(i/3.),label=r'$A={:1.1f}$'.format(a))
sm=np.average(1/S1)
h,x=np.histogram(S1*sm,np.logspace(-2,6,50),density=True)
h[h==0]=np.nan
plt.plot(x[1:],h,'-',c=plt.cm.cool(i/3.))#,label=r'$P_{\langle \rho \rangle /\rho}$')


i=2
a=0.9
Lmod=np.mod(L2,1)
dist_old=np.sum(np.diff(L2,axis=0)**2,1)**0.5
#	for i,sB in enumerate([1/100]):
N1=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1,sB)
									 ,weights=dist_old/sB,density=False)[0]
h,x=np.histogram(N1.mean()/(N1[N1>=1]),np.logspace(-2,3,50),density=True)
plt.plot(x[1:],h,M[i],c=plt.cm.cool(i/3.),label=r'$A={:1.1f}$'.format(a))
sm=np.average(1/S2)
h,x=np.histogram(S2*sm,np.logspace(-2,6,50),density=True)
h[h==0]=np.nan
plt.plot(x[1:],h,'-',c=plt.cm.cool(i/3.))#,label=r'$P_{\langle \rho \rangle /\rho}$')


i=3
a=1.8
Lmod=np.mod(L3,1)
dist_old=np.sum(np.diff(L3,axis=0)**2,1)**0.5
#	for i,sB in enumerate([1/100]):
N1=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1,sB)
									 ,weights=dist_old/sB,density=False)[0]
h,x=np.histogram(N1.mean()/(N1[N1>=1]),np.logspace(-2,3,50),density=True)
plt.plot(x[1:],h,M[i],c=plt.cm.cool(i/3.),label=r'$A={:1.1f}$'.format(a))
sm=np.average(1/S3)
h,x=np.histogram(S3*sm,np.logspace(-2,6,50),density=True)
h[h==0]=np.nan
plt.plot(x[1:],h,'-',c=plt.cm.cool(i/3.))#,label=r'$P_{\langle \rho \rangle /\rho}$')


plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\langle n \rangle/n, \langle \rho \rangle/\rho$')
plt.xlim([1e-2,1e3])
plt.ylim([1e-5,2e0])
#plt.vlines(sB*l0, 1e-3, 1e4,label='$s_B l_0/A$')
plt.legend()

plt.savefig('/home/joris/Dropbox/Articles/AggregationBatchelor/Figures/Sine_N_A.pdf',bbox_inches='tight')

#%%%% * log rho ~ log n
plt.figure(figsize=(3,2)) 
M=['d','o','s','*']


l0=0.3
sB=1/100

i=1
a=0.4
Lmod=np.mod(L1,1)
dist_old=np.sum(np.diff(L1,axis=0)**2,1)**0.5
#	for i,sB in enumerate([1/100]):
N1=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1,sB)
									 ,weights=dist_old/sB,density=False)[0]
h,x=np.histogram(np.log(N1[N1>=1]),np.linspace(0,8,50),density=True)
plt.plot(x[1:]-np.log(np.mean(N1)),h,M[i],c=plt.cm.cool(i/3.),label=r'$A={:1.1f}$'.format(a))
h,x=np.histogram(np.log(1/S1),np.linspace(-1,20,50),density=True)
h[h==0]=np.nan
plt.plot(x[1:]-np.mean(np.log(1/S1)),h,'-',c=plt.cm.cool(i/3.))#,label=r'$P_{\langle \rho \rangle /\rho}$')
plt.yscale('log')

i=2
a=0.9
Lmod=np.mod(L2,1)
dist_old=np.sum(np.diff(L2,axis=0)**2,1)**0.5
#	for i,sB in enumerate([1/100]):
N1=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1,sB)
									 ,weights=dist_old/sB,density=False)[0]
h,x=np.histogram(N1.mean()/(N1[N1>=1]),np.logspace(-2,3,50),density=True)
plt.plot(x[1:],h,M[i],c=plt.cm.cool(i/3.),label=r'$A={:1.1f}$'.format(a))
sm=np.average(1/S2)
h,x=np.histogram(S2*sm,np.logspace(-2,6,50),density=True)
h[h==0]=np.nan
plt.plot(x[1:],h,'-',c=plt.cm.cool(i/3.))#,label=r'$P_{\langle \rho \rangle /\rho}$')


i=3
a=1.8
Lmod=np.mod(L3,1)
dist_old=np.sum(np.diff(L3,axis=0)**2,1)**0.5
#	for i,sB in enumerate([1/100]):
N1=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1,sB)
									 ,weights=dist_old/sB,density=False)[0]
h,x=np.histogram(N1.mean()/(N1[N1>=1]),np.logspace(-2,3,50),density=True)
plt.plot(x[1:],h,M[i],c=plt.cm.cool(i/3.),label=r'$A={:1.1f}$'.format(a))
sm=np.average(1/S3)
h,x=np.histogram(S3*sm,np.logspace(-2,6,50),density=True)
h[h==0]=np.nan
plt.plot(x[1:],h,'-',c=plt.cm.cool(i/3.))#,label=r'$P_{\langle \rho \rangle /\rho}$')


plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\langle n \rangle/n, \langle \rho \rangle/\rho$')
plt.xlim([1e-2,1e3])
plt.ylim([1e-5,2e0])
#plt.vlines(sB*l0, 1e-3, 1e4,label='$s_B l_0/A$')
plt.legend()

plt.savefig('/home/joris/Dropbox/Articles/AggregationBatchelor/Figures/Sine_N_A.pdf',bbox_inches='tight')
#%%% * var(log n) as a function of D1
Lmax=1e7

def parrallel(s):
	AA=np.linspace(0.4,1.8,10)
	Var=[]
	for A in AA:
		L1,S1,wrapped_time,W1,t1=run_DSM(Lmax,A,s,STOP_ON_LMAX=True)
		Lmod=np.mod(L1,1)
		dist_old=np.sum(np.diff(L1,axis=0)**2,1)**0.5
		for ii,sB in enumerate([1/25,1/50,1/100,1/250,1/500]):
			N1=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1,sB)
												 ,weights=dist_old/sB,density=False)[0]
			Var.append([A,sB,np.var(N1/np.mean(N1))])
	return Var


Seeds=np.arange(10)
pool = multiprocessing.Pool(processes=len(Seeds))
Nuall_seeds=pool.map(parrallel, Seeds)
pool.close()
pool.join()

nuallmean=np.mean(np.array(Nuall_seeds),axis=0)

np.savetxt('Sine_Var(logn)_1e7.txt',nuallmean,header='# A,sB,Var(logN)')
#%%% * var(log n) as a function of sB
Lmax=1e7
A=0.9
s=3

L1,S1,wrapped_time,W1,t1=run_DSM(Lmax,A,s,STOP_ON_LMAX=True)
Lmod=np.mod(L1,1)
dist_old=np.sum(np.diff(L1,axis=0)**2,1)**0.5

SB=np.linspace(1/500,1/25,20)
Var=[]
for ii,sB in enumerate(SB):
	print(sB)
	N1=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1,sB)
										 ,weights=dist_old/sB,density=False)[0]
	Var.append([A,sB,np.var(N1/np.mean(N1))])


Lmax=1e8
L1,S1,wrapped_time,W1,t1=run_DSM(Lmax,A,s,STOP_ON_LMAX=True)
Lmod=np.mod(L1,1)
dist_old=np.sum(np.diff(L1,axis=0)**2,1)**0.5

SB=np.linspace(1/500,1/25,20)
Var2=[]
for ii,sB in enumerate(SB):
	print(sB)
	N1=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1,sB)
										 ,weights=dist_old/sB,density=False)[0]
	Var2.append([A,sB,np.var(N1/np.mean(N1))])

plt.figure()
Var=np.array(Var)
plt.plot(Var[:,1],Var[:,2],'*')
plt.xscale('log')

Var2=np.array(Var2)
plt.plot(Var2[:,1],Var2[:,2],'o')
plt.plot(SB,0.05*np.log(1/SB)**2,'-')
plt.xscale('log')
plt.yscale('log')

#%%% * var(log n) as a fct of time : Compute

ii=0
sB=1/50
s=3
l0=0.3
SB=[1/50,1/100,1/500]
AA=[0.4,0.6,0.8,1.8]
Var,Mean=[],[]
for j,A in enumerate(AA):
	for Lmax in np.logspace(4,8,15):
		AA=np.logspace(-3,np.log10(0.49),20)
		L1,S1,wrapped_time,W1,t1=run_DSM(Lmax,A,s,STOP_ON_LMAX=True)
		Lmod=np.mod(L1,1)
		dist_old=np.sum(np.diff(L1,axis=0)**2,1)**0.5
		for ii,sB in enumerate(SB):
			#	for i,sB in enumerate([1/100]):
			N1=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1,sB)
												 ,weights=dist_old/sB,density=False)[0]
			N=np.uint32(N1)
			Var.append([A,t1,sB,np.var(np.log(N[N>0])),np.var(np.log(1/S1))])

Var=np.array(Var)
#%%% * var(log n) as a fct of time : Parrallel
import multiprocessing

ii=0
sB=1/50
s=3
l0=0.3
SB=[1/50,1/100,1/500]
AA=[0.3,0.4,0.6,0.8,1.2,1.8]

def parrallel(A):
	Var=[]
	for Lmax in np.logspace(4,8,20):
		L1,S1,wrapped_time,W1,t1=run_DSM(Lmax,A,s,STOP_ON_LMAX=True)
		Lmod=np.mod(L1,1)
		dist_old=np.sum(np.diff(L1,axis=0)**2,1)**0.5
		for ii,sB in enumerate(SB):
			#	for i,sB in enumerate([1/100]):
			N1=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1,sB)
												 ,weights=dist_old/sB,density=False)[0]
			N=np.uint32(N1)
			Var.append([A,t1,sB,np.var(np.log(N[N>0])),np.var(np.log(1/S1)),np.mean(np.log(N[N>0]))])
	return Var

pool = multiprocessing.Pool(processes=len(AA))
Var=pool.map(parrallel, AA)
pool.close()
pool.join()
VarA=np.array(Var).reshape(-1,5)

np.savetxt('Sine_logN_t.txt',VarA)
#%%% * var( n) as a fct of time : Parrallel
import multiprocessing

ii=0
sB=1/50
s=3
l0=0.3
SB=[1/50,1/100,1/500]
AA=[0.3,0.4,0.6,0.8,1.2,1.8]

def parrallel(A):
	Var=[]
	for Lmax in np.logspace(4,8,20):
		L1,S1,wrapped_time,W1,t1=run_DSM(Lmax,A,s,STOP_ON_LMAX=True)
		Lmod=np.mod(L1,1)
		dist_old=np.sum(np.diff(L1,axis=0)**2,1)**0.5
		Ltot=np.sum(dist_old)
		for ii,sB in enumerate(SB):
			#	for i,sB in enumerate([1/100]):
			N1=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1,sB)
												 ,weights=dist_old/sB,density=False)[0]
			N=np.uint32(N1)
			Var.append([A,t1,sB,Ltot,np.mean(N),np.var(N),np.mean(N[N>0]),np.var(N[N>0])])
	return Var

pool = multiprocessing.Pool(processes=len(AA))
Var=pool.map(parrallel, AA)
pool.close()
pool.join()
VarA=np.array(Var).reshape(-1,8)

np.savetxt('Sine_N_t.txt',VarA)
#%%% * var(log n) as a fct of time : Plot
l0=0.3
import scipy.special
VarA=np.loadtxt('Sine_N_t.txt')
Lyap=np.loadtxt('Sine_Lyap.txt')
D1=np.loadtxt('Sine_D1.txt')
AA=np.unique(VarA[:,0])
SB=np.unique(VarA[:,2])

M=['d','o','s','*']
fig=plt.figure()
for j,A in enumerate(AA):
	idA=np.where(VarA[:,0]==A)[0]
	Var=VarA[idA,1:]
	lyap=np.interp(A,Lyap[:,0],Lyap[:,1])
	sigma2=np.interp(A,Lyap[:,0],Lyap[:,2])
	Var=np.array(Var)
	d1=np.interp(A,D1[:,0],D1[:,1])
	d1max=2.1
	for ii,sB in enumerate(SB):
		idsB=np.where(Var[:,1]==sB)[0]
		tagg=np.log(1/(sB*l0))/(lyap+sigma2/2)
	#	tagg=1
		ni=np.log(1/sB)*(2*(d1max-d1))#/(d1-d1max+1)**4
		plt.plot((Var[idsB,0]-tagg)*lyap,Var[idsB,4]/ni,M[ii]+'-',color=plt.cm.cool(((A-AA.min())/(AA.max()-AA.min()))))#,label=r'$s_B=1/{:1.0f}$'.format(1/sB))

plt.yscale('log')
plt.ylim([0,10])
plt.xlabel('$(t-t_\mathrm{agg})\mu$')
plt.ylabel(r'$\sigma^2_{\log n} / \log(1/s_B) /[2(2-D_1)]$')
[plt.plot([],[],M[k]+'-',color='k',label=r'$s_B=1/{:1.0f}$'.format(1/sb)) for k,sb in enumerate(SB)]

t=np.linspace(-3,5,100)
#plt.plot(t,((1+scipy.special.erf(t-2.5))/2)**0.2,'-',color='k',label=r'$\sim \mathrm{erf} (t)^{0.2}$')

plt.legend()
plt.ylim(1e-3,5)
plt.xlim(-3,8)

ax2 = fig.add_axes([0.6, 0.55, 0.3, 0.05])
import matplotlib as mpl
#norm = mpl.colors.Normalize(vmin=T_all[0]*u_pore/d, vmax=T_all[-1]*u_pore/d)
norm = mpl.colors.Normalize(vmin=AA.min(), vmax=AA.max())
cb1 = mpl.colorbar.ColorbarBase(ax2,cmap=plt.cm.cool,norm=norm,
								orientation='horizontal')
cb1.set_label(r'$A$', color='k',size=12,labelpad=0)
cb1.set_ticks(AA)


plt.legend()


plt.savefig('/home/joris/Dropbox/Articles/AggregationBatchelor/Figures/Sine_N_t.pdf',bbox_inches='tight')

plt.figure()
plt.plot(D1[:,1],Lyap[:,2]-Lyap[:,1]*2*(2-D1[:,1]))
plt.plot(D1[:,1],Lyap[:,2]/Lyap[:,1])
plt.plot(D1[:,1],-2*D1[:,1]+4.35)
#%%% * mean(log n) as a fct of time : Plot
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
		plt.plot(np.log(Var[idsB,2]),Var[idsB,3]/ni,M[ii]+'-',color=plt.cm.cool(((A-AA.min())/(AA.max()-AA.min()))))#,label=r'$s_B=1/{:1.0f}$'.format(1/sB))
		plt.plot(np.log(Var[idsB,2]),np.sqrt(Var[idsB,4]/nivar),M[ii]+'--',color=plt.cm.cool(((A-AA.min())/(AA.max()-AA.min()))))#,label=r'$s_B=1/{:1.0f}$'.format(1/sB))

t=np.linspace(1,14,100)
plt.yscale('log')
#plt.ylim([0,10])
plt.xlabel('$\log L(t)$')
plt.plot([],[],'--',color='k',label=r'$\sigma_n \mathcal{A} / s_B / \sqrt{s_B^{D_2-2}-1}$')
plt.plot([],[],'-',color='k',label=r'$\mu_{n} \mathcal{A} / s_B$')
plt.plot(t,np.exp(t),'-',color='k',label=r'$L(t)$',linewidth=2,zorder=-10)
[plt.plot([],[],M[k],color='k',label=r'$s_B=1/{:1.0f}$'.format(1/sb)) for k,sb in enumerate(SB)]

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



plt.savefig('/home/joris/Dropbox/Articles/AggregationBatchelor/Figures/Sine_meanN_t.pdf',bbox_inches='tight')

plt.figure()
plt.plot(D1[:,1],Lyap[:,2]-Lyap[:,1]*2*(2-D1[:,1]))
plt.plot(D1[:,1],Lyap[:,2]/Lyap[:,1])
plt.plot(D1[:,1],-2*D1[:,1]+4.35)
#%%% * p(N) for several times
from scipy import special
plt.style.use('~/.config/matplotlib/joris.mplstyle')
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
plt.savefig('/home/joris/Dropbox/Articles/AggregationBatchelor/Figures/Sine_P(N,t)_A{:1.2f}.pdf'.format(A),bbox_inches='tight')

#%%% NEgative Binomial & Gamma
from scipy import special
from scipy.stats import nbinom,gamma
x = np.arange(20)
n=3
p=0.1

k=4.0
theta=10

n=k/(1-^1/theta)
p=1/theta

plt.plot(x, nbinom.pmf(x, n, p), 'o')
k=n*(1-p)
theta=1/p

gammapdf=1/special.gamma(k)/theta**k*x**(k-1)*np.exp(-x/theta)
plt.plot(x, gammapdf, '+')
plt.yscale('log')
plt.xscale('log')




#%%% * var(log n) as a fct of log rho : Plot

import scipy.special
VarA=np.loadtxt('Sine_N_t.txt')
Lyap=np.loadtxt('Sine_Lyap.txt')
D1=np.loadtxt('Sine_D1.txt')
AA=np.unique(VarA[:,0])
SB=np.unique(VarA[:,2])

M=['d','o','s','*']
fig=plt.figure()
for j,A in enumerate(AA):
	idA=np.where(VarA[:,0]==A)[0]
	Var=VarA[idA,1:]
	lyap=np.interp(A,Lyap[:,0],Lyap[:,1])
	sigma2=np.interp(A,Lyap[:,0],Lyap[:,2])
	Var=np.array(Var)
	d1=np.interp(A,D1[:,0],D1[:,1])
	for ii,sB in enumerate(SB):
		idsB=np.where(Var[:,1]==sB)[0]
		tagg=np.log(1/(sB*l0))/(lyap+sigma2/2)
	#	tagg=1
		ni=np.log(1/sB)*(2-d1)/(d1-1)**1
		plt.plot(Var[idsB,2],(d1-1)**2*(sigma2*Var[idsB,0]-2*(2-d1)*lyap*Var[idsB,0]),M[ii]+'-',color=plt.cm.cool(((A-AA.min())/(AA.max()-AA.min()))))#,label=r'$s_B=1/{:1.0f}$'.format(1/sB))

#plt.yscale('log')
plt.ylim([0,2])
#plt.yscale('log')
plt.xlim([0,2])
plt.xlabel('$\sigma^2_{\log n}$')

plt.ylabel(r'$f(\mu_{\log\rho},\sigma^2_{\log\rho},D_1)$')
plt.legend()

#%%% Various Pe
plt.figure()

sB=1/1000

Lmod=np.mod(L1,1)
dist_old=np.sum(np.diff(L1,axis=0)**2,1)**0.5
for i,sB in enumerate([1/25,1/50,1/100,1/500,1/1000]):
	N1=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1,sB)
										 ,weights=dist_old/sB,density=False)[0]
	h,x=np.histogram(N1[N1>2],np.logspace(0,8,50),density=True)
	plt.plot(x[1:],h,'o',c=plt.cm.cool(i/5.),label=r'$P_{1/n}'+r', s_B=1/{:1.0f}$'.format(1/sB))
	h,x=np.histogram(1/S1,np.logspace(0,8,50),density=True,weights=S1)
	plt.plot(x[1:]*l0*sB,h/(l0*sB),'-',c=plt.cm.cool(i/5.),label=r'$P_{1/\rho,0}$')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$n,\rho$')

#plt.vlines(sB*l0, 1e-3, 1e4,label='$s_B l_0/A$')
#plt.legend()

#%%% ... Lagrangian
from scipy import spatial

L=L1

tree=spatial.cKDTree(np.mod(L,1))
ng=int(1/sB)
ng=50
idshuffle=np.arange(len(L)-3)
np.random.shuffle(idshuffle)
Lmod=np.mod(L[:-2],1)
Lmodint=np.uint32(Lmod*ng)
idu=np.unique(np.copy(Lmodint[idshuffle,:]), return_index=True,axis=0)
idsamples=idshuffle[idu[1]]
neighboors=tree.query_ball_point(np.mod(L[idsamples,:],1), sB/2.)
neighboors_uniq=[[] for k in range(len(idsamples))]
neighboors_all=[[] for k in range(len(idsamples))]
for ii,ns in enumerate(idsamples):
	print(ii)
	nt=np.sort(neighboors[ii])
	kk_all=np.concatenate((nt,np.array([ns])))
	neighboors_all[ii].append(kk_all)
	idgood=np.where(np.diff(nt)>3*sB/dx)[0]
	NT=[]
	for idg in idgood:
		if (np.abs(nt[idg+1]-ns)>3*sB/dx)&(np.abs(nt[idg]-ns)<3*sB/dx): # depending on the direction of the filament
			NT.append(nt[idg+1])
		else:
			NT.append(nt[idg])
	nt=np.array(NT,dtype=np.uint32)
	kk=np.concatenate((nt,np.array([ns])))
	neighboors_uniq[ii].append(kk)
nagg=np.array([len(n[0]) for n in neighboors_uniq])


nh,xh=np.histogram(np.log(nagg),50,density=True)
plt.plot(xh[1:],nh,'k*')
Lmod=np.mod(L1,1)
dist_old=np.sum(np.diff(L1,axis=0)**2,1)**0.5
N1=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1,sB)
									 ,weights=dist_old/sB,density=False)[0]
h,x=np.histogram(np.log(N1[N1>0]),50,density=True)
plt.plot(x[1:],h,'o',c=plt.cm.cool(t1/t1))
hs,xs=np.histogram((D1-1)*np.log(1/rhoc/S1),50,weights=W1,density=True)
plt.plot(xs[1:],hs,'-',c=plt.cm.cool(t1/t1))

#%%% Compare p(n) to p(s)
plt.figure()
#D1=1.77
D1=2
rhoc=250
nbin=np.logspace(0,5,50)
sbin=np.logspace(-10,0,50)
Lmod=np.mod(L1,1)
dist_old=np.sum(np.diff(L1,axis=0)**2,1)**0.5
N1=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.arange(0,1,sB)
									 ,weights=dist_old/sB,density=False)[0]
h,x=np.histogram(N1[N1>0],nbin,density=True)
plt.plot(x[1:],h,'o',c=plt.cm.cool(t1/t1))
hs,xs=np.histogram(S1*rhoc,sbin,weights=W1,density=True)
plt.plot(1/xs[1:],hs,'+',c=plt.cm.cool(t1/t1))
plt.yscale('log')
plt.xscale('log')

#%%% theoretical p(n) = p(log rho)/p(log rho|n)

t=5
mu=0.5*t
sigma2=0.5*t

D1=1.8
alpha=D1-1
beta=2(2-D1)/(D1-1)

mut=


#%%% <N(t)>

# obtained by fitting
if keyword=='sine':  
	ttttt=[0,1,2,3,4,5,6,7,8,9,10,11,12]
	nconv=[0,0,0,0,0,0,1,1,3,5,8,16,23]
if keyword=='half':
	ttttt=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
	nconv=[0,0,0,0,0,0,0,0,0,0,0 ,0 ,0 ,0 ,0 ,0 ,2 ,3 ,3 ,3 ,3 ,5 ,5, 6 ]
plt.figure(figsize=(3,3))
tv=np.array(tv)
kI=np.array(kI)
#plt.plot(tv,I.mean()/np.array(Cmaxmean),label=r'$\langle c \rangle/ \langle \theta \rangle$')
plt.plot(tv,np.array(Cmaxvar),label=r'$\langle \theta^2 \rangle_L$')
plt.plot(tv,0.6*np.exp(-2*lyap*tv),'--',label=r'$\langle \theta^2 \rangle_L$')
plt.plot(np.arange(len(Lmean)),1+np.array(Lmean)*sB,'k-',label='$L\, {s_B}/A$')
#plt.plot(np.arange(len(Lmean)),1+l0*np.exp(3*0.60/2*np.arange(len(Lmean)))*sB,'k--',label='$L\, {s_B}/A$')
plt.plot(np.arange(len(Lmean)),np.array(Nmean),'k.-',label=r'$\langle N \rangle$')
plt.plot(np.arange(len(Lmean)),np.array(Nmean_eul),'k:',label=r'$\langle N \rangle (eul)$')
plt.plot(np.arange(len(Lmean)),np.array(Nmean_eul_all),'k.--',label=r'$\langle N \rangle (eul,all)$')
plt.plot(tv,kI,'r:',label=r'$N\sim \langle c \rangle^2/\sigma^2_c$')
plt.plot(tv,kIL,'r.:',label=r'$N\sim \langle c \rangle^2/\sigma^2_c$ on $\theta$')
#plt.plot(ttttt,nconv,'ko',label='from ${\otimes N}$ convolutions')
plt.plot(tv[7:],0.003*np.exp((lyap+sigma/2)*tv[7:]),'k-',label=r'$\lambda +\sigma^2/2$')
plt.plot(tv[7:],0.01*np.exp((lyap**2./(2*sigma))*tv[7:]),'r-',label=r'${\lambda^2/(2\sigma^2)}$')
plt.yscale('log')
plt.ylabel('$N$')
plt.legend(ncol=2)
plt.xlabel('$t$')
plt.savefig(dir_out+'Nagg_Naggfit.pdf',
						bbox_inches='tight')
np.savetxt('N(t)_sine_flow_D{:1.1e}.txt'.format(Brownian),np.vstack((np.arange(len(Lmean))*lyap,np.array(Lmean)*sB,np.array(Nmean),np.array(Nmean_eul),np.array(Nmean_eul_all))).T,header='t*lyap,L*sB/A,N,Neulerian,Neulerian_all')
np.savetxt('k_sine_flow.txt',np.vstack((tv*lyap,kI)).T,header='t*lyap,<c>^2/<c^2>')
#

plt.figure()
lyap=0.6
sigma=0.55
plt.plot(np.arange(len(Lmean)),np.array(Lmean),'--')
plt.plot(np.arange(len(Lmean)),l0*np.exp(np.arange(len(Lmean))*(lyap+sigma/2)))
plt.yscale('log')
#%%% <N(t)>

plt.figure(figsize=(3,3))
tv=np.array(tv)
kI=np.array(kI)
#plt.plot(tv,I.mean()/np.array(Cmaxmean),label=r'$\langle c \rangle/ \langle \theta \rangle$')

plt.plot(np.arange(len(Lmean)),np.array(Lmean)*sB*np.sqrt(np.pi/2),'k-',label='$\sqrt{\pi/2} L\, {s_B}/A$')
#plt.plot(np.arange(len(Lmean)),1+l0*np.exp(3*0.60/2*np.arange(len(Lmean)))*sB,'k--',label='$L\, {s_B}/A$')
plt.plot(np.arange(len(Lmean)),np.array(Nmean)-1,'k.-',label=r'$\langle N \rangle (lag)$')
plt.plot(np.arange(len(Lmean)),np.array(Nmean_eul)-1,'k:',label=r'$\langle N \rangle (eul)$')
plt.plot(np.arange(len(Lmean)),np.array(Nmean_eul_all),'k.--',label=r'$\langle N \rangle (eul,all)$')

plt.yscale('log')
plt.ylabel('$N$')
plt.legend(ncol=2)
plt.xlabel('$t$')


#%%% Coalescence time 


keyword='sine'

dx=0.001

plt.style.use('~/.config/matplotlib/joris.mplstyle')
import h5py
from skimage.morphology import disk
from skimage.filters import median,gaussian
ms=['o','d','s','<','>','+','*']
radius=0.01
l0=0.3
A=1/np.sqrt(2)
#A=1.2
BR=np.logspace(-2,-4,10)
for Brownian in BR:
	D=(Brownian)**2./(2*dt)#+1e-08 # numerical diffusion ?
	dir_out='./Compare_stretching_concentration/'+keyword+'/Fourier_A{:1.1f}_l{:1.1f}_rad{:1.2f}_{:1.0e}'.format(A,l0,radius,Brownian)
	periodicity='periodic'
	
	#I0med = gaussian(I0, sigma=3)
	Imax=np.max(I0med)
	f=h5py.File('./Compare_stretching_concentration/'+keyword+'/DSM'+keyword+'_A{:1.1f}_l{:1.1f}.hdf5'.format(A,l0), 'r')
	try:
		T=np.arange(f.attrs['tmax'])
	except:
		T=np.arange(12)
	
	if keyword=='sine':
		lyap=0.65
		sigma=lyap*0.65
		
	if keyword=='half':
		lyap=0.25
		sigma=0.26
	
	sB=np.sqrt(D/lyap)
	#sB=np.mean(Sc)
	# Taking a unique sB might not be the good approach
	# We need about 2*sB to get a correct sum
	sBx=sB*1.96
	
	Nagg=[]
	for t in T:
		L=f['L_{:04d}'.format(int(t*10))][:]
		dist_old=np.sum(np.diff(L,axis=0)**2,1)**0.5
		Lmod=np.mod(L,1)
		N=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.linspace(0,1,int(1/sBx)),
										weights=dist_old/sBx,density=False)[0]
		Nagg.append(np.mean(N[N>0]))
	
	Nagg=np.array(Nagg)
	plt.plot(T,Nagg-1)
	plt.yscale('log')
plt.plot(T,np.exp(T*(lyap+sigma/2)),'k--')
plt.plot([0,t],[1,1])
#%%% Eulerian N(t)
keyword='sine'
keyword='half'
if keyword=='half': fractal=1.50
if keyword=='sine': fractal=1.7
if keyword=='single': fractal=1.7

dx=0.001

plt.style.use('~/.config/matplotlib/joris.mplstyle')
import h5py
from skimage.morphology import disk
from skimage.filters import median,gaussian
ms=['o','d','s','<','>','+','*']
radius=0.01
l0=0.3
A=1/np.sqrt(2)
#A=1.2
BR=np.logspace(-2,-4,10)
Brownian=1e-3
D=(Brownian)**2./(2*dt)#+1e-08 # numerical diffusion ?
dir_out='./Compare_stretching_concentration/'+keyword+'/Fourier_A{:1.1f}_l{:1.1f}_rad{:1.2f}_{:1.0e}'.format(A,l0,radius,Brownian)
periodicity='periodic'

#I0med = gaussian(I0, sigma=3)
#Imax=np.max(I0med)
f=h5py.File('./Compare_stretching_concentration/'+keyword+'/DSM'+keyword+'_A{:1.1f}_l{:1.1f}.hdf5'.format(A,l0), 'r')
try:
	T=np.arange(f.attrs['tmax'])
except:
	T=np.arange(12)

if keyword=='sine':
	lyap=0.65
	sigma=lyap*0.65
	
if keyword=='half':
	lyap=0.25
	sigma=0.26

sB=np.sqrt(D/lyap)
#sB=np.mean(Sc)
# Taking a unique sB might not be the good approach
# We need about 2*sB to get a correct sum
sBx=sB*1.96

Nagg=[]
for t in T:
	L=f['L_{:04d}'.format(int(t*10))][:]
	S=f['S_{:04d}'.format(int(t*10))][:]
	dist_old=np.sum(np.diff(L,axis=0)**2,1)**0.5
	Lmod=np.mod(L,1)
	Na=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.linspace(0,1,int(1/sBx)),
									density=False)[0]
	N=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.linspace(0,1,int(1/sBx)),
									weights=dist_old/sBx,density=False)[0]
	logrho=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.linspace(0,1,int(1/sBx)),
									weights=-np.log(S),density=False)[0]
	rho=np.histogram2d(Lmod[1:,0],Lmod[1:,1],bins=np.linspace(0,1,int(1/sBx)),
									weights=1/S,density=False)[0]
#Nagg.append(np.mean(N[N>0]))

binN=np.linspace(0,10,20)
rhoBin=bin_operation(np.log(N),logrho/Na,binN,np.nanmean)
plt.plot(np.log(N[N>1]),logrho[N>1]/Na[N>1],'k.',alpha=0.02)
plt.plot(np.log(N[N>1]),np.log(rho[N>1]/Na[N>1]),'r.',alpha=0.02)
plt.plot(binN[1:],rhoBin,'ro',alpha=1)
logn=np.linspace(0,6,10)
plt.plot(logn,1/(fractal-1)*logn+7.2,'r-')
plt.plot(logn,logn+7.2,'k-')
#plt.plot(logn,1.0*logn+8.3,'r--')
plt.ylim([0,30])
#plt.xscale('log')

#%% Single Bundle evolution
#%%%Plot Cmax_i(t) in a single bundle
# RUn twice with idagg initialized to a value
import h5py

#idagg=[10] # Aggregated lamellae to follow

Cmaxfollow=[]
IsinBundle=[]

Tmax=11

l0=0.3
#% Advection parameters
INTERPOLATE='NO'
CURVATURE='DIFF'

PLOT=False
dx=0.001 # Maximum distance between points, above which the line will be refined
alpha=200*dx/np.pi
dt=0.25 # dt needs to be small compare to the CFL condition
npar=6 # Number of parallel processors
tmax=Tmax # Maximum advection time
Lmax=5e7 # Maximum number of points (prevent blow up)

A=1/np.sqrt(2) # Ampluitude

radius=0.01
s0=radius
Brownian=5e-4
D=(Brownian)**2./(2*dt)#+1e-08 # numerical diffusion ?
#*np.sqrt(2)
sB=np.sqrt(D/0.5)
# Initial segment position and distance
x=np.linspace(0,2*np.pi,int(1.8e3))
#L=np.array([radius*np.cos(x),radius*np.sin(x)],dtype=np.float64).T
#L[0,:]=L[-1,:]
n=int(1e7)
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

# MAIN PARALLEL LOOP #######################
while (t<tmax)&(len(L)<Lmax):
	
	v=vel(L,t,A)
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
	Tau=D/s0**2*wrapped_time
	Cmax=1./np.sqrt(1.+4*D/s0**2*wrapped_time)
	# Check bundle
	Cmaxfollow.append(Cmax[idagg])
	IsinBundle.append(np.sqrt(np.sum((np.mod(L[idagg[:-1],:],1)-np.mod(L[idagg[-1],:],1))**2.,axis=1))<=5*sB)
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
	print('Time:',t,' - Npts:',len(L), '- Cmax:',np.mean(1./np.sqrt(1.+wrapped_time/1000)),np.max(kappa_new))
	
# End of MAIN LOOOP #######################
print('Computation time:', time.time() -ct)

# Find an aggregated zone
dx=0.01*sB # Maximum distance between points, above which the line will be refined
from scipy import spatial
tree=spatial.cKDTree(np.mod(L,1))
nsamples=int(sB/dx*10)
nsamples=10000
idsamples=np.uint32(np.linspace(0,L.shape[0]-2,nsamples))
neighboors=tree.query_ball_point(np.mod(L[idsamples,:],1), sB)
neighboors_uniq=[[] for k in range(len(idsamples))]
neighboors_all=[[] for k in range(len(idsamples))]
dist_all=np.hstack((0,np.cumsum(dist_old)))
for ii,ns in enumerate(idsamples):		
	nt=np.sort(neighboors[ii])
	kk_all=np.concatenate((nt,np.array([ns])))
	neighboors_all[ii].append(kk_all)
	idgood=np.where(np.diff(dist_all[nt])>5*sB)[0]
	kk_all=np.concatenate((nt[idgood],np.array([ns])))
	# remove extra lamella
	kk_all=np.delete(kk_all,np.where(((dist_all[kk_all[:-1]]-dist_all[kk_all[-1]])<5*sB))[0])
	neighboors_uniq[ii].append(kk_all)
nagg=np.array([len(n[0]) for n in neighboors_uniq])
idmax=np.argmax(nagg)
idagg=neighboors_uniq[idmax][0] # last id is the ref lamella

plt.style.use('~/.config/matplotlib/joris.mplstyle')
CmaxF=np.array(Cmaxfollow)
IsinBundle=np.array(IsinBundle)
t=np.arange(CmaxF.shape[0])*dt
plt.figure(figsize=(1.5,1.5))
for k in range(CmaxF.shape[1]-1):
	plt.plot(t[np.where(~IsinBundle[:,k])[0]],(CmaxF[np.where(~IsinBundle[:,k])[0],k]),'k-')
	plt.plot(t[np.where(IsinBundle[:,k])[0]],(CmaxF[np.where(IsinBundle[:,k])[0],k]),'k-')
plt.plot(t,(CmaxF[:,-1]),'r-',linewidth=1.5)
#plt.yscale('log')
plt.xlabel('$t$')
plt.ylabel(r'$\theta_i$')
plt.legend()
plt.savefig(dir_out+'./Compare_stretching_concentration/cmax_in_a_bundle_lin.pdf',bbox_inches='tight')
#%
plt.figure(figsize=(1.5,1.5))
plt.plot(t,(CmaxF[:,:-1]),'k-')
plt.plot(t,(CmaxF[:,-1]),'r-',linewidth=1.5)
plt.yscale('log')
plt.xlabel('$t$')
plt.ylabel(r'$\theta_i$')
plt.legend()
plt.savefig(dir_out+'./Compare_stretching_concentration/cmax_in_a_bundle_log.pdf',bbox_inches='tight')

#%%% check aggregation time as a function of rho
#plt.style.use('~/.config/matplotlib/joris.mplstyle')

Tmax_vec=[9]

radius =0.01
l0=0.3

#% Build KDtree 
sB=0.005
#tree=spatial.cKDTree(np.mod(L,1))
nsamples=1000

#neighboors=tree.query_ball_point(np.mod(L[idsamples,:],1), sB)
neighboors_t=[]
neighboors_rho_t=[]

for Tmax in Tmax_vec:
	#from colorline_toolbox import *
	# Inline or windows plots
	#%matplotlib auto
	#%matplotlibTrue inline
	PLOT=True
	PAR=False
	
	def curvature(L,dist):
	# Curvature via derivatives
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
	
	def find_pinch(L,dist_old,th_pinch):
	# Maximum curvature finder
		kappa=curvature(L,dist_old)
		Mkappa=maximum_filter(kappa,50) # Maximum filter on a box of 50dx
		return np.where((Mkappa==kappa)&(kappa>th_pinch))[0]
	
	#% Advection parameters
	INTERPOLATE='NO'
	CURVATURE='DIFF'
	
	PLOT=False
	dx=0.001 # Maximum distance between points, above which the line will be refined
	alpha=200*dx/np.pi
	dt=0.25 # dt needs to be small compare to the CFL condition
	npar=6 # Number of parallel processors
	tmax=Tmax # Maximum advection time
	tsave=0.1 # Time period between saving steps
	Lmax=5e7 # Maximum number of points (prevent blow up)
	th_pinch=100 # Curvature minimum (in nb of dx) to find a peak
	Pe=1e2 # Peclet
	#l0=0.1
	
	# Initial segment position and distance
	x=np.linspace(0,2*np.pi,int(1.8e3))
	L=np.array([radius*np.cos(x),radius*np.sin(x)],dtype=np.float64).T
	L[0,:]=L[-1,:]
	n=int(2e6)
	L=np.array([np.linspace(-l0/2,l0/2,n),np.zeros(n)],dtype=np.float64).T
	
	idsamples=np.arange(0,n,nsamples)
	#	L=[]
	#	for i in range(100):
	#		x,y=np.random.rand(),np.random.rand()
	#		L.append([x-dx,y-dx])
	#		L.append([x+dx,y+dx])
	#		L.append([np.nan,np.nan])
	
	L=np.array(L)
	weights=np.ones(L.shape[0]-1)
	weights=weights/np.sum(weights)
	#L=2*(np.random.rand(2)-0.5)+np.array([x,l0*np.sin(x)],dtype=np.float64).T
	
	dist_old=np.sum(np.diff(L,axis=0)**2,1)**0.5
	weights=np.ones(dist_old.shape)
	# Initial segment width
	S=np.ones(L.shape[0]-1)
	# Initial Wrapped Time
	wrapped_time=np.zeros(L.shape[0]-1)
	
	# Initialization of saved variables
	Lsum=[]
	Ssum=np.sum(dist_old*S)
	Smean=[]
	Rhomean=[]
	logRhomean=[]
	logRhovar=[]
	logKappaMean=[]
	logKappaSum=[]
	logKappaVar=[]
	KappaMean=[]
	logdKappaMean=[]
	logdKappaVar=[]
	logSmean=[]
	logSvar=[]
	Emean=[]
	Evar=[]
	Npinches=[]
	Cmax=[]
	Gamma_PDF_x=np.linspace(-10,10,1000)
	Gamma_PDF=np.zeros(Gamma_PDF_x.shape[0]-1,dtype=np.uint64)
	Lp_x=np.linspace(0,10,100)
	Lp_PDF=np.zeros(Lp_x.shape[0]-1,dtype=np.uint64)
	Lp_var=[]
	
	if PLOT:
		plt.close('all')
		#%matplotlib auto
		plt.close('all')
		plt.ion()
		fig, ax = plt.subplots(figsize=(10,10))
	#	ax.axis([-5,5,-5,5])
	#	line, = ax.plot(L[:,0], L[:,1],'-',alpha=1.,linewidth=0.5)
		ax.axis([-20,20,-3,3])
		ax.set_xlabel(r'$d\ln \kappa /dt$')
		ax.set_ylabel(r'$d\ln \rho /dt$')
		line, = ax.plot(0,0,'-',alpha=1,linewidth=1)
		ax2 = plt.axes([0.65, 0.65, 0.2, 0.2])
		ax2.axis([-np.pi/2.,np.pi/2.,-np.pi/2.,np.pi/2.])
		line2, = ax2.plot(L[:,0], L[:,1],'r-',alpha=1.,linewidth=0.5)
	
	# Initialization
	t=0
	Tv=np.arange(0,tsave+dt,dt) # Vector of time step between saving steps
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
	col=plt.cm.jet(np.linspace(0,1,len(idsamples)))
	# MAIN PARALLEL LOOP #######################
	while (t<tmax)&(len(L)<Lmax):
		v=vel(L,t)
		L+=v*dt		
		# Compute stretching rates and elongations
		dist_new=np.sum(np.diff(L,axis=0)**2,1)**0.5
		gamma=dist_new/dist_old # elongation
		S=S/gamma
		wrapped_time=wrapped_time+dt*(1./S)**2.
		#Force periodicity
		#L[0,:]=L[-1,:]		
		# Compute new aggregations	
		# Build tree
		tree=spatial.cKDTree(np.mod(L,1))
		#find neighboors of selected points
		neighboors=[]
		neighboors_rho=[]
		for ii,ns in enumerate(idsamples):		
			nt=np.sort(tree.query_ball_point(np.mod(L[ns,:],1), sB))
			idgood=np.where(np.diff(nt)>3*sB/np.mean(dist_old))[0]
			NT=[]
			for idg in idgood:
				if (np.abs(nt[idg+1]-ns)>3*sB/np.mean(dist_old))&(np.abs(nt[idg]-ns)<3*sB/np.mean(dist_old)): # depending on the direction of the filament
					NT.append(nt[idg+1])
				else:
					NT.append(nt[idg])
			nt=np.array(NT,dtype=np.uint32)
			kk=np.concatenate((nt,np.array([ns])))
			neighboors.append(kk)
			neighboors_rho.append(np.concatenate((np.array([-np.log(S[np.minimum(nsi,S.shape[0]-1)])/t for nsi in nt]),
																				 np.array([-np.log(S[np.minimum(ns,S.shape[0]-1)])/t]))))
			if t+dt>=tmax:
				#plt.plot(np.mod(L[:,0],1),np.mod(L[:,1],1),'+-',alpha=0.2)
				plt.plot(np.mod(L[kk,0],1),np.mod(L[kk,1],1),'o',color=col[ii,:])
				plt.gca().add_patch(plt.Circle((np.mod(L[ns,0],1),np.mod(L[ns,1],1)), sB,color=col[ii,:],fill=False))
		neighboors_t.append(neighboors)
		neighboors_rho_t.append(neighboors_rho)
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
		
		#Save variables
		Lsum.append(np.sum(dist_old)) # Total length
		Rhomean.append(np.average(1./S,weights=W)) # Mean width
		logRhomean.append(np.average(np.log(1./S),weights=W)) # Mean width
		logRhovar.append(np.average((np.log(1./S)-logRhomean[-1])**2./(1.-np.sum(W**2.)),weights=W)) # Var Log Rho
	
		Smean.append(np.average(S,weights=dist_old)) # Mean width
		logSmean.append(np.average(np.log(S),weights=W)) # Mean width
		logSvar.append(np.average((np.log(S)-logSmean[-1])**2./(1.-np.sum(W**2.)),weights=W)) # Var width
	
		KappaMean.append(np.average(kappa_old,weights=W)) # !!! do we take weighted average or normal average ?
		logKappaMean.append(np.average(np.log(kappa_old),weights=W))
		logKappaSum.append(np.nansum(np.log(kappa_old)))
		logKappaVar.append(np.average((np.log(kappa_old)-logKappaMean[-1])**2./(1.-np.sum(W**2.)),weights=W)) # Var Log Rho
		logdKappaMean.append(dlKMean)
		logdKappaVar.append(dlKVar) # Variance of Elongation
	#	pinches=find_pinch(L,dist_old,th_pinch)
	#	Npinches.append(len(pinches)) # Pinches
	
		Emean.append(np.average(gamma,weights=W)) # Mean Elongation
		Evar.append(np.average((gamma-Emean[-1])**2./(1.-np.sum(W**2.)),weights=W)) # Variance of Elongation
	
		Cmax.append(np.average(1./np.sqrt(1.+wrapped_time/Pe),weights=dist_old)) # Variance of Elongation
	
	#	h,hx=np.histogram(gamma,Gamma_PDF_x,weights=W) # PDF of Gamma
	#	Gamma_PDF=Gamma_PDF+h
	#	if len(pinches)>2:
	#		pinches_bnd=np.hstack((0,pinches,L.shape[0])) # PDF of pinches length
	#		L_pinches=[np.sum(dist_old[pinches_bnd[j]:pinches_bnd[j+1]-1]) for j in range(len(pinches_bnd)-1)]
	#		h,hx=np.histogram(L_pinches,Lp_x)
	#		Lp_PDF=Lp_PDF+h
	#		Lp_var.append(np.var(L_pinches))
	
		if PLOT:
	#		line.set_xdata(L[:,0])
	#		line.set_ydata(L[:,1])
			line.set_xdata(dkappa/dt)
			line.set_ydata(np.log(gamma)/dt)
			line2.set_xdata(L[:,0])
			line2.set_ydata(L[:,1])
			plt.draw()
			#plt.savefig('{:03d}.png'.format(int(t*100)))
			plt.pause(0.01)
			
			
		# Analysis of curvature production
		
	#	plt.figure()
	#	plt.
		# Update time
		t=t+dt
		print('Time:',t,' - Npts:',len(L), '- Cmax:',np.mean(1./np.sqrt(1.+wrapped_time/Pe)),np.max(kappa_new))
	
	# End of MAIN LOOOP #######################
	print('Computation time:', time.time() -ct)
	if PAR:
		mpool.close()
	#%matplotlib inline
	#%matplotlib auto
	
	#np.savetxt(INTERPOLATE+'_{:d}PTS.txt'.format(L.shape[0]),np.vstack((KappaMean,logKappaMean,logKappaVar,Rhomean,logRhomean,logRhovar)).T)
	
	logRhomean=np.array(logRhomean)
	logRhovar=np.array(logRhovar)
	Lsum=np.array(Lsum)

	plt.plot(np.mod(L[:,0],1),np.mod(L[:,1],1),'+-',alpha=0.2)
	#plt.plot(L[:,0],L[:,1],'+-',alpha=0.2)
	plt.gca().set_aspect(1)

#%%% treat previous data
rho_agg=[]
xbin=np.logspace(-1,2,100)
for n in range(len(neighboors_t[0])):
	idexist=0
	for t in np.arange(20,30):
		nnew=len(neighboors_t[t][n])-len(neighboors_t[t-1][n])
		if nnew>0:
			D=np.abs(np.repeat(neighboors_t[t][n].reshape(-1,1),len(neighboors_t[t-1][n]),axis=1)-
						np.repeat(neighboors_t[t-1][n].reshape(1,-1),len(neighboors_t[t][n]),axis=0))
			idnew=np.argsort(np.min(D,axis=1))[-nnew:]
			lambda_i=neighboors_rho_t[t][n][idnew]
			lambda_0=neighboors_rho_t[t][n][-1]
			tci=t*dt
			rho_agg.extend(np.exp(-(lambda_i-lambda_0)*tci))	
[h1,x1]=np.histogram(rho_agg,xbin,density=True)
plt.yscale('log')	
rho_agg=[]
for n in range(len(neighboors_t[0])):
	idexist=0
	for t in np.arange(30,40):
		nnew=len(neighboors_t[t][n])-len(neighboors_t[t-1][n])
		if nnew>0:
			D=np.abs(np.repeat(neighboors_t[t][n].reshape(-1,1),len(neighboors_t[t-1][n]),axis=1)-
						np.repeat(neighboors_t[t-1][n].reshape(1,-1),len(neighboors_t[t][n]),axis=0))
			idnew=np.argsort(np.min(D,axis=1))[-nnew:]
			lambda_i=neighboors_rho_t[t][n][idnew]
			lambda_0=neighboors_rho_t[t][n][-1]
			tci=t*dt
			rho_agg.extend(np.exp(-(lambda_i-lambda_0)*tci))	
[h2,x2]=np.histogram(rho_agg,x_bin,density=True)
plt.yscale('log')		
rho_agg=[]
for n in range(len(neighboors_t[0])):
	idexist=0
	for t in np.arange(40,48):
		nnew=len(neighboors_t[t][n])-len(neighboors_t[t-1][n])
		if nnew>0:
			D=np.abs(np.repeat(neighboors_t[t][n].reshape(-1,1),len(neighboors_t[t-1][n]),axis=1)-
						np.repeat(neighboors_t[t-1][n].reshape(1,-1),len(neighboors_t[t][n]),axis=0))
			idnew=np.argsort(np.min(D,axis=1))[-nnew:]
			lambda_i=neighboors_rho_t[t][n][idnew]
			lambda_0=neighboors_rho_t[t][n][-1]
			tci=t*dt
			rho_agg.extend(np.exp(-(lambda_i-lambda_0)*tci))
[h3,x3]=np.histogram(rho_agg,xbin,density=True)
plt.plot(x1[:-1],h1,label='$P(x_i), 5<t<7.5$')
plt.plot(x2[:-1],h2,label='$P(x_i), 7.5<t<10$')
plt.plot(x3[:-1],h3,label='$P(x_i), 10<t<12$')
p=np.polyfit(np.log10(x3[-20:]),np.log10(h3[-20:]),1)
plt.plot(x3,x3**p[0],'k--',label='{:1.2f}'.format(p[0]))
plt.yscale('log')		
plt.xscale('log')
plt.legend()
plt.xlabel('$x_i=\exp[-(\lambda_i-\lambda)t_{c,i}]$')
plt.ylabel('Probability')
plt.savefig('PDF_agg.pdf')

# Plot variance as a fct of time
rho_agg=[[] for t in np.arange(len(neighboors_t))]
for n in range(len(neighboors_t[0])):
	idexist=0
	for t in np.arange(len(neighboors_t)):
		nnew=len(neighboors_t[t][n])-len(neighboors_t[t-1][n])
		if nnew>0:
			D=np.abs(np.repeat(neighboors_t[t][n].reshape(-1,1),len(neighboors_t[t-1][n]),axis=1)-
						np.repeat(neighboors_t[t-1][n].reshape(1,-1),len(neighboors_t[t][n]),axis=0))
			idnew=np.argsort(np.min(D,axis=1))[-nnew:]
			lambda_i=neighboors_rho_t[t][n][idnew]
			lambda_0=neighboors_rho_t[t][n][-1]
			tci=t*dt
			rho_agg[t].extend(np.exp(-(lambda_i-lambda_0)*tci))

rho_agg_var=[np.var(rho_agg[t]) for t in np.arange(len(neighboors_t))]
plt.plot(np.arange(len(rho_agg_var))*dt,rho_agg_var)
plt.yscale('log')
plt.xlabel('$t$')
plt.ylabel('Var$[x_i]$')
plt.savefig('VAR_agg.pdf')

# Plot aggragation time as a function of rho
#%%% Plot variance as a fct of time
rho_agg=np.array([]).reshape(-1,3)
N=[[] for t in np.arange(len(neighboors_t))]
for n in range(len(neighboors_t[0])):
	idexist=0
	for t in np.arange(len(neighboors_t)):
		nnew=len(neighboors_t[t][n])-len(neighboors_t[t-1][n])
		if nnew>0:
			D=np.abs(np.repeat(neighboors_t[t][n].reshape(-1,1),len(neighboors_t[t-1][n]),axis=1)-
						np.repeat(neighboors_t[t-1][n].reshape(1,-1),len(neighboors_t[t][n]),axis=0))
			idnew=np.argsort(np.min(D,axis=1))[-nnew:]
			lambda_i=neighboors_rho_t[t][n][idnew]
			lambda_0=neighboors_rho_t[t][n][-1]
			tci=t*dt
			N[t].append(len(lambda_i))
			X=np.hstack((lambda_i.reshape(-1,1),tci.reshape(-1,1).repeat(len(lambda_i)).reshape(-1,1),lambda_0.reshape(-1,1).repeat(len(lambda_i)).reshape(-1,1)))
			rho_agg=np.vstack((rho_agg,X))

variable=np.exp(-(rho_agg[:,2]-rho_agg[:,0])*rho_agg[:,1])
plt.plot(rho_agg[:,1],variable,'.',alpha=0.01)
xbin=np.linspace(0,12,13)
tci=bin_operation(rho_agg[:,1],variable,xbin,np.var)
plt.plot(xbin[1:],tci,'ko')
plt.yscale('log')
plt.plot(rho_agg[:,2],rho_agg[:,1],'.',alpha=0.01)

Nm=[np.mean(N[t]) for t in np.arange(len(neighboors_t))]
plt.plot(Nm)
plt.yscale('log')
#%% TRASH
#%%% Make iterative smoothing of images

plt.style.use('~/.config/matplotlib/joris.mplstyle')

dt=0.25
Brownian=1e-3
D=(Brownian)**2./(2*dt)
sigma=np.sqrt(2*D)*Nbin

periodicity='periodic'
Nbin=4096
#periodicity=''
dir_out='./Compare_stretching_concentration/{:1.0e}'.format(Brownian)+periodicity

t=10
I=cv2.imread(dir_out+'/{:04d}.tif'.format(int((t)*10)),2)
I=np.float32(I)/2.**16.

# Iterative smooth
import scipy.ndimage
V=[]
Is=I
T=np.linspace(1,100,10)
for t in T:
	print(t)
	Is=scipy.ndimage.gaussian_filter(I,sigma*np.sqrt(t))
	V.append(np.var(Is.flatten()))
	
plt.figure()
plt.plot(T,V)
plt.yscale('log')

#%%% aggreagition as a fct of rho
#plt.style.use('~/.config/matplotlib/joris.mplstyle')

Tmax_vec=[11]

radius =0.01
l0=0.3
for Tmax in Tmax_vec:
	import numpy as np
	import matplotlib.pyplot as plt
	import pylab
	import multiprocessing as mp
	#from scipy.ndimage.filters import maximum_filter
	import time
	from scipy import interpolate
	from scipy.interpolate import UnivariateSpline,interp1d,griddata
	#from colorline_toolbox import *
	# Inline or windows plots
	#%matplotlib auto
	#%matplotlibTrue inline
	PLOT=True
	PAR=False
	
	def curvature(L,dist):
	# Curvature via derivatives
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
	
	def find_pinch(L,dist_old,th_pinch):
	# Maximum curvature finder
		kappa=curvature(L,dist_old)
		Mkappa=maximum_filter(kappa,50) # Maximum filter on a box of 50dx
		return np.where((Mkappa==kappa)&(kappa>th_pinch))[0]
	
	#% Advection parameters
	INTERPOLATE='LINEAR'
	CURVATURE='DIFF'
	
	PLOT=False
	dx=0.001 # Maximum distance between points, above which the line will be refined
	alpha=200*dx/np.pi
	dt=0.25 # dt needs to be small compare to the CFL condition
	npar=6 # Number of parallel processors
	tmax=Tmax # Maximum advection time
	tsave=0.1 # Time period between saving steps
	Lmax=5e7 # Maximum number of points (prevent blow up)
	th_pinch=100 # Curvature minimum (in nb of dx) to find a peak
	Pe=1e2 # Peclet
	#l0=0.1
	
	# Initial segment position and distance
	x=np.linspace(0,2*np.pi,int(1.8e3))
	L=np.array([radius*np.cos(x),radius*np.sin(x)],dtype=np.float64).T
	L[0,:]=L[-1,:]
	n=int(1.8e3)
	L=np.array([np.linspace(-l0/2,l0/2,n),np.zeros(n)],dtype=np.float64).T
	#	L=[]
	#	for i in range(100):
	#		x,y=np.random.rand(),np.random.rand()
	#		L.append([x-dx,y-dx])
	#		L.append([x+dx,y+dx])
	#		L.append([np.nan,np.nan])
	
	L=np.array(L)
	weights=np.ones(L.shape[0]-1)
	weights=weights/np.sum(weights)
	#L=2*(np.random.rand(2)-0.5)+np.array([x,l0*np.sin(x)],dtype=np.float64).T
	
	dist_old=np.sum(np.diff(L,axis=0)**2,1)**0.5
	weights=np.ones(dist_old.shape)
	# Initial segment width
	S=np.ones(L.shape[0]-1)
	# Initial Wrapped Time
	wrapped_time=np.zeros(L.shape[0]-1)
	
	# Initialization of saved variables
	Lsum=[]
	Ssum=np.sum(dist_old*S)
	Smean=[]
	Rhomean=[]
	logRhomean=[]
	logRhovar=[]
	logKappaMean=[]
	logKappaSum=[]
	logKappaVar=[]
	KappaMean=[]
	logdKappaMean=[]
	logdKappaVar=[]
	logSmean=[]
	logSvar=[]
	Emean=[]
	Evar=[]
	Npinches=[]
	Cmax=[]
	Gamma_PDF_x=np.linspace(-10,10,1000)
	Gamma_PDF=np.zeros(Gamma_PDF_x.shape[0]-1,dtype=np.uint64)
	Lp_x=np.linspace(0,10,100)
	Lp_PDF=np.zeros(Lp_x.shape[0]-1,dtype=np.uint64)
	Lp_var=[]
	
	if PLOT:
		plt.close('all')
		#%matplotlib auto
		plt.close('all')
		plt.ion()
		fig, ax = plt.subplots(figsize=(10,10))
	#	ax.axis([-5,5,-5,5])
	#	line, = ax.plot(L[:,0], L[:,1],'-',alpha=1.,linewidth=0.5)
		ax.axis([-20,20,-3,3])
		ax.set_xlabel(r'$d\ln \kappa /dt$')
		ax.set_ylabel(r'$d\ln \rho /dt$')
		line, = ax.plot(0,0,'.',alpha=0.05,linewidth=0.5,markersize=0.8)
		ax2 = plt.axes([0.65, 0.65, 0.2, 0.2])
		ax2.axis([-np.pi/2.,np.pi/2.,-np.pi/2.,np.pi/2.])
		line2, = ax2.plot(L[:,0], L[:,1],'r-',alpha=1.,linewidth=0.5)
	
	# Initialization
	t=0
	Tv=np.arange(0,tsave+dt,dt) # Vector of time step between saving steps
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
	
	# MAIN PARALLEL LOOP #######################
	while (t<tmax)&(len(L)<Lmax):
		
		v=vel(L,t,1/np.sqrt(2))
		L+=v*dt
		
		# Compute stretching rates and elongations
		dist_new=np.sum(np.diff(L,axis=0)**2,1)**0.5
		gamma=dist_new/dist_old # elongation
		S=S/gamma
		wrapped_time=wrapped_time+dt*(1./S)**2.
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
		
		# Remove strip if outside the domain
#		Id_out=np.where((np.abs(L[:,0])>2)|(np.abs(L[:,1])>2))[0]
#		L[Id_out,:]=np.nan
#		# Filter consecutive nans to save memory space
#		nnan=np.uint8(np.isnan(L[:,0]))
#		nnan2=nnan[1:]+nnan[:-1]
#		iddel=np.where(nnan2==2)[0]
#		L=np.delete(L,iddel,axis=0)
#		dist_old=np.delete(dist_old,iddel,axis=0)
#		#dist_new=np.delete(dist_new,iddel,axis=0)
#		S=np.delete(S,iddel,axis=0)
#		gamma=np.delete(gamma,iddel,axis=0)
#		kappa_old=np.delete(kappa_old,iddel,axis=0)
#		W=np.delete(W,iddel)
#		weights=np.delete(weights,iddel)
#		wrapped_time=np.delete(wrapped_time,iddel,axis=0)
#		dkappa=np.delete(dkappa,iddel)
		
		#Save variables
		Lsum.append(np.sum(dist_old)) # Total length
		Rhomean.append(np.average(1./S,weights=W)) # Mean width
		logRhomean.append(np.average(np.log(1./S),weights=W)) # Mean width
		logRhovar.append(np.average((np.log(1./S)-logRhomean[-1])**2./(1.-np.sum(W**2.)),weights=W)) # Var Log Rho
	
		Smean.append(np.average(S,weights=dist_old)) # Mean width
		logSmean.append(np.average(np.log(S),weights=W)) # Mean width
		logSvar.append(np.average((np.log(S)-logSmean[-1])**2./(1.-np.sum(W**2.)),weights=W)) # Var width
	
		KappaMean.append(np.average(kappa_old,weights=W)) # !!! do we take weighted average or normal average ?
		logKappaMean.append(np.average(np.log(kappa_old),weights=W))
		logKappaSum.append(np.nansum(np.log(kappa_old)))
		logKappaVar.append(np.average((np.log(kappa_old)-logKappaMean[-1])**2./(1.-np.sum(W**2.)),weights=W)) # Var Log Rho
		logdKappaMean.append(dlKMean)
		logdKappaVar.append(dlKVar) # Variance of Elongation
	#	pinches=find_pinch(L,dist_old,th_pinch)
	#	Npinches.append(len(pinches)) # Pinches
	
		Emean.append(np.average(gamma,weights=W)) # Mean Elongation
		Evar.append(np.average((gamma-Emean[-1])**2./(1.-np.sum(W**2.)),weights=W)) # Variance of Elongation
	
		Cmax.append(np.average(1./np.sqrt(1.+wrapped_time/Pe),weights=dist_old)) # Variance of Elongation
	
	#	h,hx=np.histogram(gamma,Gamma_PDF_x,weights=W) # PDF of Gamma
	#	Gamma_PDF=Gamma_PDF+h
	#	if len(pinches)>2:
	#		pinches_bnd=np.hstack((0,pinches,L.shape[0])) # PDF of pinches length
	#		L_pinches=[np.sum(dist_old[pinches_bnd[j]:pinches_bnd[j+1]-1]) for j in range(len(pinches_bnd)-1)]
	#		h,hx=np.histogram(L_pinches,Lp_x)
	#		Lp_PDF=Lp_PDF+h
	#		Lp_var.append(np.var(L_pinches))
	
		if PLOT:
	#		line.set_xdata(L[:,0])
	#		line.set_ydata(L[:,1])
			line.set_xdata(dkappa/dt)
			line.set_ydata(np.log(gamma)/dt)
			line2.set_xdata(L[:,0])
			line2.set_ydata(L[:,1])
			plt.draw()
			#plt.savefig('{:03d}.png'.format(int(t*100)))
			plt.pause(0.01)
			
			
		# Analysis of curvature production
		
	#	plt.figure()
	#	plt.
		# Update time
		t=t+dt
		print('Time:',t,' - Npts:',len(L), '- Cmax:',np.mean(1./np.sqrt(1.+wrapped_time/Pe)),np.max(kappa_new))
	
	# End of MAIN LOOOP #######################
	print('Computation time:', time.time() -ct)
	if PAR:
		mpool.close()
	#%matplotlib inline
	#%matplotlib auto
	
	#np.savetxt(INTERPOLATE+'_{:d}PTS.txt'.format(L.shape[0]),np.vstack((KappaMean,logKappaMean,logKappaVar,Rhomean,logRhomean,logRhovar)).T)
	
	logRhomean=np.array(logRhomean)
	logRhovar=np.array(logRhovar)
	Lsum=np.array(Lsum)
	tt=np.arange(logRhomean.shape[0])*dt
	plt.plot(tt,logRhovar);plt.plot(tt,logRhomean)
#	plt.plot(np.mod(L[:,0],1),np.mod(L[:,1],1),'.',markersize=0.01)
	#% Build KDtree 

	sB=0.01
	from scipy import spatial
	tree=spatial.cKDTree(np.mod(L,1))
	nsamples=int(sB/dx*10)
	nsamples=100
	idsamples=np.arange(0,L.shape[0],nsamples)
	neighboors=tree.query_ball_point(np.mod(L[idsamples,:],1), sB)
	#% Check number of aggregations
	nagg=np.array([len(n)*dx/sB for n in neighboors])
	plt.plot(nagg,(1./S[idsamples]),'k.',markersize=0.5,c='0.5')
	plt.yscale('log')
	plt.xscale('log')
	agg_bin=np.logspace(0,2.5,20)
	rho_bin=bin_operation(nagg,np.log(1./S[idsamples]),agg_bin,np.nanmean)
	plt.plot(agg_bin[1:],np.exp(rho_bin),'ko')
	plt.plot(agg_bin[1:],2e2*agg_bin[1:],'k--',label=r'$\rho \sim N$')
	plt.xlabel(r'$N$ Number of aggregated strip')
	plt.ylabel(r'$\rho$ Local stretching')
	plt.legend()
	plt.savefig('./Aggregation_vs_Elongation.pdf')
	
#% With unique point/lamella
	plt.style.use('~/.config/matplotlib/joris.mplstyle')
	neighboors=tree.query_ball_point(np.mod(L[idsamples,:],1), sB)
	neighboors_uniq=[[] for k in range(len(idsamples))]
	neighboors_all=[[] for k in range(len(idsamples))]
	for ii,ns in enumerate(idsamples):		
		nt=np.sort(neighboors[ii])
		kk_all=np.concatenate((nt,np.array([ns])))
		neighboors_all[ii].append(kk_all)
		idgood=np.where(np.diff(nt)>3*sB/dx)[0]
		NT=[]
		for idg in idgood:
			if (np.abs(nt[idg+1]-ns)>3*sB/dx)&(np.abs(nt[idg]-ns)<3*sB/dx): # depending on the direction of the filament
				NT.append(nt[idg+1])
			else:
				NT.append(nt[idg])
		nt=np.array(NT,dtype=np.uint32)
		kk=np.concatenate((nt,np.array([ns])))
		neighboors_uniq[ii].append(kk)
	nagg=np.array([len(n[0]) for n in neighboors_uniq])
	plt.figure(figsize=(1.5,1.5))
	#plt.plot(nagg,(1./S[idsamples]),'k.',markersize=0.5,c='0.5')
	plt.yscale('log')
	plt.xscale('log')
	agg_bin=np.logspace(0,2.5,20)
	rho_bin=bin_operation(nagg,np.log(1./S[idsamples]),agg_bin,np.nanmean)
	rho_bin_std=bin_operation(nagg,np.log(1./S[idsamples]),agg_bin,np.nanstd)
	plt.plot(agg_bin[:-1],np.exp(rho_bin),'ko')
	plt.plot(agg_bin[:-1],3.5e2*agg_bin[1:],'k--',label=r'$N\sim\rho$')
	plt.xlabel(r'$N$')
	plt.ylabel(r'$\rho$')
	plt.ylim([1e2,1e5])
	plt.xlim([0.8,1e2])
	plt.xticks([1,10,100])
	plt.legend()
	plt.savefig('./Aggregation_vs_Elongation.pdf',bbox_inches='tight')
	
	
	#% Check neighbooring elongation rates
	plt.style.use('~/.config/matplotlib/joris.mplstyle')
	neigh_elong=np.array([np.nanmean(1./S[n]) for n in neighboors])
	plt.plot(neigh_elong,1./S[idsamples],'k.',markersize=0.5,c='0.5')
	plt.yscale('log')
	plt.xscale('log')
	rho_bin=np.logspace(1,6,30)
	rho_neigh=bin_operation((neigh_elong),(1./S[idsamples]),(rho_bin),np.nanmean)
	plt.plot(rho_bin[1:],(rho_neigh),'ko')
	plt.plot(rho_bin[1:],rho_bin[1:],'k--',label=r'$\rho_i = \langle \rho_i\rangle $')
	plt.ylabel(r'$\rho_i $ Elongation of strip')
	plt.xlabel(r'$\langle \rho_i\rangle$ Average elongation of aggregated strips')
	plt.legend()
	plt.savefig('./Elongation_vs_LocalElongation.pdf')
	
#%%% Check local value of lambda_i - lambda t
# take only one sample per lamella
#% aggreagition as a fct of rho
plt.style.use('~/.config/matplotlib/joris.mplstyle')

from scipy import spatial
Tmax_vec=[12]

sB =0.01
radius=0.01
l0=0.3
keyword='sine'
for Tmax in Tmax_vec:

	
	#% Advection parameters
	INTERPOLATE='LINEAR'
	CURVATURE='DIFF'
	
	PLOT=False
	dx=0.001 # Maximum distance between points, above which the line will be refined
	alpha=200*dx/np.pi
	dt=0.25 # dt needs to be small compare to the CFL condition
	npar=6 # Number of parallel processors
	tmax=Tmax # Maximum advection time
	tsave=0.1 # Time period between saving steps
	Lmax=2e7 # Maximum number of points (prevent blow up)
	th_pinch=100 # Curvature minimum (in nb of dx) to find a peak
	Pe=1e2 # Peclet
	#l0=0.1
	
	# Initial segment position and distance
	x=np.linspace(0,2*np.pi,int(1.8e3))
	L=np.array([radius*np.cos(x),radius*np.sin(x)],dtype=np.float64).T
	L[0,:]=L[-1,:]
	n=int(1.8e3)
	L=np.array([np.linspace(-l0/2,l0/2,n),np.zeros(n)],dtype=np.float64).T
	#	L=[]
	#	for i in range(100):
	#		x,y=np.random.rand(),np.random.rand()
	#		L.append([x-dx,y-dx])
	#		L.append([x+dx,y+dx])
	#		L.append([np.nan,np.nan])
	
	L=np.array(L)
	weights=np.ones(L.shape[0]-1)
	weights=weights/np.sum(weights)
	#L=2*(np.random.rand(2)-0.5)+np.array([x,l0*np.sin(x)],dtype=np.float64).T
	
	dist_old=np.sum(np.diff(L,axis=0)**2,1)**0.5
	weights=np.ones(dist_old.shape)
	# Initial segment width
	S=np.ones(L.shape[0]-1)
	# Initial Wrapped Time
	wrapped_time=np.zeros(L.shape[0]-1)
	
	# Initialization of saved variables
	Var=[]
	Var_uniq=[]
	Meanl=[]
	Meanl_uniq=[]
	LambdaMean=[]
	LambdaVar=[]
	VarCmaxB=[]
	VarSumCmax=[]
	
	MeanCmaxB=[]
	MeanSumCmax=[]
	CgridMean=[]
	CgridVar=[]
	
	Lsum=[]
	Ssum=np.sum(dist_old*S)
	Smean=[]
	Rhomean=[]
	logRhomean=[]
	logRhovar=[]
	logKappaMean=[]
	logKappaSum=[]
	logKappaVar=[]
	KappaMean=[]
	logdKappaMean=[]
	logdKappaVar=[]
	logSmean=[]
	logSvar=[]
	Emean=[]
	Evar=[]
	Npinches=[]
	Cmax=[]
	VarCmax=[]
	CmaxN=[]
	Gamma_PDF_x=np.linspace(-10,10,1000)
	Gamma_PDF=np.zeros(Gamma_PDF_x.shape[0]-1,dtype=np.uint64)
	Lp_x=np.linspace(0,10,100)
	Lp_PDF=np.zeros(Lp_x.shape[0]-1,dtype=np.uint64)
	Lp_var=[]
	
	if PLOT:
		plt.close('all')
		#%matplotlib auto
		plt.close('all')
		plt.ion()
		fig, ax = plt.subplots(figsize=(10,10))
	#	ax.axis([-5,5,-5,5])
	#	line, = ax.plot(L[:,0], L[:,1],'-',alpha=1.,linewidth=0.5)
		ax.axis([-20,20,-3,3])
		ax.set_xlabel(r'$d\ln \kappa /dt$')
		ax.set_ylabel(r'$d\ln \rho /dt$')
		line, = ax.plot(0,0,'.',alpha=0.05,linewidth=0.5,markersize=0.8)
		ax2 = plt.axes([0.65, 0.65, 0.2, 0.2])
		ax2.axis([-np.pi/2.,np.pi/2.,-np.pi/2.,np.pi/2.])
		line2, = ax2.plot(L[:,0], L[:,1],'r-',alpha=1.,linewidth=0.5)
	
	# Initialization
	t=0
	Tv=np.arange(0,tsave+dt,dt) # Vector of time step between saving steps
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
	
	# MAIN PARALLEL LOOP #######################
	while (len(L)<Lmax):
		
		#v=vel(L,t)
		
		v=locals()['vel_'+keyword](L,t,A)
		L+=v*dt
		
		# Compute stretching rates and elongations
		dist_new=np.sum(np.diff(L,axis=0)**2,1)**0.5
		gamma=dist_new/dist_old # elongation
		S=S/gamma
		wrapped_time=wrapped_time+dt*(1./S)**2.
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

		
		#Save variables
		Lsum.append(np.sum(dist_old)) # Total length
		Rhomean.append(np.average(1./S,weights=W)) # Mean width
		logRhomean.append(np.average(np.log(1./S),weights=W)) # Mean width
		logRhovar.append(np.average((np.log(1./S)-logRhomean[-1])**2,weights=W)) # Var Log Rho
	
		Smean.append(np.average(S,weights=dist_old)) # Mean width
		logSmean.append(np.average(np.log(S),weights=W)) # Mean width
		logSvar.append(np.average((np.log(S)-logSmean[-1])**2./(1.-np.sum(W**2.)),weights=W)) # Var width
	
		KappaMean.append(np.average(kappa_old,weights=W)) # !!! do we take weighted average or normal average ?
		logKappaMean.append(np.average(np.log(kappa_old),weights=W))
		logKappaSum.append(np.nansum(np.log(kappa_old)))
		logKappaVar.append(np.average((np.log(kappa_old)-logKappaMean[-1])**2./(1.-np.sum(W**2.)),weights=W)) # Var Log Rho
		logdKappaMean.append(dlKMean)
		logdKappaVar.append(dlKVar) # Variance of Elongation
	#	pinches=find_pinch(L,dist_old,th_pinch)
	#	Npinches.append(len(pinches)) # Pinches
	
		Emean.append(np.average(gamma,weights=W)) # Mean Elongation
		Evar.append(np.average((gamma-Emean[-1])**2./(1.-np.sum(W**2.)),weights=W)) # Variance of Elongation
		cmax=1./np.sqrt(1.+wrapped_time/Pe)
		sb=np.sqrt(1.+wrapped_time/Pe)/(1/S)
		Cmax.append(np.sum(cmax*dist_old*sb)) # Mean Cmax
		VarCmax.append(np.sum((cmax)**2*dist_old*sb)) # Mean Cmax
		c0=np.average(cmax,weights=dist_old)
#		CmaxN.append([np.average(np.abs(cmax-c0)**nn,weights=dist_old) for nn in range(10)]) # Mean Cmax
		CmaxN.append([np.sum(cmax**nn*dist_old*sb) for nn in range(10)]) # Mean Cmax
#		if np.mod(t,2)==0:
#		# Other  
#		D=(Brownian)**2./(2*dt)
#		s0=radius*0.95
#		Cmaxv=1./np.sqrt(1.+4*D/s0**2*wrapped_time)
#		c0=np.nanmean(Cmaxv)
#		CmaxM.append([np.nanmean(np.abs(Cmaxv-c0)**n) for n in range(N)])
	#	h,hx=np.histogram(gamma,Gamma_PDF_x,weights=W) # PDF of Gamma
	#	Gamma_PDF=Gamma_PDF+h
	#	if len(pinches)>2:
	#		pinches_bnd=np.hstack((0,pinches,L.shape[0])) # PDF of pinches length
	#		L_pinches=[np.sum(dist_old[pinches_bnd[j]:pinches_bnd[j+1]-1]) for j in range(len(pinches_bnd)-1)]
	#		h,hx=np.histogram(L_pinches,Lp_x)
	#		Lp_PDF=Lp_PDF+h
	#		Lp_var.append(np.var(L_pinches))
	
		if PLOT:
	#		line.set_xdata(L[:,0])
	#		line.set_ydata(L[:,1])
			line.set_xdata(dkappa/dt)
			line.set_ydata(np.log(gamma)/dt)
			line2.set_xdata(L[:,0])
			line2.set_ydata(L[:,1])
			plt.draw()
			#plt.savefig('{:03d}.png'.format(int(t*100)))
			plt.pause(0.01)
			
			
		# variance of bundles
		tree=spatial.cKDTree(np.mod(L,1))
		nsamples=int(sB/dx*10)
		nsamples=500
		sB=0.01
		idsamples=np.arange(0,L.shape[0],nsamples)
		#idsamples=np.uint32(np.linspace(10,L.shape[0]-10,nsamples))
		neighboors=tree.query_ball_point(np.mod(L[idsamples,:],1), sB)
		neighboors_uniq=[[] for k in range(len(idsamples))]
		neighboors_all=[[] for k in range(len(idsamples))]
		for ii,ns in enumerate(idsamples):		
			nt=np.sort(neighboors[ii])
			kk_all=np.concatenate((nt,np.array([ns])))
			neighboors_all[ii].append(kk_all)
			idgood=np.where(np.diff(nt)>3*sB/dx)[0]
			NT=[]
			for idg in idgood:
				if (np.abs(nt[idg+1]-ns)>3*sB/dx)&(np.abs(nt[idg]-ns)<3*sB/dx): # depending on the direction of the filament
					NT.append(nt[idg+1])
				else:
					NT.append(nt[idg])
			nt=np.array(NT,dtype=np.uint32)
			kk=np.concatenate((nt,np.array([ns])))
			neighboors_uniq[ii].append(kk)
		neigh_elong_uniq=np.array([np.log(1./S[n[0][:]])-np.log(1./S[n[0][-1]]) for n in neighboors_uniq])
		neigh_elong=np.array([np.log(1./S[np.minimum(n[0][:],len(S)-1)])-np.log(1./S[n[0][-1]]) for n in neighboors_all])
		cmax=1./np.sqrt(1.+wrapped_time/Pe)
		sb=np.sqrt(1.+wrapped_time/Pe)/(1/S)
		neigh_cmax_uniq=np.array([cmax[n[0][:]]*sb[n[0][:]] for n in neighboors_uniq])
		
		Var_uniq.append(np.mean([np.nanvar(nq) for nq in neigh_elong_uniq]))
		Var.append(np.mean([np.nanvar(nq) for nq in neigh_elong]))
		
		VarSumCmax.append(np.var([np.sum(nq) for nq in neigh_cmax_uniq]))
		MeanSumCmax.append(np.mean([np.sum(nq) for nq in neigh_cmax_uniq]))
		
		VarCmaxB.append(np.var([nq[-1] for nq in neigh_cmax_uniq]))
		MeanCmaxB.append(np.mean([nq[-1] for nq in neigh_cmax_uniq]))
		
		Meanl_uniq.append(np.mean([np.nanmean(nq) for nq in neigh_elong_uniq]))
		Meanl.append(np.mean([np.nanmean(nq) for nq in neigh_elong]))

		LambdaMean.append(np.average(np.log(1./S)/t,weights=S))
		LambdaVar.append(np.average((np.log(1./S)/t-LambdaMean[-1])**2,weights=S))
		
		# Bundles formed in a grid
		sbg=0.02
		Lint=np.uint32(np.mod(L,1)/sbg)
		Cgrid=np.zeros((int(1/sbg),int(1/sbg)))
		Cgrid[Lint[1:,0],Lint[1:,1]]=Cgrid[Lint[1:,0],Lint[1:,1]]+cmax
		CgridVar.append(np.var(Cgrid[Cgrid>0].flatten()))
		CgridMean.append(np.mean(Cgrid[Cgrid>0].flatten()))
	#	plt.figure()
	#	plt.
		# Update time
		t=t+dt
		print('Time:',t,' - Npts:',len(L), '- Cmax:',np.mean(1./np.sqrt(1.+wrapped_time/Pe)),np.max(kappa_new))
	
	# End of MAIN LOOOP #######################
	print('Computation time:', time.time() -ct)
	if PAR:
		mpool.close()
	#%matplotlib inline
	#%matplotlib auto
	
	#np.savetxt(INTERPOLATE+'_{:d}PTS.txt'.format(L.shape[0]),np.vstack((KappaMean,logKappaMean,logKappaVar,Rhomean,logRhomean,logRhovar)).T)
	
	logRhomean=np.array(logRhomean)
	logRhovar=np.array(logRhovar)
	Lsum=np.array(Lsum)

#%%% Compute Lyapunov exponents

plt.style.use('~/.config/matplotlib/joris.mplstyle')
Tt=np.arange(len(logRhomean))*dt
plt.figure(figsize=(2,2))
lyapunov=np.polyfit(Tt,logRhomean,1)[0]
sigma2lyapunov=np.polyfit(Tt,logRhovar,1)[0]
plt.plot(Tt,logRhomean,'r',label=r'$\mu_{\log \rho}$')
plt.plot(Tt,logRhovar,'b',label=r'$\sigma^2_{\log \rho}$')
plt.xlabel('$t$')
#	plt.plot(Tt,-np.log(np.array(Cmax)))
#	plt.plot(Tt,(Tt*lyapunov**2/(2*sigma2lyapunov)),'k-',label=r'$-\bar{\lambda}^2/(2\sigma^2)$')
plt.plot(Tt,(Tt*lyapunov),'r--',label=r'$\mu_\lambda={:1.2f}$'.format(lyapunov))
plt.plot(Tt,(Tt*sigma2lyapunov),'b--',label=r'$\sigma^2_\lambda={:1.2f}$'.format(sigma2lyapunov))
#	plt.plot(Tt,(Tt*(lyapunov+sigma2lyapunov/2)),'k--',label=r'$-(\bar{\lambda}^2 + \sigma^2/2)$')
plt.legend()
plt.savefig('./Compare_stretching_concentration/'+keyword+'/lyapunov.pdf')
#%%% PDF of rho
#	plt.plot(np.mod(L[:,0],1),np.mod(L[:,1],1),'.',markersize=0.01)
	
	Tt=np.arange(len(Var))*dt
	lyapunov=np.polyfit(Tt,logRhomean,1)[0]
	sigma2lyapunov=np.polyfit(Tt,logRhovar,1)[0]
# make histogram
	plt.figure()
	p1,x1=np.histogram(np.log(1./S)/t,50,density=True,weights=S)
	#p2,x2=np.histogram(np.log(1./S)/t,50,density=True,weights=S)
	plt.plot(x1[1:],p1,'o',label=r'$p(\log\rho/t)$')
	#plt.plot(x2[1:],p2,'o')
	lyap=np.average(np.log(1./S)/t,weights=S)
	sigma2=np.average((np.log(1./S)/t-lyap)**2.,weights=S)
	sigma2lyap=sigma2*t
	plt.plot(x1,1/np.sqrt(2*np.pi*sigma2)*np.exp(-(x1-lyap)**2./(2*sigma2)),label='Gaussian approx.')
	plt.yscale('log')
	plt.xlabel(r'$\lambda=\log \rho / t$')
	plt.legend()
	plt.savefig('PDF_rho_A{:1.1f}.pdf'.format(A))
	
	
# Plot variance of lambda as a function of time
	plt.figure()
	plt.plot(Tt,LambdaVar,'o',label=r'$Var(\lambda)$')
	plt.plot(Tt,LambdaMean,'+',label=r'$Mean(\lambda)$')
	plt.plot(Tt,1/Tt,'-,',label=r'$1/t$')
	plt.yscale('log')
	plt.legend()
	plt.xlabel('t')
	plt.savefig('Var_lambda_t_A{:1.1f}.pdf'.format(A))
	#plt.ylabel('Var[$\lambda$]')
	
	plt.figure()
	plt.plot(Tt,Meanl_uniq,label='Mean$[(\lambda_i-\lambda)t]$')
	#plt.plot(Tt,Meanl,label='Mean$[(\lambda_i-\lambda)t]$')
	plt.plot(Tt,Var_uniq,label='Var$[(\lambda_i-\lambda)t]$')
	plt.plot(Tt,LambdaVar*Tt**2,'-',label=r'Var$(\lambda t)$')
	plt.xlabel('t')
	plt.ylim([-0.5,5])
	plt.legend()
	plt.savefig('Var_A{:1.1f}.pdf'.format(A))
	
	plt.figure()
#	plt.plot(Tt,VarCmaxB,label=r'Var$[c_{max}]$')
	plt.plot(Tt,VarCmax/VarCmax[0],label=r'Var$[c_{max}]$')
	plt.plot(Tt,VarSumCmax,'--',label=r'Var$[\sum_i c_{max,i}]$')
#	plt.plot(Tt,MeanCmaxB,label=r'Mean$[c_{max}]$')
	plt.plot(Tt,Cmax,label=r'Mean$[c_{max}]$')
	plt.plot(Tt,MeanSumCmax,'--',label=r'Mean$[\sum_i c_{max,i}]$')
	plt.plot(Tt,np.exp(-Tt*lyap**2/(2*sigma2*t)),'k-',label=r'$-\bar{\lambda}^2/(2\sigma^2)$')
	plt.plot(Tt,np.exp(-Tt*(lyap+sigma2*t/2)),'k--',label=r'$-(\bar{\lambda}^2 + \sigma^2/2)$')
	plt.yscale('log')
	plt.xlabel('t')
	plt.legend()
	plt.savefig('Var_Cmax_{:1.1f}.pdf'.format(A))
	
	CmaxN=np.array(CmaxN)
	plt.figure()
	plt.plot(Tt,CgridMean,'--',label=r'Mean$[\sum_i c_{max,i}]$')
	plt.plot(Tt,CgridVar,'--',label=r'Var$[\sum_i c_{max,i}]$')
#	plt.plot(Tt,VarCmaxB,label=r'Var$[c_{max}]$')
	#plt.plot(Tt,CmaxN/CmaxN[0,:],label='')
	plt.plot(Tt,VarCmax/VarCmax[0],label=r'$\int c_{max}^2 s dl$')
	plt.yscale('log')
#	plt.plot(Tt,MeanCmaxB,label=r'Mean$[c_{max}]$')
	plt.plot(Tt,Cmax/Cmax[0],label=r'$\int c_{max} s dl$')
#	plt.plot(Tt,np.exp(-Tt*lyap**2/(2*sigma2*t)),'k-',label=r'$-\bar{\lambda}^2/(2\sigma^2)$')
#	plt.plot(Tt,np.exp(-Tt*(lyapunov+sigma2*t/2)),'k--',label=r'$-(\bar{\lambda}^2 + \sigma^2/2)$')
	plt.plot(Tt,np.exp(-Tt*lyapunov**2/(2*sigma2lyapunov)),'k-',label=r'$-\bar{\lambda}^2/(2\sigma^2)$')
	plt.plot(Tt,np.exp(-Tt*(lyapunov+sigma2lyapunov/2)),'k--',label=r'$-(\bar{\lambda}^2 + \sigma^2/2)$')
	plt.yscale('log')
	plt.xlabel('t')
	plt.ylim([1e-6,10])
	plt.legend()
	plt.savefig('Var_CmaxGrid_{:1.1f}.pdf'.format(A))

#%% Fractal analysis
#%%% Fractal dimension of an intersection

keyword='cubic'
#keyword=''
#keyword='bifreq'
keyword='half'
keyword='single'
#keyword='cubic'
keyword='sine'
keyword='double'
keyword='halfsmooth'

dx=0.001
plt.style.use('~/.config/matplotlib/joris.mplstyle')
import h5py
from skimage.morphology import disk
from skimage.filters import median,gaussian
ms=['o','d','s','<','>','+','*']
Brownian=1e-3
radius=0.01
l0=0.3
A=1/np.sqrt(2)
#A=1.2


f=h5py.File('./Compare_stretching_concentration/'+keyword+'/DSM'+keyword+'_A{:1.1f}_l{:1.1f}.hdf5'.format(A,l0), 'r')

try:
	t=f.attrs['tmax']
except:
	t=12

L=f['L_{:04d}'.format(int(t*10))][:]
wrapped_time=f['wrapped_time_{:04d}'.format(int(t*10))][:]
S=f['S_{:04d}'.format(int(t*10))][:]
W=f['Weights_{:04d}'.format(int(t*10))][:]

nagg=[]
c_agg_sum=[]

sB=0.05
from scipy import spatial
tree=spatial.cKDTree(np.mod(L[:-2],1))
nsamples=100
sB=0.1
R=sB/np.logspace(0,4,15)
idsamples=np.uint32(np.linspace(0,L.shape[0]-2,nsamples))


# Sampling on a uniform grid

# Take idsample in a grid to respect an equi porportion in the space
ng=20
idshuffle=np.arange(len(L)-3)
np.random.shuffle(idshuffle)
Lmod=np.mod(L[:-2],1)
Lmodint=np.uint32(Lmod*ng)
idu=np.unique(np.copy(Lmodint[idshuffle,:]), return_index=True,axis=0)
idsamples=idshuffle[idu[1]]


for r in R:
	# take samples equally spaced on the final filament
	# take samples equally spaced on the initial filament
	# check if id sample far enough from boundary
	neighboors=tree.query_ball_point(np.mod(L[idsamples,:],1), r,n_jobs=4)
	nagg.append(np.array([len(n) for n in neighboors]))
	c_agg_sum.append(np.array([np.mean(S[n]**2.) for n in neighboors]))

nagg=np.array(nagg)
c_agg_sum=np.array(c_agg_sum)
p=np.array([np.polyfit(np.log(R)[:-3],np.log(nagg[:,k])[:-3],1) for k in range(nagg.shape[1])])
plt.figure()
plt.hist(p[:,0],np.linspace(1,3,40))

# Log of mean ()
idfit=(np.log(R)>-7)&(np.log(R)<-4)
p1=np.polyfit(np.log(R)[idfit],np.log(np.mean(nagg,axis=1))[idfit],1)
# Mean of log ()
p2=np.polyfit(np.log(R)[idfit],np.mean(np.log(nagg),axis=1)[idfit],1)

p=p1
N=np.log(np.mean(nagg,axis=1))
p=p2
N=np.mean(np.log(nagg),axis=1)

N=np.mean(np.log(c_agg_sum),axis=1)
plt.figure(figsize=(1.5,1.5))
plt.plot(np.log(R)[1:],np.diff(N)/np.diff(np.log(R)),'ko',alpha=1)

plt.plot(np.log(R),np.zeros(R.shape) + p[0],'k:',label=r'$\nu={:1.2f}$'.format(p[0]))
#plt.plot(np.log(R),np.log(R)*2+17,'k--',label='$D_f=2$')
plt.xlabel(r'$\log r$')
plt.ylabel(r'$d \log N / d \log r$')
plt.legend()
plt.ylim(1.1,2.2)
plt.savefig('./Compare_stretching_concentration/'+keyword+'/fractal_'+keyword+'_A{:1.1f}_T={:1.0f}.pdf'.format(A,t), bbox_inches='tight')


plt.figure(figsize=(1.5,1.5))
#plt.plot(np.log(R),np.log(nagg)-np.log(nagg)[0,:],'k-',alpha=0.01)
plt.plot(np.log(R)[1:],np.diff(N)/np.diff(np.log(R)),'ko',alpha=1)
#plt.plot(np.log(R)[:],N,'ko',alpha=1)
plt.xlabel(r'$\log r$')
plt.ylabel(r'$d \log <c^2> / d \log r$')
#plt.plot(np.log(R),np.mean(np.log(nagg),axis=1),'ko',alpha=1)


plt.figure()
plot_per(L)
plt.savefig('./Compare_stretching_concentration/'+keyword+'/lines_'+keyword+'.pdf',bbox_inches='tight')
plt.plot(np.mod(L[idsamples,1],1),np.mod(-L[idsamples,0],1),'ro')

plt.figure()
plot_wave(keyword)
plt.savefig('./Compare_stretching_concentration/'+keyword+'/wave_'+keyword+'.pdf',bbox_inches='tight')
#%%% Fractal dimension (with time)

keyword='cubic'
#keyword=''
#keyword='bifreq'
#keyword='half'
keyword='single'
#keyword='cubic'
#keyword=''

dx=0.001
plt.style.use('~/.config/matplotlib/joris.mplstyle')
import h5py
from skimage.morphology import disk
from skimage.filters import median,gaussian
ms=['o','d','s','<','>','+','*']
Brownian=1e-3
radius=0.01
l0=0.3
A=1/np.sqrt(2)
#A=1.2


f=h5py.File('./Compare_stretching_concentration/DSM'+keyword+'_A{:1.1f}_l{:1.1f}.hdf5'.format(A,l0), 'r')

tmax=f.attrs['tmax']


Df=[]
T=np.unique(np.uint8(np.linspace(tmax*0.5,tmax,6)))
#T=[2]
for t in T:
	#t=10
	L=f['L_{:04d}'.format(int(t*10))][:]
	wrapped_time=f['wrapped_time_{:04d}'.format(int(t*10))][:]
	S=f['S_{:04d}'.format(int(t*10))][:]
	W=f['Weights_{:04d}'.format(int(t*10))][:]
	
	nagg=[]
	sB=0.05
	from scipy import spatial
	tree=spatial.cKDTree(np.mod(L,1))
	nsamples=100
	sB=0.05
	R=sB/np.logspace(0,1.5,15)
	idsamples=np.uint32(np.linspace(0,L.shape[0]-2,nsamples))
	
	
	# Sampling on a uniform grid
	
	# Take idsample in a grid to respect an equi porportion in the space
	ng=20
	idshuffle=np.arange(len(L)-1)
	np.random.shuffle(idshuffle)
	Lmod=np.mod(L,1)
	Lmodint=np.uint32(Lmod*ng)
	idu=np.unique(np.copy(Lmodint[idshuffle,:]), return_index=True,axis=0)
	idsamples=idshuffle[idu[1]]
	
	
	for r in R:
		# take samples equally spaced on the final filament
		# take samples equally spaced on the initial filament
		# check if id sample far enough from boundary
		neighboors=tree.query_ball_point(np.mod(L[idsamples,:],1), r,n_jobs=4)
		nagg.append(np.array([len(n) for n in neighboors]))
	
	nagg=np.array(nagg)
	
	p=np.array([np.polyfit(np.log(R)[:-3],np.log(nagg[:,k])[:-3],1) for k in range(nagg.shape[1])])
	plt.figure()
	#plt.hist(p[:,0],np.linspace(1,3,40))
	
	# Log of mean ()
	idfit=(np.log(R)>-7)&(np.log(R)<-4)
	p1=np.polyfit(np.log(R)[idfit],np.log(np.mean(nagg,axis=1))[idfit],1)
	# Mean of log ()
	p2=np.polyfit(np.log(R)[idfit],np.mean(np.log(nagg),axis=1)[idfit],1)
	
	p=p1
	N=np.log(np.mean(nagg,axis=1))
	p=p2
	N=np.mean(np.log(nagg),axis=1)
	Df.append(np.diff(N)/np.diff(np.log(R)))
	#plt.plot(np.log(R),np.log(nagg)-np.log(nagg)[0,:],'k-',alpha=0.01)
	#plt.plot(np.log(R)[1:],np.diff(N)/np.diff(np.log(R)),'o',alpha=1,color=plt.cm.jet((t-tmax*0.7)/(0.3*tmax)))
#plt.plot(np.log(R),np.mean(np.log(nagg),axis=1),'ko',alpha=1)

plt.figure(figsize=(3,3))
[plt.plot(np.log(R)[1:],Df[k],'o-',alpha=1,
					color=plt.cm.viridis((T[k]-tmax*0.7)/(0.3*tmax)),
					label='$t={:1.0f}$'.format(T[k]))  for k in range(len(T))]

#plt.plot(np.log(R),np.log(R)*p[0]+p[1],'k:',label='$D_f={:1.2f}$'.format(p[0]))

plt.plot(np.log(R),np.zeros(R.shape) + p[0],'k:',label='$D_f={:1.2f}$'.format(p[0]))
#plt.plot(np.log(R),np.log(R)*2+17,'k--',label='$D_f=2$')
plt.xlabel(r'$\log r$')
plt.ylabel(r'$D_f$')
plt.legend()
plt.ylim(0,2)
plt.savefig('./Compare_stretching_concentration/fractal_'+keyword+'_A{:1.1f}_T={:1.0f}.pdf'.format(A,t), bbox_inches='tight')


#%%% Fractal dimension of an intersection
keyword='cubic'
#keyword=''
#keyword='bifreq'
#keyword='half'
keyword='single'
#keyword='cubic'
keyword=''

dx=0.001
plt.style.use('~/.config/matplotlib/joris.mplstyle')
import h5py
from skimage.morphology import disk
from skimage.filters import median,gaussian
ms=['o','d','s','<','>','+','*']
Brownian=1e-3
radius=0.01
l0=0.3
A=1/np.sqrt(2)
#A=1.2


f=h5py.File('./Compare_stretching_concentration/DSM'+keyword+'_A{:1.1f}_l{:1.1f}.hdf5'.format(A,l0), 'r')

t=f.attrs['tmax']
t=8
L=f['L_{:04d}'.format(int(t*10))][:]
wrapped_time=f['wrapped_time_{:04d}'.format(int(t*10))][:]
S=f['S_{:04d}'.format(int(t*10))][:]
W=f['Weights_{:04d}'.format(int(t*10))][:]

nagg=[]
rho_agg=[]
sB=0.05
from scipy import spatial
tree=spatial.cKDTree(np.mod(L[:-2,:],1))
nsamples=1000
sB=0.1
R=[sB,sB/50]
idsamples=np.uint32(np.linspace(0,L.shape[0]-2,nsamples))
for r in R:
	# take samples equally spaced on the final filament
	# take samples equally spaced on the initial filament
#	Sall=np.cumsum(S)
#	Sall,index=np.unique(np.uint32(Sall/Sall[-1]*nsamples),return_index=True)
#	idsamples=index
	# check if id sample far enough from boundary
	neighboors=tree.query_ball_point(np.mod(L[idsamples,:],1), r)
	nagg.append(np.array([len(n) for n in neighboors]))
	rho_agg.append(np.array([np.mean(1/S[n]) for n in neighboors]))

Df=np.log(nagg[0]/nagg[1])/np.log(50)
plt.plot(Df,rho_agg[0],'k.')
rhoi=np.logspace(np.log10(np.min(rho_agg)),np.log10(np.max(rho_agg)),20)
Dfi=bin_operation(rho_agg[0], Df, rhoi,
							 np.mean)
plt.plot(Dfi,rhoi[1:],'r--')
plt.yscale('log')



#%%% Fractal dimension with histogram

#keyword='cubic'
#keyword=''
#keyword='bifreq'
#keyword='half'
keyword='single'
keyword='double'
#keyword='cubic'
keyword='sine'
#keyword='halfsmooth'

dx=0.001
plt.style.use('~/.config/matplotlib/joris.mplstyle')
import h5py
from skimage.morphology import disk
from skimage.filters import median,gaussian
ms=['o','d','s','<','>','+','*']
Brownian=1e-3
radius=0.01
l0=0.3
A=1/np.sqrt(2)
#A=1.2
AA=np.linspace(0.3,1.8,8)
Nuall=[]
plt.figure(figsize=(3,3))
for a in AA:
	L,S,wrapped_time,W=run_DSM(1e7,a,1)
	# f=h5py.File('./Compare_stretching_concentration/'+keyword+'/DSM'+keyword+'_A{:1.1f}_l{:1.1f}.hdf5'.format(A,l0), 'r')
	# t=f.attrs['tmax']
	# #t=12
	# L=f['L_{:04d}'.format(int(t*10))][:]
	# wrapped_time=f['wrapped_time_{:04d}'.format(int(t*10))][:]
	# S=f['S_{:04d}'.format(int(t*10))][:]
	# W=f['Weights_{:04d}'.format(int(t*10))][:]
	
	nagg=[]
	
	from scipy import spatial
	nsamples=1000
	sB=0.2
	R=np.logspace(np.log10(sB/1e3),np.log10(sB),20)
	nagg0,nagg1,nagg2=[],[],[]
	for r in R:
		h=np.histogram2d(np.mod(L[:,0],1),np.mod(L[:,1],1),bins=int(1/r))[0]
		nagg0.append(np.sum(h>0)) # Hausdorff dimension (Grassberger 83)
		nagg1.append(np.sum(h[h>0]))
		nagg2.append(np.sum(h[h>0]**2.)) # Correlation dimension (Grassberger 83)
	#p=np.polyfit(np.log(R)[6:],np.log(nagg)[6:],1)
	nu=np.diff(np.log(nagg2))/np.diff(np.log(R))
	num=np.nanmean(nu[4:14])
	Nuall.append(num)
#	plt.figure(figsize=(1.5,1.5))
	#plt.plot(np.log(R)[1:],-np.diff(np.log(nagg0))/np.diff(np.log(R)),'r+',label=r'$d$')
	plt.plot(np.log(R)[1:],nu,'o',color=plt.cm.cool(a/2))
	plt.plot(np.log(R)[1:],np.zeros(nu.shape)+num,'--',color=plt.cm.cool(a/2),label=r'$\nu={:1.2f} (A={:1.2f})$'.format(num,a))
	#plt.plot(np.log(R),np.log(R)*p[0]+p[1],'k:',label='$D_f={:1.2f}$'.format(p[0]))
plt.xlabel(r'$\log r$')
plt.ylabel(r'$d\log N / d\log r$')
#plt.xlim([-8,-1])
plt.ylim([1,2])
plt.legend(fontsize=8)

plt.savefig('./Compare_stretching_concentration/'+keyword+'/fractal_hist_'+keyword+'_A{:1.1f}_T={:1.0f}.pdf'.format(A,t), bbox_inches='tight')

np.savetxt('Sine_D1.txt',np.vstack((AA,np.array(Nuall))).T,header='# A, D_1')

#%%% Fractal measures (run)
keyword='sine'

dx=0.001
plt.style.use('~/.config/matplotlib/joris.mplstyle')
import h5py
from skimage.morphology import disk
from skimage.filters import median,gaussian
ms=['o','d','s','<','>','+','*']
Brownian=1e-3
radius=0.01
l0=0.3
A=1/np.sqrt(2)
#A=1.2
AA=np.linspace(0.3,1.8,8)
Nuall=[]
plt.figure(figsize=(3,3))
a=0.3
L,S,wrapped_time,W,t=run_DSM(1e7,a,3)
dist_old=np.sum(np.diff(L,axis=0)**2,1)**0.5


# check if C by DNS is fractal
C_DNS=run_DNS(a,int(2**11),3,int(t),1e-6)
plt.imshow(C_DNS[0])
#%%%% Fractal measures (plot)
Nf=[]
H=[]
q=2.0
sB=0.2
N=np.logspace(1,2,5)
nagg0,nagg1,nagg2=[],[],[]
for i,n in enumerate(N):
	n=int(n)
	h=np.histogram2d(np.mod(L[1:,0],1),np.mod(L[1:,1],1),bins=n,weights=dist_old*n)[0]
	#print(n)
#	h=np.histogram(x,np.linspace(0,1,int(n)),density=False)[0]
	hlog=np.log(h[h>0])
	h_rho_1=np.histogram2d(np.mod(L[1:,0],1),np.mod(L[1:,1],1),bins=int(n),weights=np.minimum(S,1)*dist_old*n,density=False)[0]
	h_logrho=np.histogram2d(np.mod(L[1:,0],1),np.mod(L[1:,1],1),bins=int(n),weights=-np.log(S)*dist_old*n,density=False)[0]
	h_logrho_n=h_logrho/h
	h_rho=np.histogram2d(np.mod(L[1:,0],1),np.mod(L[1:,1],1),bins=int(n),weights=1/S*dist_old*n,density=False)[0]
	plt.plot(h,h/h_rho_1,'.',color=plt.cm.cool(i/10))
	plt.plot(h,h_rho/h,'+',color=plt.cm.cool(i/10))
	h=h/np.sum(h)
	hlog=hlog/np.nansum(hlog)
	h_rho=h_rho/np.nansum(h_rho)
	h_rho_1=h_rho_1/np.nansum(h_rho_1)
	h_logrho=h_logrho/np.nansum(h_logrho)
	h_logrho_n=h_logrho_n/np.nansum(h_logrho_n)
	#h=h/np.sum(h)
	#h_logrho=h_logrho/np.sum(h_logrho)
	Nf.append([np.sum(h**q),np.sum(h_rho**q),np.sum(h_rho_1**q),np.sum(h_logrho**q),np.nansum(h_logrho_n**q),np.nansum(hlog**q)])
#nu=-np.polyfit(np.log(N[:]),np.log(Nf[:]),1)[0]
Nf=np.array(Nf)
x=np.logspace(-2,2,100)
plt.plot(1e3*x,x**1,'k--',label='1')
plt.plot(1e3*x,x**1.2,'r--',label='1')
plt.yscale('log')
plt.xscale('log')
Nf=np.array(Nf)

D=np.loadtxt('Sine_D1.txt')
D0=np.interp(a,D[:,0],D[:,3])
D1=np.interp(a,D[:,0],D[:,1])
D2=np.interp(a,D[:,0],D[:,2])
plt.figure()
plt.plot(np.log(N),np.log(N)*(D0),'g--',label='$D_0 -1={:1.2f}$'.format(D0-1))
plt.plot(np.log(N),np.log(N)*(D1),'k--',label='$D_1 -1={:1.2f}$'.format(D1-1))
plt.plot(np.log(N),np.log(N)*(D2),'r--',label='$D_2 -1={:1.2f}$'.format(D2-1))
plt.plot(np.log(N),np.log(Nf[:,0])/(1-q),'.-',label=r'$p_i = n$')
plt.plot(np.log(N),np.log(Nf[:,5])/(1-q),'*-',label=r'$p_i = \log n$')
#plt.plot(np.log(N),np.log(Nf[:,1])/(1-q),'+-',label=r'$p_i = \sum \rho$')
plt.plot(np.log(N),np.log(Nf[:,2])/(1-q),'o-',label=r'$p_i = \sum \rho^{-1}$')
plt.plot(np.log(N),np.log(Nf[:,3])/(1-q),'d-',label=r'$p_i = \sum \log \rho$')
plt.plot(np.log(N),np.log(Nf[:,4])/(1-q),'d-',label=r'$p_i = n^{-1}\sum \log \rho$')


# check if C by DNS is fractal
C=C_DNS[0]
F=[]
from skimage.transform import rescale, resize
N=np.logspace(1,2,5)
for i,n in enumerate(N):
	C_r = resize(C, (C.shape[0] // n, C.shape[1] // n),anti_aliasing=False)
	C_r=C_r/np.sum(C_r)
	F.append(np.sum(C_r**q))
F=np.array(F)

plt.plot(np.log(C.shape[0]/N),np.log(F)/(1-q),'+-',label=r'$p_i = c$')
plt.legend(fontsize=6)
plt.ylabel('$1/(1-q) \log \sum p_i^q$')
plt.xlabel('$- \log s_b$')
plt.title('$q={:1.0f}$'.format(q))
plt.savefig(figdir+'fractal_scalings_sine_q{:1.0f}.pdf'.format(q))
#%%% Fractal dimension with histogram multiple seeds
keyword='sine'

dx=0.001
plt.style.use('~/.config/matplotlib/joris.mplstyle')
import h5py
from skimage.morphology import disk
import multiprocessing
from skimage.filters import median,gaussian
ms=['o','d','s','<','>','+','*']
Brownian=1e-3
radius=0.01
l0=0.3

AA=np.linspace(0.3,1.8,15)
def parrallel(s):
	Nuall=[]
	print('Seed=',s)
	for a in AA:
		print('A=',a)
		L,S,wrapped_time,W,t=run_DSM(1e7,a,s)
		from scipy import spatial
		nsamples=1000
		sB=0.2
		R=np.logspace(np.log10(sB/1e3),np.log10(sB),20)
		nagg0,nagg1,nagg2=[],[],[]
		for r in R:
			h=np.histogram2d(np.mod(L[:,0],1),np.mod(L[:,1],1),bins=int(1/r))[0]
			h=h/np.sum(h)
			nagg0.append(np.sum(h**0)) # Hausdorff dimension (Grassberger 83)
			q=1.001
			nagg1.append(np.sum(h[h>0]**q))
			nagg2.append(np.sum(h[h>0]**2.)) # Correlation dimension (Grassberger 83)
		nu=-np.diff(np.log(nagg0))/np.diff(np.log(R))
		D0=np.nanmean(nu[4:14])
		nu=np.diff(np.log(nagg1))/np.diff(np.log(R))
		D1=np.nanmean(nu[4:14])/(q-1)
		nu=np.diff(np.log(nagg2))/np.diff(np.log(R))
		D2=np.nanmean(nu[4:14])
		Nuall.append([D1,D2,D0])
	#plt.plot(np.log(R),np.log(R)*p[0]+p[1],'k:',label='$D_f={:1.2f}$'.format(p[0]))
	return Nuall


Seeds=np.arange(32)
pool = multiprocessing.Pool(processes=len(Seeds))
Nuall_seeds=pool.map(parrallel, Seeds)
pool.close()
pool.join()

nuallmean=np.mean(np.array(Nuall_seeds),axis=0)
plt.plot(AA,nuallmean,'.-')

np.savetxt('Sine_D1.txt',np.vstack((AA,nuallmean.T)).T,header='# A, D_1, D_2, D_0')
#%%% Models for fractal dimension of sine waves
alpha=np.sqrt(1.+np.pi**2/2)
#alpha=11
pi=np.array([alpha,alpha,1,1,1,1])
pi=np.array([alpha,alpha,alpha,alpha,1,1])
pi=pi/np.sum(pi)

#D2=-np.log(np.sum(pi**2.))/np.log(6)
D2=-np.log((4*alpha**2+2)/(4*alpha+2)**2)/np.log(6)
print('Sine Wave:',D2+1, '(Estimated 1.74)')
D2=-np.log((2*alpha**2+4)/(2*alpha+4)**2)/np.log(6)
print('Half Wave:',D2+1, '(Estimated 1.55)')
D2=-np.log((2*alpha**2+1)/(2*alpha+1)**2)/np.log(3)
print('Single Wave:',D2+1, '(Estimated 1.74)')
D2=-np.log((4*alpha**2+5)/(4*alpha+5)**2)/np.log(9)
print('Cubic Wave:',D2+1, '(Estimated 1.66)')
#D2=-np.log(np.sum(pi**2.))/np.log(6)


#%% Theory 

#%%% Gamma pdf of n, <c^2>

plt.style.use('~/.config/matplotlib/joris.mplstyle')
L=np.loadtxt('Sine_Lyap.txt')
DD1=np.loadtxt('Sine_D1.txt')

D11=np.linspace(np.min(DD1[:,1]),np.max(DD1[:,1]),100)
l0=0.3
Gamma2=[]
for i,D1 in enumerate(D11):
	mu=1/(D1-1)
	sigma2=2*(2-D1)/(D1-1)
	if mu-2*sigma2>0:
		M2=2*mu-2*sigma2
	else:
		M2=mu**2/(2*sigma2)
	
	
	epsilon=-M2+2
	# Fit from epsilon data
	sB=1/50
	G=np.loadtxt('Sine_scaling_N_sB1_{:1.0f}.txt'.format(1/sB))
	epsilon=np.interp(D1,G[:,0],2-G[:,2])
	
	t=10
	#sB=1/100
	
	k=-(D1-1)/2/np.log(sB)/(2-D1)
	
	from scipy.special import gamma
	def mean_n_alpha(n,k,theta,alpha):
		# Gamma distribution
		return (n**alpha)/gamma(k)/theta**k*n**(k-1)*np.exp(-n/theta)
		#return (n**alpha)/gamma(k)/theta**k*n**(k-1)*np.exp(-n/theta)

	import scipy.integrate
	alpha=-1+epsilon
	T=np.arange(5,10)
	A=np.interp(D1,DD1[:,1],DD1[:,0])
	lyap=np.interp(A,L[:,0],L[:,1])
	sigma2=np.interp(A,L[:,0],L[:,2])
	C2=[scipy.integrate.quad(mean_n_alpha,0,1000,args=(k,l0*np.exp((lyap+sigma2/2)*t)*sB*np.abs(np.log(sB))*2*(2-D1)/(D1-1),alpha))[0] for t in T]
	
	plt.plot(T,C2,'.',color=plt.cm.cool(i/len(D11)))
	plt.yscale('log')
	p=np.polyfit(T,np.log(C2),1)
	plt.plot(T,np.exp(p[0]*T+p[1]),color=plt.cm.cool(i/len(D11)))
	Gamma2.append(-p[0])

Gamma2=np.array(Gamma2)
plt.figure()
plt.plot(D11,Gamma2,'k-o',fillstyle='none')
plt.xlabel(r'$D_1$')
plt.ylabel(r'$\gamma_2$')
#plt.yscale('log')
#plt.ylim([1e-3,1e0])
plt.xlim([1.3,2])

np.savetxt('sine_gamma2_theory.txt',np.vstack((D11,Gamma2)).T)
#%%% mu and sigma2 as a function of nu

nu=np.linspace(1.2,2,1000)
mu=1/(nu-1)
sigma2=2*(2-nu)/(nu-1)
M2=2*mu-2*sigma2
M2[mu-2*sigma2<0]=mu[mu-2*sigma2<0]**2/(2*sigma2[mu-2*sigma2<0])
plt.figure(figsize=(2.5,2.5))
plt.plot(nu,mu,'k-',label=r'$\mu=1/(1-\nu)$')
plt.plot(nu,sigma2,'k--',label=r'$\sigma^2=2(2-\nu)/(1-\nu)$')
plt.plot(nu,M2,'r-',label=r'$\gamma_2$',linewidth=2)
plt.plot(nu,mu**2/(2*sigma2),'k:',label=r'$\mu^2/(2\sigma^2)$')
plt.plot(nu,2*mu-2*sigma2,'k-.',label=r'$2\mu-2\sigma^2$')
plt.ylim([0,5])
plt.legend()
plt.xlabel(r'$\nu$')
plt.savefig('nu_mi_sigma2.pdf',bbox_inches='tight')