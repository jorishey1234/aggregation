#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 15:53:46 2020

@author: joris
"""

#%% RUN FIRST : Quick Baker Flow simulations
##TODO

figdir = '/home/joris/Dropbox/Articles/AggregationBatchelor/Figures/'

def subscript(ax,i,color='k',bg='w',x=0.03,y=0.93,script=['a)','b)','c)','d)']):
	txt=ax.text(x,y,script[i],c=color,transform = ax.transAxes,backgroundcolor=bg)
	return txt

def lyapunov(A):
	# Mean and Variance of stretching rates
	return -A*np.log(A)-(1-A)*np.log(1-A),(A*np.log(A)**2.+(1-A)*np.log(1-A)**2.-(-A*np.log(A)-(1-A)*np.log(1-A))**2.)

#% Check cmax of bundles
def bin_operation(x,y,xi,op):
	r=np.zeros(xi.shape[0]-1)
	for i in range(xi.shape[0]-1):
		idx=np.where((x<=xi[i+1])&(x>=xi[i]))[0]
		r[i]=op(y[idx])
	return r


dir_out='./Compare_stretching_concentration/baker/'

dir_out='/home/joris/Dropbox/Articles/AggregationBatchelor/Figures/'
#plt.style.use('~/.config/matplotlib/joris.mplstyle')
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, misc
import numpy.fft
from scipy.ndimage import gaussian_filter

def lagrangian_reconstruct(x,Cmax,Si,n):
	# REconstruct concentration field on a n grid from position x, cmax, and s of individual lamellae
	X,Y=np.meshgrid(0,np.arange(n)) 
	C_lag=np.zeros(len(Y))
	for y in range(len(Y)):
		C_lag[y]=np.sum(Cmax*np.exp(-(x*n-y)**2./(Si*n)**2.)+
									 Cmax*np.exp(-((x-1)*n-y)**2./(Si*n)**2.)+
								 Cmax*np.exp(-((x+1)*n-y)**2./(Si*n)**2.))
## C_lag[y]=np.sum(Cmax*np.exp(-(x*n-y)**2./(Si*n)**2.))
	return C_lag

def lin_reg(xt,yt):
	# Linear regression with CI interval on slope and intercept
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

def best_polyfit(x,y):
	# Take polyfit on a sub-range (min 75% of the full range) to maximize R2
	res=[]
	nmin=int(len(x)*0.75)
	for i in range(len(x)-nmin):
		for j in range(i+nmin,len(x)):
			a,b,CIa,CIb,R2=lin_reg(x[i:j],y[i:j])
			res.append([i,j,R2,a,b,CIa,CIb])
	res=np.array(res)
	idmax=np.argmin(res[:,6])
	idmax=np.argmax(res[:,2])
	#print(res)
	idfit=np.arange(int(res[idmax,0]),int(res[idmax,1]))
	return lin_reg(x[idfit],y[idfit])

#TODO

VCall=[]
par=[[8.8,2048,1,'circle'],
		 [4.4,2048,2,'circle'],
		 [2.2,2048,4,'circle'],
		 [1.10,2048,8,'circle']]

#for pa in par:
sigma,n,P,IC=[20,2**14,4,'circle']

#	sigma=16
M=10 # number of moments to compute
dt=1/1  # Discretisation of time step
X,Y=np.meshgrid(0,np.arange(n))

def diffusion_fourier(C,sigma):
	input_ = numpy.fft.fft2(C)
	result = ndimage.fourier_gaussian(input_, sigma=sigma)
	return numpy.fft.ifft2(result).real



def diffusion(C,sigma):
	return gaussian_filter(C, sigma)
# Initial condition
#for sigma in np.linspace(3,50,4):
#for sigma in np.linspace(1,20,3):

def DNS_fourier(x0,T,sa,random):
	C=np.zeros((1,n),dtype=np.float64())
	#C[0,:int(n/2)]=1
	s0=0.05*n
	for i in range(x0.shape[0]):
		C[0,:]+=np.exp(-(Y[:,0]-x0[i]*n)**2./s0**2.)
	Cm,VC,K,Theta,H=[],[],[],[],[]
	C=C[0,:]
	Cm=C.mean()
	Cf=numpy.fft.fft(np.complex128(C-Cm))
	nn=np.fft.fftfreq(C.shape[0],d=1/C.shape[0]).reshape(1,-1)
	M,N=np.meshgrid(nn,nn)
	a=sa
	D=(sigma/n)**2/2
		#transfer matrix (see WOnhas 2002 (16))
		# n = k/2 pi mode number
	mNM=np.exp(-4*np.pi**2.*D*N**2.-1j*np.pi*a*N)*np.sin(np.pi*a*N)/np.pi*((1-2*a)*M)/(M-a*N)/(M-(1-a)*N)
	mNM[0,0]=1 # Mean
#		mNM[0,0]=1 # Mean
	mNM[M==(1-a)*N]=np.exp(-4*np.pi**2.*D*N[M==(1-a)*N]**2.-1j*np.pi*a*N[M==(1-a)*N])*(1-a)
	mNM[M==a*N]=np.exp(-4*np.pi**2.*D*N[M==a*N]**2.-1j*np.pi*a*N[M==a*N])*a
	for k in range(T):
		Cf=np.dot(mNM,Cf)
#		Cf[np.isnan(Cf)]=0
	C2=np.fft.ifft(Cf)
	return C2+Cm
	
def DNS(x0,T,sa,random):
	np.random.seed(seed=1)
	C=np.zeros((1,n),dtype=np.float64())
	#C[0,:int(n/2)]=1
	s0=0.05*n
	for i in range(x0.shape[0]):
		C[0,:]+=np.exp(-(Y[:,0]-x0[i]*n)**2./s0**2.)
	Cm,VC,K,Theta,H=[],[],[],[],[]
	C0=np.mean(C)
	Cm.append([np.mean(np.abs(C-C0)**m) for m in range(M)])
	VC.append(np.var(C))
	np.random.seed(seed=1)
	for k in range(T):
		print('t=',k)
		#MapX=np.mod(np.uint32(X+0.5*dt*a*n*np.sin(Y*P/n*2*np.pi+np.random.rand()*2*np.pi)),n)
		if random:
			a=0.5+np.random.rand()*(0.5-sa)
		else:
			a=0.5+(0.5-sa)
		print(a)
	#	a=0.4
		MapY=np.uint32(Y/a*(Y<a*n)+(Y-a*n)/(1-a)*(Y>=a*n))
		C[X.flatten(),Y.flatten()]=C[X.flatten(),MapY.flatten()]
		C=diffusion_fourier(C, sigma*np.sqrt(dt))
	return C

def DNS_sigma(x0,T,sa,random,sigma):
	np.random.seed(seed=1)
	C=np.zeros((1,n),dtype=np.float64())
	#C[0,:int(n/2)]=1
	s0=0.05*n
	for i in range(x0.shape[0]):
		C[0,:]+=np.exp(-(Y[:,0]-x0[i]*n)**2./s0**2.)
	Cm,VC,K,Theta,H=[],[],[],[],[]
	C0=np.mean(C)
	Cm.append([np.mean(np.abs(C-C0)**m) for m in range(M)])
	VC.append(np.var(C))
	np.random.seed(seed=1)
	for k in range(T):
		print('t=',k)
		#MapX=np.mod(np.uint32(X+0.5*dt*a*n*np.sin(Y*P/n*2*np.pi+np.random.rand()*2*np.pi)),n)
		if random:
			a=0.5+np.random.rand()*(0.5-sa)
		else:
			a=0.5+(0.5-sa)
		print(a)
	#	a=0.4
		MapY=np.uint32(Y/a*(Y<a*n)+(Y-a*n)/(1-a)*(Y>=a*n))
		C[X.flatten(),Y.flatten()]=C[X.flatten(),MapY.flatten()]
		C=diffusion_fourier(C, sigma*np.sqrt(dt))
	return C

def DNS_n(n,x0,T,sa,random):
	np.random.seed(seed=1)
	X,Y=np.meshgrid(0,np.arange(n))
	C=np.zeros((1,n),dtype=np.float64())
	#C[0,:int(n/2)]=1
	s0=0.05*n
	for i in range(x0.shape[0]):
		C[0,:]+=np.exp(-(Y[:,0]-x0[i]*n)**2./s0**2.)
	Cm,VC,K,Theta,H=[],[],[],[],[]
	C0=np.mean(C)
	Cm.append([np.mean(np.abs(C-C0)**m) for m in range(M)])
	VC.append(np.var(C))
	
	np.random.seed(seed=1)
	for k in range(T):
		print('t=',k)
		#MapX=np.mod(np.uint32(X+0.5*dt*a*n*np.sin(Y*P/n*2*np.pi+np.random.rand()*2*np.pi)),n)
		if random:
			a=0.5+np.random.rand()*(0.5-sa)
		else:
			a=0.5+(0.5-sa)
		print(a)
	#	a=0.4
		MapY=np.uint32(Y/a*(Y<a*n)+(Y-a*n)/(1-a)*(Y>=a*n))
		C[X.flatten(),Y.flatten()]=C[X.flatten(),MapY.flatten()]
		C=diffusion_fourier(C, sigma*np.sqrt(dt))
	return C


def DNS_ns0(n,s0,x0,T,sa,random):
	np.random.seed(seed=1)
	C=np.zeros((1,n),dtype=np.float64())
	#C[0,:int(n/2)]=1
	s0=s0*n
	for i in range(x0.shape[0]):
		C[0,:]+=np.exp(-(Y[:,0]-x0[i]*n)**2./s0**2.)+np.exp(-(Y[:,0]-(x0[i]+1)*n)**2./s0**2.)+np.exp(-(Y[:,0]-(x0[i]-1)*n)**2./s0**2.)
#		C[0,:]+=np.exp(-(Y[:,0]-x0[i]*n)**2./s0**2.)#+np.exp(-(Y[:,0]-(x0[i]+1)*n)**2./s0**2.)+np.exp(-(Y[:,0]-(x0[i]-1)*n)**2./s0**2.)
	Cm,VC,K,Theta,H=[],[],[],[],[]
	C0=np.mean(C)
	Cm.append([np.mean(np.abs(C-C0)**m) for m in range(M)])
	VC.append(np.var(C))
	np.random.seed(seed=1)
	varC=[np.var(C)]
	for k in range(T):
		print('t=',k)
		#MapX=np.mod(np.uint32(X+0.5*dt*a*n*np.sin(Y*P/n*2*np.pi+np.random.rand()*2*np.pi)),n)
		if random:
			a=0.5+np.random.rand()*(0.5-sa)
		else:
			a=0.5+(0.5-sa)
		print(a)
	#	a=0.4
		MapY=np.uint32(Y/a*(Y<a*n)+(Y-a*n)/(1-a)*(Y>=a*n))
		C[X.flatten(),Y.flatten()]=C[X.flatten(),MapY.flatten()]
		C=diffusion_fourier(C, sigma*np.sqrt(dt))
		varC.append(np.var(C))
	return C,np.array(varC)

def DNS_0mean(n,s0,x0,T,sa,sigma,random):
	np.random.seed(seed=1)
	C=np.zeros((1,n),dtype=np.float64())
	#C[0,:int(n/2)]=1
	s0=s0*n
	for i in range(x0.shape[0]):
		C[0,:]+=np.exp(-(Y[:,0]-x0[i]*n)**2./s0**2.)*(Y[:,0]-x0[i]*n)/s0+np.exp(-(Y[:,0]-(x0[i]-1)*n)**2./s0**2.)*(Y[:,0]-(x0[i]-1)*n)/s0+np.exp(-(Y[:,0]-(x0[i]+1)*n)**2./s0**2.)*(Y[:,0]-(x0[i]+1)*n)/s0
	Cm,VC,K,Theta,H=[],[],[],[],[]
	C0=np.mean(C)
	C=C-C0
	Cm.append([np.mean(np.abs(C-C0)**m) for m in range(M)])
	VC.append(np.var(C))
	varC=[np.var(C)]
	np.random.seed(seed=1)
	for k in range(T):
		print('t=',k)
		#MapX=np.mod(np.uint32(X+0.5*dt*a*n*np.sin(Y*P/n*2*np.pi+np.random.rand()*2*np.pi)),n)
		if random:
			a=0.5+np.random.rand()*(0.5-sa)
		else:
			a=0.5+(0.5-sa)
		print(a)
	#	a=0.4
		MapY=np.uint32(Y/a*(Y<a*n)+(Y-a*n)/(1-a)*(Y>=a*n))
		C[X.flatten(),Y.flatten()]=C[X.flatten(),MapY.flatten()]
		C=diffusion_fourier(C, sigma*np.sqrt(dt))
		#C=C-C.mean()
		varC.append(np.var(C))
	return C,np.array(varC)


def DNS_0mean_flipped(n,s0,x0,T,sa,sigma,random):
	np.random.seed(seed=1)
	C=np.zeros((1,n),dtype=np.float64())
	#C[0,:int(n/2)]=1
	s0=s0*n
	for i in range(x0.shape[0]):
		C[0,:]+=np.exp(-(Y[:,0]-x0[i]*n)**2./s0**2.)*(Y[:,0]-x0[i]*n)/s0+np.exp(-(Y[:,0]-(x0[i]-1)*n)**2./s0**2.)*(Y[:,0]-(x0[i]-1)*n)/s0+np.exp(-(Y[:,0]-(x0[i]+1)*n)**2./s0**2.)*(Y[:,0]-(x0[i]+1)*n)/s0
	Cm,VC,K,Theta,H=[],[],[],[],[]
	C0=np.mean(C)
#	C=C-C0
	Cm.append([np.mean(np.abs(C-C0)**m) for m in range(M)])
	VC.append(np.var(C))
	varC=[]
	np.random.seed(seed=1)
	for k in range(T):
		print('t=',k)
		#MapX=np.mod(np.uint32(X+0.5*dt*a*n*np.sin(Y*P/n*2*np.pi+np.random.rand()*2*np.pi)),n)
		if random:
			a=0.5+np.random.rand()*(0.5-sa)
		else:
			a=0.5+(0.5-sa)
		print(a)
	#	a=0.4
		MapY=np.uint32(Y/a*(Y<a*n)+(n-(Y-a*n)/(1-a))*(Y>=a*n))
		C[X.flatten(),Y.flatten()]=C[X.flatten(),MapY.flatten()]
		C=diffusion_fourier(C, sigma*np.sqrt(dt))
		#C=C-C.mean()
		varC.append(np.var(C))
	return C,np.array(varC)

def DNS_flipped(n,s0,x0,T,sa,sigma,random):
	np.random.seed(seed=1)
	C=np.zeros((1,n),dtype=np.float64())
	#C[0,:int(n/2)]=1
	s0=s0*n
	for i in range(x0.shape[0]):
		C[0,:]+=np.exp(-(Y[:,0]-x0[i]*n)**2./s0**2.)+np.exp(-(Y[:,0]-(x0[i]-1)*n)**2./s0**2.)+np.exp(-(Y[:,0]-(x0[i]+1)*n)**2./s0**2.)
	Cm,VC,K,Theta,H=[],[],[],[],[]
	C0=np.mean(C)
#	C=C-C0
	Cm.append([np.mean(np.abs(C-C0)**m) for m in range(M)])
	VC.append(np.var(C))
	varC=[]
	np.random.seed(seed=1)
	for k in range(T):
		print('t=',k)
		#MapX=np.mod(np.uint32(X+0.5*dt*a*n*np.sin(Y*P/n*2*np.pi+np.random.rand()*2*np.pi)),n)
		if random:
			a=0.5+np.random.rand()*(0.5-sa)
		else:
			a=0.5+(0.5-sa)
		print(a)
	#	a=0.4
		MapY=np.uint32(Y/a*(Y<a*n)+(n-(Y-a*n)/(1-a))*(Y>=a*n))
		C[X.flatten(),Y.flatten()]=C[X.flatten(),MapY.flatten()]
		C=diffusion_fourier(C, sigma*np.sqrt(dt))
		#C=C-C.mean()
		varC.append(np.var(C))
	return C,np.array(varC)

def DNS_flipped_rand(n,s0,T,sa,sigma,random):
	np.random.seed(seed=1)
	C=np.zeros((1,n),dtype=np.float64())
	#C[0,:int(n/2)]=1
	s0=s0*n
	C[0,:]=np.random.randn(n)
	Cm,VC,K,Theta,H=[],[],[],[],[]
	C0=np.mean(C)
#	C=C-C0
	Cm.append([np.mean(np.abs(C-C0)**m) for m in range(M)])
	VC.append(np.var(C))
	varC=[]
	np.random.seed(seed=1)
	for k in range(T):
		print('t=',k)
		#MapX=np.mod(np.uint32(X+0.5*dt*a*n*np.sin(Y*P/n*2*np.pi+np.random.rand()*2*np.pi)),n)
		if random:
			a=0.5+np.random.rand()*(0.5-sa)
		else:
			a=0.5+(0.5-sa)
		print(a)
	#	a=0.4
		MapY=np.uint32(Y/a*(Y<a*n)+(n-(Y-a*n)/(1-a))*(Y>=a*n))
		C[X.flatten(),Y.flatten()]=C[X.flatten(),MapY.flatten()]
		C=diffusion_fourier(C, sigma*np.sqrt(dt))
		#C=C-C.mean()
		varC.append(np.var(C))
	return C,np.array(varC)


def DNS_0mean_normed(n,s0,x0,T,sa,sigma,random):
	np.random.seed(seed=1)
	C=np.zeros((1,n),dtype=np.float64())
	#C[0,:int(n/2)]=1
	s0=s0*n
	for i in range(x0.shape[0]):
		C[0,:]+=np.exp(-(Y[:,0]-x0[i]*n)**2./s0**2.)*(Y[:,0]-x0[i]*n)/s0+np.exp(-(Y[:,0]-(x0[i]-1)*n)**2./s0**2.)*(Y[:,0]-(x0[i]-1)*n)/s0+np.exp(-(Y[:,0]-(x0[i]+1)*n)**2./s0**2.)*(Y[:,0]-(x0[i]+1)*n)/s0
	Cm,VC,K,Theta,H=[],[],[],[],[]
	C0=np.mean(C)
	C=C-C0
	Cm.append([np.mean(np.abs(C-C0)**m) for m in range(M)])
	VC.append(np.var(C))
	varC=[]
	Cmax=[]
	np.random.seed(seed=1)
	for k in range(T):
		Cmax.append(C.max())
		C=C/C.max()
		print('t=',k)
		#MapX=np.mod(np.uint32(X+0.5*dt*a*n*np.sin(Y*P/n*2*np.pi+np.random.rand()*2*np.pi)),n)
		if random:
			a=0.5+np.random.rand()*(0.5-sa)
		else:
			a=0.5+(0.5-sa)
		print(a)
	#	a=0.4
		MapY=np.uint32(Y/a*(Y<a*n)+(Y-a*n)/(1-a)*(Y>=a*n))
		C[X.flatten(),Y.flatten()]=C[X.flatten(),MapY.flatten()]
		C=diffusion_fourier(C, sigma*np.sqrt(dt))
		varC.append(np.var(C)*Cmax[-1]**2.)
	return C,np.array(Cmax),np.array(varC)

def DNS_nfold(x0,T,sa,random):
	np.random.seed(seed=1)
	C=np.zeros((1,n),dtype=np.float64())
	s0=0.05*n
	for i in range(len(x0)):
		C[0,:]+=np.exp(-(Y[:,0]-x0[i]*n)**2./s0**2.)
	Cm,VC,K,Theta,H=[],[],[],[],[]
	C0=np.mean(C)
	Cm.append([np.mean(np.abs(C-C0)**m) for m in range(M)])
	VC.append(np.var(C))
	np.random.seed(seed=1)
	sa=sa/np.sum(sa)
	sai=np.zeros(len(sa)+1)
	sai[1:]=sa
	sai=np.cumsum(sai)
	for k in range(T):
		print('t=',k)
		MapY=np.uint32(np.sum(np.array([((Y-sai[i-1]*n)/sa[i-1])*(Y<sai[i]*n)*(Y>=sai[i-1]*n) 
		for i in np.arange(1,len(sai))]),axis=0))
		C[X.flatten(),Y.flatten()]=C[X.flatten(),MapY.flatten()]
		C=diffusion_fourier(C, sigma*np.sqrt(dt))
	return C
#%
# T=np.arange(30)
# plt.figure()
# plt.plot(VC1,label=r'$\alpha$ random')
# plt.plot(VC3,label=r'$\alpha$ random (smaller variance)')
# plt.plot(VC0,label=r'$\alpha=0.5$')
# plt.plot(VC,label=r'$\alpha=0.4$')
# plt.yscale('log')
# plt.xlabel('$t$')
# plt.legend()
# plt.ylabel('Variance')
# plt.savefig('Baker_variance.pdf')

#% Evolution of material elements


def lagrangian(x,T,sa,random):
	#np.random.seed(seed=1)
# =============================================================================
# 	Baker Map applied to lamella located in x of size S 
# =============================================================================
	#x=np.array([0.3]) # we start with a unique lamellae
	S=np.ones(x.shape,dtype=np.float64) # we start with a unique lamellae
	wrapped_time=np.zeros(x.shape,dtype=np.float16) # we start with a unique lamellae
	dt=1.
	for k in range(T):
		#print('t=',k)
		#MapX=np.mod(np.uint32(X+0.5*dt*a*n*np.sin(Y*P/n*2*np.pi+np.random.rand()*2*np.pi)),n)	
		a=0.5+(0.5-sa)+(np.random.rand()-0.5)*random
		#print(a)
		x=np.hstack((x*a,x*(1-a)+a))
		wrapped_time=np.hstack((wrapped_time+1/(S*a)**2.*dt,wrapped_time+1/(S*(1-a))**2.*dt))
		S=np.hstack((S*a,S*(1-a)))
	return x,S,wrapped_time


def lagrangian_DSM(x,T,sa,random,sB):
	#np.random.seed(seed=1)
# =============================================================================
# 	Baker Map applied to lamella located in x of size S 
# =============================================================================
	#x=np.array([0.3]) # we start with a unique lamellae
	S=np.ones(x.shape,dtype=np.float64) # we start with a unique lamellae
	wrapped_time=np.zeros(x.shape,dtype=np.float16) # we start with a unique lamellae
	dt=1.
	
	h,xb=np.histogram(x,np.arange(0,1+sB,sB),weights=S/sB)
	C=[h]
	for k in range(T):
		#print('t=',k)
		#MapX=np.mod(np.uint32(X+0.5*dt*a*n*np.sin(Y*P/n*2*np.pi+np.random.rand()*2*np.pi)),n)	
		a=0.5+(0.5-sa)+(np.random.rand()-0.5)*random
		#print(a)
		x=np.hstack((x*a,x*(1-a)+a))
		wrapped_time=np.hstack((wrapped_time+1/(S*a)**2.*dt,wrapped_time+1/(S*(1-a))**2.*dt))
		S=np.hstack((S*a,S*(1-a)))
		h,xb=np.histogram(x,np.arange(0,1+sB,sB),weights=S/sB)
		C.append(h)
	return x,S,wrapped_time,C


def lagrangian_random(x,T,sa):
	np.random.seed(seed=1)
# =============================================================================
# 	Baker Map applied to lamella located in x of size S 
# =============================================================================
	#x=np.array([0.3]) # we start with a unique lamellae
	S=np.ones(x.shape,dtype=np.float64) # we start with a unique lamellae
	wrapped_time=np.zeros(x.shape,dtype=np.float16) # we start with a unique lamellae
	dt=1.
	for k in range(T):
		#print('t=',k)
		#MapX=np.mod(np.uint32(X+0.5*dt*a*n*np.sin(Y*P/n*2*np.pi+np.random.rand()*2*np.pi)),n)	
		a=0.5+(0.5-sa)
		#print(a)
		if np.random.rand()>0.5:
			x=np.hstack((x*a,x*(1-a)+a))
			wrapped_time=np.hstack((wrapped_time+1/(S*a)**2.*dt,wrapped_time+1/(S*(1-a))**2.*dt))
			S=np.hstack((S*a,S*(1-a)))
		else:
			x=np.hstack((x*(1-a),x*a+(1-a)))
			wrapped_time=np.hstack((wrapped_time+1/(S*(1-a))**2.*dt,wrapped_time+1/(S*(a))**2.*dt))
			S=np.hstack((S*(1-a),S*a))
	return x,S,wrapped_time

def lagrangian_reconstruct(x,T,sa,random,sB):
	#np.random.seed(seed=1)
# =============================================================================
# 	Baker Map applied to lamella located in x of size S 
# =============================================================================
	#x=np.array([0.3]) # we start with a unique lamellae
	S=np.ones(x.shape,dtype=np.float64) # we start with a unique lamellae
	wrapped_time=np.zeros(x.shape,dtype=np.float16) # we start with a unique lamellae
	dt=1.
	V=[]
	tt=np.arange(T)
	for k in tt:
		#print('t=',k)
		#MapX=np.mod(np.uint32(X+0.5*dt*a*n*np.sin(Y*P/n*2*np.pi+np.random.rand()*2*np.pi)),n)	
		a=0.5+(0.5-sa)+(np.random.rand()-0.5)*random
		#print(a)
		x=np.hstack((x*a,x*(1-a)+a))
		wrapped_time=np.hstack((wrapped_time+1/(S*a)**2.*dt,wrapped_time+1/(S*(1-a))**2.*dt))
		S=np.hstack((S*a,S*(1-a)))
		C=np.histogram(x,np.arange(0,1,sB),density=False,weights=S)[0]
		V.append(np.var(C))
	return tt,np.array(V)

#Baker map with symetry
def lagrangian_sym(x,T,sa,random):
	#np.random.seed(seed=1)
# =============================================================================
# 	Baker Map applied to lamella located in x of size S 
# =============================================================================
	#x=np.array([0.3]) # we start with a unique lamellae
	S=np.ones(x.shape,dtype=np.float64) # we start with a unique lamellae
	wrapped_time=np.zeros(x.shape,dtype=np.float16) # we start with a unique lamellae
	dt=1.
	for k in range(T):
		#print('t=',k)
		#MapX=np.mod(np.uint32(X+0.5*dt*a*n*np.sin(Y*P/n*2*np.pi+np.random.rand()*2*np.pi)),n)	
		a=0.5+(0.5-sa)+(np.random.rand()-0.5)*random
		#print(a)
		x=np.hstack(((1-x)*a,x*(1-a)+a))
		wrapped_time=np.hstack((wrapped_time+1/(S*a)**2.*dt,wrapped_time+1/(S*(1-a))**2.*dt))
		S=np.hstack((S*a,S*(1-a)))
	return x,S,wrapped_time

def lagrangian_nfold(x,T,sa,random):
	#np.random.seed(seed=1)
# =============================================================================
# 	Baker Map applied to lamella located in x of size S 
# =============================================================================
	#x=np.array([0.3]) # we start with a unique lamellae
	S=np.ones(x.shape) # we start with a unique lamellae
	wrapped_time=np.zeros(x.shape) # we start with a unique lamellae
	dt=1.
	sa=sa/np.sum(sa)
	for k in range(T):
		#print('t=',k)
		#MapX=np.mod(np.uint32(X+0.5*dt*a*n*np.sin(Y*P/n*2*np.pi+np.random.rand()*2*np.pi)),n)	
		#a=0.5+(0.5-sa)+(np.random.rand()-0.5)*random
		#print(a)
		#xc=np.copy(x)
		iorder=np.arange(len(sa))
		if shuffle: np.random.shuffle(iorder)
		sai=np.zeros(len(sa)+1)
		sai[1:]=sa[iorder]
		sai=np.cumsum(sai)
		x=np.hstack([x*sa[iorder[i]]+sai[i] for i in np.arange(len(iorder))])
		wrapped_time=np.hstack([wrapped_time+1/(S*sa[i])**2.*dt for i in iorder])
		S=np.hstack([S*sa[i] for i in iorder])
	return x,S,wrapped_time

def lagrangian_dispersive(x,T,sa,disp):
	#np.random.seed(seed=1)
# =============================================================================
# 	Baker Map applied to lamella located in x of size S 
# =============================================================================
	#x=np.array([0.3]) # we start with a unique lamellae
	S=np.ones(x.shape) # we start with a unique lamellae
	wrapped_time=np.zeros(x.shape) # we start with a unique lamellae
	dt=1.
	sa=sa/np.sum(sa)
	sai=np.zeros(len(sa)+1)
	sai[1:]=sa
	sai=np.cumsum(sai)
	for k in range(T):
		print(k)
		x+=np.random.randn(1)*disp
		cells=np.arange(np.floor(x.min()),np.ceil(x.max())) # cells
		xtemp=np.array([])
		wtemp=np.array([])
		Stemp=np.array([])
		for c in cells:
			idc=(x>c)&(x<c+1)
			xtemp=np.hstack((xtemp,
									 c+np.hstack([(x[idc]-c)*sa[i]+sai[i] for i in range(len(sa))])))
			wtemp=np.hstack((wtemp,
									 np.hstack([wrapped_time[idc]+1/(S[idc]*sa[i])**2.*dt for i in range(len(sa))])))
			Stemp=np.hstack((Stemp,np.hstack([S[idc]*sa[i] for i in range(len(sa))])))
		x=xtemp
		S=Stemp
		wrapped_time=wtemp
	return x,S,wrapped_time


def reconstruct_c(Y,x,Cmax,Si):
	n=Y.shape[0]
	C_lag=np.zeros(len(Y))
	for y in range(len(Y)):
		C_lag[y]=np.sum(Cmax*np.exp(-(x*n-y)**2./(Si*n)** 2.)+
									 Cmax*np.exp(-((x-1)*n-y)**2./(Si*n)**2.)+
									 Cmax*np.exp(-((x+1)*n-y)**2./(Si*n)**2.))
	return C_lag

#% Fractal dimensions
def fractal(xi,q):
	N=np.logspace(1,4,100)
	Nf=[]
	for n in N:
		#print(n)
		h=np.histogram(xi,np.linspace(0,1,int(n)))[0]
		h=h/np.sum(h)
		Nf.append(np.sum(h[h>0]**q))
	nu=-np.polyfit(np.log(N[:]),np.log(Nf[:]),1)[0]
	#nu=-best_polyfit(np.log(N),np.log(Nf))[1]
	return nu/(q-1),N,Nf

#% Fractal dimensions
def fractal_D1(xi):
	N=np.logspace(1,4,100)
	Nf=[]
	for n in N:
		#print(n)
		h=np.histogram(xi,np.linspace(0,1,int(n)))[0]
		h=h/np.sum(h)
		Nf.append(np.sum(h[h>0]*np.log(h[h>0])))
	nu=-np.polyfit(np.log(N[:]),Nf[:],1)[0]
	#nu=-best_polyfit(np.log(N),np.log(Nf))[1]
	return nu,N,Nf

#% Fractal dimensions centered on lamella
def fractal_centered(x,q,ng):
	N=np.logspace(1,4,100)
	Nf=[]
# =============================================================================
#	ng=1000
	tree=spatial.cKDTree(x.reshape(-1,1))
	idsamples=np.unique(np.uint16(x.reshape(-1,1)*ng), return_index=True,axis=0)[1]
	for n in N:
		neighboors=tree.query_ball_point(x.reshape(-1,1)[idsamples], 1/n)
		nagg1=np.sum([len(ni) for ni in neighboors])
		nagg=np.sum([(len(ni))**q for ni in neighboors])
		Nf.append(nagg)
	nu=np.polyfit(-np.log(N[:]),np.log(Nf[:]),1)[0]
	#nu=-best_polyfit(np.log(N),np.log(Nf))[1]
	return nu,N,Nf




#% Fractal dimensions
def fractal_best(xi,q):
	N=np.logspace(1,4,50)
	Nf=[]
	for n in N:
		#print(n)
		h=np.histogram(xi,np.linspace(0,1,int(n)))[0]
		h=h/np.sum(h)
		Nf.append(np.sum(h**q))
	#nu=-np.polyfit(np.log(N[:]),np.log(Nf[:]),1)[0]
	nu=best_polyfit(np.log(N),np.log(Nf))
	return -nu[1],N,Nf,nu[3]

def fractal2(xi):
	N=np.logspace(1,4,20)
	Nf=[]
	for n in N:
		#print(n)
		h=np.histogram(xi,np.linspace(0,1,int(n)))[0]
		h=h/np.sum(h)
		Nf.append(np.sum(h[h>0]**2.))
	#nu=-np.polyfit(np.log(N[:]),np.log(Nf[:]),1)[0]
	nu=-best_polyfit(np.log(N),np.log(Nf))[1]
	return nu,N,Nf

def fractal_weighted(xi,S,q):
# =============================================================================
# 	Compute fractal dimension from position of lamellae
# =============================================================================
	N=np.logspace(1,4,20)
	Nf=[]
	for n in N:
		#print(n)
		h=np.histogram(xi,np.linspace(0,1,int(n)),weights=S)[0]
		Nf.append(np.sum(h[h>0]**q)) # 2nd order
	nu=-np.polyfit(np.log(N),np.log(Nf),1)[0]
	return nu
#plt.plot(x)
def fractal_weighted_scale(xi,S,q,scale):
# =============================================================================
# 	Compute fractal dimension from position of lamellae
# =============================================================================
	
	N=np.logspace(np.log10(1/scale)*0.9,np.log10(1/scale)*1.1,3)
	Nf=[]
	for n in N:
		#print(n)
		h=np.histogram(xi,np.linspace(0,1,int(n)),weights=S)[0]
		Nf.append(np.sum(h[h>0]**q)) # 2nd order
	nu=-np.polyfit(np.log(N),np.log(Nf),1)[0]
	return nu
#plt.plot(x)

#%% GRAPHICAL
#%%% Flip baker map
plt.style.use('~/.config/matplotlib/joris.mplstyle')
x0=np.array([0.4],dtype=np.float64)
var_sa=0
q=2
nb=50
Res=[]
s0=0.05
x0=np.array([0.1])
from scipy import spatial
#2021-09-03_12:00:24
for T in [20]:
	for a in [0.2]:
#		C,m=DNS_flipped(2**13,s0,x0,T,a,sigma,var_sa)
		C,m=DNS_flipped_rand(2**13,s0,T,a,sigma,var_sa)
plt.plot(C.T)
#plt.ylim([C.mean()*0.99,C.mean()*1.01])
#%%% reconstruct on a grid

plt.style.use('~/.config/matplotlib/joris.mplstyle')
T=10
sa=0.21236423
random=0.0

sigma=50
#	sigma=16
M=10 # number of moments to compute
dt=1/1  # Discretisation of time step
X,Y=np.meshgrid(0,np.arange(n)) 

x0=np.array([0.3155])
C=DNS(x0,T,sa,random)
Cf=DNS_fourier(x0,T,sa,random)
x,S,wrapped_time=lagrangian(x0,T,sa,random)

xi=x
s0=0.05
D=(sigma/n)**2/2
#Sigma=np.sqrt(2*D)*n
Cmax=1./np.sqrt(1.+4*D/s0**2*wrapped_time)
Cmax_wave=1./(1.+4*D/s0**2*wrapped_time)
Si=s0*np.sqrt(1.+4*D/s0**2*wrapped_time)*S

sB=2*np.mean(Si)
plt.figure()
plt.plot(C[0,:],label='Eulerian');#plt.yscale('log')
plt.plot(Cf,':',label='Eulerian Fourier');#plt.yscale('log')
#plt.ylim([,2])
h,xh=np.histogram(x,np.arange(0,1,sB),weights=s0*S/sB*np.sqrt(np.pi))
plt.plot(xh[1:]*n,h,'r.',label='Lagrangian Grid')
n=2**13
X,Y=np.meshgrid(0,np.arange(n)) 
C_lag=np.zeros(len(Y))
for y in range(len(Y)):
	C_lag[y]=np.sum(Cmax*np.exp(-(x*n-y)**2./(Si*n)** 2.)+
								 Cmax*np.exp(-((x-1)*n-y)**2./(Si*n)**2.)+
								 Cmax*np.exp(-((x+1)*n-y)**2./(Si*n)**2.))
plt.plot(C_lag,'k--',label='Lagrangian')
plt.legend()
plt.savefig(dir_out+'c.pdf',bbox_inches='tight')
#%%% reconstruct on a grid log-scale

plt.style.use('~/.config/matplotlib/joris.mplstyle')
for T in range(40):
	sa=0.21236423
	random=0.0
	
	sigma=50
	#	sigma=16
	M=10 # number of moments to compute
	dt=1/1  # Discretisation of time step
	X,Y=np.meshgrid(0,np.arange(n)) 
	sa=0.2
	random=0
	x0=np.array([0.01])
	C=DNS_0mean(n,x0,T,sa,random)
	plt.figure()
	plt.plot(np.linspace(0,1,n),(C[0,:]),label='Eulerian');#plt.yscale('log')
	plt.plot(np.linspace(0,1,n),-(C[0,:]),label='Eulerian');#plt.yscale('log')
	plt.yscale('log')
	plt.ylim([1e-12,1e-1])
	plt.ylabel(r'$\log |c-\langle c \rangle |$')
	plt.xlabel(r'$x$')
	plt.savefig(dir_out+'logc_T{:02d}.jpg'.format(T),bbox_inches='tight')

#%%% Graphical Lagrangian and Eulerian

plt.style.use('~/.config/matplotlib/joris.mplstyle')
T=10
sa=0.2
random=0.0

fig,ax=plt.subplots(1,3,figsize=(6,2),sharey=True)

for i,sigma in enumerate([50,100,300]):
	#	sigma=16
	M=10 # number of moments to compute
	dt=1/1  # Discretisation of time step
	X,Y=np.meshgrid(0,np.arange(n)) 
	
	x0=np.array([0.3155])
	C=DNS_sigma(x0,T,sa,random,sigma)
#	Cf=DNS_fourier(x0,T,sa,random)
	x,S,wrapped_time=lagrangian(x0,T,sa,random)
	
	xi=x
	s0=0.05
	D=(sigma/n)**2/2
	#Sigma=np.sqrt(2*D)*n
	Cmax=1./np.sqrt(1.+4*D/s0**2*wrapped_time)
	Cmax_wave=1./(1.+4*D/s0**2*wrapped_time)
	Si=s0*np.sqrt(1.+4*D/s0**2*wrapped_time)*S
	
	ax[i].bar(xi,Cmax,width=0.01,color='k',alpha=.4,label=r'$\theta$')
	#plt.bar(xi,S/(2*np.sqrt(D/s0**2.)),width=0.005,color='r',alpha=0.5)
	ax[i].plot(np.linspace(0,1,len(C.T)),C.T,'r-',label=r'$c$')
	ax[i].plot([0,1],[C.mean(),C.mean()],'k--',label=r'$\langle c \rangle$')
#	ax[i].text(0.4,0.5,r'$t={:d},D={:1.1e}$'.format(T,D))
	#lt.yscale('log')
	ax[i].set_ylim(0,2*C.mean())
	ax[i].set_xlim(0,1)
	ax[i].set_xlabel('$x$')
ax[2].legend()
#plt.ylabel('$c$')
fig.savefig('/home/joris/Dropbox/Articles/AggregationBatchelor/Figures/baker_c_x.pdf',bbox_inches='tight')


#%%% Graphical Fractal patterns

plt.style.use('~/.config/matplotlib/joris.mplstyle')
T=12
sa=0.1
random=0.0
x0=np.array([0.123])
x,S,wrapped_time=lagrangian(x0,T,sa,random)
fig,ax=plt.subplots(1,3,figsize=(6.5,2),sharey=True)
w=[0.001,1e-4,1e-5]
for i,sigma in enumerate([50,100,300]):
	ax[i].bar(x,np.ones(x.shape),width=w[i],color='k',alpha=1)
	ax[i].set_ylim(0,1)
	ax[i].set_xlabel('$x$')
ax[0].set_xlim(0,1)

x1=[0.9,1.0]
y1=[0.45,0.55]

ax[1].set_xlim(0.9,1)
ax[0].plot([x1[0],x1[1],x1[1],x1[0],x1[0]],[y1[0],y1[0],y1[1],y1[1],y1[0]],'r-',linewidth=1.5)
ax[2].set_xlim(0.99,1)
x1=[0.99,1.0]
y1=[0.45,0.55]
ax[1].plot([x1[0],x1[1],x1[1],x1[0],x1[0]],[y1[0],y1[0],y1[1],y1[1],y1[0]],'r-',linewidth=1.5)
ax[0].set_xticks([0,1])
ax[0].set_yticks([])
ax[1].set_xticks([0.9,1])
ax[1].set_yticks([])
ax[2].set_yticks([])
ax[2].set_xticks([0.99,1])
fig.subplots_adjust(wspace=0.2)
fig.savefig('/home/joris/Dropbox/Articles/AggregationBatchelor/Figures/baker_c_fractals_a{:1.1f}.jpg'.format(sa),bbox_inches='tight',dpi=600)
#%%% Graphical coarsening scale

plt.style.use('~/.config/matplotlib/joris.mplstyle')
T=10
sa=0.3
random=0.0
x0=np.array([0.123])
x,S,wrapped_time=lagrangian(x0,T,sa,random)
sB=1/50
s0=0.05

lyap,sigma2=lyapunov(sa)
D=(1/50)**2/2*0.1
C=DNS_sigma(x0,T,sa,random,np.sqrt(2*D)*n)

N,xb=np.histogram(x,bins=np.linspace(0,1,int(1/sB+1)))
logrho,xb=np.histogram(x,bins=np.linspace(0,1,int(1/sB+1)),weights=np.log(1/S))
c,xb=np.histogram(x,bins=np.linspace(0,1,int(1/sB+1)),weights=S)

fig,ax=plt.subplots(1,3,figsize=(5,2))
i0=ax[0].imshow(np.log(N).reshape(1,-1),extent=[0,1,0,1],clim=[1,5])
cc0=fig.colorbar(i0,ax=ax[0],location='bottom',aspect=10,shrink=0.8,label=r'$\log n$',pad=0.05)
i1=ax[1].imshow(logrho/N.reshape(1,-1),extent=[0,1,0,1],clim=[4,10])
cc1=fig.colorbar(i1,ax=ax[1],location='bottom',aspect=10,shrink=.8,label=r'$\langle \log \rho | n\rangle $',pad=0.05)
i2=ax[2].imshow(np.log(c*s0/sB*np.sqrt(np.pi)).reshape(1,-1),extent=[0,1,0,1],clim=[-2.7,-2.1])
cc2=fig.colorbar(i2,ax=ax[2],location='bottom',aspect=10,shrink=0.8,label=r'$\log c$',pad=0.05)
#i2=ax[3].imshow(np.log(C).reshape(1,-1),extent=[0,1,0,1],clim=[-2.7,-2.1])
#cc2=fig.colorbar(i2,ax=ax[3],location='bottom',aspect=10,shrink=0.8,label=r'$\log c$ (DNS)',pad=0.05)

ax[0].axis('off')
ax[1].axis('off')
ax[2].axis('off')
#ax[3].axis('off')

script=['a.1','a.2','a.3','a.4']
subscript(ax[0],0,color='k',script=script)#,x=0.5,y=1.02)
subscript(ax[1],1,color='k',script=script)#,x=0.5,y=1.02)
subscript(ax[2],2,color='k',script=script)#,x=0.5,y=1.02)
#subscript(ax[3],3,color='k',script=script,x=0.5,y=1.02)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, hspace=0.02, wspace=0.08)

fig.savefig('/home/joris/Dropbox/Articles/AggregationBatchelor/Figures/baker_N_LAG.pdf',bbox_inches='tight')

#%%% Wave model

plt.style.use('~/.config/matplotlib/joris.mplstyle')
T=10
sa=0.21236423
#sa=0.05
random=0.0
n=2**13

sigma=50
#	sigma=16
M=10 # number of moments to compute
dt=1/1  # Discretisation of time step
X,Y=np.meshgrid(0,np.arange(n)) 

x0=np.array([0.3155])
C,m=DNS_0mean(n,s0,x0,T,sa,sigma,random)
#C,m=DNS_0mean_flipped(n,s0,x0,T,sa,sigma,random)
x,S,wrapped_time=lagrangian(x0,T,sa,random)

xi=x
s0=0.05
D=(sigma/n)**2/2
#Sigma=np.sqrt(2*D)*n
Cmax=1./np.sqrt(1.+4*D/s0**2*wrapped_time)
Cmax_wave=1./(1.+4*D/s0**2*wrapped_time) / np.sqrt(2) * np.exp(-0.5)
Si=s0*np.sqrt(1.+4*D/s0**2*wrapped_time)*S

plt.figure(figsize=(3,2))
plt.bar(xi+Si/np.sqrt(2),Cmax_wave,width=0.005,color='k',alpha=0.5,label=r'$c_\mathrm{max} e^{-1/2}/\sqrt{2}$ (wave)')
plt.bar(xi-Si/np.sqrt(2),-Cmax_wave,width=0.005,color='k',alpha=0.5)
#plt.bar(xi,S/(2*np.sqrt(D/s0**2.)),width=0.005,color='r',alpha=0.5)
plt.plot(np.linspace(0,1,len(C.T)),C.T,'r-',label=r'$c$')
plt.plot([0,1],[C.mean(),C.mean()],'k--',label=r'$\langle c \rangle$')
#plt.text(0.4,0.5,r'$t={:d},D={:1.1e}$'.format(T,D))
#lt.yscale('log')
plt.ylim(-0.01,0.01)
plt.xlabel('$x$')
plt.legend()
#plt.ylabel('$c$')
plt.savefig('/home/joris/Dropbox/Articles/AggregationBatchelor/Figures/baker_c_x_wave.pdf',bbox_inches='tight')


plt.figure(figsize=(3,2))
plt.bar(xi+Si/np.sqrt(2),Cmax_wave,width=0.001,color='k',alpha=0.5,label=r'$c_\mathrm{max} e^{-1/2}/\sqrt{2}$ (wave)')
#plt.bar(xi-Si/np.sqrt(2),-Cmax_wave,width=0.005,color='k',alpha=0.5)
#plt.bar(xi,S/(2*np.sqrt(D/s0**2.)),width=0.005,color='r',alpha=0.5)
plt.plot(np.linspace(0,1,len(C.T)),C.T-C.mean(),'r-',label=r'$c$')
#plt.text(0.4,0.5,r'$t={:d},D={:1.1e}$'.format(T,D))
plt.yscale('log')
plt.ylim(1e-7,1e-2)
plt.xlabel('$x$')
plt.ylabel(r'$c,c_\mathrm{max}$')
#plt.legend()
plt.xlim([0.3,0.41])
plt.savefig('/home/joris/Dropbox/Articles/AggregationBatchelor/Figures/baker_c_x_wave_log.pdf',bbox_inches='tight')

#%%% Plot 3 types of waves

x=np.linspace(0,1,1000)
s0=0.1
x0=0.5
cmax=1.
cm=cmax/3

fig,ax=plt.subplots(1,3,figsize=(4,1),sharey=True)
ax[0].plot(x,cmax*np.exp(-(x-x0)**2./s0**2.),'k-')
ax[0].plot(x,np.zeros(x.shape)+cm,'k--',label=r'$\langle c \rangle$')
ax[0].text(0.0,0.88,'$a.$')
ax[1].plot(x,cm+cmax/2*(x-x0)/s0*np.exp(-(x-x0)**2./s0**2.),'k-')
ax[1].plot(x,np.zeros(x.shape)+cm,'k--')
ax[1].text(0.0,0.88,'$b.$')

ax[2].plot(x,cm+cmax/8*np.sin(x*np.pi/s0),'k-')
ax[2].plot(x,np.zeros(x.shape)+cm,'k--')
ax[2].text(0.0,0.88,'$c.$')

ax[0].set_xticks([])
ax[0].set_yticks([])
ax[1].set_xticks([])
ax[1].set_yticks([])
ax[2].set_xticks([])
ax[2].set_yticks([])
plt.savefig('/home/joris/Dropbox/Articles/AggregationBatchelor/Figures/3waves.pdf',bbox_inches='tight')

#%% Compare DNS DSM
#%%% Compare DNS DSM
plt.style.use('~/.config/matplotlib/joris.mplstyle')
T=16
sa=0.3
#sa=0.05
random=0.0
n=2**13
x0=0.4
sigma=50
#	sigma=16
M=10 # number of moments to compute
dt=1/1  # Discretisation of time step
X,Y=np.meshgrid(0,np.arange(n)) 

s0=0.1

x0=np.array([0.3155],dtype=np.float128)
C,m=DNS_0mean(n,s0,x0,T,sa,sigma,random)
C=C.T-C.mean()
x,S,wrapped_time=lagrangian(x0,T,sa,random)

D=(sigma/n)**2/2
#Sigma=np.sqrt(2*D)*n
Cmax=1./np.sqrt(1.+4*D/s0**2*wrapped_time)
Cmax_wave=1./(1.+4*D/s0**2*wrapped_time) / np.sqrt(2) * np.exp(-0.5)
Si=s0*np.sqrt(1.+4*D/s0**2*wrapped_time)*S

#from scipy import ndimage
#Cmap=ndimage.map_coordinates(C[:,0], np.mod((x-Si/np.sqrt(2)),1)*(C.shape[0]-1), order=1)
# Problem here, interpolate value on location
#Ccmax=C[np.uint16(np.round(np.mod((x-Si/np.sqrt(2)),1)*(C.shape[0]-1)))]
Ccmax=C[np.uint16(np.round(np.mod((x+Si/np.sqrt(2)),1)*(C.shape[0]-1)))]

plt.figure(figsize=(3,2))
plt.plot(Cmax_wave,Ccmax,'.',alpha=0.01,label='$c_\mathrm{max}=1/\tau$')
plt.yscale('log')
plt.xscale('log')
xc=np.logspace(-15,0,50)

Cm=bin_operation(Cmax_wave,np.abs(Ccmax),xc,np.nanmean)

plt.plot(xc[1:],Cm,'o')
plt.plot(xc,xc,'k-')
plt.ylabel(r'$|c-\langle c \rangle |$')
plt.xlabel(r'$c_\mathrm{max}=1/\tau$')
plt.ylim([1e-7,1])
plt.xlim([1e-14,1])

# Check max in a grid
sB=0.01
MaxC,MeanC,MaxDNS=[],[],[]
x_DNS=np.linspace(0,1,C.shape[0])
for i in np.arange(sB,1,sB):
	if len(Cmax_wave[(x>i-sB)&(x<=i)])>0:
		MaxC.append(Cmax_wave[(x>i-sB)&(x<=i)].max())
		MeanC.append(Cmax_wave[(x>i-sB)&(x<=i)].mean())
		MaxDNS.append(np.abs(C[(x_DNS>i-sB)&(x_DNS<=i)]).max())

plt.figure(figsize=(3,2))
#plt.figure(figsize=(3,2))
plt.plot(MaxC,MaxDNS,'d',alpha=1,label=r'$\max_{s_B} ( \cdot)$')
plt.plot(MeanC,MaxDNS,'s',alpha=1,label=r'$\langle \cdot \rangle_{s_B}$')
plt.plot(xc,xc,'k-',label=r'$1/\tau$')
plt.plot(xc,xc**0.5,'k--',label=r'$1/\sqrt{\tau}$')
#plt.ylim([1e-12,1])
#plt.xlim([1e-12,1])
plt.yscale('log')
plt.xscale('log')
plt.ylabel(r'$|c-\langle c \rangle |$')
plt.xlabel(r'$1/\tau$')
plt.legend()
plt.savefig('/home/joris/Dropbox/Articles/AggregationBatchelor/Figures/baker_DSM_DNS_a{:1.2f}_T{:1.0f}.pdf'.format(sa,T),bbox_inches='tight')

#%%%  * Compare variances
plt.style.use('~/.config/matplotlib/joris.mplstyle')
T=30

fig,ax=plt.subplots(1,3,figsize=(8,3),sharey=True)

for i,sa in enumerate([0.2,0.3,0.4]):
#	sa=0.4
	#sa=0.05
	random=0.0
	n=2**13
	x0=0.4
	sigma=100
	#	sigma=16
	M=10 # number of moments to compute
	dt=1/1  # Discretisation of time step
	X,Y=np.meshgrid(0,np.arange(n)) 
	
	s0=0.05
	c0=1.
	
	x0=np.array([0.4155],dtype=np.float128)
	
	l0=len(x0)
	
	VarC,varC_lag,varC_lag_2,varC_agg_rand,varC_agg_cor,Time_lag,Time_eul=[],[],[],[],[],[],[]
	MeanRho=[]
	
	for t in range(T):
		C,m=DNS_ns0(n,s0,x0,t,sa,random)
		VarC.append(np.var(C))
		Time_eul.append(t)
		if t<25:
			x,S,wrapped_time=lagrangian(x0,t,sa,random)
			D=(sigma/n)**2/2
			#Sigma=np.sqrt(2*D)*n
			Cmax_gauss=c0/np.sqrt(1.+4*D/s0**2*wrapped_time)
			Cmax_wave=c0/(1.+4*D/s0**2*wrapped_time)
			Si=s0*np.sqrt(1.+4*D/s0**2*wrapped_time)*S
			Tau=D/s0**2*wrapped_time
			Cmax=Cmax_gauss #Gaussian
	#		Cmax=Cmax_wave #Gaussian
			Si=S*np.sqrt(1+4*Tau)*s0
			varC_lag.append(l0*s0*c0*np.average(Cmax,weights=S)*np.sqrt(np.pi/2))
			sB=np.mean(Si)
			N=len(x)*sB/1
			varC_agg_rand.append(np.var(Cmax)*N)
			mC=np.average(Cmax,weights=S)
			MeanRho.append(np.average(1/S,weights=S))
	#		print(mC)
			vC=np.average(Cmax**2.,weights=S)
			varC_agg_cor.append(vC-mC**2.)
			Time_lag.append(t)
	
	
	lyap=0.65
	sigma2=0.5
	
	sBp=np.sqrt(D/(lyap+sigma2))
	Time_lag=np.array(Time_lag)
	VarC=np.array(VarC)
	MeanRho=np.array(MeanRho)
	VarC_lag=np.array(varC_lag)
	VarC_lag_2=np.array(varC_lag_2)
	VarC_agg_rand=np.array(varC_agg_rand)
	VarC_agg_cor=np.array(varC_agg_cor)
	tt=np.linspace(15,30,100)
	lm=lyap+sigma2/2
	cm=np.mean(C)
	
	A=sa
	lyap=-A*np.log(A)-(1-A)*np.log(1-A)
	sigma2=(A*np.log(A)**2.+(1-A)*np.log(1-A)**2.-lyap**2.)
	
	tagg=1/lm*np.log(1/(sB*l0))
	Tagg=np.where(Time_lag>tagg)[0][0]
	
	factor_corr_agg=VarC_lag[Tagg]/VarC_agg_cor[Tagg]
	
	# plt.plot(tt,np.exp(-lyap**2./(2.*sigma2)*tt),'k--',label=r"$-\mu^2/(2\sigma^2)$")
	# plt.plot(tt,np.exp(-(lyap-sigma2/2)*tt),'k-',label=r"$-(\mu-\sigma^2/2)$")
	# plt.plot(tt,np.exp(-2*(lyap-sigma2)*tt),'k:',label=r"$-2(\mu-\sigma^2)$")
	print(i)
	ax[i].plot(Time_eul,VarC,'ko',label='DNS',fillstyle='full')
	ax[i].plot(Time_lag,VarC_lag,'r*',label='Isolated strip')
	#plt.plot(Time,VarC_lag_2,'g*',label='Isolated strip model')
	ax[i].plot(Time_lag,VarC_agg_rand,'rd',label='Random aggregation')
	ax[i].plot(Time_lag,VarC_agg_cor*factor_corr_agg,'b^',label='Correlated aggregation')
	ax[i].set_yscale('log')
	ax[i].set_ylim([1e-15,1e0])
	ax[i].set_xlabel('$t$')
	ax[i].plot([1/lm*np.log(1/(sB*l0)),1/lm*np.log(1/(sB*l0))],[1e-20,1e20],'k-')
	ax[i].plot([1/lm*np.log(s0/sB),1/lm*np.log(s0/sB)],[1e-20,1e20],'k--')
	ax[i].plot(tt,np.exp(np.log(1-3*sa+3*sa**2)*tt),'b-',label=r"$\log(1-3a+3a^2)$")
	ax[i].plot(tt,np.exp(np.log(1-2*sa+2*sa**2)*tt),'r-',label=r"$\log(1-2a+2a^2)$")
	#plt.title(r'Baker map, $a={:1.2f}$'.format(sa))
ax[0].legend(fontsize=8,ncol=1,fancybox='on')
ax[0].set_ylabel(r'$\sigma^2_c$')

plt.savefig('/home/joris/Dropbox/Articles/AggregationBatchelor/Figures/Baker_variances_1_sB{:1.0f}.pdf'.format(1/sB),bbox_inches='tight')

plt.figure()
plt.plot(Time_lag,MeanRho,'*')
plt.yscale('log')
plt.plot(Time_lag,np.exp(Time_lag*np.log(2)),'r--',label='$\log 2$')
plt.xlabel(r'$t$')
plt.ylabel(r'$\langle \rho \rangle$')
plt.legend()

#%%%  * Compare variances over threshold
plt.style.use('~/.config/matplotlib/joris.mplstyle')
T=35

fig,ax=plt.subplots(1,3,figsize=(8,3),sharey=True)

for i,sa in enumerate([0.2,0.3,0.4]):
#	sa=0.4
	#sa=0.05
	random=0.0
	n=2**13
	x0=0.4
	sigma=100
	#	sigma=16
	M=10 # number of moments to compute
	dt=1/1  # Discretisation of time step
	X,Y=np.meshgrid(0,np.arange(n)) 
	
	s0=0.05
	c0=1.
	

	epsilon = 0.0001
	x0=np.array([0.4155],dtype=np.float128)
	
	l0=len(x0)
	
	VarC,varC_lag,varC_lag_2,varC_agg_rand,varC_agg_cor,Time_lag,Time_eul=[],[],[],[],[],[],[]
	MeanRho=[]
	
	for t in range(T):
		C,m=DNS_ns0(n,s0,x0,t,sa,random)
		VarC.append(np.var(C[C>epsilon]))
		Time_eul.append(t)
		if t<25:
			x,S,wrapped_time=lagrangian(x0,t,sa,random)
			D=(sigma/n)**2/2
			#Sigma=np.sqrt(2*D)*n
			Cmax_gauss=c0/np.sqrt(1.+4*D/s0**2*wrapped_time)
			Cmax_wave=c0/(1.+4*D/s0**2*wrapped_time)
			Si=s0*np.sqrt(1.+4*D/s0**2*wrapped_time)*S
			Tau=D/s0**2*wrapped_time
			Cmax=Cmax_gauss #Gaussian
	#		Cmax=Cmax_wave #Gaussian
			Si=S*np.sqrt(1+4*Tau)*s0
			#varC_lag.append(l0*s0*c0*np.average(Cmax,weights=S)*np.sqrt(np.pi/2))
			varC_lag.append(np.mean(Cmax**2.)*0.12) #0.12 is a theoretical approx 
			sB=np.mean(Si)
			N=len(x)*sB/1
			varC_agg_rand.append(np.var(Cmax)*N)
			mC=np.average(Cmax,weights=S)
			MeanRho.append(np.average(1/S,weights=S))
	#		print(mC)
			vC=np.average(Cmax**2.,weights=S)
			varC_agg_cor.append(vC-mC**2.)
			Time_lag.append(t)
	
	
	lyap=0.65
	sigma2=0.5
	
	sBp=np.sqrt(D/(lyap+sigma2))
	Time_lag=np.array(Time_lag)
	VarC=np.array(VarC)
	MeanRho=np.array(MeanRho)
	VarC_lag=np.array(varC_lag)
	VarC_lag_2=np.array(varC_lag_2)
	VarC_agg_rand=np.array(varC_agg_rand)
	VarC_agg_cor=np.array(varC_agg_cor)
	VarC_agg_rand[0]=np.nan
	VarC_agg_cor[0]=np.nan
	
	tt=np.linspace(15,30,100)
	lm=lyap+sigma2/2
	cm=np.mean(C)
	
	A=sa
	lyap=-A*np.log(A)-(1-A)*np.log(1-A)
	sigma2=(A*np.log(A)**2.+(1-A)*np.log(1-A)**2.-lyap**2.)
	
	tagg=1/lm*np.log(1/(sB*l0))
	Tagg=np.where(Time_lag>tagg)[0][0]
	
	factor_corr_agg=VarC_lag[Tagg]/VarC_agg_cor[Tagg]
	
	# plt.plot(tt,np.exp(-lyap**2./(2.*sigma2)*tt),'k--',label=r"$-\mu^2/(2\sigma^2)$")
	# plt.plot(tt,np.exp(-(lyap-sigma2/2)*tt),'k-',label=r"$-(\mu-\sigma^2/2)$")
	# plt.plot(tt,np.exp(-2*(lyap-sigma2)*tt),'k:',label=r"$-2(\mu-\sigma^2)$")
	print(i)
	ax[i].plot(Time_eul,VarC,'ro',label='DNS (Pe$={:1.0e}$)'.format((1/sB)**2.),fillstyle='full')
	ax[i].plot(Time_lag,VarC_lag,'k--',label='Isolated strip',linewidth=1.1)
	#plt.plot(Time,VarC_lag_2,'g*',label='Isolated strip model')
	ax[i].plot(Time_lag,VarC_agg_rand,'k:',label='Random aggregation',linewidth=1.1)
	ax[i].plot(Time_lag,VarC_agg_cor*factor_corr_agg,'k-',label='Correlated aggregation',linewidth=1.1)
	ax[i].set_yscale('log')
	ax[i].set_ylim([1e-15,1e0])
	ax[i].set_xlabel('$t$')
# 	ax[i].plot([1/lm*np.log(1/(sB*l0)),1/lm*np.log(1/(sB*l0))],[1e-20,1e20],'k-')
# 	ax[i].plot([1/lm*np.log(s0/sB),1/lm*np.log(s0/sB)],[1e-20,1e20],'k--')
# 	ax[i].plot(tt,np.exp(np.log(1-3*sa+3*sa**2)*tt),'b-',label=r"$\log(1-3a+3a^2)$")
# 	ax[i].plot(tt,np.exp(np.log(1-2*sa+2*sa**2)*tt),'r-',label=r"$\log(1-2a+2a^2)$")
	#plt.title(r'Baker map, $a={:1.2f}$'.format(sa))
ax[0].legend(fontsize=8,ncol=1,fancybox='on')
ax[0].set_ylabel(r'$\sigma^2_{c>\varepsilon}$')

plt.savefig('/home/joris/Dropbox/Articles/AggregationBatchelor/Figures/Baker_variances_1_sB{:1.0f}.pdf'.format(1/sB),bbox_inches='tight')

plt.figure()
plt.plot(Time_lag,MeanRho,'*')
plt.yscale('log')
plt.plot(Time_lag,np.exp(Time_lag*np.log(2)),'r--',label='$\log 2$')
plt.xlabel(r'$t$')
plt.ylabel(r'$\langle \rho \rangle$')
plt.legend()

#%%%  * Compare variances as a function of Péclet
plt.style.use('~/.config/matplotlib/joris.mplstyle')
T=40
sa=0.2
#sa=0.05
epsilon=0.00001
random=0.0
n=2**13
x0=0.4
plt.figure()
for sigma in [10,100,300,600]:
		#	sigma=16
	M=10 # number of moments to compute
	dt=1/1  # Discretisation of time step
	X,Y=np.meshgrid(0,np.arange(n)) 
	
	s0=0.05
	c0=1.
	
	x0=np.array([0.4155],dtype=np.float128)
	
	l0=len(x0)
	
	VarC,varC_lag,varC_lag_2,varC_agg_rand,varC_agg_cor,Time_lag,Time_eul=[],[],[],[],[],[],[]
	MeanRho=[]
	
	for t in range(T):
		C,m=DNS_ns0(n,s0,x0,t,sa,random)
		VarC.append(np.var(C[C>epsilon]))
		Time_eul.append(t)
		Time_lag.append(t)
		if t<25:
			x,S,wrapped_time=lagrangian(x0,t,sa,random)
			D=(sigma/n)**2/2
			#Sigma=np.sqrt(2*D)*n
			Cmax_gauss=c0/np.sqrt(1.+4*D/s0**2*wrapped_time)
			Cmax_wave=c0/(1.+4*D/s0**2*wrapped_time)
			Si=s0*np.sqrt(1.+4*D/s0**2*wrapped_time)*S
			Tau=D/s0**2*wrapped_time
			Cmax=Cmax_gauss #Gaussian
	#		Cmax=Cmax_wave #Gaussian
			Si=S*np.sqrt(1+4*Tau)*s0
			varC_lag.append(l0*s0*c0*np.average(Cmax,weights=S)*np.sqrt(np.pi/2))
			sB=np.mean(Si)
			N=len(x)*sB/1
			varC_agg_rand.append(np.var(Cmax)*N)
			mC=np.average(Cmax,weights=S)
			MeanRho.append(np.average(1/S,weights=S))
	#		print(mC)
			vC=np.average(Cmax**2.,weights=S)
			varC_agg_cor.append(vC-mC**2.)
		else:
			varC_agg_cor.append(np.exp(np.log(1-3*sa+3*sa**2)*(t-24))*varC_agg_cor[24])
	
	lyap=0.65
	sigma2=0.5
	
	sBp=np.sqrt(D/(lyap+sigma2))
	VarC=np.array(VarC)
	MeanRho=np.array(MeanRho)
	VarC_lag=np.array(varC_lag)
	VarC_lag_2=np.array(varC_lag_2)
	VarC_agg_rand=np.array(varC_agg_rand)
	VarC_agg_cor=np.array(varC_agg_cor)
	tt=np.linspace(5,30,100)
	lm=lyap+sigma2/2
	cm=np.mean(C)
	
	A=sa
	lyap=-A*np.log(A)-(1-A)*np.log(1-A)
	sigma2=(A*np.log(A)**2.+(1-A)*np.log(1-A)**2.-lyap**2.)
	VarC_agg_cor[0]=np.nan
	# plt.plot(tt,np.exp(-lyap**2./(2.*sigma2)*tt),'k--',label=r"$-\mu^2/(2\sigma^2)$")
	# plt.plot(tt,np.exp(-(lyap-sigma2/2)*tt),'k-',label=r"$-(\mu-\sigma^2/2)$")
	# plt.plot(tt,np.exp(-2*(lyap-sigma2)*tt),'k:',label=r"$-2(\mu-\sigma^2)$")
	plt.plot(Time_eul,VarC,'o',fillstyle='full',color=plt.cm.cool((np.log(1/sB)-1)/5),label=r'DNS, Pe$={:1.0e}$'.format((1/sB)**2.))
# 	plt.plot(Time_lag,VarC_lag,'r*',label='Isolated strip')
# 	#plt.plot(Time,VarC_lag_2,'g*',label='Isolated strip model')
# 	plt.plot(Time_lag,VarC_agg_rand,'rd',label='Random aggregation')
	plt.plot(Time_lag,0.05*VarC_agg_cor*(1-cm/c0)**2.,'-',color=plt.cm.cool((np.log(1/sB)-1)/5))
	plt.yscale('log')
	plt.ylim([1e-15,1e0])
	plt.xlabel('$t$')
	plt.ylabel(r'$\sigma^2_{c>\varepsilon}$')
# 	plt.plot([1/lm*np.log(1/(sB*l0)),1/lm*np.log(1/(sB*l0))],[1e-20,1e20],'-',color=plt.cm.cool(1/sB/100))
# 	plt.plot([1/lm*np.log(s0/sB),1/lm*np.log(s0/sB)],[1e-20,1e20],'--',color=plt.cm.cool(1/sB/100))
# 	plt.plot(tt,np.exp(np.log(1-3*sa+3*sa**2)*tt),'b-',label=r"$\log(1-3a+3a^2)$")
# 	plt.plot(tt,np.exp(np.log(1-2*sa+2*sa**2)*tt),'r-',label=r"$\log(1-2a+2a^2)$")
	#plt.title(r'Baker map, $a={:1.2f}$'.format(sa))
	plt.legend(fontsize=8,ncol=1,fancybox='on')
	plt.savefig('/home/joris/Dropbox/Articles/AggregationBatchelor/Figures/Baker_Peclet_a{:1.2f}_1_sB{:1.0f}.pdf'.format(A,1/sB),bbox_inches='tight')

#%%% 
#%% Distance between lamellae
#%%% Geometry of s

random=0
shuffle=1
a=0.3
x0=np.array([0.999])
Sn,N=[],[]
Tv=[6]
Tv=np.arange(10)
plt.figure()
for T in Tv:
	#	sigma=16
	M=10 # n%umber of moments to compute
	dt=1/1  # Discretisation of time step
	X,Y=np.meshgrid(0,np.arange(n)) 
	
	x,S,wrapped_time=lagrangian_nfold(x0,T,np.array([a,1-a]),random)
	
	xi=x
	s0=0.05
	D=(sigma/n)**2/2
	#Sigma=np.sqrt(2*D)*n
	Cmax=1./np.sqrt(1.+4*D/s0**2*wrapped_time)
	Si=s0*np.sqrt(1.+4*D/s0**2*wrapped_time)*S
	plt.plot(x,S,'o',color=plt.cm.jet(T/Tv.max()))
	Su=np.unique(np.float16(S))
#	np.sort(Su)
	Sn.append(list(Su))
	N.append([len(np.where(np.float16(S)==s)[0]) for s in Su])
plt.yscale('log')

# Pascal Triangle
print(N)
print(Sn)
Sk=[]
for T in np.arange(10):
	Sk.append([np.float16((1-a)**k*a**(T-k)) for k in np.arange(T+1)])

print(Sk)

# Distances
plt.figure()
xs1=x[np.where(np.float16(S)==Su[-2])[0]]
plt.plot(np.sort(np.diff(xs1))[::-1],'*')
plt.plot(Tv-1,a*(1-a)**Tv)
plt.xlabel('$j$')
plt.ylabel('$x_{1,j}-x_{1,j-1}$')

plt.figure()
xs2=x[np.where(np.float16(S)==Su[-3])[0]]
plt.plot(np.diff(xs2),'*')
dk=[]
for k  in range(10):
	dk+=list(a**2.*(1-a)**(Tv[:-k-2])*(1-a)**k)
plt.plot(dk[1:],'-o')
plt.xlabel('$j$')
plt.ylabel('$x_{2,j}-x_{2,j-1}$')

def hist_uniq(x):
	x=np.float16(x)
	xu=np.sort(np.unique(x))
	n=[]
	for i,u in enumerate(xu):
		n.append(np.sum(x==u))
	return xu,np.array(n)
	
plt.figure()
hk,dkk=hist_uniq(dk)
plt.plot(hk,dkk,'o')
hk,dkk=hist_uniq(np.diff(xs2))
plt.plot(hk,dkk,'*')

xs2=x[np.where(np.float16(S)==Su[-3])[0]]
hk,dkk=hist_uniq(np.diff(xs2)/a**2)
plt.plot(hk,dkk,'s')

xs3=x[np.where(np.float16(S)==Su[-4])[0]]
hk,dkk=hist_uniq(np.diff(xs3)/a**3)
plt.plot(hk,dkk,'o')

xs4=x[np.where(np.float16(S)==Su[-5])[0]]
hk,dkk=hist_uniq(np.diff(xs4)/a**4)
plt.plot(hk,dkk,'*')
plt.xlim([0,1])
#%%% Distance between lamellae

random=0
shuffle=0
a=0.1
x0=np.array([1.0])
Sn,N=[],[]
Tv=[10]
#Tv=np.arange(10)
plt.figure()
for T in Tv:
	#	sigma=16
	M=10 # n%umber of moments to compute
	dt=1/1  # Discretisation of time step
	X,Y=np.meshgrid(0,np.arange(n)) 
	
	#x,S,wrapped_time=lagrangian_nfold(x0,T,np.array([a,1-a]),random)
	x,S,wrapped_time=lagrangian_sym(x0,T,a,random)
	
	xi=x
	s0=0.05
	D=(sigma/n)**2/2
	#Sigma=np.sqrt(2*D)*n
	Cmax=1./np.sqrt(1.+4*D/s0**2*wrapped_time)
	Si=s0*np.sqrt(1.+4*D/s0**2*wrapped_time)*S
	Su=np.unique(np.float16(S))
#	np.sort(Su)
	Sn.append(list(Su))
	N.append([len(np.where(np.float16(S)==s)[0]) for s in Su])

x1=np.concatenate((np.array([x[-1]])-1,np.array(x)))
x0=np.concatenate((np.array(x),np.array([x[0]])+1))
d1=np.diff(x1)
d0=np.diff(x0)
S=np.array(S)
plt.plot(S,d1,'d',label='left',alpha=0.01)
plt.plot(S,d0,'*',label='right',alpha=0.01)
plt.plot([S.min(),S.max()],[S.min(),S.max()],'k--')
plt.yscale('log')
plt.xscale('log')
plt.xlabel(r'$1/\rho$')
plt.ylabel('Distance')
plt.title(r'$a={:1.2f}$'.format(a))
plt.legend()

#%% Distance between lamellae
#%%% Geometry of s

random=0
shuffle=1
a=0.3
x0=np.array([0.999])
Sn,N=[],[]
Tv=[6]
Tv=np.arange(10)
plt.figure()
for T in Tv:
	#	sigma=16
	M=10 # n%umber of moments to compute
	dt=1/1  # Discretisation of time step
	X,Y=np.meshgrid(0,np.arange(n)) 
	
	x,S,wrapped_time=lagrangian_nfold(x0,T,np.array([a,1-a]),random)
	
	xi=x
	s0=0.05
	D=(sigma/n)**2/2
	#Sigma=np.sqrt(2*D)*n
	Cmax=1./np.sqrt(1.+4*D/s0**2*wrapped_time)
	Si=s0*np.sqrt(1.+4*D/s0**2*wrapped_time)*S
	plt.plot(x,S,'o',color=plt.cm.jet(T/Tv.max()))
	Su=np.unique(np.float16(S))
#	np.sort(Su)
	Sn.append(list(Su))
	N.append([len(np.where(np.float16(S)==s)[0]) for s in Su])
plt.yscale('log')

# Pascal Triangle
print(N)
print(Sn)
Sk=[]
for T in np.arange(10):
	Sk.append([np.float16((1-a)**k*a**(T-k)) for k in np.arange(T+1)])

print(Sk)

# Distances
plt.figure()
xs1=x[np.where(np.float16(S)==Su[-2])[0]]
plt.plot(np.sort(np.diff(xs1))[::-1],'*')
plt.plot(Tv-1,a*(1-a)**Tv)
plt.xlabel('$j$')
plt.ylabel('$x_{1,j}-x_{1,j-1}$')

plt.figure()
xs2=x[np.where(np.float16(S)==Su[-3])[0]]
plt.plot(np.diff(xs2),'*')
dk=[]
for k  in range(10):
	dk+=list(a**2.*(1-a)**(Tv[:-k-2])*(1-a)**k)
plt.plot(dk[1:],'-o')
plt.xlabel('$j$')
plt.ylabel('$x_{2,j}-x_{2,j-1}$')

def hist_uniq(x):
	x=np.float16(x)
	xu=np.sort(np.unique(x))
	n=[]
	for i,u in enumerate(xu):
		n.append(np.sum(x==u))
	return xu,np.array(n)
	
plt.figure()
hk,dkk=hist_uniq(dk)
plt.plot(hk,dkk,'o')
hk,dkk=hist_uniq(np.diff(xs2))
plt.plot(hk,dkk,'*')

xs2=x[np.where(np.float16(S)==Su[-3])[0]]
hk,dkk=hist_uniq(np.diff(xs2)/a**2)
plt.plot(hk,dkk,'s')

xs3=x[np.where(np.float16(S)==Su[-4])[0]]
hk,dkk=hist_uniq(np.diff(xs3)/a**3)
plt.plot(hk,dkk,'o')

xs4=x[np.where(np.float16(S)==Su[-5])[0]]
hk,dkk=hist_uniq(np.diff(xs4)/a**4)
plt.plot(hk,dkk,'*')
plt.xlim([0,1])
#%%% Distance between lamellae

random=0
shuffle=0
a=0.1
x0=np.array([1.0])
Sn,N=[],[]
Tv=[10]
#Tv=np.arange(10)
plt.figure()
for T in Tv:
	#	sigma=16
	M=10 # n%umber of moments to compute
	dt=1/1  # Discretisation of time step
	X,Y=np.meshgrid(0,np.arange(n)) 
	
	#x,S,wrapped_time=lagrangian_nfold(x0,T,np.array([a,1-a]),random)
	x,S,wrapped_time=lagrangian_sym(x0,T,a,random)
	
	xi=x
	s0=0.05
	D=(sigma/n)**2/2
	#Sigma=np.sqrt(2*D)*n
	Cmax=1./np.sqrt(1.+4*D/s0**2*wrapped_time)
	Si=s0*np.sqrt(1.+4*D/s0**2*wrapped_time)*S
	Su=np.unique(np.float16(S))
#	np.sort(Su)
	Sn.append(list(Su))
	N.append([len(np.where(np.float16(S)==s)[0]) for s in Su])

x1=np.concatenate((np.array([x[-1]])-1,np.array(x)))
x0=np.concatenate((np.array(x),np.array([x[0]])+1))
d1=np.diff(x1)
d0=np.diff(x0)
S=np.array(S)
plt.plot(S,d1,'d',label='left',alpha=0.01)
plt.plot(S,d0,'*',label='right',alpha=0.01)
plt.plot([S.min(),S.max()],[S.min(),S.max()],'k--')
plt.yscale('log')
plt.xscale('log')
plt.xlabel(r'$1/\rho$')
plt.ylabel('Distance')
plt.title(r'$a={:1.2f}$'.format(a))
plt.legend()

#%% GLOBAL Moments
#%%% Moments of rho as a function of a and sa

plt.style.use('~/.config/matplotlib/joris.mplstyle')
A=np.logspace(-3,np.log10(0.5),100)
sa=0
s0=0.05
D=(sigma/n)**2/2
#Sigma=np.sqrt(2*D)*n
T=12
Nu=[]
RhoM=[]
RhoV=[]
x0=np.array([0.1])
for a in A:
	x,S,wrapped_time=lagrangian(x0,T,a,sa)
	#nu=fractal(x)
	W=np.ones(S.shape)
	W=S
	RhoM.append(np.average(np.log(1/S),weights=W)/T)
	RhoV.append((np.average((np.log(1/S)/T)**2.,weights=W)-np.average(np.log(1/S)/T,weights=W)**2.))


RhoM=np.array(RhoM)
RhoV=np.array(RhoV)
dq=2*3/4
D2=(1+dq)*np.log(2)/np.log(1/A**dq+1/(1-A)**dq)
dq=1.
plt.figure(figsize=(4,3))
D1=(1+dq)*np.log(2)/np.log(1/A**dq+1/(1-A)**dq)
plt.plot(A,RhoM,label=r'$\mu_{\lambda}=-a \log a - (1-a)\log (1-a)$')
M=-A*np.log(A)-(1-A)*np.log(1-A)
plt.plot(A,M,'bo')
plt.plot(A,RhoV*T,label=r'$\sigma^2_{\lambda}=-a \log^2 a - (1-a)\log^2 (1-a) - \mu_\lambda^2$')
V=(A*np.log(A)**2.+(1-A)*np.log(1-A)**2.-M**2.)
plt.plot(A,V,'ro')
plt.plot(A,V/M,'g',label=r'$\sigma^2_\lambda /\mu_\lambda$')
nu_th=D2
V2=2*(2-(nu_th+1))
#plt.plot(A,V2,'g:',label=r'$2(2-D_2)\approx\sigma^2_{\lambda,B}/\mu_{\lambda,B}$')
plt.plot(A,D2,'k:',label=r'$D_2-1$')
plt.plot([0,0.5],[np.log(2),np.log(2)],'k--',label='$\log 2$')
plt.xlabel(r'$a$')
plt.legend(fancybox=False)
plt.ylim([0,2.])
plt.savefig(dir_out+'/moment_of_rho.pdf')
#plt.yscale('log')
#plt.xscale('log')

#%%% Comparison of -mu+sigma2/2  et mu^2/(2*sigma2)
A=np.logspace(-3,np.log10(0.5),100)
M=-A*np.log(A)-(1-A)*np.log(1-A)
V=(A*np.log(A)**2.+(1-A)*np.log(1-A)**2.-M**2.)

plt.plot(A,M-V/2,label='$\mu-\sigma^2/2$')
plt.plot(A,M**2./(2*V),label='$\mu^2/(2\sigma^2)$')
plt.yscale('log')
plt.legend()

#%%% Second Moments with Random aggregation model
plt.style.use('~/.config/matplotlib/joris.mplstyle')
T=16
a=0.25
n=2**15
x0=0.1
sigma=10
D=(sigma/n)**2/2
lyap,sigma2=lyapunov(a)
sB=np.sqrt(D/lyap)
#sa=0.05
random=0.0
#	sigma=16
dt=1/1  # Discretisation of time step
X,Y=np.meshgrid(0,np.arange(n)) 
s0=2*sB

x0=np.array([0.3155],dtype=np.float128)
t=np.arange(40)

mean_rho_2,mean_n,L=[],[],[]
for T in t:
	if T<25:
		x,S,wrapped_time=lagrangian(x0,T,a,random)
		Cmax=1./np.sqrt(1.+4*D/s0**2*wrapped_time)
		Si=s0*np.sqrt(1.+4*D/s0**2*wrapped_time)*S
		mean_rho_2.append(np.mean(S**2.))
		L.append(x.shape[0])
		h,xh=np.histogram(x,np.arange(0,1,sB))
		mean_n.append(np.mean(h))
C,V1=DNS_ns0(n,s0,x0,t[-1],a,0)
C,V=DNS_0mean(n,s0,x0,t[-1],a,sigma,0)

V=np.array(V)
V1=np.array(V1)
mean_n=np.array(mean_n)
mean_rho_2=np.array(mean_rho_2)
L=np.array(L)

plt.figure()
plt.plot(V,label='wave')
plt.plot(V1,label='gaussian')
plt.plot(mean_n*mean_rho_2,'-',label=r'$\langle n \rangle \langle \rho^{-2} \rangle_L$')
plt.plot(L*mean_rho_2*sB,'.',label=r'$L s_B \langle \rho^{-2} \rangle_L$')
#plt.plot(t,np.exp(-lyap**2/(2*sigma2)*t),'k--')
plt.yscale('log')
plt.legend()
#%%% Moments of Cmax
plt.style.use('~/.config/matplotlib/joris.mplstyle')
x0=np.array([0.1])
var_sa=0

s0=0.01
D=1e-6
Nmean=[]
Lmean=[]
k=[]
Tv=np.arange(0,20)
MCmax,VC,VCe,Lmean=[],[],[],[]

Moments=np.arange(1,4)
for T in Tv:
	a=0.17
	lyap=-a*np.log(a)-(1-a)*np.log(1-a)
	sigma2=a*np.log(a)**2.+(1-a)*np.log(1-a)**2.-lyap**2.
	C=DNS_n(2**13,x0,T,a,var_sa)
	k.append(np.mean(C)**2/np.var(C))
	if T<28:
		x,S,wrapped_time=lagrangian(x0,T,a,var_sa)
		Cmax=1./np.sqrt(1.+4*D/s0**2*wrapped_time)
		Si=s0*np.sqrt(1.+4*D/s0**2*wrapped_time)*S
		sB=np.mean(Si)
		MCmax.append([np.mean(Cmax**n) for n in Moments])
		VC.append(np.sum(Cmax**2.*S))
		VCe.append(np.var(C.T))
		Lmean.append(len(x))


MCmax=np.array(MCmax)
VC=np.array(VC)
VCe=np.array(VCe)

plt.text(10,2,'$a={:1.2f}$'.format(a))
[plt.plot(Tv,MCmax[:,q],label=r'$\langle \theta^{:1.0f} \rangle_L$'.format(Moments[q]),linewidth=1.5) for q in range(MCmax.shape[1])]
plt.plot(Tv,1/np.array(Lmean),'c--',label=r'$1/L$',linewidth=1.5)
plt.plot(Tv,VC/VC[0],'r-',label=r'$\int_L s c^2 = L \langle \theta^2 \rangle_L$',linewidth=1.5)
plt.plot(Tv,VCe/VCe[0],'r--',label=r'$\sigma^2_c$',linewidth=1.5)
plt.plot(Tv,5*np.exp(-lyap/2*Tv),'k-',label=r'$\lambda/2$')
plt.plot(Tv,10*np.exp(-3*lyap/2*Tv),'k--',label=r'$-3/2 \lambda$')
plt.plot(Tv,5*np.exp(-2*lyap*Tv),'k:',label=r'$-2 \lambda$')
plt.yscale('log')
plt.legend()
plt.xlabel('Time')
#%%% * Scalar decay reconstructed Eulerian grid
import scipy.optimize

plt.style.use('~/.config/matplotlib/joris.mplstyle')
a=0.4
#Cf=DNS_fourier(x0,T,a,var_sa)
x0=np.random.rand(1)
x0=np.array([0.04])
T=22
sB=1/100
s0=0.1
AA=np.linspace(0.01,0.49,50)
SB=[1/50,1/100,1/500]
SB=[1/50]
M=['d','o','s','*']

plt.figure(figsize=(2,2))
for i,sB in enumerate(SB):
	decay_rate=[]
	for a in AA:
		print(a)
		V=[]
		x,S,wrapped_time,C=lagrangian_DSM(x0,T,a,0,sB)
		V=[np.var(c) for c in C]
		p=np.polyfit(np.arange(T+1),np.log(V),1)
		decay_rate.append([a,sB,-p[0]])
	
	decay_rate=np.array(decay_rate)
	dq=1
	A=decay_rate[:,0]
	
	D1=(1+dq)*np.log(2)/np.log(1/A**dq+1/(1-A)**dq)
	lyap,sigma2=lyapunov(A)
	 #plt.plot(1+D1,decay_rate[:,2],'k'+M[i],label='Baker map'.format(1/sB),color='plt.cm.cool(i/3)',fillstyle='full')
	plt.plot(1+D1,decay_rate[:,2],'ko',label='Data',zorder=100)

d1=1+D1
#kn=(d1-1)/(2*np.log(1/sB)*(2-d1))
dq=2.0
D2=[]
for a in AA:
	f = lambda Dq : dq*np.log(2)-np.log(1/a**((dq-1)*Dq)+1/(1-a)**((dq-1)*Dq))
	D2.append(scipy.optimize.broyden1(f, 1, f_tol=1e-14)+1)
d2=np.array(D2)
kn=1/(sB**(d2-2)-1)

nu=np.linspace(1.2,2,1000)
mu=1/(nu-1)
sigma2=2*(2-nu)/(nu-1)
M2=2*mu-2*sigma2
M2[mu-2*sigma2<0]=mu[mu-2*sigma2<0]**2/(2*sigma2[mu-2*sigma2<0])
ksi=np.interp(d1,nu,M2)

KSI=np.loadtxt('Baker_xi.txt')
ksi=np.interp(1+D1,KSI[:,0],KSI[:,1])

p=np.polyfit(KSI[:-1,0],KSI[:-1,1],1)
ksi=(1+D1)*p[0]+p[1]

ksi[kn<ksi]=kn[kn<ksi]

idint=np.where(kn>ksi)[0][1]

idint=1

plt.plot(1+D1,-np.log(1-2*A+2*A**2.),'-',color='seagreen',label=r'Isolated strip',linewidth=1.2)
plt.plot([],[],'-',color='w',label=r'\textbf{Aggregation models:}',linewidth=1.2)
plt.plot(1+D1,-np.log(1-3*A+3*A**2.),'--',color='darkorange',label=r'Fully correlated',linewidth=1.2)
plt.plot(1+D1,np.zeros(D1.shape)+np.log(2),':',color='indianred',label=r'Fully random',linewidth=1.2)
plt.plot(1+D1,np.log(2)*ksi,'-',color='blueviolet',label=r'Correlated',linewidth=1.2) #kn>ksi
#plt.plot(1+D1[:idint],np.log(2)*(ksi[:idint]-kn[:idint]),'-',color='blueviolet',linewidth=1.2) # kn<ksi
#plt.yscale('log')
plt.xlabel(r'$D_1$')
plt.ylabel(r'$\gamma_{2,c}$')

# G2=np.loadtxt('Baker_gamma2_theory2.txt').T
# D1=(1+dq)*np.log(2)/np.log(1/G2[:,0]**dq+1/(1-G2[:,0])**dq)
# plt.plot(D1+1,G2[:,1],'k--',label=r'Theory (empirical $P_n$)')
# plt.ylim([1e-2,2e0])


# G2=np.loadtxt('gamma2_theory.txt')
# plt.plot(G2[:,0],G2[:,1],'k-',label=r'Theory (Gamma $P_n$)')
# plt.ylim([1e-2,2e0])

plt.legend(frameon=False,fontsize=6)
plt.yscale('log')
plt.ylim([1e-2,2])
plt.savefig('/home/joris/Dropbox/Articles/AggregationBatchelor/Figures/gamma2_baker_log.pdf',bbox_inches='tight')

#%%% DNS Scalar Decay rate as a function of t
plt.style.use('~/.config/matplotlib/joris.mplstyle')
random=0

def mom(x,k):
	return np.mean(np.abs(x-np.mean(x))**k)
x0=np.array([1.0],dtype=np.float64)
#A=np.logspace(-7,np.log10(0.49),100)
var_sa=0
shuffle=0
q=2
plt.figure(figsize=(5,5))
s0=0.05
sigma=20
n=2**13
D=(sigma/n)**2/2
X,Y=np.meshgrid(0,np.arange(n)) 
V,Vm=[],[]
t=np.arange(25)
for T in t:
	a=0.4
	C=DNS_n(2**13,x0,T,a,var_sa)
	#Cf=DNS_fourier(x0,T,a,var_sa)
	shuffle=0
	#x,S,wrapped_time=lagrangian_nfold(x0,T,np.array([a,1-a]),random)
	x,S,wrapped_time=lagrangian_sym(x0,T,a,random)
	Cmax=1./np.sqrt(1.+4*D/s0**2*wrapped_time)
	Si=s0*np.sqrt(1.+4*D/s0**2*wrapped_time)*S
	sB=2*np.mean(Si)
	bins=np.arange(0,1,sB)
	ceul,cb=np.histogram(x,bins,weights=s0*S/sB*np.sqrt(np.pi))
	ceul2,cb=np.histogram(x,bins,weights=(s0*S/sB*np.sqrt(np.pi))**2.)
	
	
	xi=x
	Cmax=1./np.sqrt(1.+4*D/s0**2*wrapped_time)
	Si=s0*np.sqrt(1.+4*D/s0**2*wrapped_time)*S
	sB=2*np.mean(Si)
	if T<14:
		C_lag=np.zeros(len(Y))
		for y in range(len(Y)):
			C_lag[y]=np.sum(Cmax*np.exp(-(x*n-y)**2./(Si*n)** 2.)+
										 Cmax*np.exp(-((x-1)*n-y)**2./(Si*n)**2.)+
										 Cmax*np.exp(-((x+1)*n-y)**2./(Si*n)**2.))
	else:
		C_lag=np.nan
	
	shuffle=1
	x,S,wrapped_time=lagrangian_nfold(x0,T,np.array([a,1-a]),random)
	Cmax=1./np.sqrt(1.+4*D/s0**2*wrapped_time)
	Si=s0*np.sqrt(1.+4*D/s0**2*wrapped_time)*S
	sB=2*np.mean(Si)
	bins=np.arange(0,1,sB)
	ceul3,cb=np.histogram(x,bins,weights=s0*S/sB*np.sqrt(np.pi))
	
	V.append([np.var(ceul),np.var(ceul),np.var(C_lag),np.var(ceul3)])
	m=2
	Vm.append([mom(C,m),mom(ceul,m),mom(C_lag,m),mom(ceul3,m)])
	
plt.figure()
V=np.array(V)
plt.plot(t,V[:,0]/V[0,0],label='Eulerien')
plt.plot(t,V[:,3]/V[0,3],label='Lag shuffle')
plt.plot(t,V[:,1]/V[0,1],label='Lagrangian Grid')
plt.plot(t,V[:,2]/V[0,2],'k--',label='Lagrangian')
lyap=-a*np.log(a)-(1-a)*np.log(1-a)
sigma2=a*np.log(a)**2.+(1-a)*np.log(1-a)**2.-lyap**2.
#plt.plot(t,np.exp(-lyap**2./(2*sigma2)*t),'k:')
#plt.plot(t,np.exp(2*np.log(1-a)*t),'r--')
plt.legend()
plt.yscale('log')
plt.ylabel('Var(c)')
plt.xlabel('$t$')


plt.figure()
V=np.array(Vm)
plt.plot(t,V[:,0]/V[0,0],label='Eulerien')
plt.plot(t,V[:,3]/V[0,3],label='Lag shuffle')
plt.plot(t,V[:,1]/V[0,1],label='Lagrangian Grid')
plt.plot(t,V[:,2]/V[0,2],'k--',label='Lagrangian')
lyap=-a*np.log(a)-(1-a)*np.log(1-a)
sigma2=a*np.log(a)**2.+(1-a)*np.log(1-a)**2.-lyap**2.
plt.plot(t,np.exp(-lyap**2./(2*sigma2)*t),'k:')
#plt.plot(t,np.exp(2*np.log(1-a)*t),'r--')
plt.legend()
plt.yscale('log')
plt.ylabel('$c^m=$')
plt.xlabel('$t$')


#%%%  DNS  Scalar Decay rate as a function of  sigma (2)
x0=np.array([0.1546],dtype=np.float64)
#A=np.logspace(-7,np.log10(0.49),100)
var_sa=0
q=2
plt.figure()
x0=np.random.rand(16)

n=2**15
T=80
A=np.linspace(0.1,0.4,4)
#A=[0.1]
for a in A:
	X,Y=np.meshgrid(0,np.arange(n)) 
	V=[]
	lyap,sigma2=lyapunov(a)
	Va,Vb=[],[]
	Sigma=np.logspace(1,3,20)
	Pe=1/(Sigma/n)**2*2
	for sigma in Sigma:
		D=(sigma/n)**2/2
		s0=2*np.sqrt(D/lyap)
#		s0=0.05
		t=np.arange(int(T/lyap))
		C,V=DNS_ns0(n,s0,x0,t[-1],a,0)
		tfit=(t*lyap>5)&(np.log10(V)>-28)
		Va.append(-np.polyfit(t[tfit],np.log(V[tfit]),1)[0])
		#Vb.append(-np.polyfit(t[-15:],np.log(M4[-15:]),1)[0])
		#Va.append(-best_polyfit(t[:],np.log(V[:]))[0])
	D=(Sigma/n)**2/2
	norm=(np.array(Va)-np.array(Va)[0])/np.array(Va)[5]
	norm[norm==0]=np.nan
	plt.plot(Pe,norm,'-',color=plt.cm.jet(a*2),label='a={:1.2}'.format(a))
plt.xscale('log')
plt.yscale('log')
#plt.ylim([1e-2,2])
plt.xlabel('Pe')
plt.ylabel('$(\gamma_2-\gamma_{2,\infty})/\gamma_{2,\infty}$')
plt.plot(Pe,1e2*Pe**-0.5,'k-',label='Pe$^{-1/2}$')
plt.plot(Pe,1e3*Pe**-1,'k--',label='Pe$^{-1}$')
plt.legend()
plt.savefig('baker_peclet_gamma2.pdf')
#%%%  DNS  Scalar Decay rate as a function of  sigma (3)
x0=np.array([0.1546],dtype=np.float64)
#A=np.logspace(-7,np.log10(0.49),100)
var_sa=0
q=2
plt.figure()
x0=np.random.rand(16)

n=2**13
T=80
A=np.linspace(0.1,0.4,4)
for a in A:
	X,Y=np.meshgrid(0,np.arange(n)) 
	V=[]
	lyap,sigma2=lyapunov(a)
	Va,Vb=[],[]
	Sigma=np.array([2,4,8,16,32,64,128,256,512])
	for sigma in Sigma:
		D=(sigma/n)**2/2
		s0=2*np.sqrt(D/lyap)
#		s0=0.05
		t=np.arange(int(T/lyap))
		C,V=DNS_ns0(n,s0,x0,t[-1],a,0)
		tfit=(t*lyap>10)&(np.log10(V)>-28)
		Va.append(-np.polyfit(t[tfit],np.log(V[tfit]),1)[0])
		#Vb.append(-np.polyfit(t[-15:],np.log(M4[-15:]),1)[0])
		#Va.append(-best_polyfit(t[:],np.log(V[:]))[0])
	D=(Sigma/n)**2/2
	plt.plot(1/np.log(D)**2,(np.array(Va)-np.array(Va)[0]),'o-',color=plt.cm.jet(a*2),label='a={:1.2}'.format(a))
#plt.xscale('log')
#plt.yscale('log')
plt.ylim([0,1])
plt.xlabel('$1/(\log \kappa)^2$')
plt.ylabel('$(\gamma_2-\gamma_{2,\infty})$')
#plt.plot(D,20*D**0.5,'k--',label='$20 \kappa^{1/2}$')
plt.legend()
plt.title('baker map')
#%%%  DNS  Scalar Decay rate as a function of  sigma
x0=np.array([0.1546],dtype=np.float64)
#A=np.logspace(-7,np.log10(0.49),100)
var_sa=0
q=2
plt.figure()
s0=0.05
T=80
A=np.linspace(0.01,0.4,10)
for a in A:
	X,Y=np.meshgrid(0,np.arange(n)) 
	V=[]
	t=np.arange(25)
	Va,Vb=[],[]
	Sigma=np.uint16(np.logspace(np.log10(2),np.log10(2e3),10))
	D=(Sigma/n)**2/2
	Pe=1/D
	for sigma in Sigma:
		n=2**14
		D=(Sigma/n)**2/2
		V=[]
		
		lyap,sigma2=lyapunov(a)
		V=[]
		s0=2*np.sqrt(D/lyap)
		t=np.arange(T/lyap)
		
		C,V=DNS_ns0(n,s0,x0,t[-1],a,var_sa)
		
			#M4.append(np.mean((C-np.mean(C))**4))
		Va.append(-np.polyfit(t[-15:],np.log(V[-15:]),1)[0])
		#Vb.append(-np.polyfit(t[-15:],np.log(M4[-15:]),1)[0])
		#Va.append(-best_polyfit(t[:],np.log(V[:]))[0])
	plt.plot(Pe,np.array(Va)-np.array(Va)[0],'-',color=plt.cm.jet(a*2),label='a={:1.2}'.format(a))
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.ylim([0,10])
plt.xlabel('Pe')
plt.ylabel('$\gamma_2-\gamma_{2,\infty}$')
plt.plot(Pe,100 * D**0.5,'k--',label='Pe$^{-0.5}$')
plt.plot(Pe,100* D**1,'k-',label='Pe$^{-1}$')
#%%%  DNS  Scalar Decay rate as a function of a
x0=np.array([0.23],dtype=np.float64)
#A=np.logspace(-7,np.log10(0.49),100)
var_sa=0
q=2
plt.figure()
s0=0.05
for sigma in [50]:
	n=2**15
	D=(sigma/n)**2/2
	X,Y=np.meshgrid(0,np.arange(n)) 
	V=[]
	t=np.arange(40)
	A=np.linspace(0.01,0.49,20)
	Va,Vb=[],[]
	for a in A:
#		V=[]
#		for T in t:
		lyap=-a*np.log(a)-(1-a)*np.log(1-a)
		s0=2*np.sqrt(D/lyap)
		C,V=DNS_ns0(n,s0,x0,t[-1],a,var_sa)
#			V.append(np.var(C))
			#M4.append(np.mean((C-np.mean(C))**4))
		Va.append(-np.polyfit(t[10:],np.log(V[10:]),1)[0])
		#Vb.append(-np.polyfit(t[-15:],np.log(M4[-15:]),1)[0])
		#Va.append(-best_polyfit(t[:],np.log(V[:]))[0])
		plt.plot(t,V,color=plt.cm.jet(a*2))
plt.yscale('log')

plt.figure()
plt.plot(A,Va,'*',label='D={:1.1e}'.format(D))


lyap=-A*np.log(A)-(1-A)*np.log(1-A)
sigma2=A*np.log(A)**2.+(1-A)*np.log(1-A)**2.-lyap**2.
plt.plot(A,lyap**2./(2*sigma2),label=r'$\lambda^2/(2\sigma^2)$')
decay_rate=-(-np.log(1-A)-lyap)**2./(2*sigma2)+2*np.log(1-A) # Gaussian approx at -log(1-a)
decay_rate2=(2*np.log(1-A)-2) # Binomial at minimum
plt.plot(A,lyap**2./(2*sigma2),label=r'$\lambda^2/(2\sigma^2)$')
plt.plot(A,-decay_rate2,label=r'$\Lambda^*=-\log(1-a)$')
plt.plot(A,2*lyap-2*sigma2,label=r'$2\lambda-2\sigma^2$')
plt.plot(A,lyap-sigma2/2.,label=r'$\lambda-\sigma^2/2$')
plt.plot(A,-(4*np.log(1-A)))
#plt.plot(A,3*lyap-9*sigma2/2,label=r'$3\lambda-9\sigma^2/2)$')
plt.ylabel(r'$\gamma_2$')
plt.xlabel(r'$a$')
plt.legend()
plt.ylim([0,2])


#%%%  DNS  Scalar Decay rate as a function of a, zero Mean

plt.style.use('~/.config/matplotlib/joris.mplstyle')
x0=np.array([0.124,0.75],dtype=np.float64)
#A=np.logspace(-7,np.log10(0.49),100)
var_sa=0.0
q=2
plt.figure()
T=80
#for sigma in [5,10,20]:
VaD=[]
sigma=10
x0=np.array([0.1])
for sigma in [200]:
	n=2**15
	D=(sigma/n)**2/2
	s0=0.05

	X,Y=np.meshgrid(0,np.arange(n)) 
	V=[]
	
	A=np.linspace(0.01,0.49,100)
	Va,Vb=[],[]
	for a in A:
		lyap,sigma2=lyapunov(a)
		V=[]
		s0=2*np.sqrt(D/lyap)
		t=np.arange(int(T/lyap))
		C,V=DNS_0mean(n,s0,x0,t[-1],a,sigma,var_sa)
#		C,V=DNS_ns0(n,s0,x0,t[-1],a,var_sa)
#		C,V=DNS_flipped_rand(n,s0,T,a,sigma,var_sa)
#		C,Cmax,V=DNS_0mean_normed(n,s0,x0,T,a,sigma,var_sa)
			#plt.plot(t,np.log(V))
			#M4.append(np.mean((C-np.mean(C))**4))
		tfit=(t*lyap>10)&(np.log10(V)>-35)
		Va.append(-np.polyfit(t[tfit],np.log(V[tfit]),1)[0])
		#Vb.append(-np.polyfit(t[-15:],np.log(M4[-15:]),1)[0])
		#Va.append(-best_polyfit(t[:],np.log(V[:]))[1])
		plt.plot(t[tfit],V[tfit],color=plt.cm.jet(a*2))
#	plt.plot(A,Va,'*',label='D={:1.1e}'.format(D))
	VaD.append(Va)
plt.yscale('log')

plt.figure(figsize=(3,2))
plt.text(-0.1,0.9,'(a)',transform=plt.gca().transAxes)

lyap=-A*np.log(A)-(1-A)*np.log(1-A)
sigma2=A*np.log(A)**2.+(1-A)*np.log(1-A)**2.-lyap**2.
plt.plot(A,np.array(VaD).T,'ro',label='DNS ($D={:1.0e}$)'.format(D))
#plt.plot(A,lyap**2./(2*sigma2),label=r'$\lambda^2/(2\sigma^2)$')
decay_rate=-(-np.log(1-A)-lyap)**2./(2*sigma2)+2*np.log(1-A) # Gaussian approx at -log(1-a)
decay_rate2=(4*np.log(1-A)) # Binomial at minimum
decay_rate3=-(np.log(1-3*A+3*A**2.)) # correlated agg
ac=0.28
plt.plot(A[A<ac],lyap[A<ac]**2./(2*sigma2[A<ac]),'k--',label=r'$\lambda^2/(2\sigma^2)$')
plt.plot(A[A>ac],lyap[A>ac]**2./(2*sigma2[A>ac]),'k--',alpha=0.2)
plt.plot(A,lyap-sigma2/2,label=r'$\lambda-\sigma^2/2$')
plt.plot(A,2*(lyap-sigma2),label=r'$2(\lambda-\sigma^2)$')
plt.plot(A[A<ac],3*lyap[A<ac]-9*sigma2[A<ac]/2,'k:',alpha=0.2,linewidth=1.5)
plt.plot(A[A>ac],3*lyap[A>ac]-9*sigma2[A>ac]/2,'k:',label=r'$3\lambda-9\sigma^2/2$',linewidth=1.5)
plt.plot(A,-decay_rate2,'r',label=r'$-4 \log(1-a)$')
plt.plot(A,-decay_rate2/2,'k',label=r'$-2 \log(1-a)$')
plt.plot(A,-np.log(1-3*A+3*A**2.),'g',label=r'$\log(1-3a+3a^2)$')
plt.plot(A,-np.log(1-2*A+2*A**2.),'c',label=r'$\log(1-2a+2a^2)$')
plt.plot([ac,ac],[0,2.5],'k-',alpha=0.5)
plt.ylabel('$\gamma_2$')
plt.xlabel('$a$')
plt.legend(fontsize=8,loc=2)
plt.ylim([0,2.5])
plt.savefig('/home/joris/Dropbox/Articles/AggregationBatchelor/Figures/gamma2_baker.pdf',bbox_inches='tight')

#%%%  DNS versus DSM Scalar Decay rate as a function of a

plt.style.use('~/.config/matplotlib/joris.mplstyle')
x0=np.array([0.124,0.75],dtype=np.float64)
#A=np.logspace(-7,np.log10(0.49),100)
var_sa=0.0
q=2
plt.figure()
T=23
#for sigma in [5,10,20]:
VaD=[]
sigma=10
x0=np.array([0.1])
for sigma in [200]:
	n=2**15
	D=(sigma/n)**2/2
	s0=0.05

	X,Y=np.meshgrid(0,np.arange(n)) 
	V=[]
	
	A=np.linspace(0.01,0.49,20)
	Va,Vb=[],[]
	for a in A:
		lyap,sigma2=lyapunov(a)
		V=[]
		s0=2*np.sqrt(D/lyap)
		t=np.arange(T)
#		C,V=DNS_0mean(n,s0,x0,t[-1],a,sigma,var_sa)
		C,V=DNS_ns0(n,s0,x0,t[-1],a,var_sa)
#		C,V=DNS_flipped_rand(n,s0,T,a,sigma,var_sa)
#		C,Cmax,V=DNS_0mean_normed(n,s0,x0,T,a,sigma,var_sa)
			#plt.plot(t,np.log(V))
			#M4.append(np.mean((C-np.mean(C))**4))
		tfit=(t>5)&(np.log10(V)>-35)
		pDNS=-np.polyfit(t[tfit],np.log(V[tfit]),1)[0]
		plt.plot(t[tfit],V[tfit],'--',color=plt.cm.jet(a*2))
		
		sB=np.sqrt(2*D/lyap)
		t,VDSM=lagrangian_reconstruct(x0,T,a,0,sB)
		
		tfit=(t>5)&(np.log10(VDSM)>-35)
		pDSM=-np.polyfit(t[tfit],np.log(VDSM[tfit]),1)[0]
		Va.append([pDNS,pDSM])
		#Vb.append(-np.polyfit(t[-15:],np.log(M4[-15:]),1)[0])
		#Va.append(-best_polyfit(t[:],np.log(V[:]))[1])
		plt.plot(t[tfit],VDSM[tfit],color=plt.cm.jet(a*2))
#	plt.plot(A,Va,'*',label='D={:1.1e}'.format(D))
	VaD.append(Va)
plt.yscale('log')

plt.figure(figsize=(3,2))
#plt.text(-0.1,0.9,'(a)',transform=plt.gca().transAxes)

lyap=-A*np.log(A)-(1-A)*np.log(1-A)
sigma2=A*np.log(A)**2.+(1-A)*np.log(1-A)**2.-lyap**2.
plt.plot(A,np.array(VaD[0])[:,0],'ro',label='DNS ($D={:1.0e}$)'.format(D))
plt.plot(A,np.array(VaD[0])[:,1],'k+',label='DSM ($D={:1.0e}$)'.format(D))

plt.plot(A,-np.log(1-3*A+3*A**2.),'k--',label=r'$<1/\rho^2>$')

#plt.plot(A,-np.log(1-2*A+2*A**2.),'k-',label=r'$\log(1-2a+2a^2)$')
plt.plot(A,-2*np.log(1-A),'k-',label=r'min$(1/\rho^2)$')
plt.legend()
#plt.plot(A,-np.log(1-2*A+2*A**2.),'k-',label=r'$\log(1-2a+2a^2)$')
plt.ylabel('$\gamma_2$')
plt.yscale('log')
plt.xlabel('$a$')
plt.ylim([1e-2,5e0])
#%%%  DNS  c^n as a function nof a
x0=np.array([0.1546],dtype=np.float64)
#A=np.logspace(-7,np.log10(0.49),100)
var_sa=0
q=2
plt.figure()
s0=0.05
for sigma in [10]:
	n=2**15
	D=(sigma/n)**2/2
	X,Y=np.meshgrid(0,np.arange(n)) 
	V=[]
	t=np.arange(25)
	A=np.linspace(0.01,0.49,20)
	Va=[]
	for a in A:
		V=[]
		for T in t:
			C=DNS_n(n,x0,T,a,var_sa)
			V.append([np.mean((C-np.mean(C))**k) for k in np.arange(2,10,2)])
		V=np.array(V)
		Va.append(-np.polyfit(t[-15:],np.log(V[-15:,:]),1)[0])
		#Va.append(-best_polyfit(t[:],np.log(V[:]))[0])
	plt.plot(A,Va,'*')


lyap=-A*np.log(A)-(1-A)*np.log(1-A)
sigma2=A*np.log(A)**2.+(1-A)*np.log(1-A)**2.-lyap**2.
plt.plot(A,lyap**2./(2*sigma2),'k',label=r'$\lambda^2/(2\sigma^2)$')
plt.gca().set_prop_cycle(None)
[plt.plot(A,k*lyap-k**2/2*sigma2,'--',label=r'$n\lambda-n^2 \sigma^2/2, n={:1.0f}$'.format(k)) for k in np.arange(2,10,2)]
plt.gca().set_prop_cycle(None)
[plt.plot(A,(-np.log(1-A)-lyap)**2./(2*sigma2)-k*np.log(1-A),':',label=r'$\Lambda^* = -\log(1-a), n={:1.0f}$'.format(k)) for k in np.arange(2,10,2)]
plt.ylabel('$\gamma_n$')
plt.xlabel('$a$')
plt.legend()
plt.ylim([0,6])
#%%% Scaling of max spatial concentration with Rho
plt.style.use('~/.config/matplotlib/joris.mplstyle')
x0=np.array([0.4],dtype=np.float64)
var_sa=0
q=2
nb=50
Res=[]
x0=np.array([0.1])
from scipy import spatial
#2021-09-03_12:00:24
for TT in range(24):
	for a in [0.3]:
		lyap=-a*np.log(a)-(1-a)*np.log(1-a)
		sigma2=a*np.log(a)**2.+(1-a)*np.log(1-a)**2.-lyap**2.
		#T=int((2*sigma2/lyap**2.)*T)
		T=TT
		D=(sigma/2**13)**2/2
		s0=np.sqrt(2)*np.sqrt(D/a)
		x,S,wrapped_time=lagrangian(x0,T,a,var_sa)
#		C,m=DNS_0mean(2**13,s0,x0,T,a,sigma,var_sa)
# flipped baker map
		C,m=DNS_0mean_flipped(2**13,s0,x0,T,a,sigma,var_sa)
		C=C-C.mean()
		C_gauss=DNS_ns0(2**13,s0,x0,T,a,var_sa)
		Cmax=1./(1.+4*D/s0**2*wrapped_time)
		Cmax_2=1./(1.+4*D/s0**2*wrapped_time)**0.5
		Res.append([np.max(C.T),
							np.max(Cmax),
							np.max(Cmax_2),
							np.max(S),
							np.nanmax(C_gauss.T-np.mean(C_gauss.T))])
Res=np.array(Res)
plt.figure()
plt.plot(Res[:,0]/Res[0,0],Res[:,1],'dk',label=r'Wave')
plt.plot(Res[:,4]/Res[0,4],Res[:,1],'sr',label=r'Gaussian')
plt.plot([1,1e-8],[1,1e-8],'k--',label='1:1')
plt.ylabel(r'max$(1/\rho^2)$')
plt.xlabel(r'max$(c)$')
plt.yscale('log')
plt.xscale('log')
plt.legend()

plt.figure()
plt.plot(Res[:,0]/Res[0,0],Res[:,2],'dk',label=r'Wave')
plt.plot(Res[:,4]/Res[0,4],Res[:,2],'sr',label=r'Gaussian')
plt.plot([1,1e-8],[1,1e-8],'k--',label='1:1')
plt.ylabel(r'max$(1/\rho)$')
plt.xlabel(r'max$(c)$')
plt.yscale('log')
plt.xscale('log')
plt.legend()

plt.figure()
plt.plot(np.abs(C.T))
plt.plot(np.abs(C_gauss.T-np.mean(C_gauss)))
plt.yscale('log')
#%%% Scaling of max concentration with min stretching (without lagrangian)
plt.style.use('~/.config/matplotlib/joris.mplstyle')
x0=np.array([0.4],dtype=np.float64)
var_sa=0
q=2
nb=50
Res=[]
x0=np.array([0.1])
from scipy import spatial
#2021-09-03_12:00:24
for TT in np.arange(1,60,2):
	for a in [0.1]:
		lyap=-a*np.log(a)-(1-a)*np.log(1-a)
		sigma2=a*np.log(a)**2.+(1-a)*np.log(1-a)**2.-lyap**2.
		#T=int((2*sigma2/lyap**2.)*T)
		T=TT
		D=(sigma/2**13)**2/2
		s0=np.sqrt(2)*np.sqrt(D/a)
#		C,m=DNS_0mean(2**13,s0,x0,T,a,sigma,var_sa)
# flipped baker map
		C0m,m=DNS_0mean_flipped(2**13,s0,x0,T,a,sigma,var_sa)
		Cflipped,m=DNS_flipped(2**13,s0,x0,T,a,sigma,var_sa)
		Crand,m=DNS_flipped_rand(2**13,s0,T,a,sigma,var_sa)
		C_gauss=DNS_ns0(2**13,s0,x0,T,a,var_sa)
		rho_min=1./(1-a)**TT
		Cmax=1./(rho_min)
		Cmax_2=1./(rho_min)**2.
		Res.append([np.nanmax(C_gauss.T-np.mean(C_gauss.T)),
							np.nanmax(Cflipped.T-np.mean(Cflipped.T)),
							np.nanmax(Crand.T-np.mean(Crand.T)),
							np.nanmax(C0m.T-np.mean(C0m.T)),
							np.nanstd(Crand.T),
							Cmax,
							Cmax_2])
Res=np.array(Res)
plt.figure()
plt.plot(Res[:,1]/Res[0,1],Res[:,-1],'dk-',label=r'flipped')
plt.plot(Res[:,0]/Res[0,0],Res[:,-1],'sr-',label=r'unflipped')
plt.plot(Res[:,2]/Res[0,2],Res[:,-1],'*b-',label=r'rand')
plt.plot(Res[:,3]/Res[0,3],Res[:,-1],'*g-',label=r'0mean')
plt.plot(Res[:,4]/Res[0,4],Res[:,-1],'*y-',label=r'std')
plt.plot([1,1e-5],[1,1e-5],'k--',label='1:1')
plt.plot([1,1e-5],np.array([1,1e-10]),'k--',label='1:2')
plt.ylabel(r'max$(1/\rho^2)$')
plt.xlabel(r'max$(c)$')
plt.yscale('log')
plt.xscale('log')
plt.legend()


#%%% Moments of rho Nfold
plt.style.use('~/.config/matplotlib/joris.mplstyle')
A=np.logspace(-3,np.log10(0.5),100)
A=np.random.rand(100,10)
A=(A.T/np.sum(A,axis=1)).T
#A=np.ones((100,3))
#A[:,0]=np.linspace(0.01,0.99,100)
#A[:,1]=np.linspace(0.4,0.7,100)
sa=0
s0=0.05
shuffle=0
D=(sigma/n)**2/2
#Sigma=np.sqrt(2*D)*n
T=6
Nu=[]
RhoM=[]
RhoV=[]
Nu=[]
x0=np.array([0.1])
for a in A:
	print(a)
	x,S,wrapped_time=lagrangian_nfold(x0,T,a,sa)
	#nu=fractal(x,2)
	W=np.ones(S.shape)
	W=S
	#Nu.append(nu)
	RhoM.append(np.average(np.log(1/S),weights=W)/T)
	RhoV.append((np.average((np.log(1/S)/T)**2.,weights=W)-np.average(np.log(1/S),weights=W)**2.))


RhoM=np.array(RhoM)
RhoV=np.array(RhoV)
b=2*3/4
D2=(b+1)*np.log(A.shape[1])/np.log(np.sum(1/A**b,axis=1))
dq=3/4
plt.figure(figsize=(4,3))
D1=(1+dq)*np.log(2)/np.log(1/A**dq+1/(1-A)**dq)
plt.plot(A,RhoM,label=r'$\mu_{\lambda}=-a \log a - (1-a)\log (1-a)$')
M=-A*np.log(A)-(1-A)*np.log(1-A)
plt.plot(A,M,'b--')
plt.plot(A,RhoV*T,label=r'$\sigma^2_{\lambda}=-a \log^2 a - (1-a)\log^2 (1-a) - \mu_\lambda^2$')
V=A*np.log(A)**2.+(1-A)*np.log(1-A)**2.-M**2.
plt.plot(A,V,'r--')
plt.plot(A,V/M,'g',label=r'$\sigma^2_\lambda /\mu_\lambda$')
nu_th=D2
V2=2*(2-(nu_th+1))
#plt.plot(A,V2,'g:',label=r'$2(2-D_2)\approx\sigma^2_{\lambda,B}/\mu_{\lambda,B}$')
plt.plot(A,D2,'k:',label=r'$D_2-1$')
plt.plot([0,0.5],[np.log(2),np.log(2)],'k--',label='$\log 2$')
plt.xlabel(r'$a$')
plt.legend(fancybox=False)
plt.ylim([0,2.])
plt.savefig(dir_out+'/moment_of_rho.pdf')
#plt.yscale('log')
#plt.xscale('log')

plt.figure()
plt.plot(D2,V/M,'r*')
plt.plot([0,1],[1,1],'k--')
plt.xlabel('$D_2-1$')
plt.ylabel('$\sigma^2_\lambda / \mu_\lambda$')
plt.title('{:1.0f} folds'.format(A.shape[1]))
plt.savefig(dir_out+'/D_mu.pdf')
#%% GLOBAL PDF

#%%% PDF of rho, 1/s & c

# 	RhoM.append(np.average(np.log(1/S),weights=W)/T)
# 	RhoV.append((np.average(np.log(1/S)**2.,weights=W)-np.average(np.log(1/S),weights=W)**2.)/T)

plt.style.use('~/.config/matplotlib/joris.mplstyle')
A=np.logspace(-3,np.log10(0.5),100)
sa=0
s0=0.05
D=(sigma/n)**2/2
#Sigma=np.sqrt(2*D)*n
T=13
Nu=[]
RhoM=[]
RhoV=[]
x0=np.array([0.1])
a=0.2
x,S,wrapped_time=lagrangian(x0,T,a,sa)
Cmax=1./np.sqrt(1.+4*D/s0**2*wrapped_time)
Si=s0*np.sqrt(1.+4*D/s0**2*wrapped_time)*S
sB=np.mean(Si)


nbin=100
h,xb=np.histogram(np.log(1/S)/T,nbin,density=True,weights=S/np.sum(S))
plt.figure()
plt.plot(xb[:-1],h/np.sum(np.diff(xb)[0]*h),'ko',label='Simulation')
plt.yscale('log')
plt.xlabel(r'$\log \rho / t$')
plt.ylabel(r'$P(\log \rho/t)$')

# Theorical binomial
import scipy.special
#N=np.array([scipy.special.binom(T,i) for i in range(T+1)])
N2=np.array([scipy.special.binom(T,i)*a**i*(1-a)**(T-i) for i in range(T+1)])
pN=np.array([-np.log((a**i*(1-a)**(T-i)))/T for i in range(T+1)])
#plt.bar(pN,N/np.sum(N)*T/(np.log(1-a)-np.log(a)),color='k',width=0.01,label='Binomial distribution')
plt.bar(pN,N2/np.sum(N2)*T/(np.log(1-a)-np.log(a)),color='r',width=0.01,label='Binomial distribution')
# Normal approximation
logrho_T=np.linspace(-1,2,100)
# From lyapunov exponents
mu=-a*np.log(a)-(1-a)*np.log(1-a)
sigma2=(a*np.log(a)**2.+(1-a)*np.log(1-a)**2.-mu**2)/T

#mu=np.mean(np.log(1/S)/T)
#sigma2=np.var(np.log(1/S)/T)
p_logrho_T=1/np.sqrt(2*np.pi*sigma2)*np.exp(-0.5*(logrho_T-mu)**2./(sigma2))
plt.plot(logrho_T,p_logrho_T,'k--',label='Normal approximation')
plt.ylim([1e-10,1e3])
# T=30
# N=np.array([scipy.special.binom(T,i) for i in range(T+1)])
# k=np.arange(T+1)
# plt.plot(k,N/float(2**T))
# sigma2=T*a*(1-a)
# mu=T*((a)+(1-a))/2.
# napp=1/np.sqrt(2*np.pi*sigma2)*np.exp(-(k-mu)**2./(2*sigma2))
# plt.plot(k,napp/np.sum(napp),'r--',label='Normal approximation')
# plt.yscale('log')
plt.legend()

# Theorical binomial
nbin=100
h,xb=np.histogram((1/S),nbin,density=True,weights=S/np.sum(S))
plt.figure()
plt.plot(xb[:-1],h/np.sum(np.diff(xb)[0]*h),'ko',label='Simulation')
plt.yscale('log')
plt.xlabel(r'$ \rho $')
plt.ylabel(r'$P( \rho)$')
import scipy.special
#N=np.array([scipy.special.binom(T,i) for i in range(T+1)])
N2=np.array([scipy.special.binom(T,i)*a**i*(1-a)**(T-i) for i in range(T+1)])
pN=np.array([1/(a**i*(1-a)**(T-i)) for i in range(T+1)])
#plt.bar(pN,N/np.sum(N)*T/(np.log(1-a)-np.log(a)),color='k',width=0.01,label='Binomial distribution')
plt.plot(pN,N2/np.sum(N2),'ro',label='Binomial distribution')
# Normal approximation
plt.yscale('log')
plt.xscale('log')
plt.ylim([1e-10,1e3])
# T=30
# N=np.array([scipy.special.binom(T,i) for i in range(T+1)])
# k=np.arange(T+1)
# plt.plot(k,N/float(2**T))
# sigma2=T*a*(1-a)
# mu=T*((a)+(1-a))/2.
# napp=1/np.sqrt(2*np.pi*sigma2)*np.exp(-(k-mu)**2./(2*sigma2))
# plt.plot(k,napp/np.sum(napp),'r--',label='Normal approximation')
# plt.yscale('log')
plt.legend()

xbin=np.logspace(np.log10(S.min()),np.log10(S.max()),nbin)
h,xb=np.histogram(S,xbin,density=True)
plt.figure()
plt.plot(xb[1:],h,'ko')
N2=np.array([scipy.special.binom(T,i)/(a**i*(1-a)**(T-i)) for i in range(T+1)])
pN=np.array([(a**i*(1-a)**(T-i)) for i in range(T+1)])
#plt.bar(pN,N/np.sum(N)*T/(np.log(1-a)-np.log(a)),color='k',width=0.01,label='Binomial distribution')
plt.plot(pN,N2/np.sum(N2),'ro',label='Binomial distribution')
plt.yscale('log')
plt.xscale('log')
plt.xlabel(r'$1/\rho$')
plt.ylabel(r'$P(1/ \rho)$')

xbin=np.logspace(np.log10(Cmax.min()),np.log10(Cmax.max()),nbin)
h,xb=np.histogram(Cmax,xbin,density=True)
plt.figure()
plt.plot(xb[1:],h,'ko')
plt.yscale('log')
plt.xscale('log')
plt.xlabel(r'$\theta$')
plt.ylabel(r'$P(\theta)$')

# Reconstruct C
n=2**13
X,Y=np.meshgrid(0,np.arange(n)) 
C_lag=np.zeros(len(Y))
for y in range(len(Y)):
	C_lag[y]=np.sum(Cmax*np.exp(-(x*n-y)**2./(Si*n)** 2.)+
								 Cmax*np.exp(-((x-1)*n-y)**2./(Si*n)**2.)+
								 Cmax*np.exp(-((x+1)*n-y)**2./(Si*n)**2.))

xbin=np.logspace(np.log10(C_lag.min()),np.log10(C_lag.max()),nbin)
h,xb=np.histogram(C_lag,xbin,density=True)
plt.plot(xb[1:],h,'rs',label='$c$')

#%%% pdf of sB

h,xh=np.histogram(Si,50,density=True)
plt.plot(xh[1:],h,'ko')
plt.yscale('log')
plt.xscale('log')


nu,N,Nf=fractal(xi)
print(nu)
#Theory aker map
a=sa
dq=2*3/4.
D2=(1+dq)*np.log(2)/np.log(1/a**dq+1/(1-a)**dq)
print('Baker map:',D2)

plt.plot(1/N,Nf,'*')
plt.yscale('log')
plt.xscale('log')

#%%% Compare p(C,t) DSM/DNS for several a

plt.style.use('~/.config/matplotlib/joris.mplstyle')
x0=np.random.rand(5)
var_sa=0
q=2
nb=70
from scipy import spatial
#2021-09-03_12:00:24
fig,ax=plt.subplots(1,1,figsize=(2,2))
fig2,ax2=plt.subplots(1,1,figsize=(2,2))
for a in [0.1,0.2,0.3,0.4]:
	T=15
	
	lyap=-a*np.log(a)-(1-a)*np.log(1-a)
	#T=int(1/lyap*12)
	C=DNS_n(2**13,x0,T,a,var_sa)
	s0=0.05
	D=(sigma/2**13)**2/2
	x,S,wrapped_time=lagrangian(x0,T,a,var_sa)
	Cmax=1./(1.+4*D/s0**2*wrapped_time)**0.5
# 	x,S,wrapped_time=lagrangian(x0,T,a,var_sa)
# 	Cmax=1./np.sqrt(1.+4*D/s0**2*wrapped_time)
# 	Si=s0*np.sqrt(1.+4*D/s0**2*wrapped_time)*S
# 	sB=np.mean(Si)
# 	#sB=2.2*np.mean(Si)
# 	# Lagrangian
# 	tree=spatial.cKDTree(x.reshape(-1,1))
# 	# Take samples in within cmax
# 	neighboors=tree.query_ball_point(x.reshape(-1,1)[np.abs(x-0.5)<0.5-2*sB], sB)
# 	cm=[np.sum(Cmax[n]*np.mean(Si)/sB*np.sqrt(np.pi)) for n in neighboors] # is subjected to noise! 
# 	cm2=np.array([np.sum(s0*S[n]/sB*np.sqrt(np.pi)) for n in neighboors])
# 	nagg_lag=np.array([len(n) for n in neighboors])
# 	cmm=2*s0*np.sqrt(np.pi)
# 	import matplotlib as mpl
# 	
# 	nb=70
# 	cmabs=np.abs(cm2-cmm)
	cmabs=np.abs(C-np.mean(C))
	bin_c=np.logspace(-8,0,nb)
	h2_lag=np.histogram(cmabs,bin_c,density=True)[0]
	h3_lag=np.histogram(Cmax,bin_c,density=True)[0]
	ax.plot(bin_c[1:]/np.mean(C),h3_lag,'*',color=plt.cm.jet(a/0.5))
#	ax.plot(bin_c[1:],h2_lag,'o-',color=plt.cm.jet(a/0.5))
	ax.plot(bin_c[1:]/np.mean(C),h2_lag,'o-',color=plt.cm.jet(a/0.5))
#	bin_c=np.linspace(np.mean(C)*0.6,np.mean(C)*1.2,nb)
	h2_lag,bin_c=np.histogram(C,nb,density=True)
	ax2.plot(bin_c[1:]/np.mean(C),h2_lag*np.mean(C),'o-',color=plt.cm.jet(a/0.5))
ax.set_ylabel(r'$P$')
ax.set_xlabel(r'$c_B$')
ax.set_yscale('log')
ax2.set_yscale('log')
ax.set_xscale('log')
ax.set_xlabel(r'$|c-\langle c \rangle|$')
ax.set_ylabel('$p(c)$')
ax2.set_xlabel(r'$c / \langle c \rangle$')
ax2.set_ylabel('$p(c)$')
#plt.xlim([0,0.1])

ax2.set_xlim(np.array([0.9,1.1]))
#ax.set_ylim([1e-2,1e6])
ax.set_xlim([1e-10,1e2])

#%%% DNS p(C,t) with 0-mean

plt.style.use('~/.config/matplotlib/joris.mplstyle')
x0=np.array([0.4],dtype=np.float64)
var_sa=0
q=2
nb=100
s0=0.05
from scipy import spatial
#2021-09-03_12:00:24
for TT in [30]:
	fig,ax=plt.subplots(1,1,figsize=(2,2))
	fig2,ax2=plt.subplots(1,1,figsize=(2,2))
	for a in [0.1,0.2,0.3,0.40]:
		lyap=-a*np.log(a)-(1-a)*np.log(1-a)
		sigma2=a*np.log(a)**2.+(1-a)*np.log(1-a)**2.-lyap**2.
		#T=int((2*sigma2/lyap**2.)*T)
		T=int((1/lyap)*TT)
		C=DNS_0mean(2**13,s0,x0,T,a,var_sa)
		
	# 	s0=0.05
	# 	D=(sigma/2**13)**2/2
	# 	x,S,wrapped_time=lagrangian(x0,T,a,var_sa)
	# 	Cmax=1./(1.+4*D/s0**2*wrapped_time)
		
	# 	Si=s0*np.sqrt(1.+4*D/s0**2*wrapped_time)*S
	# 	sB=np.mean(Si)
	# 	#sB=2.2*np.mean(Si)
	# 	# Lagrangian
	# 	tree=spatial.cKDTree(x.reshape(-1,1))
	# 	# Take samples in within cmax
	# 	neighboors=tree.query_ball_point(x.reshape(-1,1)[np.abs(x-0.5)<0.5-2*sB], sB)
	# 	cm=[np.sum(Cmax[n]*np.mean(Si)/sB*np.sqrt(np.pi)) for n in neighboors] # is subjected to noise! 
	# 	cm2=np.array([np.sum(s0*S[n]/sB*np.sqrt(np.pi)) for n in neighboors])
	# 	nagg_lag=np.array([len(n) for n in neighboors])
	# 	cmm=2*s0*np.sqrt(np.pi)
	# 	import matplotlib as mpl
	# 	
	# 	nb=70
	# 	cmabs=np.abs(cm2-cmm)
		#cmabs=C[C>0]
		cmabs=np.abs(C-C.mean())
		cmabs_std=np.std(np.abs(C))
		#cmabs=np.abs(C)
		bin_c=np.logspace(-14,0,nb)
		h2_lag=np.histogram(cmabs,bin_c,density=True)[0]
		#h3_lag=np.histogram(Cmax,bin_c,density=True)[0]
		#ax.plot(bin_c[1:]/cmabs_std,h3_lag*cmabs_std,'*',color=plt.cm.jet(a/0.5),label='$\theta$')
		ax.plot(bin_c[1:]/cmabs_std,h2_lag*cmabs_std,'o-',color=plt.cm.jet(a/0.5),label=r'$a={:1.1f}$'.format(a))
	#	bin_c=np.linspace(np.mean(C)*0.6,np.mean(C)*1.2,nb)
		h2_lag,bin_c=np.histogram(C,nb,density=True)
		ax2.plot(bin_c[1:],h2_lag,'o-',color=plt.cm.jet(a/0.5),label=r'$a={:1.1f}$'.format(a))
	
	#ax.plot(bin_c,bin_c**(-1-(lyap+sigma2)/sigma2),'k--')
	ax.set_ylabel(r'$P$')
	ax.set_xlabel(r'$c_B$')
	ax.set_yscale('log')
	ax2.set_yscale('log')
	ax.set_xscale('log')
	ax.set_xlabel(r'$\tilde{c}=|c-\langle c \rangle|/\sigma_c$')
	ax.set_ylabel('$p(c)$')
	ax2.set_xlabel('$c$')
	ax2.set_ylabel('$p(c)$')
	#ax.set_xlim([0,0.1])
	binexp=np.logspace(-3,1,100)
	ax.plot(binexp,np.exp(-binexp),'k--',label=r'$e^{-\tilde{c}}$')
	
	ax.text(5e-3,1e-2,r'$t={:1.0f}/\lambda$'.format(TT))
	ax2.set_xlim(np.array([0.99,1.01])*np.mean(C))
	ax.set_xlim([1e-3,1e2])
	ax.set_ylim([1e-4,1e2])
	ax.legend(fontsize=6)
	fig.savefig(dir_out+'/{:1.0f}-pdf.png'.format(TT),bbox_inches='tight')

#%%% DNS p(C,t) with 0-mean without scaling
plt.style.use('~/.config/matplotlib/joris.mplstyle')
x0=np.array([0.4],dtype=np.float64)
var_sa=0
q=2
nb=100
s0=0.05
from scipy import spatial
#2021-09-03_12:00:24
for TT in [20]:
	fig,ax=plt.subplots(1,1,figsize=(2,2))
	fig2,ax2=plt.subplots(1,1,figsize=(2,2))
	for a in [0.1,0.2,0.3,0.40]:
		lyap=-a*np.log(a)-(1-a)*np.log(1-a)
		sigma2=a*np.log(a)**2.+(1-a)*np.log(1-a)**2.-lyap**2.
		#T=int((2*sigma2/lyap**2.)*T)
		T=int((1/lyap)*TT)
		C=DNS_0mean(2**13,s0,x0,T,a,var_sa)
		#cmabs=C[C>0]
		cmabs=np.abs(C-C.mean())
		cmabs_std=np.std(np.abs(C))
		#cmabs=np.abs(C)
		bin_c=np.logspace(-14,0,nb)
		h2_lag=np.histogram(cmabs,bin_c,density=True)[0]
		#h3_lag=np.histogram(Cmax,bin_c,density=True)[0]
		#ax.plot(bin_c[1:]/cmabs_std,h3_lag*cmabs_std,'*',color=plt.cm.jet(a/0.5),label='$\theta$')
		ax.plot(bin_c[1:],h2_lag,'o',color=plt.cm.jet(a/0.5),label=r'$a={:1.1f}$'.format(a))
		alpha=-1-(lyap+sigma2)/(2*sigma2)
		ax.plot(bin_c,bin_c**alpha,'-',color=plt.cm.jet(a/0.5),label=r'$\alpha={:1.1f}$'.format(alpha))
	#	bin_c=np.linspace(np.mean(C)*0.6,np.mean(C)*1.2,nb)
		h2_lag,bin_c=np.histogram(C,nb,density=True)
		ax2.plot(bin_c[1:],h2_lag,'o-',color=plt.cm.jet(a/0.5),label=r'$a={:1.1f}$'.format(a))
	
	#ax.plot(bin_c,bin_c**(-1-(lyap+sigma2)/sigma2),'k--')
	ax.set_ylabel(r'$P$')
	ax.set_xlabel(r'$c_B$')
	ax.set_yscale('log')
	ax2.set_yscale('log')
	ax.set_xscale('log')
	ax.set_xlabel(r'$\tilde{c}=|c-\langle c \rangle|$')
	ax.set_ylabel('$p(c)$')
	ax2.set_xlabel('$c$')
	ax2.set_ylabel('$p(c)$')
	#ax.set_xlim([0,0.1])
	binexp=np.logspace(-3,1,100)
#	ax.plot(binexp,np.exp(-binexp),'k--',label=r'$e^{-\tilde{c}}$')
	
	ax.text(5e-5,1e10,r'$t={:1.0f}/\lambda$'.format(TT))
	ax2.set_xlim(np.array([0.99,1.01])*np.mean(C))
	#ax.set_xlim([1e-3,1e2])
	ax.set_ylim([1e0,1e12])
	ax.legend(fontsize=6)
	fig.savefig(dir_out+'/{:1.0f}-pdf.png'.format(TT),bbox_inches='tight')
	
#%%% p(c) ~ c^(-1-(lyap+sigma^2)/(2sigma2))
a=np.linspace(0,0.5,100)
lyap=-a*np.log(a)-(1-a)*np.log(1-a)
sigma2=a*np.log(a)**2.+(1-a)*np.log(1-a)**2.-lyap**2.

alpha=-1-(lyap+sigma2)/(2*sigma2)

plt.plot(a,-alpha,'+-')
plt.yscale('log')
#%%% DNS p(C,t) with convolution of sine wave

plt.style.use('~/.config/matplotlib/joris.mplstyle')
x0=np.array([0.4],dtype=np.float64)
var_sa=0
q=2
nb=50
from scipy import spatial
#2021-09-03_12:00:24
for TT in [22]:
	fig,ax=plt.subplots(1,1,figsize=(4,4))
	for a in [0.4]:
		lyap=-a*np.log(a)-(1-a)*np.log(1-a)
		sigma2=a*np.log(a)**2.+(1-a)*np.log(1-a)**2.-lyap**2.
		#T=int((2*sigma2/lyap**2.)*T)
		T=TT
		s0=0.05
		C=DNS_0mean(2**13,s0,x0,T,a,var_sa)
		
		D=(sigma/2**13)**2/2
		x,S,wrapped_time=lagrangian(x0,T,a,var_sa)
		Cmax=1./(1.+4*D/s0**2*wrapped_time)
		Cmax_2=1./(1.+4*D/s0**2*wrapped_time)**0.5
		
# 		Si=s0*np.sqrt(1.+4*D/s0**2*wrapped_time)*S
# 		sB=np.mean(Si)
# 		#sB=2.2*np.mean(Si)
# 		# Lagrangian
# 		tree=spatial.cKDTree(x.reshape(-1,1))
# 		# Take samples in within cmax
# 		neighboors=tree.query_ball_point(x.reshape(-1,1)[np.abs(x-0.5)<0.5-2*sB], sB)
# 		cm=[np.sum(Cmax[n]*np.mean(Si)/sB*np.sqrt(np.pi)) for n in neighboors] # is subjected to noise! 
# 		cm2=np.array([np.sum(s0*S[n]/sB*np.sqrt(np.pi)) for n in neighboors])
# 		nagg_lag=np.array([len(n) for n in neighboors])
# 		cmm=2*s0*np.sqrt(np.pi)
# 		import matplotlib as mpl
	# 	
	# 	nb=70
	# 	cmabs=np.abs(cm2-cmm)
		#cmabs=C[C>0]
		cmabs=np.abs(C-C.mean())
		cmabs_std=np.std(np.abs(C))
		#cmabs=np.abs(C)
		bin_c=np.logspace(-14,0,nb)
		h2_lag=np.histogram(cmabs,bin_c,density=True)[0]
		h3_lag=np.histogram(Cmax,bin_c,density=True)[0] # Wave
		h4_lag=np.histogram(Cmax_2,bin_c,density=True)[0] # Gaussian peak
		ax.plot(bin_c[1:],h2_lag,'ko-',label=r'$p(c)$'.format(a))

	#	bin_c=np.linspace(np.mean(C)*0.6,np.mean(C)*1.2,nb)
		h2_lag,bin_c=np.histogram(C,nb,density=True)
	# Convolution with sine wave
	# p(c) = 1/np.sqrt(cmax**2. - c**2.)/np.pi*2 for c=[0,cmax]
	
		bin_c=np.logspace(-16,0,nb)
		h3_lag=np.histogram(Cmax,bin_c,density=True)[0] # Wave
		ax.plot(bin_c[1:],h3_lag,'r*',label=r'$\theta \sim 1/\rho^2$')

		def p_sine(c):
			bin_mid=(bin_c[1:]+bin_c[:-1])/2.
			idc=np.where((bin_mid>c)&(bin_mid>0))[0]
			#print(idc)
			return np.nansum(1/np.pi*2/np.sqrt(bin_mid[idc]**2. - c**2.)*h3_lag[idc]*np.diff(bin_c)[idc])
		pc_conv=np.array([p_sine(c) for c in bin_c])
		pc_conv=pc_conv/np.nansum(pc_conv[1:]*np.diff(bin_c))
		ax.plot(bin_c,pc_conv,'r--',label=r'$\int P(c|\theta) P(\theta) d\theta$')
	
		bin_c=np.logspace(-16,0,nb)
		h3_lag=np.histogram(Cmax**0.5,bin_c,density=True)[0] # Wave
		def p_sine(c):
			bin_mid=(bin_c[1:]+bin_c[:-1])/2.
			idc=np.where((bin_mid>c)&(bin_mid>0))[0]
			#print(idc)
			return np.nansum(1/np.pi*2/np.sqrt(bin_mid[idc]**2. - c**2.)*h3_lag[idc]*np.diff(bin_c)[idc])
		pc_conv=np.array([p_sine(c) for c in bin_c])
		pc_conv=pc_conv/np.nansum(pc_conv[1:]*np.diff(bin_c))
		ax.plot(bin_c[1:],h3_lag,'bs',label=r'$\theta\sim 1/\rho$')
		ax.plot(bin_c,pc_conv,'b--',label=r'$\int P(c|\theta) P(\theta) d\theta$')

	#ax.plot(bin_c,bin_c**(-1-(lyap+sigma2)/sigma2),'k--')
	ax.set_xlabel(r'$c$')
	ax.set_yscale('log')
	ax.set_xscale('log')
	ax.set_xlabel(r'$|c-\langle c \rangle|/\sigma_c$')

	#ax.set_xlim([0,0.1])
	binexp=np.logspace(-3,1,100)
	leg=plt.legend(title=r'$a={:1.2f}, t={:1.0f}$'.format(a,TT))


#%%% P(C,t) with partial reconstruction

plt.style.use('~/.config/matplotlib/joris.mplstyle')
T=14
sa=0.2
random=0.0

sigma=50
#	sigma=16
M=10 # number of moments to compute
dt=1/1  # Discretisation of time step
X,Y=np.meshgrid(0,np.arange(n)) 

x0=np.array([0.3])
C=DNS(x0,T,sa,random)
x,S,wrapped_time=lagrangian(x0,T,sa,random)

xi=x
s0=0.05
D=(sigma/n)**2/2
#Sigma=np.sqrt(2*D)*n
Cmax=1./np.sqrt(1.+4*D/s0**2*wrapped_time)
Si=s0*np.sqrt(1.+4*D/s0**2*wrapped_time)*S

sB=2*np.mean(Si)
plt.figure()
#plt.plot(C[0,:],label='Eulerian');#plt.yscale('log')
#plt.ylim([,2])

n=2**13
X,Y=np.meshgrid(0,np.arange(n)) 
C_lag=np.zeros(len(Y))
for y in range(len(Y)):
	C_lag[y]=np.sum(Cmax*np.exp(-(x*n-y)**2./(Si*n)** 2.)+
								 Cmax*np.exp(-((x-1)*n-y)**2./(Si*n)**2.)+
								 Cmax*np.exp(-((x+1)*n-y)**2./(Si*n)**2.))
h,cb=np.histogram(np.log(C_lag),100,density='True')
plt.plot(cb[1:],h,'k-',label='$c_B$')
cmax,cb=np.histogram(np.log(Cmax),100,density='True')
#plt.plot(cb[1:],cmax,'r-',label=r'$\theta$')
plt.yscale('log')
plt.xlim([-10,0])
plt.legend(title='$t={:1.0f}$'.format(T))

#%%% P(C,t) on sB

plt.style.use('~/.config/matplotlib/joris.mplstyle')
T=24
sa=0.2
random=0.0

sigma=50
#	sigma=16
M=10 # number of moments to compute
dt=1/1  # Discretisation of time step

sB=1/100
s0=0.05
cb=np.linspace(s0*0.8,s0*1.2,100)
seeds=np.arange(100)
H=[]
for s in seeds:
	print(s)
	np.random.seed(s)
	x0=np.random.rand(1)
	x,S,wrapped_time=lagrangian(x0,T,sa,random)
	
	Cmax=s0/sB*S
	
	C,xb=np.histogram(x,np.arange(0,1+sB,sB),weights=Cmax)
	
	hc,cb=np.histogram(C,cb,density=True)
	H.append(hc)

plt.figure()

H=np.array(H)
plt.plot(cb[1:],np.mean(H,axis=0),'ko-',label='$c_B$')
plt.yscale('log')
#plt.xscale('log')
plt.legend(title='$t={:1.0f}$'.format(T))
#%% Statistics in bundles

#%%% * 2D Eulerian histgrams N / cmax / 1/rho
plt.style.use('~/.config/matplotlib/joris.mplstyle')
random=0.0
a=0.1
T=24
x0=np.float128(np.random.rand(1))
x,S,wrapped_time=lagrangian(x0,T,a,random)
nu=1+2*np.log(2)/np.log(a**(-1)+(1-a)**(-1))
sB=1/100
k=25

# Lagrangian
#from scipy import spatial
# tree=spatial.cKDTree(xi_per.reshape(-1,1))
# # Take samples in within cmax
# neighboors=tree.query_ball_point(xi.reshape(-1,1), sB)
# nagg=np.array([len(n) for n in neighboors])
# cmaxagg_mean=np.array([np.nanmean((Cmax[n])) for n in neighboors])
# logrho=np.array([np.nanmean(np.log(1/S[n])) for n in neighboors])
# s_mean=np.array([np.nanmean((S[n])) for n in neighboors])

# #Eulerian
NH=np.histogram(x,np.arange(0,1+sB,sB),density=False)[0]
logrhoH=np.histogram(x,np.arange(0,1+sB,sB),density=False,weights=np.log(1/S))[0]
inv_rhoH=np.histogram(x,np.arange(0,1+sB,sB),density=False,weights=S)[0]

idgood=inv_rhoH>0
nagg=NH[idgood].flatten()
s_mean=(inv_rhoH[idgood]/NH[idgood]).flatten()
logrho=(logrhoH[idgood]/NH[idgood]).flatten()

np.savetxt('Baker_Scaling_N_invrho_logrho_a{:1.2f}.txt'.format(a),np.vstack((nagg,s_mean,logrho)).T)

Df=nu
fig,ax=plt.subplots(1,2,figsize=(3,1.5), sharey=True)
nbin=np.log(np.unique(np.uint32(np.logspace(np.log10(np.min(nagg))*0.9,np.log10(np.max(nagg))*1.1,k)))-0.5)
sbin=np.log(np.logspace(np.log10(s_mean.min())*1.1,np.log10(s_mean.max())*0.9,k))
ax[0].hist2d(np.log(s_mean),np.log(nagg),[sbin,nbin],norm=mpl.colors.LogNorm(),cmap=plt.cm.Greys)
ax[0].patch.set_facecolor(plt.cm.Greys(0))
ax[0].set_xlabel(r'$\log\langle \rho^{-1}(\textbf{x}) \rangle_{s_B}$')
rx=sbin
ax[0].plot(rx,-rx-4,'r-',label=r'$-1$')
legend=ax[0].legend(frameon=False,fontsize=6)
#plt.setp(legend.get_texts(), color='r')


nbin=np.log(np.unique(np.uint32(np.logspace(np.log10(np.min(nagg))*0.9,np.log10(np.max(nagg))*1.1,k)))-0.5)
rhobin=np.linspace(logrho.min()*0.9,logrho.max()*1.1,k)
ax[0].set_ylabel(r'$ \log n(\textbf{x}) $')
ax[1].hist2d(logrho,np.log(nagg),[rhobin,nbin],norm=mpl.colors.LogNorm(),cmap=plt.cm.Greys)
# plt.ylim([0,6])
# plt.xlim([0,20])
ax[1].set_xlabel(r'$\langle \log  \rho(\textbf{x}) \rangle_{s_B}$')
ax[1].patch.set_facecolor(plt.cm.Greys(0))
rx=rhobin
ax[1].plot(rx,rx-3.0,'r--',label=r'$1$')
ax[1].plot(rx,rx*(Df-1)-2.0,'r-',label=r'$D_1-1$')
legend=ax[1].legend(frameon=False,fontsize=6)
#plt.setp(legend.get_texts(), color='r')

fig.savefig('/home/joris/Dropbox/Articles/AggregationBatchelor/Figures/agg_1.pdf',bbox_inches='tight')


#%%% <log rho> ~ <log n> Lagrangian

plt.style.use('~/.config/matplotlib/joris.mplstyle')
T=9
sa=0.12
random=0

x0=np.random.rand(1)
#x0=np.random.rand(10)
x,S,wrapped_time=lagrangian(x0,T,sa,random)
nu,N,Nf=fractal(x,2)
a=sa
dq=2*3/4.
D2=(1+dq)*np.log(2)/np.log(1/a**dq+1/(1-a)**dq)
print(nu)
nu=D2

from scipy import spatial
xi=x
#xi_per=np.hstack((xi-1,xi,xi+1))
xi_per=xi

s0=0.05
sigma=20
D=(sigma/n)**2/2
#Sigma=np.sqrt(2*D)*n
Cmax=1./np.sqrt(1.+4*D/s0**2*wrapped_time)
Si=s0*np.sqrt(1.+4*D/s0**2*wrapped_time)*S

sB=np.mean(Si)

tree=spatial.cKDTree(xi_per.reshape(-1,1))
# Take samples in within cmax
neighboors=tree.query_ball_point(xi.reshape(-1,1), sB)


nagg=np.array([len(n) for n in neighboors])

#% Check cmax of bundles
def bin_operation(x,y,xi,op):
	r=np.zeros(xi.shape[0]-1)
	for i in range(xi.shape[0]-1):
		idx=np.where((x<=xi[i+1])&(x>=xi[i]))[0]
		r[i]=op(y[idx])
	return r

def bundle(neighboors,variable,operator,binarize,nbin):
	v=np.array([operator(variable[n]) for n in neighboors])
	if binarize=='N':
		nagg=np.array([len(n) for n in neighboors])
		nagg_bin=np.logspace(0,np.max(np.log10(nagg)),nbin)
		return bin_operation(nagg,v,nagg_bin,np.nanmean),nagg_bin
	if binarize=='logN':
		nagg=np.array([np.log(len(n)) for n in neighboors])
		nagg_bin=np.linspace(0,np.max(nagg),nbin)
		return bin_operation(nagg,v,nagg_bin,np.nanmean),nagg_bin
	if binarize=='Rho':
		rhoagg_mean=np.array([np.nanmean((1/S[n])) for n in neighboors])
		rho_bin=np.logspace(np.log10(rhoagg_mean.min()),np.log10(rhoagg_mean.max()),nbin)
		return bin_operation(rhoagg_mean,v,rho_bin,np.nanmean),rho_bin
	if binarize=='logRho':
		rhoagg_mean=np.array([np.nanmean(np.log(1/S[n])) for n in neighboors])
		rho_bin=np.linspace((rhoagg_mean.min()),(rhoagg_mean.max()),nbin)
		return bin_operation(rhoagg_mean,v,rho_bin,np.nanmean),rho_bin
	if binarize=='Cmax':
		c_mean=np.array([np.nanmean((Cmax[n])) for n in neighboors])
		c_bin=np.logspace(np.log10(c_mean.min()),np.log10(c_mean.max()),nbin)
		return bin_operation(c_mean,v,c_bin,np.nanmean),c_bin


h,x=bundle(neighboors,np.log(1/S),np.mean,'logN',30)
plt.figure(figsize=(3,2))
plt.plot((x[1:]),h,'ko')
#plt.yscale('log')
#plt.xscale('log')
plt.xlabel(r'$\log N$')
plt.ylabel(r'$\langle \log \rho \rangle$')
#plt.ylim([1e-1,1e2])
#plt.ylim([1e-1,1e2])
isgood=np.where(np.isfinite(h))[0]
p=np.polyfit((x[isgood]),(h[isgood]),1)
#plt.plot(x,p[0]*(x)+p[1],'r--',label='{:1.2f}'.format(p[0]))
plt.plot(x,1/(nu+1-1)*x+4,'r--',label=r'$1/(D_1-1)={:1.2f}$'.format(1/(nu+1-1)))
plt.legend()
plt.savefig(dir_out+'logNlogrho.pdf',bbox_inches='tight')

h,x=bundle(neighboors,np.log(nagg),np.mean,'logRho',30)
plt.figure()
plt.plot((x[1:]),h,'ko')
#plt.yscale('log')
#plt.xscale('log')
plt.ylabel(r'$\log N$')
plt.xlabel(r'$\mu_{ \log \rho}$')
#plt.ylim([1e-1,1e2])
#plt.ylim([1e-1,1e2])
isgood=np.where(np.isfinite(h))[0]
p=np.polyfit((x[isgood]),(h[isgood]),1)
#plt.plot(x,p[0]*(x)+p[1],'r--',label='{:1.2f}'.format(p[0]))
plt.plot(x,(nu+1-1)*x,'--',label=r'$(\nu-1)={:1.2f}$'.format((nu+1-1)))
plt.legend()
plt.savefig(dir_out+'logNlogrho2.pdf',bbox_inches='tight')



h,x=bundle(neighboors,np.log(1/S),np.var,'logN',30)
plt.figure()
plt.plot(x[1:],h,'ko')
#plt.yscale('log')
#plt.xscale('log')
plt.xlabel(r'$\log N$')
plt.ylabel(r'$\sigma^2_{\log \rho}$')
#plt.ylim([1e-1,1e2])
#plt.ylim([1e-1,1e2])
isgood=np.where(np.isfinite(h))[0]
p=np.polyfit((x[isgood]),(h[isgood]),1)
#plt.plot(x,p[0]*(x)+p[1],'r--',label='{:1.2f}'.format(p[0]))
plt.plot(x,x*2*(2-(nu+1))/(nu+1-1),'--',label=r'$2(2-\nu)/(\nu-1)={:1.2f}$'.format(2*(2-(nu+1))/(nu+1-1)))
plt.legend()
plt.savefig(dir_out+'logNsigma2rho.pdf',bbox_inches='tight')
#%%% * <sigma C | theta> for several times
plt.style.use('~/.config/matplotlib/joris.mplstyle')

cbin=np.unique(np.logspace(-5,0.,30))
from scipy import spatial ,ndimage
#x0=np.array([0.32,0.18,0.95])
x0=np.random.rand(1)
var_sa=0
sigma=30
s0=0.05
D=(sigma/n)**2/2
Tvec=np.arange(1,16,1)
plt.figure(figsize=(2,2))
cbin=np.unique(np.logspace(-5,0.,20))
ft=[]
for i,T in enumerate(Tvec):
	a=0.35
	C=DNS_n(2**13,x0,T,a,var_sa)
	x,S,wrapped_time=lagrangian(x0,T,a,var_sa)
	Cmax=1./np.sqrt(1.+4*D/s0**2*wrapped_time)
	Si=s0*np.sqrt(1.+4*D/s0**2*wrapped_time)*S
	sB=np.mean(Si)
	# sB=2.2*np.mean(Si)
	# Lagrangian
	tree=spatial.cKDTree(x.reshape(-1,1))
	# # Take samples in within cmax
	npts=1000
	idpts=np.uint16(np.linspace(0,len(x)-1,npts))
	neighboors=tree.query_ball_point(x[idpts].reshape(-1,1), sB/2.)
	cm=[np.mean(Cmax[n]) for n in neighboors] # is subjected to noise! 
	# cm2=np.array([np.sum(s0*S[n]/sB*np.sqrt(np.pi)) for n in neighboors])
	# nagg_lag=np.array([len(n) for n in neighboors])
# 	ceul=np.interp(x,np.linspace(0,1,C.shape[1]),C.flatten())
# 	cbin=np.unique(np.logspace(-5,0,30))
# 	ceulbin=bin_operation(Cmax,ceul,cbin,np.mean)
	ceul=np.interp(x[idpts],np.linspace(0,1,C.shape[1]),C.flatten())
	ceulbin=bin_operation(cm,ceul,cbin,np.mean)
	stdceulbin=bin_operation(cm,ceul,cbin,np.std)
	plt.plot(cbin[:-1]/np.mean(C),stdceulbin,'o-',c=plt.cm.cool(float(i)/len(Tvec)))
	ft.append([stdceulbin[stdceulbin>0].mean(),np.std(C),np.std(cm)])
# for i,T in enumerate(Tvec):
# 	a=0.4
# 	C=DNS_n(2**13,x0,T,a,var_sa)
# 	x,S,wrapped_time=lagrangian(x0,T,a,var_sa)
# 	Cmax=1./np.sqrt(1.+4*D/s0**2*wrapped_time)
# 	Si=s0*np.sqrt(1.+4*D/s0**2*wrapped_time)*S
# 	sB=np.mean(Si)
# 	# sB=2.2*np.mean(Si)
# 	# Lagrangian
# 	tree=spatial.cKDTree(x.reshape(-1,1))
# 	# # Take samples in within cmax
# 	cbin=np.unique(np.logspace(-5,0.,30))
# 	npts=1000
# 	idpts=np.uint16(np.linspace(0,len(x)-1,npts))
# 	neighboors=tree.query_ball_point(x[idpts].reshape(-1,1), sB/2.)
# 	cm=[np.mean(Cmax[n]) for n in neighboors] # is subjected to noise! 
# 	# cm2=np.array([np.sum(s0*S[n]/sB*np.sqrt(np.pi)) for n in neighboors])
# 	# nagg_lag=np.array([len(n) for n in neighboors])
# # 	ceul=np.interp(x,np.linspace(0,1,C.shape[1]),C.flatten())
# # 	ceulbin=bin_operation(Cmax,ceul,cbin,np.mean)
# 	ceul=np.interp(x[idpts],np.linspace(0,1,C.shape[1]),C.flatten())
# 	ceulbin=bin_operation(cm,ceul,cbin,np.mean)
# 	stdceulbin=bin_operation(cm,ceul,cbin,np.std)
# 	plt.plot(cbin[:-1]/np.mean(C),stdceulbin/ceulbin,'s',c=plt.cm.cool(float(i)/len(Tvec)))

plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\langle \theta \rangle_{s_B}/\langle c\rangle$')
plt.ylabel(r'$\sigma_{c|\theta}$')
plt.xlim([1e-4,1e1])
plt.plot(cbin/np.mean(C),np.zeros(cbin.shape)+0.2,'k-')
plt.xticks([1e-3,1e-2,1e-1,1e0,10])
plt.ylim([1e-5,1])
#plt.plot(cbin/np.mean(C),(cbin/C.mean()+cbin*(1/cbin-1)),'k--')
plt.savefig('/home/joris/Dropbox/Articles/AggregationBatchelor/Figures/sigmaccmax_baker.pdf',bbox_inches='tight')

ft=np.array(ft)
plt.figure()
plt.plot(Tvec,ft[:,0],'o-')
plt.plot(Tvec,ft[:,1],'s-')
plt.plot(Tvec,ft[:,2],'*-')
plt.yscale('log')
#%%% * sigma C | n> for several times
plt.style.use('~/.config/matplotlib/joris.mplstyle')

cbin=np.unique(np.logspace(-5,0.,30))
from scipy import spatial ,ndimage
#x0=np.array([0.32,0.18,0.95])
x0=np.random.rand(1)
var_sa=0
sigma=50
s0=0.05
D=(sigma/n)**2/2
Tvec=np.arange(1,12)
plt.figure(figsize=(2,2))
nbin=np.unique(np.floor(np.logspace(0,3,30)))
for i,T in enumerate(Tvec):
	a=0.35
	C=DNS_n(2**13,x0,T,a,var_sa)
	x,S,wrapped_time=lagrangian(x0,T,a,var_sa)
	Cmax=1./np.sqrt(1.+4*D/s0**2*wrapped_time)
	Si=s0*np.sqrt(1.+4*D/s0**2*wrapped_time)*S
	sB=np.mean(Si)
	# sB=2.2*np.mean(Si)
	# Lagrangian
	tree=spatial.cKDTree(x.reshape(-1,1))
	# # Take samples in within cmax
	npts=1000
	idpts=np.uint16(np.linspace(0,len(x)-1,npts))
	neighboors=tree.query_ball_point(x[idpts].reshape(-1,1), sB/2.)
	cm=[np.mean(Cmax[n]) for n in neighboors] # is subjected to noise! 
	# cm2=np.array([np.sum(s0*S[n]/sB*np.sqrt(np.pi)) for n in neighboors])
	nagg_lag=np.array([len(n) for n in neighboors])
	ceul=np.interp(x[idpts],np.linspace(0,1,C.shape[1]),C.flatten())
# 	cbin=np.unique(np.logspace(-5,0,30))
# 	ceulbin=bin_operation(Cmax,ceul,cbin,np.mean)
	ceulbin=bin_operation(nagg_lag,ceul,nbin,np.mean)
	stdceulbin=bin_operation(nagg_lag,ceul,nbin,np.std)
	plt.plot(nbin[:-1],stdceulbin,'o-',c=plt.cm.cool(float(i)/len(Tvec)))

# for i,T in enumerate(Tvec):
# 	a=0.1
# 	C=DNS_n(2**13,x0,T,a,var_sa)
# 	x,S,wrapped_time=lagrangian(x0,T,a,var_sa)
# 	Cmax=1./np.sqrt(1.+4*D/s0**2*wrapped_time)
# 	Si=s0*np.sqrt(1.+4*D/s0**2*wrapped_time)*S
# 	sB=np.mean(Si)
# 	# sB=2.2*np.mean(Si)
# 	# Lagrangian
# 	tree=spatial.cKDTree(x.reshape(-1,1))
# 	# # Take samples in within cmax
# 	cbin=np.unique(np.logspace(-5,0.,30))
# 	npts=1000
# 	idpts=np.uint16(np.linspace(0,len(x)-1,npts))
# 	neighboors=tree.query_ball_point(x[idpts].reshape(-1,1), sB/2.)
# 	cm=[np.mean(Cmax[n]) for n in neighboors] # is subjected to noise! 
# 	nagg_lag=np.array([len(n) for n in neighboors])
# 	ceul=np.interp(x[idpts],np.linspace(0,1,C.shape[1]),C.flatten())
# # 	cbin=np.unique(np.logspace(-5,0,30))
# # 	ceulbin=bin_operation(Cmax,ceul,cbin,np.mean)
# 	ceulbin=bin_operation(nagg_lag,ceul,nbin,np.mean)
# 	stdceulbin=bin_operation(nagg_lag,ceul,nbin,np.std)
# 	plt.plot(nbin[:-1],stdceulbin/ceulbin,'o',c=plt.cm.cool(float(i)/len(Tvec)))

plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$n$')
plt.ylabel(r'$\sigma_{c|n}$')
#plt.xlim([1e-3,1e1])
#plt.plot(cbin/np.mean(C),np.zeros(cbin.shape)+0.2,'k-')
#plt.xticks([1e-3,1e-2,1e-1,1e0,10])
plt.ylim([1e-5,1])
#plt.plot(cbin/np.mean(C),(cbin/C.mean()+cbin*(1/cbin-1)),'k--')
plt.savefig('/home/joris/Dropbox/Articles/AggregationBatchelor/Figures/sigmacN_baker.pdf',bbox_inches='tight')

#%%% var(cb) for several times
#2021-09-03_12:05:38
from scipy import spatial
plt.style.use('~/.config/matplotlib/joris.mplstyle')
import scipy
x0=np.array([0.1546],dtype=np.float64)
#A=np.logspace(-7,np.log10(0.49),100)
var_sa=0
q=2
plt.figure(figsize=(5,5))
s0=0.05
sigma=20
n=2**13
D=(sigma/n)**2/2

#for T in range(5,17):
for T in [8,12,16]:
#	a=np.array([0.05,0.1,0.35])
#	x,S,wrapped_time=lagrangian_nfold(x0,T,a,var_sa)
	a=0.49
	a=np.array([a,1-a])
	x,S,wrapped_time=lagrangian_nfold(x0,T,a,var_sa)
	Cmax=1./np.sqrt(1.+4*D/s0**2*wrapped_time)
	Si=s0*np.sqrt(1.+4*D/s0**2*wrapped_time)*S
	sB=np.mean(Si)
	#sB=2.2*np.mean(Si)
	
	# Lagrangian
	tree=spatial.cKDTree(x.reshape(-1,1))
	# Take samples in within cmax
	neighboors=tree.query_ball_point(x.reshape(-1,1)[np.abs(x-0.5)<0.5-2*sB], sB)
	cm=[np.sum(Cmax[n]*np.mean(Si)/sB*np.sqrt(np.pi)) for n in neighboors] # is subjected to noise! 
	cm2=np.array([np.sum(s0*S[n]/sB*np.sqrt(np.pi)) for n in neighboors])
	nagg_lag=np.array([len(n) for n in neighboors])
	var_1_rho=[np.mean(S[n]**2.) for n in neighboors]
	k_1_rho=[np.mean(S[n]**3.) for n in neighboors]
	max_rho=[np.max(S[n]) for n in neighboors]
	mean_1_rho=[np.mean(S[n]) for n in neighboors]
	
	var2=cm2**2.-np.mean(cm2)**2.
	
	bins=np.arange(0,1,sB)
	hn,xb=np.histogram(x,bins)
	hS2,xb=np.histogram(x,bins,weights=S**2.)
	hSc,xb=np.histogram(x,bins,weights=S)
	
	#plt.figure()
	plt.plot(nagg_lag,var2,'*',label=r'$(c_B^2-<c_B>^2)$,'+' $t={:1.0f}$'.format(T))
	#plt.scatter(nagg_lag,var_1_rho,c=cm2,alpha=0.5,
	#					 label=r'$\sigma^2_{1/\rho}$')
	#plt.scatter(hn,hS2/hn,c=np.abs(hSc-np.mean(hSc)),alpha=1.,
#+
#						 label=r'$\sigma^2_{1/\rho}$')
	#plt.scatter(hn,hS2/hn,c=np.arange(len(hn)),alpha=0.5)
	#plt.plot(nagg_lag,mean_1_rho,'o',label=r'$\mu_{1/\rho}$')
	#plt.scatter(nagg_lag,mean_1_rho,c=cm2,alpha=0.5,
	#					 label=r'$\mu_{1/\rho}$')
N=np.array([1,1000])
plt.plot(N,1e-2/N,'k--',label='$N^{-1}$')
plt.plot(N,1e-2/N**2.,'k--',label='$N^{-2}$')
plt.yscale('log')
plt.xscale('log')
plt.xlabel(r'$N$')
plt.legend()
	#plt.xlim([1,1e4])
	#plt.ylim([1e-10,1e-4])
plt.text(2,1e-8,r'$t={:d}$'.format(T),color='k')
plt.savefig(dir_out+'/sigma_1_rho,t={:02d}.png'.format(T),bbox_inches='tight')
#%%% max(1/rho) for several times
#2021-09-03_12:05:38
from scipy import spatial
plt.style.use('~/.config/matplotlib/joris.mplstyle')
import scipy
x0=np.array([0.45],dtype=np.float64)
#A=np.logspace(-7,np.log10(0.49),100)
var_sa=0
q=2
plt.figure(figsize=(5,5))
s0=0.05
sigma=20
n=2**13
D=(sigma/n)**2/2

#for T in range(5,17):
for T in [8,12,16]:
#	a=np.array([0.05,0.1,0.35])
#	x,S,wrapped_time=lagrangian_nfold(x0,T,a,var_sa)
	a=0.4
	a=np.array([a,1-a])
	x,S,wrapped_time=lagrangian_nfold(x0,T,a,var_sa)
	Cmax=1./np.sqrt(1.+4*D/s0**2*wrapped_time)
	Si=s0*np.sqrt(1.+4*D/s0**2*wrapped_time)*S
	sB=np.mean(Si)
	#sB=2.2*np.mean(Si)
	
	# Lagrangian
	tree=spatial.cKDTree(x.reshape(-1,1))
	# Take samples in within cmax
	neighboors=tree.query_ball_point(x.reshape(-1,1)[np.abs(x-0.5)<0.5-2*sB], sB)
	cm=[np.sum(Cmax[n]*np.mean(Si)/sB*np.sqrt(np.pi)) for n in neighboors] # is subjected to noise! 
	cm2=np.array([np.sum(s0*S[n]/sB*np.sqrt(np.pi)) for n in neighboors])
	nagg_lag=np.array([len(n) for n in neighboors])
	var_1_rho=[np.mean(S[n]**2.) for n in neighboors]
	k_1_rho=[np.mean(S[n]**3.) for n in neighboors]
	max_rho=[np.max(S[n]) for n in neighboors]
	mean_1_rho=[np.mean(S[n]) for n in neighboors]
	
	bins=np.arange(0,1,sB)
	hn,xb=np.histogram(x,bins)
	hS2,xb=np.histogram(x,bins,weights=S**2.)
	hSc,xb=np.histogram(x,bins,weights=S)
	
	#plt.figure()
	plt.plot(nagg_lag,max_rho,'*-',label=r'$\max({1/\rho})$,'+' $t={:1.0f}$'.format(T))
	#plt.scatter(nagg_lag,var_1_rho,c=cm2,alpha=0.5,
	#					 label=r'$\sigma^2_{1/\rho}$')
	#plt.scatter(hn,hS2/hn,c=np.abs(hSc-np.mean(hSc)),alpha=1.,
#+
#						 label=r'$\sigma^2_{1/\rho}$')
	#plt.scatter(hn,hS2/hn,c=np.arange(len(hn)),alpha=0.5)
	#plt.plot(nagg_lag,mean_1_rho,'o',label=r'$\mu_{1/\rho}$')
	#plt.scatter(nagg_lag,mean_1_rho,c=cm2,alpha=0.5,
	#					 label=r'$\mu_{1/\rho}$')
N=np.array([1,1000])
plt.plot(N,1e-2/N,'k--',label='$N^{-1}$')
plt.plot(N,1e-2/N**2.,'k--',label='$N^{-2}$')
plt.yscale('log')
plt.xscale('log')
plt.xlabel(r'$N$')
plt.legend()
	#plt.xlim([1,1e4])
	#plt.ylim([1e-10,1e-4])
plt.text(2,1e-8,r'$t={:d}$'.format(T),color='k')
plt.savefig(dir_out+'/sigma_1_rho,t={:02d}.png'.format(T),bbox_inches='tight')
#%%% var(1/rho)  for several times Eulerian
#2021-09-03_12:05:38
from scipy import spatial
plt.style.use('~/.config/matplotlib/joris.mplstyle')
import scipy
x0=np.array([0.1546],dtype=np.float64)
#A=np.logspace(-7,np.log10(0.49),100)
var_sa=0
q=2
plt.figure(figsize=(5,5))
s0=0.05
sigma=100
n=2**13
D=(sigma/n)**2/2
shuffle=False
#for T in range(5,17):
nufit=[]
Varreconstruct=[]
for T in [15,16,17,18,19,20,21,22,23]:
#	a=np.array([0.05,0.1,0.35])
#	x,S,wrapped_time=lagrangian_nfold(x0,T,a,var_sa)
	a=0.2
	a=np.array([a,1-a])
	x,S,wrapped_time=lagrangian_nfold(x0,T,a,var_sa)
	Cmax=1./np.sqrt(1.+4*D/s0**2*wrapped_time)
	Si=s0*np.sqrt(1.+4*D/s0**2*wrapped_time)*S
	sB=np.mean(Si)
	#sB=2.2*np.mean(Si)
	
	# Lagrangian
# 	tree=spatial.cKDTree(x.reshape(-1,1))
# 	# Take samples in within cmax
# 	neighboors=tree.query_ball_point(x.reshape(-1,1)[np.abs(x-0.5)<0.5-2*sB], sB/2)
# 	cm=[np.sum(Cmax[n]*np.mean(Si)/sB*np.sqrt(np.pi)) for n in neighboors] # is subjected to noise! 
# 	cm2=np.array([np.sum(s0*S[n]/sB*np.sqrt(np.pi)) for n in neighboors])
# 	nagg_lag=np.array([len(n) for n in neighboors])
# 	var_1_rho=[np.mean(S[n]**2.) for n in neighboors]
# 	k_1_rho=[np.mean(S[n]**3.) for n in neighboors]
# 	mean_1_rho=[np.mean(S[n]) for n in neighboors]
# 	
	# Grid based
	bins=np.arange(0,1,sB)
	hn,xb=np.histogram(x,bins)
	hS2,xb=np.histogram(x,bins,weights=S**2.)
	hSc,xb=np.histogram(x,bins,weights=S)
	V=hS2/hn-(hSc/hn)**2.
	#plt.figure()
	hC,xb=np.histogram(x,bins,weights=Cmax)
	Varreconstruct.append(np.var(hC))
	#plt.plot(nagg_lag,var_1_rho,'*',label=r'$\sigma^2_{1/\rho}$')
	#plt.scatter(nagg_lag,var_1_rho,c=cm2,alpha=0.5,
	#					 label=r'$\sigma^2_{1/\rho}$')
	#plt.scatter(hn,hS2/hn,c=np.abs(hSc-np.mean(hSc)),alpha=1.,
	#					 label=r'$\sigma^2_{1/\rho}$')
	plt.plot(np.log(hn),np.log(V),'*',label=r'$\sigma^2_{1/\rho}$')
	#plt.scatter(hn,hS2/hn,c=np.arange(len(hn)),alpha=0.5)
	#plt.plot(nagg_lag,mean_1_rho,'o',label=r'$\mu_{1/\rho}$')
	#plt.scatter(nagg_lag,mean_1_rho,c=cm2,alpha=0.5,
	#					 label=r'$\mu_{1/\rho}$')
	n=np.exp(5)
	nufit.append(np.polyfit(np.log(hn[(V>0)&(hn>n)]),np.log(V[(V>0)&(hn>n)]),1))
	N=np.array([1,20])
	plt.plot(N,-13-N,'k-',label='$-1$')
	plt.plot(N,-13-2*N,'k--',label='$-2$')
	plt.plot(N,nufit[-1][0]*N+nufit[-1][1])

	plt.xlabel(r'$ \log N$')
#	plt.legend()
	#plt.xlim([1,1e4])
	#plt.ylim([1e-10,1e-4])
	plt.savefig(dir_out+'/sigma_1_rho,t={:02d}.png'.format(T),bbox_inches='tight')
plt.figure()
plt.plot(Varreconstruct)
#%%% var(1/rho)  for several times Eulerian
#2021-09-03_12:05:38
from scipy import spatial
plt.style.use('~/.config/matplotlib/joris.mplstyle')
import scipy
x0=np.array([0.1546],dtype=np.float64)
#A=np.logspace(-7,np.log10(0.49),100)
var_sa=0
q=2
plt.figure(figsize=(5,5))
s0=0.05
sigma=100
n=2**13
D=(sigma/n)**2/2
shuffle=False
#for T in range(5,17):
nufit=[]
Varreconstruct=[]
TT=np.arange(2,20,1)
for T in TT:
#	a=np.array([0.05,0.1,0.35])
#	x,S,wrapped_time=lagrangian_nfold(x0,T,a,var_sa)
	a=0.4
	a=np.array([a,1-a])
	x,S,wrapped_time=lagrangian_nfold(x0,T,a,var_sa)
	Cmax=1./np.sqrt(1.+4*D/s0**2*wrapped_time)
	Si=s0*np.sqrt(1.+4*D/s0**2*wrapped_time)*S
#	sB=np.mean(Si)
	sB=0.02
	#sB=2.2*np.mean(Si)
	
	# Lagrangian
# 	tree=spatial.cKDTree(x.reshape(-1,1))
# 	# Take samples in within cmax
# 	neighboors=tree.query_ball_point(x.reshape(-1,1)[np.abs(x-0.5)<0.5-2*sB], sB/2)
# 	cm=[np.sum(Cmax[n]*np.mean(Si)/sB*np.sqrt(np.pi)) for n in neighboors] # is subjected to noise! 
# 	cm2=np.array([np.sum(s0*S[n]/sB*np.sqrt(np.pi)) for n in neighboors])
# 	nagg_lag=np.array([len(n) for n in neighboors])
# 	var_1_rho=[np.mean(S[n]**2.) for n in neighboors]
# 	k_1_rho=[np.mean(S[n]**3.) for n in neighboors]
# 	mean_1_rho=[np.mean(S[n]) for n in neighboors]
# 	
	# Grid based
	bins=np.arange(0,1,sB)
	hn,xb=np.histogram(x,bins)
	hS2,xb=np.histogram(x,bins,weights=S**2.)
	hSc,xb=np.histogram(x,bins,weights=S)
	V=hS2/hn-(hSc/hn)**2.
	#plt.figure()
	hC,xb=np.histogram(x,bins,weights=Cmax)
	Varreconstruct.append(np.var(hSc))
	#plt.plot(nagg_lag,var_1_rho,'*',label=r'$\sigma^2_{1/\rho}$')
	#plt.scatter(nagg_lag,var_1_rho,c=cm2,alpha=0.5,
	#					 label=r'$\sigma^2_{1/\rho}$')
	#plt.scatter(hn,hS2/hn,c=np.abs(hSc-np.mean(hSc)),alpha=1.,
	#					 label=r'$\sigma^2_{1/\rho}$')
	plt.plot(hSc,c=plt.cm.cool(T/25))


	plt.xlabel(r'$x$')
#	plt.legend()
	#plt.xlim([1,1e4])
	#plt.ylim([1e-10,1e-4])
	plt.savefig(dir_out+'/sigma_1_rho,t={:02d}.png'.format(T),bbox_inches='tight')
plt.figure()
plt.plot(TT,Varreconstruct)
plt.yscale('log')
#%%% std(1/rho)/mean(1/rho) versus N   Eulerian
#2021-09-03_12:05:38
from scipy import spatial
plt.style.use('~/.config/matplotlib/joris.mplstyle')
import scipy
x0=np.array([0.1546],dtype=np.float64)
#A=np.logspace(-7,np.log10(0.49),100)
var_sa=0
q=2
plt.figure(figsize=(5,5))
s0=0.05
sigma=20
n=2**13
D=(sigma/n)**2/2
shuffle=False
#for T in range(5,17):
for T in [10]:
#	a=np.array([0.05,0.1,0.35])
#	x,S,wrapped_time=lagrangian_nfold(x0,T,a,var_sa)
	a=0.1
	a=np.array([a,1-a])
	x,S,wrapped_time=lagrangian_nfold(x0,T,a,var_sa)
	Cmax=1./np.sqrt(1.+4*D/s0**2*wrapped_time)
	Si=s0*np.sqrt(1.+4*D/s0**2*wrapped_time)*S
	sB=np.mean(Si)
	#sB=2.2*np.mean(Si)
	
	# Lagrangian
	tree=spatial.cKDTree(x.reshape(-1,1))
	# Take samples in within cmax
	neighboors=tree.query_ball_point(x.reshape(-1,1)[np.abs(x-0.5)<0.5-2*sB], sB)
	cm=[np.sum(Cmax[n]*np.mean(Si)/sB*np.sqrt(np.pi)) for n in neighboors] # is subjected to noise! 
	cm2=np.array([np.sum(s0*S[n]/sB*np.sqrt(np.pi)) for n in neighboors])
	nagg_lag=np.array([len(n) for n in neighboors])
	var_1_rho=[np.mean(S[n]**2.) for n in neighboors]
	k_1_rho=[np.mean(S[n]**3.) for n in neighboors]
	mean_1_rho=[np.mean(S[n]) for n in neighboors]
	
	bins=np.arange(0,1,sB)
	hn,xb=np.histogram(x,bins)
	hS2,xb=np.histogram(x,bins,weights=S**2.)
	hSc,xb=np.histogram(x,bins,weights=S)
	
	plt.figure()
	#plt.plot(nagg_lag,var_1_rho,'*',label=r'$\sigma^2_{1/\rho}$')
	#plt.scatter(nagg_lag,var_1_rho,c=cm2,alpha=0.5,
	#					 label=r'$\sigma^2_{1/\rho}$')
	#plt.scatter(hn,hS2/hn,c=np.abs(hSc-np.mean(hSc)),alpha=1.,
	#					 label=r'$\sigma^2_{1/\rho}$')
#	plt.plot(hn,hS2/hn-(hSc/hn)**2.,'*',
#						 label=r'$\sigma^2_{1/\rho}$')
	plt.plot(hn,(hS2/hn-(hSc/hn)**2.)**0.5/(hSc/hn),'*',
						 label=r'$\sigma_{1/\rho}/<1/\rho>$')
	#plt.scatter(hn,hS2/hn,c=np.arange(len(hn)),alpha=0.5)
	#plt.plot(nagg_lag,mean_1_rho,'o',label=r'$\mu_{1/\rho}$')
	#plt.scatter(nagg_lag,mean_1_rho,c=cm2,alpha=0.5,
	#					 label=r'$\mu_{1/\rho}$')
	N=np.array([1,1000])
	plt.plot(N,1e-0*N**0.5,'k--',label='$N^{1/2}$')
	plt.yscale('log')
	plt.xscale('log')
	plt.xlabel(r'$N$')
	plt.legend()
	#plt.xlim([1,1e4])
	plt.ylim([1e-1,1e2])
	plt.text(2,1e-0,r'$t={:d}$'.format(T),color='k')
	plt.savefig(dir_out+'/sigma_1_rho,t={:02d}.png'.format(T),bbox_inches='tight')
#%%% var(1/rho) * N for several times Lagrangian
#2021-09-03_12:05:38
from scipy import spatial
plt.style.use('~/.config/matplotlib/joris.mplstyle')
import scipy
x0=np.array([0.1546],dtype=np.float64)
#A=np.logspace(-7,np.log10(0.49),100)
var_sa=0
q=2
plt.figure(figsize=(5,5))
s0=0.05
sigma=20
n=2**13
D=(sigma/n)**2/2

for T in range(5,17):
	#a=np.array([0.1,0.4,0.35])
	#x,S,wrapped_time=lagrangian_nfold(x0,T,a,var_sa)
	a=np.array([0.3])
	x,S,wrapped_time=lagrangian(x0,T,a,var_sa)
	Cmax=1./np.sqrt(1.+4*D/s0**2*wrapped_time)
	Si=s0*np.sqrt(1.+4*D/s0**2*wrapped_time)*S
	sB=np.mean(Si)
	#sB=2.2*np.mean(Si)
	# Lagrangian
	tree=spatial.cKDTree(x.reshape(-1,1))
	# Take samples in within cmax
	neighboors=tree.query_ball_point(x.reshape(-1,1)[np.abs(x-0.5)<0.5-2*sB], sB)
	cm=[np.sum(Cmax[n]*np.mean(Si)/sB*np.sqrt(np.pi)) for n in neighboors] # is subjected to noise! 
	cm2=np.array([np.sum(s0*S[n]/sB*np.sqrt(np.pi)) for n in neighboors])
	nagg_lag=np.array([len(n) for n in neighboors])
	var_1_rho=[np.var(S[n]) for n in neighboors]
	mean_1_rho=[np.mean(S[n]) for n in neighboors]
#	plt.figure()
	plt.plot(nagg_lag,var_1_rho,'*',color=plt.cm.jet(T/15))
	plt.plot(nagg_lag,mean_1_rho,'o',color=plt.cm.jet(T/15))
	N=np.array([1,1000])
	plt.plot(N,1e-5/N,'k--')
	plt.yscale('log')
	plt.xscale('log')
	plt.xlabel(r'$N$')
	plt.ylabel(r'$\sigma^2_{1/\rho}$')
	plt.legend()
	plt.xlim([1,1e4])
	plt.ylim([1e-10,1e-1])
#plt.text(2,1e-8,r'$t={:d}$'.format(T),color='k')


#%%% sigma^2_n for several times Eulerian
plt.style.use('~/.config/matplotlib/joris.mplstyle')
import scipy.spatial as spatial
import scipy.optimize
plt.style.use('~/.config/matplotlib/joris.mplstyle')
import scipy
x0=np.array([0.1546],dtype=np.float64)
#A=np.logspace(-7,np.log10(0.49),100)
var_sa=0
q=2
plt.figure(figsize=(2,2))

SB=1/np.array([20,50,200,500])
for j,sB in enumerate(SB):
	
	a=np.array([0.2])
	SA=np.linspace(0.05,0.49,50)
	Mean=[]
	T=20
	
	
	
	for i,a in enumerate(SA):
		print(a)
		x,S,wrapped_time=lagrangian(x0,T,a,var_sa)
		# Grid estimates
		NH=np.histogram(x,np.arange(0,1+sB,sB),density=False)[0]
		pi=NH/np.sum(NH)
		Mean.append([np.sum(pi**2.),np.var(NH/np.mean(NH))])
		
	
	dq=2.0
	D2=[]
	for a in SA:
		f = lambda Dq : dq*np.log(2)-np.log(1/a**((dq-1)*Dq)+1/(1-a)**((dq-1)*Dq))
		D2.append(scipy.optimize.broyden1(f, 1, f_tol=1e-14)+1)
	D2=np.array(D2)
	dq=1
	D1=(1+dq)*np.log(2)/np.log(1/SA**dq+1/(1-SA)**dq)+1
	
	Mean=np.array(Mean)
	#plt.plot(SA,Mean[:,0]/sB-1,'o',label=r'$ \sum_i (n_i/\sum_i n_i)^2$')
	plt.plot(SA,Mean[:,1],'o',color=plt.cm.cool(j/len(SB)))
	plt.plot(SA,sB**(D2-2)-1,'-',color=plt.cm.cool(j/len(SB)))
	
plt.plot([],[],'k-',label=r'$s_B^{D_2-2}-1$')
plt.plot([],[],'ko',label=r'$ \sigma^2_{n/<n>} $')
plt.plot([],[],'o',label=r'$ s_B=1/20 $',color=plt.cm.cool(0/4))
plt.plot([],[],'o',label=r'$ s_B=1/50 $',color=plt.cm.cool(1/4))
plt.plot([],[],'o',label=r'$ s_B=1/200 $',color=plt.cm.cool(2/4))
plt.plot([],[],'o',label=r'$ s_B=1/500 $',color=plt.cm.cool(3/4))

l,s=lyapunov(a)
plt.xlabel('$a$')
plt.ylabel('Moments of $n$')
plt.legend(loc=3,fontsize=6)
plt.yscale('log')
subscript(plt.gca(),0)
plt.savefig('/home/joris/Dropbox/Articles/AggregationBatchelor/Figures/sigma2_n_baker.pdf',bbox_inches='tight')
#%%% <var(log rho) | n>n for several times Eulerian
from scipy import spatial
plt.style.use('~/.config/matplotlib/joris.mplstyle')
import scipy
x0=np.array([0.1546],dtype=np.float64)
#A=np.logspace(-7,np.log10(0.49),100)
var_sa=0
q=2
plt.figure(figsize=(5,5))

sB=1/100
a=np.array([0.4])

Mean=[]
TT=np.arange(5,24,4)

fig,ax=plt.subplots(3,len(TT),sharex=True,sharey=True)
for i,T in enumerate(TT):
	x,S,wrapped_time=lagrangian(x0,T,a,var_sa)
	# Grid estimates
	NH=np.histogram(x,np.arange(0,1+sB,sB),density=False)[0]
	logrho=np.histogram(x,np.arange(0,1+sB,sB),density=False,weights=-np.log(S))[0]/NH
	log2rho=np.histogram(x,np.arange(0,1+sB,sB),density=False,weights=np.log(S)**2)[0]/NH
	sigma2logrho=log2rho-logrho**2.
	ax[2][i].plot(sigma2logrho,color=plt.cm.cool(T/TT.max()))
	ax[0][i].plot(np.log(NH),color=plt.cm.cool(T/TT.max()))
	ax[1][i].plot(logrho,color=plt.cm.cool(T/TT.max()))
	Mean.append([np.nanmean(logrho),np.nanmean(sigma2logrho),np.nanmean(np.log(NH[NH>0])),np.nanvar(np.log(NH[NH>0]))])

ax[0][0].set_ylim([-1,20])
ax[1,0].set_ylabel(r'$\mu_{\log \rho | n}$')
ax[0,0].set_ylabel(r'$\log n$')
ax[2,0].set_ylabel(r'$\sigma^2_{\log \rho|n}$')
[ax[2,i].set_xlabel(r'$x/s_B$') for i in range(ax.shape[1])]

plt.figure()
Mean=np.array(Mean)
plt.plot(TT,Mean[:,0],'*-',label=r'$ <\mu_{\log \rho | n}>_n$')
plt.plot(TT,Mean[:,1],'o-',label=r'$ <\sigma^2_{\log \rho | n }>_n$')
plt.plot(TT,Mean[:,2],'+-',label=r'$ \mu_{\log n}$')
plt.plot(TT,Mean[:,3],'+-',label=r'$ \sigma^2_{\log n}$')

l,s=lyapunov(a)
plt.xlabel('t')
plt.legend()
#%%% * <var(1/rho) * N> for several times Lagrangian
#2021-09-03_12:05:38
from scipy import spatial
plt.style.use('~/.config/matplotlib/joris.mplstyle')
import scipy
x0=np.array([0.1546],dtype=np.float64)
#A=np.logspace(-7,np.log10(0.49),100)
var_sa=0
q=2
plt.figure(figsize=(5,5))
s0=0.05
sigma=20
n=2**13
D=(sigma/n)**2/2
Tv=np.arange(1,25,2)
A=np.linspace(0.01,0.49,10)
VMa=[]
for a in A:
	print(a)
	VM=[]
	for T in Tv:
		#a=np.array([0.1,0.4,0.35])
		#x,S,wrapped_time=lagrangian_nfold(x0,T,a,var_sa)
		x,S,wrapped_time=lagrangian(x0,T,a,var_sa)
		Cmax=1./np.sqrt(1.+4*D/s0**2*wrapped_time)
		Si=s0*np.sqrt(1.+4*D/s0**2*wrapped_time)*S
		sB=np.mean(Si)
		#sB=2.2*np.mean(Si)
		# Lagrangian
	#	tree=spatial.cKDTree(x.reshape(-1,1))
		# Take samples in within cmax
	#	neighboors=tree.query_ball_point(x.reshape(-1,1)[np.abs(x-0.5)<0.5-2*sB], sB)
	#	cm=[np.sum(Cmax[n]*np.mean(Si)/sB*np.sqrt(np.pi)) for n in neighboors] # is subjected to noise! 
	#	cm2=np.array([np.sum(s0*S[n]/sB*np.sqrt(np.pi)) for n in neighboors])
	#	nagg_lag=np.array([len(n) for n in neighboors])
	#	var_1_rho=[np.var(S[n]) for n in neighboors]
	#	mean_1_rho=[np.mean(S[n]) for n in neighboors]
	# Eulerian
		bins=np.arange(0,1,sB)
		hn,xb=np.histogram(x,bins)
		hS2,xb=np.histogram(x,bins,weights=S**2.)
		hSc,xb=np.histogram(x,bins,weights=S)
		
		nagg_eul=hn
		mean_1_rho=hSc/hn
		var_1_rho=hS2/hn-mean_1_rho**2.
		
		
		VM.append(np.mean(hS2))
#		VM.append(np.mean(hS2[nagg_eul>1]))
#		VM.append(np.mean(var_1_rho[nagg_eul>1]*nagg_eul[nagg_eul>1]))
	#	plt.figure()
		plt.plot(hn,var_1_rho,'*',color=plt.cm.jet(T/Tv.max()))
	##	plt.plot(nagg_lag,mean_1_rho,'o',color=plt.cm.jet(T/15))
	#	N=np.array([1,1000])
	#	plt.plot(N,1e-5/N,'k--')
		plt.yscale('log')
		plt.xscale('log')
	#	plt.xlabel(r'$N$')
	#	plt.ylabel(r'$\sigma^2_{1/\rho} n$')
	#	plt.legend()
	#	plt.xlim([1,1e4])
	#	plt.ylim([1e-10,1e-1])
	#plt.text(2,1e-8,r'$t={:d}$'.format(T),color='k')
	#plt.savefig(dir_out+'/sigma_1_rho,a={:1.1f}.png'.format(a[0]),bbox_inches='tight')
	VMa.append(VM)

plt.style.use('~/.config/matplotlib/joris.mplstyle')
plt.figure(figsize=(3,2))
[plt.plot(Tv,v,'-',color=plt.cm.cool(A[i]*2.),label='$a={:1.2f}$'.format(A[i])) for i,v in enumerate(VMa)]
plt.yscale('log')
plt.xlabel(r'$t$')
plt.ylabel(r'$\delta^2(t) = \langle \langle \theta^2 \rangle_\mathcal{B} \, n(x) \rangle$')
#plt.plot(Tv,np.exp(-Tv*np.log(2)))
plt.legend(ncol=2,fontsize=6)
plt.savefig('/home/joris/Dropbox/Articles/AggregationBatchelor/Figures/delta_baker.pdf',bbox_inches='tight')

alpha=[]
for i,a in enumerate(A):
	alpha.append(np.polyfit(Tv[-5:],np.log(VMa[i][-5:]),1)[0])

plt.figure(figsize=(3,2))
plt.plot(A,alpha,'o-')
plt.xlabel(r'$a$')
plt.ylabel(r'$\log \delta^2(t) /t$')
plt.savefig('/home/joris/Dropbox/Articles/AggregationBatchelor/Figures/logdelta_t_baker.pdf',bbox_inches='tight')


lyap,sigma2=lyapunov(A)
plt.figure(figsize=(3,2))
plt.plot(sigma2/lyap,alpha,'o-')
plt.xlabel(r'$\sigma^2/\mu$')
plt.ylabel(r'$\log \delta^2(t) /t$')
#%%% * <var(1/rho) * n>_B / N <var(1/rho)> 
#2021-09-03_12:05:38
from scipy import spatial
plt.style.use('~/.config/matplotlib/joris.mplstyle')
import scipy
x0=np.array([0.1546],dtype=np.float64)
#A=np.logspace(-7,np.log10(0.49),100)
var_sa=0
q=2
plt.figure(figsize=(5,5))
s0=0.05
sigma=20
n=2**13
D=(sigma/n)**2/2
Tv=np.arange(10,30,5)
A=np.linspace(0.01,0.49,10)
VMa=[]
for a in A:
	print(a)
	VM=[]
	for T in Tv:
		#a=np.array([0.1,0.4,0.35])
		#x,S,wrapped_time=lagrangian_nfold(x0,T,a,var_sa)
		x,S,wrapped_time=lagrangian(x0,T,a,var_sa)
		Cmax=1./np.sqrt(1.+4*D/s0**2*wrapped_time)
		Si=s0*np.sqrt(1.+4*D/s0**2*wrapped_time)*S
		sB=np.mean(Si)
		#sB=2.2*np.mean(Si)
		# Lagrangian
	#	tree=spatial.cKDTree(x.reshape(-1,1))
		# Take samples in within cmax
	#	neighboors=tree.query_ball_point(x.reshape(-1,1)[np.abs(x-0.5)<0.5-2*sB], sB)
	#	cm=[np.sum(Cmax[n]*np.mean(Si)/sB*np.sqrt(np.pi)) for n in neighboors] # is subjected to noise! 
	#	cm2=np.array([np.sum(s0*S[n]/sB*np.sqrt(np.pi)) for n in neighboors])
	#	nagg_lag=np.array([len(n) for n in neighboors])
	#	var_1_rho=[np.var(S[n]) for n in neighboors]
	#	mean_1_rho=[np.mean(S[n]) for n in neighboors]
	# Eulerian
		bins=np.arange(0,1,sB)
		hn,xb=np.histogram(x,bins)
		hS2,xb=np.histogram(x,bins,weights=S**2.)
		hSc,xb=np.histogram(x,bins,weights=S)
		nagg_eul=hn
		mean_1_rho=hSc/hn
		var_1_rho=hS2/hn-mean_1_rho**2.
		
		# Weighted
		var_1_rho_x_N=np.var(S)*np.mean(hn[hn>0])
		# Unweighted
#		var_1_rho_x_N=(np.average(S**2,weights=S)-np.average(S,weights=S)**2.)*np.mean(hn[hn>0])
		
		VM.append(np.nanmean(var_1_rho*hn)/var_1_rho_x_N)

	VMa.append(VM)

plt.style.use('~/.config/matplotlib/joris.mplstyle')
plt.figure(figsize=(3,2))
[plt.plot(Tv,v,'-',color=plt.cm.cool(A[i]*2.),label='$a={:1.2f}$'.format(A[i])) for i,v in enumerate(VMa)]
plt.yscale('log')
plt.xlabel(r'$t$')
plt.ylabel(r'$ \langle \theta^2 n \rangle_\mathcal{B} / \langle n \rangle_A \langle \theta^2 \rangle_L $')
#plt.plot(Tv,np.exp(-Tv*np.log(2)))
plt.legend(ncol=2,fontsize=6)
plt.savefig('/home/joris/Dropbox/Articles/AggregationBatchelor/Figures/delta_baker.pdf',bbox_inches='tight')

#%%% var(1/rho) * N for several times
#2021-09-03_12:05:38
from scipy import spatial
plt.style.use('~/.config/matplotlib/joris.mplstyle')
import scipy
x0=np.array([0.1546],dtype=np.float64)
#A=np.logspace(-7,np.log10(0.49),100)
var_sa=0
q=2
plt.figure(figsize=(5,5))
s0=0.05
sigma=20
n=2**13
D=(sigma/n)**2/2
R=[]
for T in range(5,17):
	#a=np.array([0.1,0.4,0.35])
	#x,S,wrapped_time=lagrangian_nfold(x0,T,a,var_sa)
	a=np.array([0.3])
	x,S,wrapped_time=lagrangian(x0,T,a,var_sa)
	Cmax=1./np.sqrt(1.+4*D/s0**2*wrapped_time)
	Si=s0*np.sqrt(1.+4*D/s0**2*wrapped_time)*S
	sB=np.mean(Si)
	#sB=2.2*np.mean(Si)
	# Lagrangian
	tree=spatial.cKDTree(x.reshape(-1,1))
	# Take samples in within cmax
	neighboors=tree.query_ball_point(x.reshape(-1,1)[np.abs(x-0.5)<0.5-2*sB], sB)
	cm=[np.sum(Cmax[n]*np.mean(Si)/sB*np.sqrt(np.pi)) for n in neighboors] # is subjected to noise! 
	cm2=np.array([np.sum(s0*S[n]/sB*np.sqrt(np.pi)) for n in neighboors])
	nagg_lag=np.array([len(n) for n in neighboors])
	var_1_rho=[np.var(S[n]) for n in neighboors]
	mean_1_rho=[np.mean(S[n]) for n in neighboors]
#	plt.figure()
	R.append(np.mean(var_1_rho*nagg_lag))
#plt.text(2,1e-8,r'$t={:d}$'.format(T),color='k')
#plt.savefig(dir_out+'/sigma_1_rho,a={:1.1f}.png'.format(a[0]),bbox_inches='tight')

plt.plot(R,'*')
plt.yscale('log')

#%%% Compare Lagrangian and Eulerian scaling with N

plt.style.use('~/.config/matplotlib/joris.mplstyle')
K=30 # Realisations
import scipy.spatial as spatial

T=14
SA=np.logspace(-3,np.log10(0.49),30)
nbin=30
#A=np.random.rand(100,2)
#A=(A.T/np.sum(A,axis=1)).T
#SA=A
SA=np.linspace(0.01,0.49,20)
Res=[]
sa=0.2
random=0.0

x0=np.random.rand(1)
#x0=np.random.rand(10)
x,S,wrapped_time=lagrangian(x0,T,sa,random)
#	x,S,wrapped_time=lagrangian_nfold(x0,T,sa,random)
a=sa
dq=2*3/4.
dq=1
#D2=(1+dq)*np.log(A.shape[1])/np.log(np.sum(1/sa**dq))
D2=(1+dq)*np.log(2)/np.log(1/a**dq+1/(1-a)**dq)
nu,N,Nf=fractal(x,1.001)
D1,N,Nf=fractal_D1(x)
nu=D1
print(nu)
#nu=D2
xi=x
#xi_per=np.hstack((xi-1,xi,xi+1))
xi_per=xi

s0=0.02
sigma=20
n=2**13
D=(sigma/n)**2/2
#Sigma=np.sqrt(2*D)*n
Cmax=1./np.sqrt(1.+4*D/s0**2*wrapped_time)
Si=s0*np.sqrt(1.+4*D/s0**2*wrapped_time)*S

sB=np.mean(Si)

tree=spatial.cKDTree(xi_per.reshape(-1,1))
# Take samples in within cmax
neighboors=tree.query_ball_point(x.reshape(-1,1)[np.abs(x-0.5)<0.5-2*sB], sB/2)

NH=np.histogram(xi,np.arange(0,1,sB),density=False)[0]
logrhoH=np.histogram(xi,np.arange(0,1,sB),density=False,weights=np.log(1/S))[0]

nagg=np.array([len(n) for n in neighboors])
logrho=np.array([(np.mean(np.log(1/S[n]))) for n in neighboors])

lyap=-a*np.log(a)-(1-a)*np.log(1-a)
sigma2=a*np.log(a)**2.+(1-a)*np.log(1-a)**2.-lyap**2.
dq=2*3/4.
dq=1
rhoc=1/sB
#nu=(1+dq)*np.log(2)/np.log(1/a**dq+1/(1-a)**dq)+1
l=(nu-1)*(lyap*T-np.log(rhoc))
s=sigma2*(nu-1)**2.*T

nn=np.array([0,6])
plt.plot(np.log(nagg),logrho,'k+',alpha=1,label='Lagrangian')
plt.plot(nn,1/(nu)*nn+np.log(rhoc),'k--',alpha=1,label=r'$1/(D_1-1)$')
#plt.plot(nn,nn,'k-',alpha=1)
plt.plot(np.log(NH),logrhoH/NH,'r.',alpha=1,label='Eulerian')



plt.legend(title='$a={:1.2f}$'.format(sa))
plt.xlabel(r'$\log N$')
plt.ylabel(r'$\langle \log \rho \rangle$')
#%%% Compare Lagrangian and Eulerian scaling with N log <rho>

plt.style.use('~/.config/matplotlib/joris.mplstyle')
K=30 # Realisations
import scipy.spatial as spatial

T=14
SA=np.logspace(-3,np.log10(0.49),30)
nbin=30
#A=np.random.rand(100,2)
#A=(A.T/np.sum(A,axis=1)).T
#SA=A
SA=np.linspace(0.01,0.49,20)
Res=[]
sa=0.2
random=0.0

x0=np.random.rand(1)
#x0=np.random.rand(10)
x,S,wrapped_time=lagrangian(x0,T,sa,random)
#	x,S,wrapped_time=lagrangian_nfold(x0,T,sa,random)
a=sa
dq=2*3/4.
#D2=(1+dq)*np.log(A.shape[1])/np.log(np.sum(1/sa**dq))
D2=(1+dq)*np.log(2)/np.log(1/a**dq+1/(1-a)**dq)
nu,N,Nf=fractal(x,2)
print(nu)
#nu=D2
xi=x
#xi_per=np.hstack((xi-1,xi,xi+1))
xi_per=xi

s0=0.02
sigma=20
n=2**13
D=(sigma/n)**2/2
#Sigma=np.sqrt(2*D)*n
Cmax=1./np.sqrt(1.+4*D/s0**2*wrapped_time)
Si=s0*np.sqrt(1.+4*D/s0**2*wrapped_time)*S

sB=np.mean(Si)

tree=spatial.cKDTree(xi_per.reshape(-1,1))
# Take samples in within cmax
neighboors=tree.query_ball_point(x.reshape(-1,1)[np.abs(x-0.5)<0.5-2*sB], sB/2)

NH=np.histogram(xi,np.arange(0,1,sB),density=False)[0]
logrhoH=np.histogram(xi,np.arange(0,1,sB),density=False,weights=S)[0]

nagg=np.array([len(n) for n in neighboors])
logrho=np.array([(-np.log(np.mean(S[n]))) for n in neighboors])

lyap=-a*np.log(a)-(1-a)*np.log(1-a)
sigma2=a*np.log(a)**2.+(1-a)*np.log(1-a)**2.-lyap**2.
dq=2*3/4.
rhoc=1/sB
#nu=(1+dq)*np.log(2)/np.log(1/a**dq+1/(1-a)**dq)+1
l=(nu-1)*(lyap*T-np.log(rhoc))
s=sigma2*(nu-1)**2.*T

nn=np.array([0,6])
plt.plot(np.log(nagg),logrho,'k+',alpha=1,label='Lagrangian')
plt.plot(nn,nn+np.log(rhoc),'k--',alpha=1,label=r'$1:1$')
#plt.plot(nn,nn,'k-',alpha=1)
plt.plot(np.log(NH),-np.log(logrhoH/NH),'r.',alpha=1,label='Eulerian')
plt.legend(title='$a={:1.2f}$'.format(sa))
plt.xlabel(r'$\log N$')
plt.ylabel(r'$ -\log \langle \rho^{-1} \rangle$')


#%%% * Grid-based statistics: log- rho versus log n
plt.style.use('~/.config/matplotlib/joris.mplstyle')
import scipy.spatial as spatial
import scipy.optimize

SA=np.logspace(-2,np.log10(0.48),70)
nbin=30
#A=np.random.rand(100,2)
#A=(A.T/np.sum(A,axis=1)).T
#SA=A
#SA=np.linspace(0.01,0.49,20)
Res=[]
nufit=[]
nufit2=[]
nufit_lag=[]
for sa in SA:
	random=0.0
	T=23
	x0=np.random.rand(1)
	#x0=np.random.rand(10)
	x,S,wrapped_time=lagrangian(x0,T,sa,random)
	#	x,S,wrapped_time=lagrangian_nfold(x0,T,sa,random)
	a=sa
	dq=2
	#D2=(1+dq)*np.log(A.shape[1])/np.log(np.sum(1/sa**dq))
	D2=(1+dq)*np.log(2)/np.log(1/a**dq+1/(1-a)**dq)
	#nu,N,Nf=fractal(x,2)
	print(sa)
	#nu=D2
	xi=x
	#xi_per=np.hstack((xi-1,xi,xi+1))
	xi_per=xi
	
	s0=0.02
	sigma=100
	n=2**13
	D=(sigma/n)**2/2
	#Sigma=np.sqrt(2*D)*n
	Cmax=1./np.sqrt(1.+4*D/s0**2*wrapped_time)
	Si=s0*np.sqrt(1.+4*D/s0**2*wrapped_time)*S
	
	sB=np.mean(Si)
	
	NH=np.histogram(xi,np.arange(0,1,sB),density=False)[0]
	logrhoH=np.histogram(xi,np.arange(0,1,sB),density=False,weights=np.log(1/S))[0]/NH
	logrhoH2=np.histogram(xi,np.arange(0,1,sB),density=False,weights=np.log(1/S)**2.)[0]/NH
	varlogRho=logrhoH2-logrhoH**2.
	
	n=15
	logrhoH=logrhoH[NH>n]
	logrhoH2=logrhoH2[NH>n]
	NH=NH[NH>n]
	
#	tree=spatial.cKDTree(x.reshape(-1,1))
#	# Take samples in within cmax
#	neighboors=tree.query_ball_point(x.reshape(-1,1)[np.abs(x-0.5)<0.5-2*sB], sB/2)
#	nagg=np.array([len(n) for n in neighboors])
#	logrho=np.array([(np.mean(np.log(1/S[n]))) for n in neighboors])
#	nufit_lag.append(np.polyfit(np.log(nagg),logrho,1))

	nufit.append(np.polyfit(np.log(NH),logrhoH,1))
#	nufit.append(np.polyfit(np.log(NH)+np.log(np.log(NH)),logrhoH,1))
#	nufit2.append(np.polyfit(np.log(NH),logrhoH2,1))
	nufit2.append([np.mean((logrhoH2-logrhoH**2.)/np.log(NH)),0]) # Assume intercept is null


lyap=-SA*np.log(SA)-(1-a)*np.log(1-SA)
sigma2=a*np.log(SA)**2.+(1-SA)*np.log(1-SA)**2.-lyap**2.
dq=1 #!! Why 1 ?
rhoc=1/sB
D1=(1+dq)*np.log(2)/np.log(1/SA**dq+1/(1-SA)**dq)+1
dq=2 #!! Why 1 ?
D2=(1+dq)*np.log(2)/np.log(1/SA**dq+1/(1-SA)**dq)+1
D0=np.zeros(len(D2))+2

dq=2.0
nu=[]
for a in SA:
	f = lambda Dq : dq*np.log(2)-np.log(1/a**((dq-1)*Dq)+1/(1-a)**((dq-1)*Dq))
	nu.append(scipy.optimize.broyden1(f, 1, f_tol=1e-14))
D2=np.array(nu)+1.

Q=np.loadtxt('Sine_scaling_N_sB1_75.txt')

nufit=np.array(nufit)
nufit2=np.array(nufit2)
nufit_lag=np.array(nufit_lag)

fig,ax=plt.subplots(1,2,figsize=(5,2))
ax[0].plot(D1,nufit[:,0],'ko')
ax[1].plot(D1,nufit2[:,0],'ko')

ax[0].plot(Q[:,0],Q[:,3],'ko',fillstyle='full')
ax[1].plot(Q[:,0],Q[:,4],'ko',fillstyle='full')

ax[0].plot(D1,np.log(2)/lyap,'g-',label=r'$1/(1+\sigma^2/(2*\mu))$ ')

ax[0].plot(D1,np.log((1-a)/a)/(3/2-a)/np.log((3-2*a)/(1+2*a)),'g--',label=r'$1/(1+\sigma^2/(2*\mu))$ ')
ax[0].plot(D1,1/(D0-1),'r:',label=r'$(D_0-1)^{-1}$ ')
ax[0].plot(D1,1/(D1-1),'r-',label=r'$(D_1-1)^{-1}$ ')
ax[1].plot(D1,2*(2-D1)/(D1-1),'r-',label=r'$2(2-D_1)/(D_1-1)$ ')
#ax[1].plot(D1,3*(2-D1)/(D1*(D2-1)),'r:',label=r'$2(2-D_1)/(D_2-1)$ ')
ax[0].plot(D1,1/(D2-1),'r--',label=r'$(D_2-1)^{-1}$ ')
#ax[1].plot(D1,0.1**(D2-2)-1,'r--',label=r'$2(2-D_2)/(D_2-1)$ ')
#plt.plot(nu-1,(2*(2-nu)),'b--',label=r'$2(2-D_1)/(D_1-1)$ ')
#plt.plot(nu-1,nufit[:,1],'ro',alpha=1)
#plt.plot(nu-1,np.log(rhoc)+np.zeros(SA.shape),'r--',label=r'$\langle \rho_c \rangle_{s_B}=-\log s_B$',linewidth=1.)
ax[1].set_xlabel('$D_1$')
ax[0].set_xlabel('$D_1$')
ax[0].set_ylabel(r'$\mu_{\log \rho | n} / \log n $ ')
#ax[0].set_ylabel(r'$\mu_{\log \rho | n} / (\log n +\log\log n)$ ')
ax[1].set_ylabel(r'$\sigma^2_{\log \rho | n} /  \log n$ ')
#plt.yscale('log')
ax[0].legend(fontsize=6)
ax[1].legend(fontsize=6)
ax[0].set_ylim([0,4])
ax[1].set_ylim([1e-3,10])
ax[1].set_yscale('log')
plt.subplots_adjust(wspace=0.4)
subscript(ax[0],0)
subscript(ax[1],1)
plt.savefig('/home/joris/Dropbox/Articles/AggregationBatchelor/Figures/statbundles_baker.pdf',bbox_inches='tight')

#plt.savefig('statbundles_baker.pdf',bbox_inches='tight')


#plt.figure(figsize=(2,2))
#plt.xscale('log')
#plt.yscale('log')

#%%% * Grid-based statistics of bundles dependence on Pe
plt.style.use('~/.config/matplotlib/joris.mplstyle')
from scipy import spatial
def m3(x):
	return np.nanmean(x**3)

def m4(x):
	return np.nanmean(x**4)
	
dic={'nu':[],'D2':[],'D1':[],'mean_logRho1':[],'mean_logRho2':[],'var_logRho':[],
	 'mean_Cmax':[],'var_Cmax':[],'m3_Cmax':[],'m4_Cmax':[],
	 'mean_S':[],'var_S':[],'m3_S':[],'m4_S':[],'mean_sumC':[],'var_sumC':[]}

T=20
SA=np.logspace(-3,np.log10(0.49),30)
nbin=30
A=np.random.rand(100,2)
#A=(A.T/np.sum(A,axis=1)).T
#SA=A
SA=np.linspace(0.01,0.49,50)
Res=[]
P1,P2=[],[]
SB=1/np.logspace(1,3,20)
sa=0.2
random=0.0

x0=np.random.rand(1)
#x0=np.random.rand(10)
x,S,wrapped_time=lagrangian(x0,T,sa,random)
for sB in SB:
#	x,S,wrapped_time=lagrangian_nfold(x0,T,sa,random)
	a=sa
	dq=2*3/4.
	#D2=(1+dq)*np.log(A.shape[1])/np.log(np.sum(1/sa**dq))
	D2=(1+dq)*np.log(A.shape[1])/np.log(1/a**dq+1/(1-a)**dq)
	dq=1
	#D2=(1+dq)*np.log(A.shape[1])/np.log(np.sum(1/sa**dq))
	D1=(1+dq)*np.log(A.shape[1])/np.log(1/a**dq+1/(1-a)**dq)
	#nu,N,Nf=fractal(x,2)
	nu=D2
	dic['nu'].append(nu)
	dic['D2'].append(D2)
	dic['D1'].append(D1)

	nlog=np.linspace(1,10,100)
	# Grid estimates
	NH=np.histogram(x,np.arange(0,1,sB),density=False)[0]
	thetaH=np.histogram(x,np.arange(0,1,sB),density=False,weights=S)[0]
	theta2H=np.histogram(x,np.arange(0,1,sB),density=False,weights=S**2)[0]
	MH=thetaH/NH
	VARH=(theta2H/NH)#-(thetaH/NH)**2
	CVH=np.sqrt(VARH)/(thetaH/NH)
	G=(NH>1)*np.isfinite(VARH)
	pH=np.polyfit(np.log(NH[G]), np.log(VARH[G]),1)
	P2.append(pH)
	plt.plot(np.log(NH[NH>1]),np.log(VARH[NH>1]),'*',color=plt.cm.jet(sa));plt.plot(nlog,pH[0]*nlog+pH[1],'--',color=plt.cm.jet(sa));
	pH=np.polyfit(np.log(NH[G]), np.log(MH[G]),1)
	P1.append(-pH[0])
	nlog=np.linspace(5,15,100)
	plt.plot(np.log(NH[NH>1]),np.log(MH[NH>1]),'*',color=plt.cm.jet(sa));plt.plot(nlog,pH[0]*nlog+pH[1],'--',color=plt.cm.jet(sa));

P2=np.array(P2)

plt.plot(nlog,-nlog-10,'k-')
plt.plot(nlog,-2*nlog-10,'r-')

#Pe=1/(Sigma/n)**2/2
nu=np.linspace(1.2,2,1000)
mu=1/(nu-1)
sigma2=2*(2-nu)/(nu-1)
M2=2*mu-2*sigma2
M2[mu-2*sigma2<0]=mu[mu-2*sigma2<0]**2/(2*sigma2[mu-2*sigma2<0])
plt.figure()
#plt.plot(Pe,P1,'ko')
plt.plot(np.log(1/SB),-P2[:,0],'kd',label=r'$\tilde\gamma$')
plt.plot(np.log(1/SB),-P2[:,1],'rs',label=r'$\tilde\omega$')
plt.xlabel('$  \log (\mathcal{A}/(\ell_0 s_B))$')
plt.ylabel('Exponents')
#plt.ylim([0.5,2.0])
#plt.xscale('log')

# Load and plot results of sine flow
P=np.loadtxt('Sine_scaling_N_A_1.2.txt')
SB=P[:,0]
l0=0.3
plt.plot(np.log(1/(l0*SB)),P[:,2],'kd',fillstyle='full')
plt.plot(np.log(1/(l0*SB)),-P[:,5],'rs',fillstyle='full')
s=np.linspace(2,8)
plt.plot(s,2*s,'r-',label='$2 \log (  \mathcal{A} / (\ell_0 s_B))$')
plt.ylim([0,20])
plt.legend()

plt.savefig('/home/joris/Dropbox/Articles/AggregationBatchelor/Figures/gamma2bundles_Peclet.pdf',bbox_inches='tight')

#%%% * Grid-based statistics of bundles  1/N ~ <1/ RHO> dependence on a
plt.style.use('~/.config/matplotlib/joris.mplstyle')
from scipy import spatial
def m3(x):
	return np.nanmean(x**3)

def m4(x):
	return np.nanmean(x**4)
	
dic={'nu':[],'D2':[],'D1':[],'mean_logRho1':[],'mean_logRho2':[],'var_logRho':[],
	 'mean_Cmax':[],'var_Cmax':[],'m3_Cmax':[],'m4_Cmax':[],
	 'mean_S':[],'var_S':[],'m3_S':[],'m4_S':[],'mean_sumC':[],'var_sumC':[]}

T=20
SA=np.logspace(-3,np.log10(0.49),10)
nbin=30
A=np.random.rand(100,2)
#A=(A.T/np.sum(A,axis=1)).T
#SA=A
SA=np.linspace(0.01,0.499,20)
Res=[]
P1,P2=[],[]

for sa in SA:
	print(sa)
	random=0.0
	
	x0=np.random.rand(2)
	l0=len(x0)
	#x0=np.random.rand(10)
	x,S,wrapped_time=lagrangian(x0,T,sa,random)
#	x,S,wrapped_time=lagrangian_nfold(x0,T,sa,random)
	a=sa
	dq=2*3/4.
	#D2=(1+dq)*np.log(A.shape[1])/np.log(np.sum(1/sa**dq))
	D2=(1+dq)*np.log(A.shape[1])/np.log(1/a**dq+1/(1-a)**dq)
	dq=1
	#D2=(1+dq)*np.log(A.shape[1])/np.log(np.sum(1/sa**dq))
	D1=(1+dq)*np.log(A.shape[1])/np.log(1/a**dq+1/(1-a)**dq)
	#nu,N,Nf=fractal(x,2)
	nu=D2
	dic['nu'].append(nu)
	dic['D2'].append(D2)
	dic['D1'].append(D1)
	
	xi=x
	#xi_per=np.hstack((xi-1,xi,xi+1))
	xi_per=xi
	
	s0=0.05
	sigma=200
	n=2**13
	D=(sigma/n)**2/2
	#Sigma=np.sqrt(2*D)*n
	Cmax=1./np.sqrt(1.+4*D/s0**2*wrapped_time)
	Cmax=S
	Si=s0*np.sqrt(1.+4*D/s0**2*wrapped_time)*S
	
	sB=1/50
	
	nmin=np.exp(3)
	nlog=np.linspace(0,15,100)
	# Grid estimates
	NH=np.histogram(xi,np.arange(0,1,sB),density=False)[0]
	thetaH=np.histogram(xi,np.arange(0,1,sB),density=False,weights=Cmax)[0]
	MH=thetaH/NH
	theta2H=np.histogram(xi,np.arange(0,1,sB),density=False,weights=Cmax**2)[0]
	VARH=(theta2H/NH)-(thetaH/NH)**2
	CVH=np.sqrt(VARH)/(thetaH/NH)
	G=(NH>nmin)*np.isfinite(VARH)
	pH=np.polyfit(np.log(NH[G]), np.log((theta2H/NH)[G]),1)
#	pH=np.polyfit(np.log(NH[G]), np.log(VARH[G]),1)
	P2.append([-pH[0],pH[1]])
#	P2.append([-np.mean(np.log(VARH[G])/np.log(NH[G])),pH[1]])
#	plt.plot(np.log(NH[NH>1]),np.log(VARH[NH>1]),'o',color=plt.cm.jet(sa));
	plt.plot(np.log(NH[NH>1]),np.log((theta2H/NH)[NH>1]),'o',color=plt.cm.jet(sa));
	plt.plot(nlog,pH[0]*nlog+pH[1],'--',color=plt.cm.jet(sa));
	pH=np.polyfit(np.log(NH[G]), np.log(MH[G]),1)
	P1.append(-pH[0])
	nlog=np.linspace(5,15,100)
	plt.plot(np.log(NH[NH>1]),np.log(MH[NH>1]),'*',color=plt.cm.jet(sa));plt.plot(nlog,pH[0]*nlog+pH[1],'--',color=plt.cm.jet(sa));

plt.plot(nlog,-nlog+np.log(sB*l0),'k-')
plt.plot(nlog,-2*(nlog-np.log(sB*l0)),'r-')

plt.plot(nlog,-nlog-10,'k-')


# Load Sine flow equivalent
Q=np.loadtxt('Sine_scaling_N_sB1_{:1.0f}.txt'.format(1/sB))

P2=np.array(P2)
nu=np.linspace(1.2,2,1000)
mu=1/(nu-1)
sigma2=2*(2-nu)/(nu-1)
M2=2*mu-2*sigma2
M2[mu-2*sigma2<0]=mu[mu-2*sigma2<0]**2/(2*sigma2[mu-2*sigma2<0])

fig,ax=plt.subplots(1,2,figsize=(7,3))
ax[0].plot(np.array(dic['D1'])+1,P1,'ko')
ax[0].plot(np.array(dic['D1'])+1,P2[:,0],'rd')

np.savetxt('Baker_epsilon.txt',np.vstack((np.array(dic['D1'])+1,2-P2[:,0])).T)
# Sine
ax[0].plot(Q[:,0],Q[:,1],'ko',fillstyle='full')
ax[0].plot(Q[:,0],Q[:,2],'rd',fillstyle='full')

ax[0].plot(nu,np.zeros(nu.shape)+1,'k--',label=r'$\gamma_{1,\rho^{-1}|n}=1$')
ax[0].plot(nu[mu-2*sigma2<0],M2[mu-2*sigma2<0],'r-',label=r'$\gamma_{2,\rho^{-1}|n}=\tilde{\mu}^2/(2\tilde{\sigma^2})$')
ax[0].plot(nu[mu-2*sigma2>0],M2[mu-2*sigma2>0],'r--',label=r'$\gamma_{2,\rho^{-1}|n}=2\tilde{\mu}-2\tilde{\sigma^2}$')
ax[0].set_xlabel('$D_1$')
ax[0].set_ylabel(r'$\gamma_{q,\rho^{-1}|n}$')
ax[0].set_ylim([0.5,2.75])
ax[0].legend(loc=1)
subscript(ax[0],0)
#plt.savefig('/home/joris/Dropbox/Articles/AggregationBatchelor/Figures/gamma2bundles.pdf',bbox_inches='tight')
#plt.savefig('gamma2bundles.pdf',bbox_inches='tight')



#plt.figure()
ax[1].plot(np.array(dic['D1'])+1,-P2[:,1]/2/np.log(1/(sB)),'rs')
ax[1].plot(Q[:,0],-Q[:,-1]/2/np.log(1/(sB*0.3)),'rs',fillstyle='full')
ax[1].plot([1.3,2.0],[1,1],'k-')
ax[1].set_ylim([0,2])
ax[1].set_ylabel(r'$ \tilde\omega / 2 \log (\mathcal{A}/(\ell_0 s_B))$')
ax[1].set_xlabel('$D_1$')
subscript(ax[1],1)
plt.subplots_adjust(wspace=0.2)
plt.savefig('/home/joris/Dropbox/Articles/AggregationBatchelor/Figures/gamma2bundles.pdf',bbox_inches='tight')

#%%% Why rho ^-1 | n do not predict c | n ?
plt.style.use('~/.config/matplotlib/joris.mplstyle')
from scipy import spatial
x0=np.array([0.2])
T=20
nbin=30
sB=1/100
A=np.random.rand(100,2)
#A=(A.T/np.sum(A,axis=1)).T
#SA=A
SA=np.linspace(0.01,0.499,40)
#SA=np.array([0.1,0.49])
Res=[]
P1,P2=[],[]
X0=np.linspace(0.01,0.49,15)
X0=np.random.rand(20)
nmin=np.exp(0)
nlog=np.linspace(0,15,100)
l0=1
sa=0.45
print(sa)
NH,cH,vH=[],[],[]
x,S,wrapped_time=lagrangian(np.array([0.1]),T,sa,0)
#x0=np.random.rand()
NH=np.histogram(x,np.arange(0,1+sB,sB),density=False)[0]
cH=np.histogram(x,np.arange(0,1+sB,sB),density=False,weights=S)[0]
rho2H=np.histogram(x,np.arange(0,1+sB,sB),density=False,weights=S**2)[0]
vrho2=rho2H/NH-cH**2/NH**2
vH=(cH-np.mean(cH))**2.
#vH=cH**2.

#plt.plot(np.log(NH),np.log(cH),'*')
#plt.plot(np.log(NH),np.log(rho2H),'*',label='\sum_n \rho^{-2}')
plt.figure(figsize=(4,3))
plt.plot(np.log(NH),np.log(vrho2*NH/(NH-1)),'k*',label=r'$n/(n-1) \sigma^2_{\rho^{-1}|n}$')
plt.plot(np.log(NH),np.log(vH),'o',label=r'$ \sigma^2_{c|n}$')
#plt.ylim([-30,-10])
plt.xlabel('$\log n$')
plt.ylabel('$\log \sigma^2$')
plt.legend()

plt.figure(figsize=(4,3))
plt.plot(np.log(NH),np.log(cH),'k*',label=r'$\sum s$')
plt.plot(np.log(NH),np.log(np.sqrt(vrho2)),'bo',label=r'$\sum s$')
#plt.ylim([-30,-10])
plt.xlabel('$\log n$')
plt.ylabel('$\log \sum s$')
plt.legend()




plt.figure()
plt.plot(np.linspace(0,1,len(vrho2)),np.log(vrho2),'--',label=r'$\sigma^2_{\rho^{-1}|n}$')
plt.plot(np.linspace(0,1,len(vrho2)),np.log(NH*vrho2),':',label=r'$n \sigma^2_{\rho^{-1}|n}$')
plt.plot(np.linspace(0,1,len(vrho2)),np.log(vH),label=r'$(\sum \rho^{-1}-\langle \sum \rho^{-1} \rangle)^2$')
plt.xlabel('$x$')
plt.legend()
#plt.plot(cH)
# plt.plot([0,100],[np.mean(np.log(vH)),np.mean(np.log(vH))])
# plt.plot([0,100],[np.mean(np.log(vrho2)),np.mean(np.log(vrho2))])

x0=0.1
idb=np.where((x<x0+sB)&(x>x0))[0]
nh=len(idb)
vrho2=np.var(S[idb])
vH=(np.sum(S[idb])-np.mean(cH))**2

print(nh,nh*vrho2,vrho2,vH)

# Is the distribution of s lognormal in a bundle: yes
plt.figure()
plt.hist((S[idb]),20)
plt.xlabel(r'$\log s$')
plt.ylabel(r'$P$')

# what are the mean and variance
mu=np.mean(np.log(S[idb]))
sigma2=np.var(np.log(S[idb]))

print(np.mean((S[idb])),np.exp(mu+sigma2/2))
print(np.var((S[idb])),np.exp(2*mu+sigma2)*(np.exp(sigma2)-1),np.exp(2*mu+sigma2)*(np.exp(sigma2)))

print(np.sum(S[idb]),nh*np.mean((S[idb])))
print(vH,nh*np.var((S[idb])))

nh_cible=350
xx=np.arange(0,1+sB,sB)
idcible=np.where((NH<nh_cible+20)&(NH>nh_cible-20))[0]

for xi in xx[idcible]:
	idb=np.where((x<xi+sB)&(x>xi))[0]
	nh=len(idb)
	vrho2=np.var(S[idb])
	vH=(np.sum(S[idb])-np.mean(cH))**2
	print(nh,nh*vrho2,vrho2,vH)



#%%%
vsx=[]
for k in np.arange(1000):
	x=np.random.rand(1000)
	x[x>0.5]=0
	vsx.append(np.sum(x))

print(np.var(vsx))
v=np.var(x)
print(v*len(x))

#%%% * Grid-based statistics of bundles  1/N ~ c^2 dependence on a
plt.style.use('~/.config/matplotlib/joris.mplstyle')
from scipy import spatial

T=22
nbin=30
sB=1/100
A=np.random.rand(100,2)
#A=(A.T/np.sum(A,axis=1)).T
#SA=A
SA=np.linspace(0.01,0.499,40)
#SA=np.array([0.1,0.49])
Res=[]
P1,P2=[],[]
X0=np.linspace(0.01,0.49,15)
X0=np.random.rand(20)
nmin=np.exp(0)
nlog=np.linspace(0,15,100)
l0=1
for sa in SA:
	print(sa)
	NH,cH,vH=[],[],[]
	x,S,wrapped_time=lagrangian(np.array([0.12]),T,sa,0)
	for x0 in X0:
		NH.append(np.histogram(np.mod(x-x0,1),np.arange(0,1+sB,sB),density=False)[0])
		cH.append(np.histogram(np.mod(x-x0,1),np.arange(0,1+sB,sB),density=False,weights=S)[0])
		vH.append((cH[-1]-np.mean(cH[-1]))**2.)
	NH=np.array(NH).flatten()
	vH=np.array(vH).flatten()
	G=(NH>nmin)*np.isfinite(vH)
	# With intercept free
	#pH=np.polyfit(np.log(NH[G]), np.log(vH[G]),1)
	# With fixed intercept
	intercept=-(2*np.log(1/(sB)))
	#intercept=10
	pH=[np.nanmean((np.log(vH[G])-intercept)/np.log(NH[G])),intercept]
	P2.append([-pH[0],pH[1]])
	plt.plot(np.log(NH[NH>1]),np.log(vH[NH>1]),'o',color=plt.cm.jet(sa));
	plt.plot(nlog,pH[0]*nlog+pH[1],'--',color=plt.cm.jet(sa));
	
plt.plot(nlog,-nlog+np.log(sB*l0),'k-')
plt.plot(nlog,-2*(nlog-np.log(sB*l0)),'r-')
plt.plot(nlog,-nlog-10,'k-')
plt.ylim([-50,-6])

dq=1
D1=1+(1+dq)*np.log(2)/np.log(1/SA**dq+1/(1-SA)**dq)
P2=np.array(P2)



plt.figure(figsize=(3,3))
plt.plot(D1,P2[:,0],'ko',label=r'Baker map')
plt.ylim([0,2.5])
plt.xlabel('$D_1$')
np.savetxt('Baker_xi.txt',np.vstack((D1,P2[:,0])).T)

#E=np.loadtxt('Baker_epsilon.txt')
#plt.plot(E[:,0],2-E[:,1]-1.0,'k--',label=r'$\gamma_{2,\rho^{-1}|n}-1$ (baker map)')

#Load sine flow
sB=1/50
#A=np.loadtxt('Sine_scaling_C|N_sB1_{:1.0f}.txt'.format(1/sB))
#plt.plot(A[:,0],A[:,1],'ko',label='Sine flow',linewidth=1.1,fillstyle='full')

A=np.loadtxt('Sine_scaling_with_n.txt')
D1=np.loadtxt('Sine_D1.txt')

d1=np.interp(A[:,1],D1[:,0],D1[:,1])
plt.plot(d1,-A[:,4],'ko',label='Sine flow',linewidth=1.1,fillstyle='full')

#B=np.loadtxt('Sine_scaling_N_sB1_{:1.0f}.txt'.format(1/sB))
#plt.plot(B[:,0],B[:,2]-1,'r--',label=r'$\gamma_{2,\rho^{-1}|n}-1$ (sine flow)',linewidth=1.1)

# Model 
nu=np.linspace(1.2,2,1000)
mu=1/(nu-1)
sigma2=2*(2-nu)/(nu-1)
M2=2*mu-2*sigma2
M2[mu-2*sigma2<0]=mu[mu-2*sigma2<0]**2/(2*sigma2[mu-2*sigma2<0])
plt.plot(nu,M2,'r--',label=r'$\gamma_{2,\rho^{-1}|n}$')
plt.plot(nu,M2-1,'r-',label=r'$\gamma_{2,\rho^{-1}|n}-1$')
#plt.plot(nu,(M2-1)*2,'r-',label=r'$\gamma_{2,\rho^{-1}|n}-1$')

plt.legend()
plt.xlabel('$D_1$')
plt.ylim([0,2.5])
plt.ylabel(r'$\gamma_{2,c|n}$')
#Theory
# nu=np.linspace(1.2,2,1000)
# mu=1/(nu-1)
# sigma2=2*(2-nu)/(nu-1)
# M2=2*mu-2*sigma2
# M2[mu-2*sigma2<0]=mu[mu-2*sigma2<0]**2/(2*sigma2[mu-2*sigma2<0])
# plt.plot(nu,M2-1,'r-',label='Theory')

plt.legend()
plt.savefig(figdir+'gamma2_1_rho&c.pdf')

plt.figure()
D1=1+(1+dq)*np.log(2)/np.log(1/SA**dq+1/(1-SA)**dq)
plt.plot(D1,-P2[:,1]/(2*np.log(1/(sB))),label=r'$\omega_{2,c|n}$')
plt.ylim([0,2])
plt.xlabel('$D_1$')
plt.legend()

#%%% Initiation statistics of bundles dependence on a
from scipy import spatial
def m3(x):
	return np.nanmean(x**3)

def m4(x):
	return np.nanmean(x**4)

dic={'nu':[],'D2':[],'D1':[],'mean_logRho1':[],'mean_logRho2':[],'var_logRho':[],
	 'mean_Cmax':[],'var_Cmax':[],'m3_Cmax':[],'m4_Cmax':[],
	 'mean_S':[],'var_S':[],'m3_S':[],'m4_S':[],'mean_sumC':[],'var_sumC':[]}

T=20
SA=np.logspace(-3,np.log10(0.49),30)
nbin=30
A=np.random.rand(100,2)
#A=(A.T/np.sum(A,axis=1)).T
#SA=A
SA=np.linspace(0.01,0.49,20)
Res=[]
for sa in SA:
	print(sa)
	random=0.0
	
	x0=np.random.rand(1)
	#x0=np.random.rand(10)
	x,S,wrapped_time=lagrangian(x0,T,sa,random)
#	x,S,wrapped_time=lagrangian_nfold(x0,T,sa,random)
	a=sa
	dq=2*3/4.
	#D2=(1+dq)*np.log(A.shape[1])/np.log(np.sum(1/sa**dq))
	D2=(1+dq)*np.log(A.shape[1])/np.log(1/a**dq+1/(1-a)**dq)
	dq=1
	#D2=(1+dq)*np.log(A.shape[1])/np.log(np.sum(1/sa**dq))
	D1=(1+dq)*np.log(A.shape[1])/np.log(1/a**dq+1/(1-a)**dq)
	#nu,N,Nf=fractal(x,2)
	nu=D2
	dic['nu'].append(nu)
	dic['D2'].append(D2)
	dic['D1'].append(D1)
	
	xi=x
	#xi_per=np.hstack((xi-1,xi,xi+1))
	xi_per=xi
	
	s0=0.05
	sigma=20
	n=2**13
	D=(sigma/n)**2/2
	#Sigma=np.sqrt(2*D)*n
	Cmax=1./np.sqrt(1.+4*D/s0**2*wrapped_time)
	Si=s0*np.sqrt(1.+4*D/s0**2*wrapped_time)*S
	
	sB=np.mean(Si)
	
	# Grid estimates
	NH=np.histogram(xi,np.arange(0,1,sB),density=False)[0]
	thetaH=np.histogram(xi,np.arange(0,1,sB),density=False,weights=Cmax)[0]
	theta2H=np.histogram(xi,np.arange(0,1,sB),density=False,weights=Cmax**2)[0]
	VARH=(theta2H/NH)-(thetaH/NH)**2
	CVH=np.sqrt(VARH)/(thetaH/NH)
	pH=np.polyfit(np.log(NH), np.log(VARH),1)
	nlog=np.linspace(5,15,100)
	plt.plot(np.log(NH),np.log(VARH),'*');plt.plot(nlog,pH[0]*nlog+pH[1],'--');
	
	# Centered on lamellae
	tree=spatial.cKDTree(xi_per.reshape(-1,1))
	# Take samples in within cmax
	neighboors=tree.query_ball_point(xi.reshape(-1,1), sB/2)
	
	nagg=np.array([len(n) for n in neighboors])
	
	#% Check cmax of bundles
	def bin_operation(x,y,xi,op):
		r=np.zeros(xi.shape[0]-1)
		for i in range(xi.shape[0]-1):
			idx=np.where((x<=xi[i+1])&(x>=xi[i]))[0]
			r[i]=op(y[idx])
		return r
	
	def bundle(neighboors,variable,operator,binarize,nbin):
		v=np.array([operator(variable[n]) for n in neighboors])
		if binarize=='N':
			nagg=np.array([len(n) for n in neighboors])
			nagg_bin=np.logspace(0,np.max(np.log10(nagg)),nbin)
			return bin_operation(nagg,v,nagg_bin,np.nanmean),nagg_bin
		if binarize=='logN':
			nagg=np.array([np.log(len(n)) for n in neighboors])
			nagg_bin=np.linspace(0,np.max(nagg),nbin)
			return bin_operation(nagg,v,nagg_bin,np.nanmean),nagg_bin
		if binarize=='Rho':
			rhoagg_mean=np.array([np.nanmean((1/S[n])) for n in neighboors])
			rho_bin=np.logspace(np.log10(rhoagg_mean.min()),np.log10(rhoagg_mean.max()),nbin)
			return bin_operation(rhoagg_mean,v,rho_bin,np.nanmean),rho_bin
		if binarize=='logRho':
			rhoagg_mean=np.array([np.nanmean(np.log(1/S[n])) for n in neighboors])
			rho_bin=np.linspace((rhoagg_mean.min()),(rhoagg_mean.max()),nbin)
			return bin_operation(rhoagg_mean,v,rho_bin,np.nanmean),rho_bin
		if binarize=='Cmax':
			c_mean=np.array([np.nanmean((Cmax[n])) for n in neighboors])
			c_bin=np.logspace(np.log10(c_mean.min()),np.log10(c_mean.max()),nbin)
			return bin_operation(c_mean,v,c_bin,np.nanmean),c_bin
	
	
	def bundle_var(neighboors,variable,operator,binarize,nbin):
		v=np.array([operator(variable[n]) for n in neighboors])
		if binarize=='N':
			nagg=np.array([len(n) for n in neighboors])
			nagg_bin=np.logspace(0,np.max(np.log10(nagg)),nbin)
			return bin_operation(nagg,v,nagg_bin,np.nanvar),nagg_bin
		if binarize=='logN':
			nagg=np.array([np.log(len(n)) for n in neighboors])
			nagg_bin=np.linspace(0,np.max(nagg),nbin)
			return bin_operation(nagg,v,nagg_bin,np.nanvar),nagg_bin
		if binarize=='Rho':
			rhoagg_mean=np.array([np.nanmean((1/S[n])) for n in neighboors])
			rho_bin=np.logspace(np.log10(rhoagg_mean.min()),np.log10(rhoagg_mean.max()),nbin)
			return bin_operation(rhoagg_mean,v,rho_bin,np.nanvar),rho_bin
		if binarize=='logRho':
			rhoagg_mean=np.array([np.nanmean(np.log(1/S[n])) for n in neighboors])
			rho_bin=np.linspace((rhoagg_mean.min()),(rhoagg_mean.max()),nbin)
			return bin_operation(rhoagg_mean,v,rho_bin,np.nanvar),rho_bin
		if binarize=='Cmax':
			c_mean=np.array([np.nanmean((Cmax[n])) for n in neighboors])
			c_bin=np.logspace(np.log10(c_mean.min()),np.log10(c_mean.max()),nbin)
			return bin_operation(c_mean,v,c_bin,np.nanvar),c_bin
		
							
	h,x=bundle(neighboors,np.log(1/S),np.nanmean,'logN',30)
	#isgood=np.where(np.isfinite(h))[0]
	#p0=np.polyfit((x[isgood]),(h[isgood]),1)
	dic['mean_logRho1'].append(best_polyfit(x[1:],h))
	
	h,x=bundle(neighboors,np.log(nagg),np.nanmean,'logRho',30)
	#isgood=np.where(np.isfinite(h))[0]
	#p1=np.polyfit((x[isgood]),(h[isgood]),1)
	dic['mean_logRho2'].append(best_polyfit(x[1:],h))
	
	h,x=bundle(neighboors,np.log(1/S),np.nanvar,'logN',30)
	#isgood=np.where(np.isfinite(h))[0]
	#p2=np.polyfit((x[isgood]),(h[isgood]),1)
	dic['var_logRho'].append(best_polyfit(x[1:],h))
	
	
	h,x=bundle(neighboors,Cmax,np.nanmean,'logN',30)
	#isgood=np.where(np.isfinite(h))[0]
	#p0=np.polyfit((x[isgood]),(h[isgood]),1)
	dic['mean_Cmax'].append(best_polyfit(x[1:],np.log(h)))
	
	h,x=bundle(neighboors,Cmax,np.nanvar,'logN',30)
	dic['var_Cmax'].append(best_polyfit(x[1:],np.log(h)))
	h,x=bundle(neighboors,Cmax,m3,'logN',30)
	dic['m3_Cmax'].append(best_polyfit(x[1:],np.log(h)))
	h,x=bundle(neighboors,Cmax,m4,'logN',30)
	dic['m4_Cmax'].append(best_polyfit(x[1:],np.log(h)))
	h,x=bundle(neighboors,Cmax,np.mean,'logN',30)
	#isgood=np.where(np.isfinite(h))[0]
	#p0=np.polyfit((x[isgood]),(h[isgood]),1)
	h,x=bundle(neighboors,S,np.nanmean,'logN',30)
	dic['mean_S'].append(best_polyfit(x[1:],np.log(h)))
	h,x=bundle(neighboors,S,np.nanvar,'logN',30)
	dic['var_S'].append(best_polyfit(x[1:],np.log(h)))
	h,x=bundle(neighboors,S,m3,'logN',30)
	dic['m3_S'].append(best_polyfit(x[1:],np.log(h)))
	h,x=bundle(neighboors,S,m4,'logN',30)
	dic['m4_S'].append(best_polyfit(x[1:],np.log(h)))
	
	h,x=bundle(neighboors,Cmax,np.sum,'logN',30)
	v=np.array([np.sum(Cmax[n]) for n in neighboors])
	nagg=np.array([len(n) for n in neighboors])
	nagg_bin=np.logspace(0,3,20)
	dic['mean_sumC'].append(bin_operation(nagg,v,nagg_bin,np.nanmean))
	dic['var_sumC'].append(bin_operation(nagg,v,nagg_bin,np.nanvar))
	
	print(dic['mean_logRho2'][-1])
	
for keys in dic.keys():
	dic[keys]=np.array(dic[keys])
	
plt.plot(SA,dic['nu'])
plt.plot(SA,dic['D2'])
plt.xscale('log')
plt.yscale('log')
#
#bins=np.arange(0,1,sB)
#hn,x=np.histogram(xi,bins)
#hlogrho,x=np.histogram(xi,bins,weights=np.log(1/S))
#hvarlogrho,x=np.histogram(xi,bins,weights=np.log(1/S)**2.)
#hlogrho/hn
#%%%  + Plots
plt.style.use('~/.config/matplotlib/joris.mplstyle')
fig=plt.figure(figsize=(4,3))
[plt.plot(nagg_bin[1:],dic['var_sumC'][k],'o',c=plt.cm.viridis(dic['D2'][k]),alpha=0.5,markersize=5,fillstyle='full')
 for k in range(len(dic['var_sumC']))] 
plt.yscale('log')
plt.xscale('log')
plt.ylabel(r'Var $c_B$')
plt.xlabel(r'$N$')
plt.ylim([1e-6,1e-1])
ax2 = fig.add_axes([0.6, 0.8, 0.2, 0.05])
import matplotlib as mpl
#norm = mpl.colors.Normalize(vmin=T_all[0]*u_pore/d, vmax=T_all[-1]*u_pore/d)
norm = mpl.colors.Normalize(vmin=0, vmax=1)
cb1 = mpl.colorbar.ColorbarBase(ax2, 
																cmap=plt.cm.viridis,norm=norm,
																orientation='horizontal')
cb1.set_label(r'$D_2$', color='k',size=8,labelpad=0)
plt.savefig(dir_out+'varsumC.pdf',bbox_inches='tight')


plt.style.use('~/.config/matplotlib/joris.mplstyle')
fig=plt.figure(figsize=(4,3))
[plt.plot(nagg_bin[1:],dic['mean_sumC'][k],'o',c=plt.cm.viridis(dic['D2'][k]),alpha=0.5,markersize=5,fillstyle='full')
 for k in range(len(dic['mean_sumC']))] 
plt.yscale('log')
plt.xscale('log')
plt.ylabel(r'$\langle c_B | N \rangle$')
plt.xlabel(r'$N$')
plt.ylim([1e-2,1e0])
ax2 = fig.add_axes([0.6, 0.8, 0.2, 0.05])
import matplotlib as mpl
#norm = mpl.colors.Normalize(vmin=T_all[0]*u_pore/d, vmax=T_all[-1]*u_pore/d)
norm = mpl.colors.Normalize(vmin=0, vmax=1)
cb1 = mpl.colorbar.ColorbarBase(ax2, 
																cmap=plt.cm.viridis,norm=norm,
																orientation='horizontal')
cb1.set_label(r'$D_2$', color='k',size=8,labelpad=0)
plt.savefig(dir_out+'meansumC.pdf',bbox_inches='tight')


plt.style.use('~/.config/matplotlib/joris.mplstyle')
fig=plt.figure(figsize=(4,3))
[plt.plot(nagg_bin[1:],np.array(dic['var_sumC'][k])/np.array(dic['mean_sumC'][k]),'o',c=plt.cm.viridis(dic['D2'][k]),alpha=0.5,markersize=5,fillstyle='full')
 for k in range(len(dic['mean_sumC']))] 
plt.yscale('log')
plt.xscale('log')
plt.ylabel(r'STD $c_B /\langle c_B | N \rangle$')
plt.xlabel(r'$N$')
plt.ylim([1e-5,1e0])
ax2 = fig.add_axes([0.6, 0.8, 0.2, 0.05])
import matplotlib as mpl
#norm = mpl.colors.Normalize(vmin=T_all[0]*u_pore/d, vmax=T_all[-1]*u_pore/d)
norm = mpl.colors.Normalize(vmin=0, vmax=1)
cb1 = mpl.colorbar.ColorbarBase(ax2, 
																cmap=plt.cm.viridis,norm=norm,
																orientation='horizontal')
cb1.set_label(r'$D_2$', color='k',size=8,labelpad=0)
plt.savefig(dir_out+'STDmeansumC.pdf',bbox_inches='tight')

plt.figure()
mv=[np.nanmean(np.array(dic['var_sumC'][k])) for k in range(len(dic['mean_sumC']))]
plt.plot(dic['D2'],mv,'o')
#plt.yscale('log')
plt.ylabel(r'Var $ c_B$')
plt.plot(dic['D2'],s0**2.*2*(2-(dic['D2']+1))/dic['D2'],'r*',label=r'$\langle c \rangle^2 2(2-\nu)/(\nu-1)$ ?')
plt.xlabel(r'$D_2$')
plt.legend()
plt.savefig(dir_out+'sumC.pdf',bbox_inches='tight')
#%%% + Plots Scaling log rho as log N
plt.style.use('~/.config/matplotlib/joris.mplstyle')
nu=dic['nu']
nu_th=dic['D1']
fig,ax=plt.subplots(1,1,figsize=(2,2))
#ax[0].plot(SA,1/nu,'k--',label=r'$1/(\nu-1)$')
#ax[1].plot(SA,2*(2-(nu+1))/nu,'r--',label=r'$2(2-\nu)/(\nu-1)$')
ax.plot(nu_th,1/nu_th,'k-',label=r'$1/(D_1-1)$')
ax.plot(nu_th,2*(2-(nu_th+1))/nu_th,'r--',label=r'$2(2-D_1)/(D_1-1)$')
#ax.errorbar(nu_th,1/dic['mean_logRho2'][:,1],yerr=dic['mean_logRho2'][:,3],label=r'$\alpha, \mu_{\log \rho} \propto \alpha \log N$')
#ax.errorbar(nu_th,dic['mean_logRho1'][:,1],yerr=dic['mean_logRho1'][:,3],label=r'$\alpha, \mu_{\log \rho} \propto \alpha \log N$')
##ax[0].plot(SA,1/Res[:,3],'+',label=r'$\alpha, \mu_{\log \rho} \propto \alpha \log N$')
#ax.errorbar(nu_th,dic['var_logRho'][:,1],yerr=dic['var_logRho'][:,3],label=r'$\beta, \sigma^2_{\log \rho} \propto \beta \log N$')
ax.plot(nu_th,1/dic['mean_logRho2'][:,1],'ko',label=r'$\mu_{\rho,s_B}/\log n$')
ax.plot(nu_th,dic['var_logRho'][:,1],'ro',fillstyle='top',label=r'$\sigma^2_{\rho,s_B}/\log n$')

#ax.errorbar(nu_th,dic['mean_logRho1'][:,1],yerr=dic['mean_logRho1'][:,3],label=r'$\alpha, \mu_{\log \rho} \propto \alpha \log N$')
##ax[0].plot(SA,1/Res[:,3],'+',label=r'$\alpha, \mu_{\log \rho} \propto \alpha \log N$')
#ax.errorbar(nu_th,dic['var_logRho'][:,1],yerr=dic['var_logRho'][:,3],label=r'$\beta, \sigma^2_{\log \rho} \propto \beta \log N$')
#ax[1].plot(SA,-Res[:,8],label=r'$\beta, \sigma^2_{\log \rho} \propto \beta \log N$')
ax.set_xlabel('$D_1$')
ax.legend(fontsize=8)
#ax.set_yscale('log')
#ax.set_yscale('log')
#plt.yscale('log')
ax.set_ylim([1e-2,10])
ax.set_ylim([0,5])
plt.savefig('/home/joris/Dropbox/Articles/AggregationBatchelor/Figures/statbundles_baker.pdf',bbox_inches='tight')
#%%% + Plots Scaling log rho as log N
plt.style.use('~/.config/matplotlib/joris.mplstyle')
nu=dic['nu']
nu_th=dic['D1']
fig,ax=plt.subplots(2,1,figsize=(3,4))
#ax[0].plot(SA,1/nu,'k--',label=r'$1/(\nu-1)$')
#ax[1].plot(SA,2*(2-(nu+1))/nu,'r--',label=r'$2(2-\nu)/(\nu-1)$')
ax[0].plot(SA,1/nu_th,'k--',label=r'$1/(\nu-1)$')
ax[1].plot(SA,2*(2-(nu_th+1))/nu_th,'r--',label=r'$2(2-\nu)/(\nu-1)$')
ax[0].errorbar(SA,1/dic['mean_logRho2'][:,1],yerr=dic['mean_logRho2'][:,3],label=r'$\alpha, \mu_{\log \rho} \propto \alpha \log N$')
ax[0].errorbar(SA,dic['mean_logRho1'][:,1],yerr=dic['mean_logRho1'][:,3],label=r'$\alpha, \mu_{\log \rho} \propto \alpha \log N$')
#ax[0].plot(SA,1/Res[:,3],'+',label=r'$\alpha, \mu_{\log \rho} \propto \alpha \log N$')
ax[1].errorbar(SA,dic['var_logRho'][:,1],yerr=dic['var_logRho'][:,3],label=r'$\beta, \sigma^2_{\log \rho} \propto \beta \log N$')
#ax[1].plot(SA,-Res[:,8],label=r'$\beta, \sigma^2_{\log \rho} \propto \beta \log N$')
ax[1].set_xlabel('$a$')
ax[0].legend()
ax[1].legend()
ax[1].set_yscale('log')
ax[0].set_yscale('log')
#plt.yscale('log')

plt.savefig(dir_out+'rho_Df.pdf',bbox_inches='tight')
#%%% + Plots
plt.style.use('~/.config/matplotlib/joris.mplstyle')
nu=dic['nu']
nu_th=dic['D2']
nu=dic['D2']
fig,ax=plt.subplots(2,1,figsize=(3,4))
#ax[0].plot(SA,1/nu,'k--',label=r'$1/(\nu-1)$')
#ax[1].plot(SA,2*(2-(nu+1))/nu,'r--',label=r'$2(2-\nu)/(\nu-1)$')
ax[0].plot(nu,1/nu,'ko',label=r'$1/(\nu-1)$')
ax[1].plot(nu,2*(2-(nu+1))/(nu),'ko',label=r'$2(2-\nu)/(\nu-1)$')
ax[0].plot(nu,1/dic['mean_logRho2'][:,1],'rs',label=r'$\alpha, \mu_{\log \rho} \propto \alpha \log N$')
#ax[0].plot(nu,dic['mean_logRho1'][:,1],'ro',label=r'$\alpha, \mu_{\log \rho} \propto \alpha \log N$')
#ax[0].plot(SA,1/Res[:,3],'+',label=r'$\alpha, \mu_{\log \rho} \propto \alpha \log N$')
ax[1].plot(nu,dic['var_logRho'][:,1],'rs',label=r'$\beta, \sigma^2_{\log \rho} \propto \beta \log N$')
#ax[1].plot(SA,-Res[:,8],label=r'$\beta, \sigma^2_{\log \rho} \propto \beta \log N$')
ax[1].set_xlabel(r'$\nu$')
ax[0].legend()
ax[1].legend()
#ax[1].set_yscale('log')
#ax[0].set_yscale('log')
plt.savefig(dir_out+'rho_Df_{:1.0f}folds.pdf'.format(A.shape[1]),bbox_inches='tight')
#%%% + Plots
plt.style.use('~/.config/matplotlib/joris.mplstyle')
fig,ax=plt.subplots(2,1,figsize=(3,4))
#ax[0].plot(SA,1/dic['D2'],'k--',label=r'$1/(\nu-1)$')
#ax[1].plot(SA,2*(2-(dic['D2']+1))/dic['D2'],'r--',label=r'$2(2-\nu)/(\nu-1)$')
ax[0].errorbar(SA,dic['mean_Cmax'][:,1],yerr=dic['mean_Cmax'][:,3],
	label=r'$\alpha, \log \mu_{\theta} \propto \alpha \log N$')
#ax[0].plot(SA,1/Res[:,3],'+',label=r'$\alpha, \mu_{\log \rho} \propto \alpha \log N$')
ax[1].errorbar(SA,dic['var_Cmax'][:,1],yerr=dic['var_Cmax'][:,3],
	label=r'$\beta, \log \sigma^2_{\theta} \propto \beta \log N$')
#ax[1].plot(SA,-Res[:,8],label=r'$\beta, \sigma^2_{\log \rho} \propto \beta \log N$')
ax[1].set_xlabel('$a$')
ax[0].legend()
ax[1].legend()
ax[0].set_ylim([0,-2])
ax[1].set_ylim([0,-2])
plt.savefig(dir_out+'cmax_Df.pdf',bbox_inches='tight')
#%%% * + Plots
variable='S'
variable_latex=r'1/\rho'
plt.style.use('~/.config/matplotlib/joris.mplstyle')
plt.figure()
plt.errorbar(dic['D2']+1,-dic['mean_'+variable][:,1],yerr=dic['mean_Cmax'][:,2],
						 color='k',fmt='o',
					 label=r'$\gamma_1, \langle'+variable_latex+r'\rangle \propto N^{-\gamma_1}$')
plt.errorbar(dic['D2']+1,-dic['var_'+variable][:,1],yerr=2*dic['var_Cmax'][:,2],
						 color='r',fmt='o',
						 label=r'$\gamma_2, \langle'+variable_latex+r'^2\rangle \propto N^{-\gamma_2}$')
plt.errorbar(dic['D2']+1,-dic['m3_'+variable][:,1],yerr=2*dic['m3_Cmax'][:,2],
						 color='g',fmt='o',
						 label=r'$\gamma_3, \langle'+variable_latex+r'^3\rangle \propto N^{-\gamma_3}$')
plt.errorbar(dic['D2']+1,-dic['m4_'+variable][:,1],yerr=2*dic['m4_Cmax'][:,2],
						 color='b',fmt='o',
						 label=r'$\gamma_4, \langle'+variable_latex+r'^3\rangle \propto N^{-\gamma_4}$')
plt.legend()

plt.ylim([0,4])
plt.hlines(1.0,1.2,2)

nu=np.linspace(1.2,2,1000)
mu=1/(nu-1)
sigma2=2*(2-nu)/(nu-1)
M2=2*mu-2*sigma2
M2[mu-2*sigma2<0]=mu[mu-2*sigma2<0]**2/(2*sigma2[mu-2*sigma2<0])
n=3
M3=n*mu-n**2/2*sigma2
M3[mu-n*sigma2<0]=mu[mu-n*sigma2<0]**2/(2*sigma2[mu-n*sigma2<0])
n=4
M4=n*mu-n**2/2*sigma2
M4[mu-n*sigma2<0]=mu[mu-n*sigma2<0]**2/(2*sigma2[mu-n*sigma2<0])

n=1
M1=n*mu-n**2/2*sigma2
M1[mu-n*sigma2<0]=mu[mu-n*sigma2<0]**2/(2*sigma2[mu-n*sigma2<0])
#plt.figure(figsize=(2.5,2.5))
#plt.plot(nu,mu,'k-',label=r'$\mu=1/(1-\nu)$')
#plt.plot(nu,sigma2,'k--',label=r'$\sigma^2=2(2-\nu)/(1-\nu)$')
plt.plot(nu,M1,'k-',label=r'$\gamma_1$',linewidth=1.5)
plt.plot(nu,M2,'r-',label=r'$\gamma_2$',linewidth=1.5)
plt.plot(nu,M3,'g-',label=r'$\gamma_3$',linewidth=1.5)
plt.plot(nu,M4,'b-',label=r'$\gamma_4$',linewidth=1.5)
#plt.plot(nu,mu**2/(2*sigma2),'k:',label=r'$\mu^2/(2\sigma^2)$')
#plt.plot(nu,2*mu-2*sigma2,'k-.',label=r'$2\mu-2\sigma^2$')
#plt.ylim([0,5])
#plt.legend()
#plt.xlabel(r'$\nu$')
#plt.savefig('nu_mi_sigma2.pdf',bbox_inches='tight')
plt.xlabel(r'$\nu$')

plt.savefig(dir_out+'Moments1_S_Df_bis.pdf',bbox_inches='tight')

variable='Cmax'
variable_latex=r'\theta'
plt.style.use('~/.config/matplotlib/joris.mplstyle')
plt.figure()
plt.errorbar(dic['D2']+1,-dic['mean_'+variable][:,1],yerr=dic['mean_Cmax'][:,2],
					 fmt='o',label=r'$\gamma_1, \langle'+variable_latex+r'\rangle \propto N^{-\gamma_1}$')
plt.errorbar(dic['D2']+1,-dic['var_'+variable][:,1],yerr=2*dic['var_Cmax'][:,2],
						 fmt='o',label=r'$\gamma_2, \langle'+variable_latex+r'^2\rangle \propto N^{-\gamma_2}$')
plt.errorbar(dic['D2']+1,-dic['m3_'+variable][:,1],yerr=2*dic['m3_Cmax'][:,2],
						fmt='o', label=r'$\gamma_3, \langle'+variable_latex+r'^3\rangle \propto N^{-\gamma_3}$')
plt.errorbar(dic['D2']+1,-dic['m4_'+variable][:,1],yerr=2*dic['m4_Cmax'][:,2],
						 fmt='o',label=r'$\gamma_4, \langle'+variable_latex+r'^3\rangle \propto N^{-\gamma_4}$')
plt.legend()

plt.ylim([0,4])
plt.hlines(1.0,1.2,2)
plt.savefig(dir_out+'MomentsCmax_Df_bis.pdf',bbox_inches='tight')

variable='Cmax'
variable_latex=r'\theta'
plt.style.use('~/.config/matplotlib/joris.mplstyle')
plt.figure(figsize=(2,2))
plt.plot(dic['D2']+1,-dic['mean_'+variable][:,1],'ko',label=r'$\gamma_{1,s_B}$')
plt.plot(dic['D2']+1,-dic['var_'+variable][:,1],'ro',fillstyle='top',label=r'$\gamma_{2,s_B}$')
plt.plot(nu,M2,'r--')
#plt.errorbar(dic['D2']+1,-dic['m3_'+variable][:,1],yerr=2*dic['m3_Cmax'][:,2],
#						fmt='o', label=r'$\gamma_3, \langle'+variable_latex+r'^3\rangle \propto N^{-\gamma_3}$')
#plt.errorbar(dic['D2']+1,-dic['m4_'+variable][:,1],yerr=2*dic['m4_Cmax'][:,2],
#						 fmt='o',label=r'$\gamma_4, \langle'+variable_latex+r'^3\rangle \propto N^{-\gamma_4}$')
plt.legend(fontsize=8)
plt.ylim([0,3])
plt.hlines(1.0,1.2,2)
plt.xlabel(r'$D_1$')
plt.savefig('/home/joris/Dropbox/Articles/AggregationBatchelor/Figures/gamma2bundles.pdf',bbox_inches='tight')
#%%% statistics of bundles dependence on var_sa

def m3(x):
	return np.nanmean(x**3)

def m4(x):
	return np.nanmean(x**4)

dic={'nu':[],'D2':[],'mean_logRho1':[],'mean_logRho2':[],'var_logRho':[],
	 'mean_Cmax':[],'var_Cmax':[],'m3_Cmax':[],'m4_Cmax':[],
	 'mean_S':[],'var_S':[],'m3_S':[],'m4_S':[]}

T=12
a=0.5
SA=np.logspace(-3,np.log10(0.49),30)
SA=np.linspace(0.0,0.5,10)
Res=[]
for sa in SA:
	random=sa
	
	x0=np.random.rand(1)
	#x0=np.random.rand(10)
	x,S,wrapped_time=lagrangian(x0,T,a,random)
	nu,N,Nf=fractal(x,2)
	a=sa
	dq=2*3/4.
	D2=(1+dq)*np.log(2)/np.log(1/a**dq+1/(1-a)**dq)
	print(nu)
	#nu=D2
	dic['nu'].append(nu)
	dic['D2'].append(D2)
	
	from scipy import spatial
	xi=x
	#xi_per=np.hstack((xi-1,xi,xi+1))
	xi_per=xi
	
	s0=0.05
	sigma=20
	D=(sigma/2**13)**2/2
	#Sigma=np.sqrt(2*D)*n
	Cmax=1./np.sqrt(1.+4*D/s0**2*wrapped_time)
	Si=s0*np.sqrt(1.+4*D/s0**2*wrapped_time)*S
	
	sB=np.mean(Si)
	
	tree=spatial.cKDTree(xi_per.reshape(-1,1))
	# Take samples in within cmax
	neighboors=tree.query_ball_point(xi.reshape(-1,1), sB)
	
	
	nagg=np.array([len(n) for n in neighboors])
	
	#% Check cmax of bundles
	def bin_operation(x,y,xi,op):
		r=np.zeros(xi.shape[0]-1)
		for i in range(xi.shape[0]-1):
			idx=np.where((x<=xi[i+1])&(x>=xi[i]))[0]
			r[i]=op(y[idx])
		return r
	
	def bundle(neighboors,variable,operator,binarize,nbin):
		v=np.array([operator(variable[n]) for n in neighboors])
		if binarize=='N':
			nagg=np.array([len(n) for n in neighboors])
			nagg_bin=np.logspace(0,np.max(np.log10(nagg)),nbin)
			return bin_operation(nagg,v,nagg_bin,np.nanmean),nagg_bin
		if binarize=='logN':
			nagg=np.array([np.log(len(n)) for n in neighboors])
			nagg_bin=np.linspace(0,np.max(nagg),nbin)
			return bin_operation(nagg,v,nagg_bin,np.nanmean),nagg_bin
		if binarize=='Rho':
			rhoagg_mean=np.array([np.nanmean((1/S[n])) for n in neighboors])
			rho_bin=np.logspace(np.log10(rhoagg_mean.min()),np.log10(rhoagg_mean.max()),nbin)
			return bin_operation(rhoagg_mean,v,rho_bin,np.nanmean),rho_bin
		if binarize=='logRho':
			rhoagg_mean=np.array([np.nanmean(np.log(1/S[n])) for n in neighboors])
			rho_bin=np.linspace((rhoagg_mean.min()),(rhoagg_mean.max()),nbin)
			return bin_operation(rhoagg_mean,v,rho_bin,np.nanmean),rho_bin
		if binarize=='Cmax':
			c_mean=np.array([np.nanmean((Cmax[n])) for n in neighboors])
			c_bin=np.logspace(np.log10(c_mean.min()),np.log10(c_mean.max()),nbin)
			return bin_operation(c_mean,v,c_bin,np.nanmean),c_bin
	
							
	h,x=bundle(neighboors,np.log(1/S),np.nanmean,'logN',30)
	#isgood=np.where(np.isfinite(h))[0]
	#p0=np.polyfit((x[isgood]),(h[isgood]),1)
	dic['mean_logRho1'].append(best_polyfit(x[1:],h))
	
	h,x=bundle(neighboors,np.log(nagg),np.nanmean,'logRho',30)
	#isgood=np.where(np.isfinite(h))[0]
	#p1=np.polyfit((x[isgood]),(h[isgood]),1)
	dic['mean_logRho2'].append(best_polyfit(x[1:],h))
	
	h,x=bundle(neighboors,np.log(1/S),np.nanvar,'logN',30)
	#isgood=np.where(np.isfinite(h))[0]
	#p2=np.polyfit((x[isgood]),(h[isgood]),1)
	dic['var_logRho'].append(best_polyfit(x[1:],h))
	
	h,x=bundle(neighboors,Cmax,np.nanmean,'logN',30)
	#isgood=np.where(np.isfinite(h))[0]
	#p0=np.polyfit((x[isgood]),(h[isgood]),1)
	dic['mean_Cmax'].append(best_polyfit(x[1:],np.log(h)))
	
	h,x=bundle(neighboors,Cmax,np.nanvar,'logN',30)
	dic['var_Cmax'].append(best_polyfit(x[1:],np.log(h)))
	h,x=bundle(neighboors,Cmax,m3,'logN',30)
	dic['m3_Cmax'].append(best_polyfit(x[1:],np.log(h)))
	h,x=bundle(neighboors,Cmax,m4,'logN',30)
	dic['m4_Cmax'].append(best_polyfit(x[1:],np.log(h)))
	h,x=bundle(neighboors,Cmax,np.mean,'logN',30)
	#isgood=np.where(np.isfinite(h))[0]
	#p0=np.polyfit((x[isgood]),(h[isgood]),1)
	h,x=bundle(neighboors,S,np.nanmean,'logN',30)
	dic['mean_S'].append(best_polyfit(x[1:],np.log(h)))
	h,x=bundle(neighboors,S,np.nanvar,'logN',30)
	dic['var_S'].append(best_polyfit(x[1:],np.log(h)))
	h,x=bundle(neighboors,S,m3,'logN',30)
	dic['m3_S'].append(best_polyfit(x[1:],np.log(h)))
	h,x=bundle(neighboors,S,m4,'logN',30)
	dic['m4_S'].append(best_polyfit(x[1:],np.log(h)))

for keys in dic.keys():
	dic[keys]=np.array(dic[keys])
	
plt.plot(SA,dic['nu'])
plt.plot(SA,dic['D2'])
plt.xscale('log')
plt.yscale('log')

#%%% + Plots
plt.style.use('~/.config/matplotlib/joris.mplstyle')
nu=dic['nu']
nu_th=dic['D2']
fig,ax=plt.subplots(2,1,figsize=(3,4))
#ax[0].plot(SA,1/nu,'k--',label=r'$1/(\nu-1)$')
#ax[1].plot(SA,2*(2-(nu+1))/nu,'r--',label=r'$2(2-\nu)/(\nu-1)$')
ax[0].plot(nu,1/nu_th,'k--',label=r'$1/(\nu-1)$')
ax[1].plot(SA,2*(2-(nu_th+1))/nu_th,'r--',label=r'$2(2-\nu)/(\nu-1)$')
ax[0].errorbar(nu,dic['mean_logRho1'][:,1],yerr=dic['mean_logRho1'][:,3],label=r'$\alpha, \mu_{\log \rho} \propto \alpha \log N$')
#ax[0].plot(SA,1/Res[:,3],'+',label=r'$\alpha, \mu_{\log \rho} \propto \alpha \log N$')
ax[1].errorbar(nu,dic['var_logRho'][:,1],yerr=dic['var_logRho'][:,3],label=r'$\beta, \sigma^2_{\log \rho} \propto \beta \log N$')
#ax[1].plot(SA,-Res[:,8],label=r'$\beta, \sigma^2_{\log \rho} \propto \beta \log N$')
ax[1].set_xlabel('$\sigma^2_a$')
ax[0].legend()
ax[1].legend()
#ax[1].set_yscale('log')
#ax[0].set_yscale('log')
#plt.yscale('log')

plt.savefig(dir_out+'rho_var_sa.pdf',bbox_inches='tight')
#%%% + Plots
plt.style.use('~/.config/matplotlib/joris.mplstyle')
fig,ax=plt.subplots(2,1,figsize=(3,4))
#ax[0].plot(SA,1/dic['D2'],'k--',label=r'$1/(\nu-1)$')
#ax[1].plot(SA,2*(2-(dic['D2']+1))/dic['D2'],'r--',label=r'$2(2-\nu)/(\nu-1)$')
ax[0].errorbar(SA,dic['mean_Cmax'][:,1],yerr=dic['mean_Cmax'][:,3],
	label=r'$\alpha, \log \mu_{\theta} \propto \alpha \log N$')
#ax[0].plot(SA,1/Res[:,3],'+',label=r'$\alpha, \mu_{\log \rho} \propto \alpha \log N$')
ax[1].errorbar(SA,dic['var_Cmax'][:,1],yerr=dic['var_Cmax'][:,3],
	label=r'$\beta, \log \sigma^2_{\theta} \propto \beta \log N$')
#ax[1].plot(SA,-Res[:,8],label=r'$\beta, \sigma^2_{\log \rho} \propto \beta \log N$')
ax[1].set_xlabel('$a$')
ax[0].legend()
ax[1].legend()
ax[0].set_ylim([0,-2])
ax[1].set_ylim([0,-2])
plt.savefig(dir_out+'cmax_Df.pdf',bbox_inches='tight')
#%% FRACTAL DIMENSIONS
#%%% Ott antonsen
import scipy.optimize

a=0.1
gamma=-1.0
q=2.001

def transcendental(Dq):
	sigma=(q-1)*(Dq-2)+gamma*q
	return a**(1-sigma)+(1-a)**(1-sigma)-(a**(1-gamma)+(1-a)**(1-gamma))**q

D=scipy.optimize.broyden1(transcendental, 1, f_tol=1e-14)

print(D)

#%%% variance of c as a function of sB

plt.style.use('~/.config/matplotlib/joris.mplstyle')
import scipy
a=0.2
x0=np.float64(np.random.rand(1))
for j,T in enumerate([18,20,22]): # several times
	#x,S,wrapped_time=lagrangian(x0,T,a,0)
	x,S,wrapped_time=lagrangian(x0,T,a,0)
	N=np.logspace(1,3,20)
	Nf=[]
	H=[]
	
	N=np.logspace(1,3,100)
	Nf=[]
	VarC=[]
	q=2.001
	for i,n in enumerate(N): # several sB
		print(n)
		vc=[]
		for k in range(2): # several grids
			c=np.histogram(np.mod(x+np.random.rand(1),1),np.linspace(0,1,int(n)),weights=S*s0/sB,density=False)[0]
			vc.append(np.var(c))
		VarC.append(np.mean(vc))
	
	plt.plot(1/N,VarC,color=plt.cm.cool(j/3),label='$T={:1.0f}$'.format(T))
plt.yscale('log')
plt.xscale('log')
plt.xlabel('$s_B$')
plt.ylabel(r'$\sigma^2_c$')
plt.legend()
plt.savefig(dir_out+'variance_c_sB.pdf',bbox_inches='tight')


#%%% Fractal scaling of different measures

plt.style.use('~/.config/matplotlib/joris.mplstyle')
import scipy
a=0.2
x0=np.float64(np.random.rand(1))
T=19
#x,S,wrapped_time=lagrangian(x0,T,a,0)
x,S,wrapped_time=lagrangian_random(x0,T,a)

N=np.logspace(1,3,20)
Nf=[]
H=[]
q=1.001
for i,n in enumerate(N):
	#print(n)
	h=np.histogram(x,np.linspace(0,1,int(n)),density=False)[0]
	hlog=np.log(h)
	hhlog=np.log(h)*h
	h_rho_1=np.histogram(x,np.linspace(0,1,int(n)),weights=S,density=False)[0]
	h_logrho=np.histogram(x,np.linspace(0,1,int(n)),weights=-np.log(S),density=False)[0]
	h_logrho_n=h_logrho/h
	h_rho=np.histogram(x,np.linspace(0,1,int(n)),weights=1/S,density=False)[0]
	h_rho_n=h_rho/h
	plt.plot(h,h/h_rho_1,'.',color=plt.cm.cool(i/10))
	plt.plot(h,h_rho/h,'+',color=plt.cm.cool(i/10))
	h=h/np.sum(h)
	hlog=hlog/np.nansum(hlog)
	hhlog=hhlog/np.nansum(hhlog)
	h_rho=h_rho/np.sum(h_rho)
	h_rho_n=h_rho_n/np.sum(h_rho_n)
	h_rho_1=h_rho_1/np.sum(h_rho_1)
	h_logrho=h_logrho/np.sum(h_logrho)
	h_logrho_n=h_logrho_n/np.sum(h_logrho_n)
	#h=h/np.sum(h)
	#h_logrho=h_logrho/np.sum(h_logrho)
	Nf.append([np.sum(h**q),np.sum(h_rho**q),np.sum(h_rho_1**q),np.sum(h_logrho**q),np.nansum(h_logrho_n**q),np.nansum(hlog**q),np.nansum(h_rho_n**q),np.nansum(hhlog**q)])
#nu=-np.polyfit(np.log(N[:]),np.log(Nf[:]),1)[0]
Nf=np.array(Nf)
x=np.logspace(-2,2,100)
plt.plot(1e3*x,x**1,'k--',label='1')
plt.plot(1e3*x,x**1.2,'r--',label='1')
plt.yscale('log')
plt.xscale('log')
nu=[best_polyfit(np.log(N),Nf[:,k])[1] for k in range(len(Nf[0]))]
Nf=np.array(Nf)

dq=2
f = lambda Dq : dq*np.log(2)-np.log(1/a**((dq-1)*Dq)+1/(1-a)**((dq-1)*Dq))
D2=scipy.optimize.broyden1(f, 1, f_tol=1e-14)+1

dq=4
f = lambda Dq : dq*np.log(2)-np.log(1/a**((dq-1)*Dq)+1/(1-a)**((dq-1)*Dq))
D4=scipy.optimize.broyden1(f, 1, f_tol=1e-14)+1


dq=1.001
f = lambda Dq : (a**2+(1-a)**2)**dq-a**(2*dq+(1-dq)*Dq)-(1-a)**(2*dq+(1-dq)*Dq)
Drho2p=scipy.optimize.broyden1(f, 1, f_tol=1e-14)+1
print(Drho2p)

D0=2
D1=1+2*np.log(2)/np.log(a**(-1)+(1-a)**(-1))


dq=1.001
f = lambda Dq : 1-a**((1-dq)*Dq)*(1-a)**dq-(1-a)**((1-dq)*Dq)*(a)**dq
Drho1=scipy.optimize.broyden1(f, 1, f_tol=1e-14)+1

dq=2
f = lambda Dq : 1-a**((1-dq)*Dq)*(1-a)**dq-(1-a)**((1-dq)*Dq)*(a)**dq
Drho2=scipy.optimize.broyden1(f, 1, f_tol=1e-14)+1

dq=1.001
f = lambda Dq : 2**dq-a**((1-dq)*Dq)-(1-a)**((1-dq)*Dq)
Dmrho1=scipy.optimize.broyden1(f, 1, f_tol=1e-14)+1

dq=1.001
f = lambda Dq : 2**(dq)-a**((1-dq)*Dq)*(1-a)**dq-(1-a)**((1-dq)*Dq)*a**dq
Dmrho1=scipy.optimize.broyden1(f, 1, f_tol=1e-14)+1

dq=2
f = lambda Dq : 2**dq-a**((1-dq)*Dq)*(1-a)**dq-(1-a)**((1-dq)*Dq)*(a)**dq
Dmrho2=scipy.optimize.broyden1(f, 1, f_tol=1e-14)+1

plt.figure()
plt.plot(np.log(N),np.log(N)*(D0-1),'g--',label='$D_0 -1={:1.2f}$'.format(D0-1))
plt.plot(np.log(N),np.log(N)*(D1-1),'k--',label='$D_1 -1={:1.2f}$'.format(D1-1))
plt.plot(np.log(N),np.log(N)*(D2-1),'r--',label='$D_2 -1={:1.2f}$'.format(D2-1))
plt.plot(np.log(N),np.log(N)*(Drho1-1),'g-',label=r'$D_{\rho,1} -1'+'={:1.2f}$'.format(Drho1-1))
plt.plot(np.log(N),np.log(N)*(Drho2-1),'r-',label=r'$D_{\rho,2} -1'+'={:1.2f}$'.format(Drho2-1))
plt.plot(np.log(N),np.log(Nf[:,0])/(1-q),'.-',label=r'$p_i = n$')
plt.plot(np.log(N),np.log(Nf[:,5])/(1-q),'*-',label=r'$p_i = \log n$')
plt.plot(np.log(N),np.log(Nf[:,7])/(1-q),'k*-',label=r'$p_i = n\log n$')
plt.plot(np.log(N),np.log(Nf[:,1])/(1-q),'+-',label=r'$p_i =  \sum \rho$')
plt.plot(np.log(N),np.log(Nf[:,6])/(1-q),'+-',label=r'$p_i = n^{-1} \sum \rho$')
plt.plot(np.log(N),np.log(Nf[:,2])/(1-q),'o-',label=r'$p_i = \sum \rho^{-1}$')
plt.plot(np.log(N),np.log(Nf[:,3])/(1-q),'d-',label=r'$p_i = \sum \log \rho$')
plt.plot(np.log(N),np.log(Nf[:,4])/(1-q),'d-',label=r'$p_i = n^{-1}\sum \log \rho$')
plt.legend(fontsize=6,ncol=3)
plt.ylabel('$1/(1-q) \log \sum p_i^q$')
plt.xlabel('$- \log s_b$')
plt.title('$q={:1.0f}$'.format(q))
plt.savefig(dir_out+'fractal_scalings_baker_q{:1.0f}.pdf'.format(q))

#%%% log n / log(<1/rho>)

plt.style.use('~/.config/matplotlib/joris.mplstyle')
import scipy
a=0.05
x0=np.float64(np.random.rand(1))
T=20
x,S,wrapped_time=lagrangian(x0,T,a,0)

N=np.logspace(1,3,20)
N=[100]
Nf=[]
H=[]
q=2.0
for i,n in enumerate(N):
	#print(n)
	h=np.histogram(x,np.linspace(0,1,int(n)),density=False)[0]
	h_rho_1=np.histogram(x,np.linspace(0,1,int(n)),weights=S,density=False)[0]/h
	h_logrho=np.exp(np.histogram(x,np.linspace(0,1,int(n)),weights=np.log(1/S),density=False)[0]/h)
	plt.plot(h,1/h_rho_1,'ro',label=r'$<\rho^{-1}|n>$')
	plt.plot(h,h_logrho,'k+',label=r'$\exp <\log \rho|n>$')
x=np.logspace(-2,2,100)

dq=2
f = lambda Dq : dq*np.log(2)-np.log(1/a**((dq-1)*Dq)+1/(1-a)**((dq-1)*Dq))
D2=scipy.optimize.broyden1(f, 1, f_tol=1e-14)+1

D1=1+2*np.log(2)/np.log(a**(-1)+(1-a)**(-1))
plt.plot(1e3*x,1e5*x**1,'r--',label='1')
plt.plot(1e3*x,1e8*x**(1/(D1-1)),'k-',label='$(D_1-1)^{-1}$')
#plt.plot(1e3*x,1e9*x**(1/(D2-1)),'b:',label='$(D_2-1)^{-1}$')
plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.xlabel('$n$')

#%%%

#% Fractal dimensions
def fractal_nonnormalized(x,q):
	N=np.logspace(1,4,50)
	Nf=[]
	H=[]
	for n in N:
		#print(n)
		h=np.histogram(x,np.linspace(0,1,int(n)))[0]
		hn=h/np.sum(h)
		loghn=np.log(h[h>0])/np.sum(np.log(h[h>0]))
		H.append(np.sum(hn**2.))
		Nf.append([np.log(np.nansum(hn[hn>0]**0))/(-1),np.nansum(loghn**2),np.log(np.nansum(hn[hn>0]**1.001))/(0.001),np.log(np.nansum(hn[hn>0]**2.))/(1)])
	#nu=-np.polyfit(np.log(N[:]),np.log(Nf[:]),1)[0]
	H=np.array(H)
	nu=[best_polyfit(np.log(N),np.array(Nf)[:,k])[1] for k in range(len(Nf[0]))]
	print(nu)
	return -nu[1],N,Nf,nu[3]

N=np.logspace(1,4,50)
Nf=[]
H=[]
for n in N:
	#print(n)
	h=np.histogram(x,np.linspace(0,1,int(n)))[0]
	hn=h/np.sum(h)
	loghn=np.log(h[h>0])/np.sum(np.log(h[h>0]))
	H.append(np.sum(hn**2.))
	Nf.append([np.log(np.nansum(hn[hn>0]**0))/(-1),np.nansum(loghn**2),np.log(np.nansum(hn[hn>0]**1.001))/(0.001),np.log(np.nansum(hn[hn>0]**2.))/(1)])
#nu=-np.polyfit(np.log(N[:]),np.log(Nf[:]),1)[0]
H=np.array(H)
nu=[best_polyfit(np.log(N),np.array(Nf)[:,k])[1] for k in range(len(Nf[0]))]
Nf=np.array(Nf)
plt.plot(np.log(N),Nf-Nf[0,:])
plt.yscale('log')

plt.style.use('~/.config/matplotlib/joris.mplstyle')
import scipy
#A=np.logspace(-7,np.log10(0.49),100)
var_sa=0.0
A=np.linspace(var_sa,0.5-var_sa,100)
A=np.logspace(-1,np.log10(0.5),100)
A=np.linspace(0.01,0.49,100)
x0=np.float64(np.random.rand(1))
#x0=np.random.rand(100)
T=20
plt.figure(figsize=(5,5))
Q=np.linspace(2,10,10)
Q=[1.00]
for q in Q:
	Nu=[]
	Dq=[]
	for a in A:

		x,S,wrapped_time=lagrangian(x0,T,a,var_sa)
		nu,N,Nf,snu=fractal_nonnormalized(x,q)
		#nu,N,Nf=fractal(x,q)
		nuS=fractal_weighted(x,S,q)
		print(nu)
		Nu.append([nu,snu,nuS])
		#f = lambda Dq : 2**q-1/a**((q-1)*Dq)+1/(1-a)**((q-1)*Dq)
		dq=2
		f = lambda Dq : dq*np.log(2)-np.log(1/a**((dq-1)*Dq)+1/(1-a)**((dq-1)*Dq))
		Dq.append(scipy.optimize.broyden1(f, 1, f_tol=1e-14))
	
	b=q
	Nu=np.array(Nu)
#	plt.errorbar(A,Nu[:,0],yerr=2*Nu[:,1],label=r'$D_{:1.0f}(1)$'.format(q),color=plt.cm.viridis(q/5.))
	plt.plot(A,2-Nu[:,0],'o',label=r'$d \log n / d r$'.format(q),color=plt.cm.viridis(q/10.))
	#plt.plot(A,Nu[:,1],label=r'$D_2(s)$')
	#plt.plot(A,-np.log(A**2.+(1-A)**2.)/np.log(2),'k--',label='$(\log(a^2+(1-a)^2)/\log 2$')
	plt.plot(A,(b+1)*np.log(2)/np.log(1/A**b+1./(1-A)**b),'-',color=plt.cm.viridis(q/10.),label='$D_1-1$')
	plt.plot(A,Dq,'--',color=plt.cm.viridis(q/10.),label='$D_2-1$')

plt.xlabel(r'$a$')
#plt.xscale('log')
#plt.yscale('log')
plt.legend()

#%%% Check dependence of nu parameter a
plt.style.use('~/.config/matplotlib/joris.mplstyle')
import scipy
#A=np.logspace(-7,np.log10(0.49),100)
var_sa=0.0
A=np.linspace(var_sa,0.5-var_sa,100)
A=np.logspace(-1,np.log10(0.5),100)
A=np.linspace(0.01,0.49,100)
x0=np.float64(np.random.rand(1))
#x0=np.random.rand(100)
T=15
plt.figure(figsize=(5,5))
Q=np.linspace(2,10,10)
Q=[1.01]
for q in Q:
	Nu=[]
	Dq=[]
	for a in A:
		x,S,wrapped_time=lagrangian(x0,T,a,var_sa)
		nu,N,Nf,snu=fractal_best(x,q)
		#nu,N,Nf=fractal(x,q)
		nuS=fractal_weighted(x,S,q)
		print(nu)
		Nu.append([nu/(q-1),snu/(q-1),nuS/(q-1)])
		#f = lambda Dq : 2**q-1/a**((q-1)*Dq)+1/(1-a)**((q-1)*Dq)
		dq=q
		f = lambda Dq : dq*np.log(2)-np.log(1/a**((dq-1)*Dq)+1/(1-a)**((dq-1)*Dq))
		Dq.append(scipy.optimize.broyden1(f, 1, f_tol=1e-14))
	
	b=q
	Nu=np.array(Nu)
#	plt.errorbar(A,Nu[:,0],yerr=2*Nu[:,1],label=r'$D_{:1.0f}(1)$'.format(q),color=plt.cm.viridis(q/5.))
	plt.plot(A,Nu[:,0]/Nu[:,0].max(),'o',label=r'$D_{:1.0f}(1)$'.format(q),color=plt.cm.viridis(q/10.))
	#plt.plot(A,Nu[:,1],label=r'$D_2(s)$')
	#plt.plot(A,-np.log(A**2.+(1-A)**2.)/np.log(2),'k--',label='$(\log(a^2+(1-a)^2)/\log 2$')
	plt.plot(A,(b+1)*np.log(2)/np.log(1/A**b+1./(1-A)**b),'+',color=plt.cm.viridis(q/10.))
	plt.plot(A,Dq,'--',color=plt.cm.viridis(q/10.))

plt.plot([],[],'k--',label=r'$D_q=(d_q+1)\log 2/(\log(\bar{a}^{-d_q}+(1-\bar{a})^{-d_q}),'+r'd_q=q$'.format(b))
plt.plot([],[],'k-',label=r'$2^{d_q}=1/a^{(d_q-1)D_q}+1/(1-a)^{(d_q-1)D_q}, d_q = q$')
plt.xlabel(r'$a$')
#plt.xscale('log')
#plt.yscale('log')
plt.legend()
plt.title('Var(a)={:1.2}'.format(var_sa))

plt.savefig(dir_out+'a.pdf',bbox_inches='tight')

plt.plot(A,(b+1)*np.log(2)/np.log(1/A**b+1./(1-A)**b)-Dq)
#%%% f(alpha) LEgendre transform
# b = q

Q=np.logspace(-3,1,100)
A=np.linspace(0.1,0.4,5)

fig,ax=plt.subplots(1,1,figsize=(2,2))
fig1,ax1=plt.subplots(1,1,figsize=(2,2))
for a in A:
	Dq=[]
	for dq in Q:
		f = lambda Dq : dq*np.log(2)-np.log(1/a**((dq-1)*Dq)+1/(1-a)**((dq-1)*Dq))
		Dq.append(scipy.optimize.broyden1(f, 1, f_tol=1e-14))
	
	# plt.figure()
	ax.plot(Q,np.array(Dq),'*-',label='$a={:1.2f}$'.format(a),color=plt.cm.jet(a*2.))
	Dq=np.array(Dq)
	alpha=np.diff((Q-1)*Dq)/np.diff(Q)
	f_alpha=Q[:-1]*alpha-(Q[:-1]-1)*Dq[:-1]
	ax1.plot(f_alpha,alpha,'*-',label='$a={:1.2f}$'.format(a),color=plt.cm.jet(a*2.))
ax1.set_xlabel(r'$\alpha$')
ax1.set_ylabel(r'$f(\alpha)$')
ax.set_xlabel('$q$')
ax.set_ylabel('$D_q$')
ax.legend()
ax1.legend()
#FIXME

#%%% Check dependence of nu_s parameter a
plt.style.use('~/.config/matplotlib/joris.mplstyle')
import scipy
var_sa=0.0
A=np.linspace(0.01,0.49,100)
x0=np.float64(np.random.rand(1))

T=15
plt.figure(figsize=(5,5))
for q in [2,4,8]:
	Nu=[]
	Dq=[]
	for a in A:
		print(q,a)
		x,S,wrapped_time=lagrangian(x0,T,a,var_sa)
		nuS=fractal_weighted(x,S,q)
		Nu.append([nuS/(q-1)])
	b=q
	Nu=np.array(Nu)
	plt.plot(A,Nu[:,0],'o',label=r'$D_{:1.0f}(1)$'.format(q),color=plt.cm.viridis(q/10.))
plt.xlabel(r'$a$')
plt.legend()
plt.title('Var(a)={:1.2}'.format(var_sa))
plt.savefig(dir_out+'s_a.pdf',bbox_inches='tight')
#%%% Fractal dimension time dependence

#2021-08-31_07:39:14
plt.style.use('~/.config/matplotlib/joris.mplstyle')
import scipy

x0=np.array([0.1546],dtype=np.float64)
#A=np.logspace(-7,np.log10(0.49),100)
var_sa=0
q=2
plt.figure(figsize=(5,5))
s0=0.05
sigma=20
n=2**14
D=(sigma/n)**2/2
X,Y=np.meshgrid(0,np.arange(n)) 
#Sigma=np.sqrt(2*D)*n
for a in [0.4]:
	Nu=[]
	t=np.arange(1,25)
	for T in t:
		print(T)
		x,S,wrapped_time=lagrangian(x0,T,a,var_sa)
		C=DNS_n(2**14,x0,T,a,var_sa)
		Cmax=1./np.sqrt(1.+4*D/s0**2*wrapped_time)
		Si=s0*np.sqrt(1.+4*D/s0**2*wrapped_time)*S
		sB=np.mean(Si)
		#sB=0.001		#h,xh=np.histogram(x,np.arange(0,1,sB),weights=Cmax*Si/np.sqrt(2*np.pi))
		#Clag=reconstruct_c(Y,x,Cmax,Si)
		bins=np.arange(0,1,sB)
		#hn,x=np.histogram(xi,bins)
		#hcmax,xb=np.histogram(x,bins,weights=Cmax*Si/sB*np.sqrt(np.pi))
		hcmax,xb=np.histogram(x,bins,weights=s0*S/sB*np.sqrt(np.pi))
		hcmax1,xb=np.histogram(x,bins,weights=Cmax*np.sqrt(np.pi))
# 		plt.figure()
# 		plt.plot(np.linspace(0,1,C.shape[1]),C.T,'r-')
# 		plt.plot(np.linspace(0,1,Clag.shape[0]),Clag,'m--')
# 		plt.savefig('img{:1.0f}.tif'.format(T))
		nuS=fractal_weighted_scale(x,S,q,sB)
#		Nu.append([nuS,np.var(C),np.var(hcmax),np.var(hcmax1),np.var(Clag)])
		Nu.append([nuS,np.var(C),np.var(hcmax),np.var(hcmax1),
						 np.mean(hcmax),np.mean(hcmax1),np.mean(C)])
	b=q
	Nu=np.array(Nu)
#	plt.plot(t,Nu[:,0].max()-Nu[:,0],'o-',label=r'$1-D_2,a={:1.1f}$'.format(a),color=plt.cm.viridis(a*2))
#	plt.plot(t,Nu[:,1]/Nu[0,1],'+--',label=r'$\sigma^2_c, a={:1.1f}$'.format(a),color=plt.cm.viridis(a*2))
# plt.yscale('log')
# plt.xlabel(r'$t$')
# #plt.ylabel(r'$1-D_{:1.0f}(1)$'.format(q))
# plt.legend()
# plt.savefig(dir_out+'s_t.pdf',bbox_inches='tight')

plt.figure()
plt.plot(t,Nu[:,2],'--',label=r'Var $ (\sqrt{\pi}s_B)^{-1} \sum \theta s $')
plt.plot(t,Nu[:,3],'-',label=r'Var $ (\sqrt{\pi})^{-1} \sum \theta $')
plt.plot(t,Nu[:,4],'--',label=r'Mean $ (\sqrt{\pi}s_B)^{-1} \sum \theta s $')
plt.plot(t,Nu[:,5],'-',label=r'Mean $ (\sqrt{\pi})^{-1} \sum \theta $')
plt.plot(t,Nu[:,1],label=r'Var $c$')
plt.plot(t,Nu[:,6],label=r'Mean $c$')
#plt.plot(t,Nu[:,4],'+-',label=r'Var $ c_{lag} $')
plt.yscale('log')
plt.legend()
plt.xlabel('Time')


plt.figure()
plt.plot(xb[1:],hcmax,label=r'$\sum \theta s$')
#plt.plot(xb[1:],hcmax1,label=r'$s_B \sum \theta$')
plt.ylabel('$c(x)$')
plt.xlabel('$x$')
plt.legend()

#%%% Check dependence on parameter a nfolds
plt.style.use('~/.config/matplotlib/joris.mplstyle')
import scipy
#A=np.logspace(-7,np.log10(0.49),100)
var_sa=0.0
A=np.random.rand(50,3)
A=(A.T/np.sum(A,axis=1)).T
x0=np.float64(np.random.rand(1))
#x0=np.random.rand(100)
T=7
plt.figure(figsize=(3,3))
for q in [2,4,8]:
	Nu=[]
	Dq=[]
	for a in A:
		print(a)
		x,S,wrapped_time=lagrangian_nfold(x0,T,a,var_sa)
		nu,N,Nf,snu=fractal_best(x,q)
		#nu,N,Nf=fractal(x,q)
		nuS=fractal_weighted(x,S)
		print(nu)
		Nu.append([nu/(q-1),snu/(q-1),nuS])
		#f = lambda Dq : 2**q-1/a**((q-1)*Dq)+1/(1-a)**((q-1)*Dq)
		dq=q
#		f = lambda Dq : dq*np.log(2)-np.log(1/a**((dq-1)*Dq)+1/(1-a)**((dq-1)*Dq))
#		Dq.append(scipy.optimize.broyden1(f, 1, f_tol=1e-14))
	
	b=q*3/4
	Dqth=(b+1)*np.log(A.shape[1])/np.log(np.sum(1/A**b,axis=1))
	Nu=np.array(Nu)
#	plt.errorbar(A,Nu[:,0],yerr=2*Nu[:,1],label=r'$D_{:1.0f}(1)$'.format(q),color=plt.cm.viridis(q/5.))
	plt.plot(Dqth,Nu[:,0],'o',label=r'',color=plt.cm.viridis(q/10.))
	#plt.plot(A,Nu[:,1],label=r'$D_2(s)$')
	#plt.plot(A,-np.log(A**2.+(1-A)**2.)/np.log(2),'k--',label='$(\log(a^2+(1-a)^2)/\log 2$')
#	plt.plot(A,(b+1)*np.log(2)/np.log(1/A**b+1./(1-A)**b),'--',color=plt.cm.viridis(q/10.))
#	plt.plot(A,Dq,'-',color=plt.cm.viridis(q/10.))
plt.plot([0,1],[0,1],'k--')
#plt.plot([],[],'k--',label=r'$D_q=(d_q+1)\log 2/(\log(\bar{a}^{-d_q}+(1-\bar{a})^{-d_q}),'+r'd_q=q$'.format(b))
#plt.plot([],[],'k-',label=r'$2^{d_q}=1/a^{(d_q-1)D_q}+1/(1-a)^{(d_q-1)D_q}, d_q = q$')
plt.xlabel(r'$(d_q+1)\log(n) / \log( \sum^{n} a_i ^{d_q} )$')
plt.ylabel(r'$D_q$')
#plt.xscale('log')
#plt.yscale('log')
plt.legend()
plt.title('Var(a)={:1.2}'.format(var_sa))

plt.savefig(dir_out+'a.pdf',bbox_inches='tight')

plt.plot(A,(b+1)*np.log(2)/np.log(1/A**b+1./(1-A)**b)-Dq)

#%%% Check dependence on a
A=np.linspace(0.01,0.49,50)
s0=0.05
D=(sigma/n)**2/2
#Sigma=np.sqrt(2*D)*n
T=13
Nu=[]
for a in A:
	x,S,wrapped_time=lagrangian(T,a,0,0)
	nu=fractal(x)
	print(nu)
	xi_per=x
	Cmax=1./np.sqrt(1.+4*D/s0**2*wrapped_time)
	Si=s0*np.sqrt(1.+4*D/s0**2*wrapped_time)*S
	sB=np.mean(Si)
	tree=spatial.cKDTree(xi_per.reshape(-1,1))
	# Take samples in within cmax
	neighboors=tree.query_ball_point(x.reshape(-1,1), sB)
	nbin=50
	nagg=np.array([len(n) for n in neighboors])
	rhoagg_mean=np.array([np.nanmean((1/S[n])) for n in neighboors])
	rho_bin=np.logspace(np.log10(rhoagg_mean.min()),np.log10(rhoagg_mean.max()),nbin)
	h=bin_operation(rhoagg_mean,nagg,rho_bin,np.nanmean)
	isgood=np.where(np.isfinite(h))[0]
	p=np.polyfit(np.log(rho_bin[isgood]),np.log(h[isgood]),1)
	print(p[0])
	Nu.append([nu,p[0]])

Nu=np.array(Nu)
plt.figure()
plt.plot(A,Nu[:,0],label=r'Correlation dimension')
#plt.plot(A,Nu[:,1],label=r'Exponent $N\sim \rho^\alpha$')
plt.plot(A,-np.log(A**2.+(1-A)**2.)/np.log(2),'k--',label='$(\log(a^2+(1-a)^2)/\log 2$')
plt.xlabel(r'$a$')
plt.legend()




#%%% D2 = f(sigma2/lambda)

plt.style.use('~/.config/matplotlib/joris.mplstyle')
A=np.linspace(0,0.5,1000)
M,V=lyapunov(A)
D1=1+2*np.log(2)/np.log(A**(-1)+(1-A)**(-1))
plt.figure(figsize=(3,2))
plt.plot(D1,V/M,'k-',label='Baker map $a\in[0,0.5]$')

#plt.plot(D1,2*(2-D1)+0.35,'r--',label='2(2-D1)+0.35')
#plt.plot(D1,2*(2-D1),'g--',label='2(2-D1)')



#plt.plot(A,V-2*(2-D1)*M)

SINE=np.loadtxt('Sine_Lyap.txt')
SINE_D1=np.loadtxt('Sine_D1.txt')
plt.plot(SINE_D1[:,1],SINE[:,2]/SINE[:,1],'r-',label='Sine flow $A\in[0.3,1.8]$',linewidth=1.5)
#plt.plot(SINE_D1[:,1]+0.1,(SINE[:,2]/SINE[:,1]-0.6),'r--',label='Sine flow (modified)',linewidth=1.5)

plt.plot(SINE_D1[:,1]-1,(SINE[:,1]+SINE[:,2]/2)/(SINE[:,1]+SINE[:,2]),'*')
plt.plot(D1-1,2*np.log(2)/np.log(A**(-1)+(1-A)**(-1)),'o')

ratio=np.linspace(0,4,1000)
#plt.plot(1+(1+ratio/2)/(1+ratio),ratio,'r--',label='Sine Flow')
plt.legend()
plt.xlabel(r'$D_1$')
plt.ylabel(r'$\sigma^2/\mu$')
plt.ylim([0,4])
plt.xlim([1.2,2])
#plt.yscale('log')

plt.savefig('/home/joris/Dropbox/Articles/AggregationBatchelor/Figures/fractal_lyapunov.pdf',bbox_inches='tight')

A=np.linspace(0,0.5,100)
M,V=lyapunov(A)
D1=1+2*np.log(2)/np.log(A**(-1)+(1-A)**(-1))
plt.figure()
plt.plot(2-D1,V-2*(2-D1)*M,'k-',label=r'$\sigma^2_{\log\rho}/t-2(2-D_1)\mu_{\log\rho}/t$')
plt.plot(2-D1,1.2*(2-D1)**2,'r--',label=r'$1.2(2-D_1)^2$')
#plt.ylim([0,0.25])
plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.xlabel('$2-D_1$')
#%%%
SINE=np.loadtxt('lyap_sigma_nu_half.txt')
plt.plot(SINE[:,2]-1,SINE[:,1]/SINE[:,0],'bo',label='Sine flow (half)')
#plt.plot([0,1],[1,1],'k--')
plt.xlabel('$D_2-1$')
plt.ylabel('$\sigma^2_\lambda / \mu_\lambda$')
plt.yscale('log')
plt.ylim([1e-2,1e1])
plt.legend()
plt.savefig(dir_out+'/D_mu.pdf')
#%%% dependance on var_sa
a=0.5
plt.style.use('~/.config/matplotlib/joris.mplstyle')
K=30 # Realisations
x0=np.float64(np.random.rand(1))

T=12 # Maximum time
import scipy.integrate as integrate
c=plt.cm.viridis(np.linspace(0,1,10))
b=1.5
D2=lambda a : (b+1)*np.log(2)/np.log(1/a**b+1./(1-a)**b)
for a in [0.1,0.2,0.3,0.4,0.5]:
	Nu=[]
	V=np.linspace(0,2*a,50)
	for var_sa in V:
		Nutemp=[]
		for k in range(K):
			x,S,wrapped_time=lagrangian(x0,T,a,var_sa)
			nu,N,Nf=fractal(x)
			Nutemp.append(nu)
		D2m=integrate.quad(D2,a-var_sa/2,a+var_sa/2)[0]/(var_sa)
		Nu.append([np.mean(Nutemp),D2m])
		print(a,np.mean(nu))
	Nu=np.array(Nu)
	plt.plot(1/12*V**2.,Nu[:,0]-Nu[0,0],label=r'$a={:1.1}$'.format(a),color=plt.cm.viridis(2*a))
	plt.plot(1/12*V**2.,Nu[:,1]-D2(a),'k--',label=r'$\overline{D_2}$',color=plt.cm.viridis(2*a))


V=np.linspace(0,1,50)

plt.xlabel(r'$\sigma^2_a=1/12(a_*-b_*)^2$')
plt.ylabel(r'$D_2 - D_2(\sigma^2_a=0)$')
plt.legend()
plt.savefig(dir_out+'vara.pdf',bbox_inches='tight')

#plt.xscale('log')
#%% N(t)
#%%% * N(t) 
plt.style.use('~/.config/matplotlib/joris.mplstyle')
x0=np.array([0.1])
var_sa=0

s0=0.01
D=1e-8
Nmean=[]
Lmean=[]
k=[]
TT=np.arange(0,40)
for T in TT:
	a=0.1
	lyap=-a*np.log(a)-(1-a)*np.log(1-a)
	sigma2=a*np.log(a)**2.+(1-a)*np.log(1-a)**2.-lyap**2.
	C=DNS_n(2**13,x0,T,a,var_sa)
	k.append(np.mean(C)**2/np.var(C))
	if T<28:
		x,S,wrapped_time=lagrangian(x0,T,a,var_sa)
		Cmax=1./np.sqrt(1.+4*D/s0**2*wrapped_time)
		Si=s0*np.sqrt(1.+4*D/s0**2*wrapped_time)*S
		sB=np.mean(Si)
		sB=np.sqrt(D/lyap)
		nagg_lag=np.histogram(x,np.linspace(0,1,int(1/sB)))[0]
		Nmean.append(np.mean(nagg_lag[nagg_lag>0]))
		Lmean.append(len(x))
		

# Read sine flow
Nsine=np.loadtxt('N(t)_sine_flow.txt')
Ksine=np.loadtxt('k_sine_flow.txt')

#a=0.45
lt=np.linspace(0,30,100)
lyap=-a*np.log(a)-(1-a)*np.log(1-a)
sigma2=a*np.log(a)**2.+(1-a)*np.log(1-a)**2.-lyap**2.
fig,ax=plt.subplots(1,2,figsize=(4,1.5))
ax[0].plot(np.arange(len(Nmean))*lyap,Nmean,'ko')
ax[0].plot(np.arange(len(k))*lyap,k,'rs')
ax[0].plot(np.arange(len(Lmean))*lyap,np.array(Lmean)*sB,'k-')
ax[0].plot(lt,1e-2*np.exp(lt*0.8),'r-')
#ax[0].plot(np.array([20,30])*lyap,1e3*np.exp(np.array([0,10])*lyap**2/(2*sigma2)),'k-',label=r'$e^{\lambda^2/(2\sigma^2_\lambda)t}$')
#ax[0].plot(np.array([20,30])*lyap,1e3*np.exp(np.array([0,10])*(lyap+sigma2/2)),'k-',linewidth=1.2,label=r'$e^{(\lambda+\sigma^2_\lambda/2)t}$')
ax[0].set_ylim([1e-1,1e5])
ax[0].legend()
ax[0].set_yscale('log')
ax[0].set_xlabel('$\lambda t$')
ax[0].set_xlim([0,15])
ax[0].text(0.05,0.9,'a.',transform=ax[0].transAxes)


ax[1].plot(Nsine[:,0],Nsine[:,2],'ko',label=r'$\langle n \rangle$')
ax[1].plot(Ksine[:,0],Ksine[:,1],'rs',label=r'$k=\langle c \rangle^2/\langle c^2 \rangle $')
ax[1].plot(Nsine[:,0],Nsine[:,1],'k-')
ax[1].set_yscale('log')
ax[1].set_xlabel('$\lambda t$')
ax[1].set_xlabel('$\lambda t$')
lyap=0.54
sigma2=0.4
ax[1].plot(lt,1e-2*np.exp(lt*0.8),'r-')
#ax[1].plot(np.array([0,5])*lyap,1e-2*np.exp(np.array([00,5])*(lyap+sigma2/2)),'k-',linewidth=1.2)
ax[1].set_ylim([1e-2,1e2])
ax[1].set_xlim([0,10])
ax[1].text(0.05,0.9,'b.',transform=ax[1].transAxes)

ax[1].legend(fontsize=8, handletextpad=0.02,frameon=False)
fig.subplots_adjust(wspace=0.25)

fig.savefig('/home/joris/Dropbox/Articles/AggregationBatchelor/Figures/aggregation_scalings.pdf',bbox_inches='tight')
#%%% * Spatial mean of mean(log rho|n) and var(log rho|n)
s=3

Res=[]
Res2=[]

plt.style.use('~/.config/matplotlib/joris.mplstyle')
x0=np.random.rand(1)
var_sa=0

s0=0.01
D=1e-8
Nmean=[]
Lmean=[]
k=[]
TT=np.arange(0,24)
for T in TT:
	a=0.15
	
	dq=1.
	d1=(1+dq)*np.log(2)/np.log(1/a**dq+1/(1-a)**dq)+1
	lyap=-a*np.log(a)-(1-a)*np.log(1-a)
	sigma2=a*np.log(a)**2.+(1-a)*np.log(1-a)**2.-lyap**2.
	x,S1,wrapped_time=lagrangian(x0,T,a,var_sa)
	sB=1/50
	#	for i,sB in enumerate([1/100]):
	N1=np.histogram(x,np.arange(0,1+sB,sB),density=False)[0]
	logrho=np.histogram(x,np.arange(0,1+sB,sB),density=False
						  ,weights=np.log(1/S1))[0]
	logrho2=np.histogram(x,np.arange(0,1+sB,sB),density=False
						  ,weights=np.log(1/S1)**2)[0]
	
	Mlogrho=logrho/N1
	Vlogrho=logrho2/N1-Mlogrho**2.
	
	# compare with moments of the strip
	
	ML=np.average(np.log(1/S1),weights=S1)
	L2=np.average(np.log(1/S1)**2.,weights=S1)
	VL=L2-ML**2.
	
	MLL=np.mean(np.log(1/S1))
	VLL=np.var(np.log(1/S1))
	
	Res.append([T,np.nanmean(Mlogrho),np.average(Mlogrho[N1>0],weights=N1[N1>0]),ML,MLL,np.nanmean(Vlogrho),np.average(Vlogrho[N1>0],weights=N1[N1>0]),VL,VLL])
	# Difference
	diff1=np.nanmean(Vlogrho)-np.nanmean(2*(2-d1)*Mlogrho)
	diff2=VL-2*(2-d1)*ML
	diff3=VLL-2*(2-d1)*MLL
	Res2.append([T,diff1,diff2,diff3])
	
Res=np.array(Res)
i=1
plt.plot(Res[:,0],Res[:,i],'b-',label=r'$\mu_B$'); i+=1
plt.plot(Res[:,0],Res[:,i],'bo',label=r'$\mu_{n,B}$'); i+=1
plt.plot(Res[:,0],Res[:,i],'b--',label=r'$\mu_0$'); i+=1
plt.plot(Res[:,0],Res[:,i],'b:',label=r'$\mu_L$'); i+=1

plt.xlabel('$t$')
plt.legend()

plt.title(r'Mean of $\log \rho \, (a={:1.1f})$'.format(a))

plt.figure()
plt.plot(Res[:,0],Res[:,i],'r-',label=r'$\sigma^2_B$'); i+=1
plt.plot(Res[:,0],Res[:,i],'ro',label=r'$\sigma^2_{n,B}$'); i+=1
plt.plot(Res[:,0],Res[:,i],'r--',label=r'$\sigma^2_0$'); i+=1
plt.plot(Res[:,0],Res[:,i],'r:',label=r'$\sigma^2_L$'); i+=1

plt.xlabel('$t$')
plt.legend()

plt.title(r'Variance of $\log \rho \, (a={:1.2f})$'.format(a))

# Theory works well
# mu_0=-a*np.log(a)-(1-a)*np.log(1-a)
# sigma_0=a*np.log(a)**2.+(1-a)*np.log(1-a)**2.-mu_0**2.
# mu_L=(-np.log(a)-np.log(1-a))/2.
# sigma_L=(np.log(a)**2.+np.log(1-a)**2.)/2-mu_L**2.
# plt.plot(Res[:,0],mu_L*Res[:,0],'bs')
# plt.plot(Res[:,0],mu_0*Res[:,0],'bd')
# plt.plot(Res[:,0],sigma_0*Res[:,0],'rs')
# plt.plot(Res[:,0],sigma_L*Res[:,0],'rd')

Res2=np.array(Res2)
plt.figure()
plt.plot(Res2[:,0],(d1-1)**2.*Res2[:,1],'-+',label='$x = B$')
plt.plot(Res2[:,0],(d1-1)**2.*Res2[:,2],'-d',label='$x = 0$')
plt.plot(Res2[:,0],(d1-1)**2.*Res2[:,3],'-o',label='$x = L$')
plt.ylabel('$\sigma^2_x-2(2-D_1)\mu_x$')
plt.xlabel('$t$')
plt.legend()
plt.title(r'Variance of $\log \rho \, (a={:1.2f})$'.format(a))
#%%% p(N) on t
#2021-09-02_11M14:12
from scipy import spatial
plt.style.use('~/.config/matplotlib/joris.mplstyle')
import scipy

x0=np.array([0.1546],dtype=np.float64)
#A=np.logspace(-7,np.log10(0.49),100)
var_sa=0
q=2
plt.figure(figsize=(5,5))
s0=0.05
sigma=20
n=2**13
D=(sigma/n)**2/2
X,Y=np.meshgrid(0,np.arange(n)) 
#Sigma=np.sqrt(2*D)*n
for a in [0.49]:
	Nu=[]
	t=np.arange(1,10)
	t=np.array([14])
	for T in t:
		print(T)
		C=DNS_n(2**13,x0,T,a,var_sa)
		x,S,wrapped_time=lagrangian(x0,T,a,var_sa)
		Cmax=1./np.sqrt(1.+4*D/s0**2*wrapped_time)
		Si=s0*np.sqrt(1.+4*D/s0**2*wrapped_time)*S
		sB=np.mean(Si)
		#sB=2.2*np.mean(Si)
		# Lagrangian
		tree=spatial.cKDTree(x.reshape(-1,1))
# =============================================================================
# 	# All sampling
# =============================================================================
		#neighboors=tree.query_ball_point(x.reshape(-1,1)[np.abs(x-0.5)<0.5-2*sB], sB)
# =============================================================================
# 		# tAKE SAMPLES UNIFORMLY DISTRIBUTED
# =============================================================================
		ng=1000
		idsamples=np.unique(np.uint16(x.reshape(-1,1)*ng), return_index=True,axis=0)[1]
		neighboors=tree.query_ball_point(x.reshape(-1,1)[idsamples], sB/2.)
		
		cm=[np.sum(Cmax[n]*np.mean(Si)/sB*np.sqrt(np.pi)) for n in neighboors] # is subjected to noise! 
		cm2=[np.sum(s0*S[n]/sB*np.sqrt(np.pi)) for n in neighboors]
		nagg_lag=[len(n) for n in neighboors]
		cmm=2*s0*np.sqrt(np.pi)
		bin_c=np.linspace(cmm-0.1,cmm+0.1,100)
		bin_n=np.unique(np.uint16(np.logspace(0,3,100)))
		h2_lag=np.histogram2d(cm2,nagg_lag,[bin_c,bin_n],density=True)[0]
		# Eulerian
		bins=np.arange(0,1,sB)
		nagg_eul,xb=np.histogram(x,bins)
		nagg_eul=nagg_eul[nagg_eul>0]
				#sB=0.001		#h,xh=np.histogram(x,np.arange(0,1,sB),weights=Cmax*Si/np.sqrt(2*np.pi))
		# C, reconstructed
		hcmax,xb=np.histogram(x,bins,weights=s0*S/sB*np.sqrt(np.pi))
		# p(C|N)
		
		coeff_var=hcmax.std()/hcmax.mean()
		Nu.append([np.mean(cm),np.var(cm),np.mean(cm2),np.var(cm2),
						 np.var(C),np.var(hcmax),coeff_var])
		
Nu=np.array(Nu)
plt.figure()
#plt.plot(t,Nu[:,0],label=r'Mean $\sum \theta$')
#plt.plot(t,Nu[:,1],label=r'Var $\sum \theta$')
#plt.plot(t,Nu[:,2],'--',label=r'Mean $s_B^{-1}\sum \theta s$')
plt.plot(t,Nu[:,3],'b--',label=r'Var $s_B^{-1}\sum \theta s$  (Lag)')
plt.plot(t,Nu[:,5],'r:',label=r'Var $s_B^{-1}\sum \theta s$ (Eul)')
plt.plot(t,Nu[:,4],'k+-',label=r'Var $ c $')
plt.yscale('log')
plt.legend()
plt.xlabel('Time')
plt.ylim([1e-12,1e0])


bins=np.logspace(np.log10(min(nagg_eul)),np.log10(max(nagg_lag)),20)
pdf_N_lag,b=np.histogram(nagg_lag,bins,density=True)
pdf_N_eul,b=np.histogram(nagg_eul,bins,density=True)

lyap=-a*np.log(a)-(1-a)*np.log(1-a)
sigma2=a*np.log(a)**2.+(1-a)*np.log(1-a)**2.-lyap**2.
dq=2*3/4.
logrhoc=5
nu=(1+dq)*np.log(2)/np.log(1/a**dq+1/(1-a)**dq)+1
l=(nu-1)*(lyap*T-logrhoc)
s=sigma2*(nu-1)**2.*T
#!!! no dependence on fractal dimension !?
l=(lyap*T-logrhoc)
s=sigma2*T

plt.figure()
plt.plot(bins,1/np.sqrt(2*np.pi*s)/bins*np.exp(-(np.log(bins)-l)**2./(2*s)),'k--')
plt.plot(bins[1:],pdf_N_lag,'r+-',label=r'$P_{l}(N)$')
plt.plot(bins[1:],pdf_N_eul,'b+-',label=r'$P_{e}(N)$')
pdfeul_from_lag=pdf_N_lag/bins[1:]**3
pdfeul_from_lag=pdfeul_from_lag/np.sum(pdfeul_from_lag*np.diff(bins))
plt.plot(bins[1:],pdfeul_from_lag,'r--',label=r'$P_{l}(N)/N^3$')
pdfeul_from_lag=pdf_N_lag/bins[1:]
pdfeul_from_lag=pdfeul_from_lag/np.sum(pdfeul_from_lag*np.diff(bins))
plt.plot(bins[1:],pdfeul_from_lag,'r:',label=r'$P_{l}(N)/N$')
plt.xscale('log')
#plt.xlim([1,5000])
#plt.ylim([1e-4,1e-1])
plt.yscale('log')
plt.legend(title='$a={:1.2f}$'.format(a))

plt.figure()
plt.plot(bins,1/np.sqrt(2*np.pi*s)/bins*np.exp(-(np.log(bins)-l)**2./(2*s)),'k--')
plt.plot(bins[1:],pdf_N_lag,'r+-',label=r'$P_{l}(N)$')
plt.plot(bins[1:],pdf_N_eul,'b+-',label=r'$P_{e}(N)$')
pdfeul_from_lag=pdf_N_lag/bins[1:]**3
pdfeul_from_lag=pdfeul_from_lag/np.sum(pdfeul_from_lag*np.diff(bins))
plt.plot(bins[1:],pdfeul_from_lag,'r--',label=r'$P_{l}(N)/N^3$')
pdfeul_from_lag=pdf_N_lag/bins[1:]
pdfeul_from_lag=pdfeul_from_lag/np.sum(pdfeul_from_lag*np.diff(bins))
plt.plot(bins[1:],pdfeul_from_lag,'r:',label=r'$P_{l}(N)/N$')
plt.xscale('log')
#plt.xlim([1,5000])
#plt.ylim([1e-4,1e-1])
plt.yscale('log')
plt.legend(title='$a={:1.2f}$'.format(a))


#%%% Other plots
plt.figure()
#plt.plot(cm,label=r'$\sum \theta$')
plt.plot(cm2,label=r'$s_B \sum \theta s$')
plt.ylabel('$c(x)$')
plt.xlabel('lamella id')
plt.legend()



plt.figure()
#plt.plot(cm,label=r'$\sum \theta$')
plt.plot(nagg_lag,label=r'$N$')
plt.ylabel('$N$')
plt.xlabel('lamella id')
plt.legend()

plt.figure()
#plt.plot(cm,label=r'$\sum \theta$')
plt.plot(np.arange(0,1,sB)[1:],nagg_eul,label=r'$N$')
plt.ylabel('$N(x)$')
plt.xlabel('$x$')
plt.legend()


fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.set_xlabel('$x$')
ax1.set_ylabel('$N$', color=color)
ax1.plot(np.arange(0,1,sB)[1:], nagg_eul, color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:blue'
ax2.set_ylabel(r'$\sum \theta s$', color=color)  # we already handled the x-label with ax1
ax2.plot(np.arange(0,1,sB)[1:], hcmax, color=color)
ax2.tick_params(axis='y', labelcolor=color)
fig.tight_layout()  # otherwise the right y-label is slightly clipped

plt.figure()
plt.plot(nagg_eul,np.abs(hcmax-hcmax.mean()),'.')
plt.xlabel(r'$N$')
plt.ylabel(r'$|\sum \theta s - \langle c \rangle |$')
plt.yscale('log')
plt.xscale('log')

#%%% * p(C,N) for several times
plt.style.use('~/.config/matplotlib/joris.mplstyle')
from scipy import spatial
x0=np.array([0.32])
var_sa=0
sigma=50
s0=0.05
D=(sigma/n)**2/2
Tvec=[8,12,15]
fig,ax=plt.subplots(1,len(Tvec),figsize=(len(Tvec)*1.5,1.5),sharey=True)
for i,T in enumerate(Tvec):
	a=0.3
	C=DNS_n(2**13,x0,T,a,var_sa)
	x,S,wrapped_time=lagrangian(x0,T,a,var_sa)
	Cmax=1./np.sqrt(1.+4*D/s0**2*wrapped_time)
	Si=s0*np.sqrt(1.+4*D/s0**2*wrapped_time)*S
	sB=np.mean(Si)
	#sB=2.2*np.mean(Si)
	# Lagrangian
	tree=spatial.cKDTree(x.reshape(-1,1))
	# Take samples in within cmax
	neighboors=tree.query_ball_point(x.reshape(-1,1)[np.abs(x-0.5)<0.5-2*sB], sB)
	cm=[np.sum(Cmax[n]*np.mean(Si)/sB*np.sqrt(np.pi)) for n in neighboors] # is subjected to noise! 
	cm2=np.array([np.sum(s0*S[n]/sB*np.sqrt(np.pi)) for n in neighboors])
	nagg_lag=np.array([len(n) for n in neighboors])
	cmm=2*s0*np.sqrt(np.pi)
	import matplotlib as mpl
	
	nb=10
	cnorm=cm2-np.mean(cm2)
	bin_c=np.linspace(cnorm.min(),cnorm.max(),nb)
	bin_n=np.unique(np.uint16(np.logspace(np.log10(nagg_lag.min()),
																				np.log10(nagg_lag.max()*2),nb)))
	h2_lag=np.histogram2d(cm2,nagg_lag,[bin_c,bin_n],density=True)[0]
#	bin_c=np.linspace(0.1,0.25,40)
#	bin_n=np.unique(np.uint16(np.logspace(0,3.5,40)))
	ax[i].hist2d(cnorm,nagg_lag,[bin_c,bin_n],norm=mpl.colors.LogNorm(),cmap=plt.cm.Greys,
					 density=True)
	ax[i].patch.set_facecolor(plt.cm.Greys(0))
	ax[i].set_xlabel(r'$c(\textbf{x})-\langle c \rangle$')
	ax[i].set_yscale('log')
	ax[i].set_xlim([-0.1,0.1])
	ax[i].set_ylim([1,5000])
	
ax[0].set_ylabel(r'$n(\textbf{x})$')
	#plt.text(0.12,2,r'$P(c,N,t={:d})$'.format(T),color='w')
plt.savefig('/home/joris/Dropbox/Articles/AggregationBatchelor/Figures/P(C,N,t).pdf',bbox_inches='tight')
#%%% * <C|N> for several times
plt.style.use('~/.config/matplotlib/joris.mplstyle')
from scipy import spatial ,ndimage
#x0=np.array([0.32,0.18,0.95])
x0=np.random.rand(1)
var_sa=0
sigma=50
s0=0.05
D=(sigma/n)**2/2
Tvec=np.arange(1,12)
plt.figure(figsize=(2,2))
cbin=np.unique(np.logspace(-5,0.,30))
for i,T in enumerate(Tvec):
	a=0.35
	C=DNS_n(2**13,x0,T,a,var_sa)
	x,S,wrapped_time=lagrangian(x0,T,a,var_sa)
	Cmax=1./np.sqrt(1.+4*D/s0**2*wrapped_time)
	Si=s0*np.sqrt(1.+4*D/s0**2*wrapped_time)*S
	sB=np.mean(Si)
	# sB=2.2*np.mean(Si)
	# Lagrangian
	tree=spatial.cKDTree(x.reshape(-1,1))
	# # Take samples in within cmax
	npts=1000
	idpts=np.uint16(np.linspace(0,len(x)-1,npts))
	neighboors=tree.query_ball_point(x[idpts].reshape(-1,1), sB/2.)
	cm=[np.mean(Cmax[n]) for n in neighboors] # is subjected to noise! 
	# cm2=np.array([np.sum(s0*S[n]/sB*np.sqrt(np.pi)) for n in neighboors])
	# nagg_lag=np.array([len(n) for n in neighboors])
# 	ceul=np.interp(x,np.linspace(0,1,C.shape[1]),C.flatten())
# 	cbin=np.unique(np.logspace(-5,0,30))
# 	ceulbin=bin_operation(Cmax,ceul,cbin,np.mean)
	ceul=np.interp(x[idpts],np.linspace(0,1,C.shape[1]),C.flatten())
	ceulbin=bin_operation(cm,ceul,cbin,np.mean)
	plt.plot(cbin[:-1]/np.mean(C),ceulbin/np.mean(C),'o-',c=plt.cm.cool(float(i)/len(Tvec)))
	
for i,T in enumerate(Tvec):
	a=0.1
	C=DNS_n(2**13,x0,T,a,var_sa)
	x,S,wrapped_time=lagrangian(x0,T,a,var_sa)
	Cmax=1./np.sqrt(1.+4*D/s0**2*wrapped_time)
	Si=s0*np.sqrt(1.+4*D/s0**2*wrapped_time)*S
	sB=np.mean(Si)
	# sB=2.2*np.mean(Si)
	# Lagrangian
	tree=spatial.cKDTree(x.reshape(-1,1))
	# # Take samples in within cmax
	npts=1000
	idpts=np.uint16(np.linspace(0,len(x)-1,npts))
	neighboors=tree.query_ball_point(x[idpts].reshape(-1,1), sB/2.)
	cm=[np.mean(Cmax[n]) for n in neighboors] # is subjected to noise! 
	# cm2=np.array([np.sum(s0*S[n]/sB*np.sqrt(np.pi)) for n in neighboors])
	# nagg_lag=np.array([len(n) for n in neighboors])
# 	ceul=np.interp(x,np.linspace(0,1,C.shape[1]),C.flatten())
# 	ceulbin=bin_operation(Cmax,ceul,cbin,np.mean)
	ceul=np.interp(x[idpts],np.linspace(0,1,C.shape[1]),C.flatten())
	ceulbin=bin_operation(cm,ceul,cbin,np.mean)
	plt.plot(cbin[:-1]/np.mean(C),ceulbin/np.mean(C),'s-',c=plt.cm.cool(float(i)/len(Tvec)))

plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\langle \theta \rangle_{s_B}/\langle c\rangle$')
plt.ylabel(r'$c/\langle c \rangle$')
plt.xlim([1e-3,1e1])
plt.xticks([1e-3,1e-2,1e-1,1e0,10])
plt.ylim([1e-1,20])
plt.plot(cbin/np.mean(C),(cbin/C.mean()+cbin*(1/cbin-1)),'k--')
plt.savefig('/home/joris/Dropbox/Articles/AggregationBatchelor/Figures/ccmax_baker.pdf',bbox_inches='tight')


#%%% * p(C,N) for several times, log scale)
import matplotlib as mpl
plt.style.use('~/.config/matplotlib/joris.mplstyle')
from scipy import spatial
x0=np.array([0.32])
var_sa=0
sigma=50
s0=0.05
D=(sigma/n)**2/2
Tvec=[8,12,15]
fig,ax=plt.subplots(1,len(Tvec),figsize=(len(Tvec)*1.5,1.5),sharey=True)
for i,T in enumerate(Tvec):
	a=0.3
	C=DNS_n(2**13,x0,T,a,var_sa)
	x,S,wrapped_time=lagrangian(x0,T,a,var_sa)
	Cmax=1./np.sqrt(1.+4*D/s0**2*wrapped_time)
	Si=s0*np.sqrt(1.+4*D/s0**2*wrapped_time)*S
	sB=np.mean(Si)
	#sB=2.2*np.mean(Si)
	# Lagrangian
	tree=spatial.cKDTree(x.reshape(-1,1))
	# Take samples in within cmax
	neighboors=tree.query_ball_point(x.reshape(-1,1)[np.abs(x-0.5)<0.5-2*sB], sB)
	cm=[np.sum(Cmax[n]*np.mean(Si)/sB*np.sqrt(np.pi)) for n in neighboors] # is subjected to noise! 
	cm2=np.array([np.sum(s0*S[n]/sB*np.sqrt(np.pi)) for n in neighboors])
	nagg_lag=np.array([len(n) for n in neighboors])
	cmm=2*s0*np.sqrt(np.pi)
	import matplotlib as mpl
	
	nb=20
	cnorm=np.abs(cm2-np.mean(cm2))
	bin_c=np.logspace(-3,-1,nb)
	bin_n=np.unique(np.uint16(np.logspace(np.log10(nagg_lag.min()),
																				np.log10(nagg_lag.max()*2),nb)))
	h2_lag=np.histogram2d(cm2,nagg_lag,[bin_c,bin_n],density=True)[0]
#	bin_c=np.linspace(0.1,0.25,40)
#	bin_n=np.unique(np.uint16(np.logspace(0,3.5,40)))
	ax[i].hist2d(cnorm,nagg_lag,[bin_c,bin_n],norm=mpl.colors.LogNorm(),cmap=plt.cm.Greys,
					 density=True)
	ax[i].patch.set_facecolor(plt.cm.Greys(0))
	ax[i].set_xlabel(r'$|c(\textbf{x})-\langle c \rangle|$')
	ax[i].set_yscale('log')
	ax[i].set_xscale('log')
	ax[i].set_xlim([1e-3,1e-1])
	ax[i].set_ylim([1,5000])
	
ax[0].set_ylabel(r'$n(\textbf{x})$')
	#plt.text(0.12,2,r'$P(c,N,t={:d})$'.format(T),color='w')
plt.savefig('/home/joris/Dropbox/Articles/AggregationBatchelor/Figures/P(C,N,t)_log.pdf',bbox_inches='tight')
#%%% * p(N) for several times
from scipy import spatial,optimize
plt.style.use('~/.config/matplotlib/joris.mplstyle')
x0=np.random.rand(1)
var_sa=0
nb=20
s0=0.05
D=1e-4
TT=np.array([10,15,20])
plt.figure(figsize=(2,2))
M=['d','o','s','*']

#TT=np.arange(0,20,5)
for i,T in enumerate(TT):
	a=0.3
	x,S,wrapped_time=lagrangian(x0,T,a,var_sa)
	sB=1/100
	# Lagrangian
# =============================================================================
# 	tree=spatial.cKDTree(x.reshape(-1,1))
# 	neighboors=tree.query_ball_point(x.reshape(-1,1)[np.abs(x-0.5)<0.5-2*sB], sB/2.)
# 	nagg_lag=np.array([len(n) for n in neighboors])
#	nb=20
#	bin_n=np.unique(np.uint16(np.logspace(np.log10(nagg_lag.min()),
#																				np.log10(nagg_lag.max()+5),nb)))
#	hN,n=np.histogram(nagg_lag,bin_n,density=True)
#	plt.plot(n[:-1],hN,'o',color=plt.cm.jet(T/TT.max()))
# =============================================================================

	#fig,ax=plt.subplots(1,1,figsize=(2,2))
	import matplotlib as mpl
# =================Eulerian =================================================
	he,ne=np.histogram(x,np.arange(0,1,sB),density=False)
	he=he[he>0]
	bin_n=np.unique(np.uint16(np.logspace(np.log10(he.min()),np.log10(he.max()+5),nb)))
	hE,n=np.histogram(he,bin_n,density=True)
	plt.plot(n[:-1],hE,M[i],color=plt.cm.viridis(T/TT.max()),label='$t={:1.0f}$'.format(T),fillstyle='full')
# =============================================================================
	lyap=-a*np.log(a)-(1-a)*np.log(1-a)
	sigma2=a*np.log(a)**2.+(1-a)*np.log(1-a)**2.-lyap**2.
	dq=1.
	rhoc=1/sB
	nu=(1+dq)*np.log(2)/np.log(1/a**dq+1/(1-a)**dq)+1
	l=(nu-1)*(lyap*T-np.log(rhoc))
	s=sigma2*(nu-1)**2.*T
	n=np.logspace(0,5,200)
#	plt.plot(n,1/np.sqrt(2*np.pi*s)/n*np.exp(-(np.log(n)-l)**2./(2*s)),'-',color=plt.cm.viridis(T/TT.max()))
	import scipy.special
#!!! no dependence on fractal dimension !?
	l=((lyap)*T-np.log(rhoc))
	s=sigma2*T
	n=np.logspace(0,5,200)
#	plt.plot(n,1/np.sqrt(2*np.pi*s)/n*np.exp(-(np.log(n)-l)**2./(2*s)),'--',color=plt.cm.viridis(T/TT.max()))
	dq=2	
	f = lambda Dq : dq*np.log(2)-np.log(1/a**((dq-1)*Dq)+1/(1-a)**((dq-1)*Dq))
	D2=scipy.optimize.broyden1(f, 1, f_tol=1e-14)+1
		
	# Theory
	mu_n=2**T*sB
	sigma2_n=mu_n**2.*(sB**(D2-2)-1)
	# gamma
	k=mu_n**2./sigma2_n
	theta=sigma2_n/mu_n
	pdfn=1/scipy.special.gamma(k)/theta**k*n**(k-1)*np.exp(-n/theta)
	plt.plot(n,pdfn,'-',color=plt.cm.viridis(T/TT.max()))
	# Neg Binomial
	from scipy.stats import nbinom,gamma
	r=mu_n**2/(sigma2_n-mu_n)
	p=mu_n/sigma2_n
	nn=np.uint32(n)
	#plt.plot(nn,nbinom.pmf(nn,r,p),'--',color=plt.cm.viridis(T/TT.max()))
	
	plt.xscale('log')
	plt.yscale('log')
	plt.xlabel('$n$')
	plt.ylabel('$P(n,t)$')
subscript(plt.gca(),0)
plt.ylim([1e-12,2])
plt.xlim([1e0,1e5])
plt.legend(fontsize=8)
plt.savefig('/home/joris/Dropbox/Articles/AggregationBatchelor/Figures/Baker_P(N,t)_a{:1.2f}.pdf'.format(a),bbox_inches='tight')
#%%% * Integral of p(N)*n^-1+epsilon for several times
from scipy import spatial
plt.style.use('~/.config/matplotlib/joris.mplstyle')
x0=np.random.rand(1)
var_sa=0
nb=20
s0=0.05
D=1e-4
TT=np.arange(15,24)
AA=np.linspace(0.01,0.49,20)

plt.figure(figsize=(3,2))
M=['d','o','s','*']

#TT=np.arange(0,20,5)
P=[]
for a in AA:
	Var=[]
	for i,T in enumerate(TT):
		x,S,wrapped_time=lagrangian(x0,T,a,var_sa)
		sB=1/200
		# Lagrangian
	# =============================================================================
	# 	tree=spatial.cKDTree(x.reshape(-1,1))
	# 	neighboors=tree.query_ball_point(x.reshape(-1,1)[np.abs(x-0.5)<0.5-2*sB], sB/2.)
	# 	nagg_lag=np.array([len(n) for n in neighboors])
	#	nb=20
	#	bin_n=np.unique(np.uint16(np.logspace(np.log10(nagg_lag.min()),
	#																				np.log10(nagg_lag.max()+5),nb)))
	#	hN,n=np.histogram(nagg_lag,bin_n,density=True)
	#	plt.plot(n[:-1],hN,'o',color=plt.cm.jet(T/TT.max()))
	# =============================================================================
	
	# =================Eulerian =================================================
		he,ne=np.histogram(x,np.arange(0,1,sB),density=False)
		he=he[he>0]
		bin_n=np.unique(np.uint16(np.logspace(np.log10(he.min()-0.5),np.log10(he.max()+5),nb)))
		hE,n=np.histogram(he,bin_n,density=True)
	# =============================================================================
		lyap=-a*np.log(a)-(1-a)*np.log(1-a)
		sigma2=a*np.log(a)**2.+(1-a)*np.log(1-a)**2.-lyap**2.
		dq=1.
		rhoc=1/sB
		nu=(1+dq)*np.log(2)/np.log(1/a**dq+1/(1-a)**dq)+1
		l=(nu-1)*(lyap*T-np.log(rhoc))
		s=sigma2*(nu-1)**2.*T
		#n=np.logspace(0,5,200)
	#	plt.plot(n,1/np.sqrt(2*np.pi*s)/n*np.exp(-(np.log(n)-l)**2./(2*s)),'-',color=plt.cm.viridis(T/TT.max()))
		import scipy.special
	#!!! no dependence on fractal dimension !?
		l=((lyap)*T-np.log(rhoc))
		s=sigma2*T
		#n=np.logspace(0,5,200)
	#	plt.plot(n,1/np.sqrt(2*np.pi*s)/n*np.exp(-(np.log(n)-l)**2./(2*s)),'--',color=plt.cm.viridis(T/TT.max()))
		
		# Gamma
# 		mu_n=2**T*sB
# 		sigma2_n=mu_n**2.*(-np.log(sB))*2*(2-nu)/(nu-1)**2.
# 		# gamma
# 		k=mu_n**2./sigma2_n
# 		theta=sigma2_n/mu_n
		n=np.float128(bin_n[1:])
# 		pdfn=1/scipy.special.gamma(k)/theta**k*n**(k-1)*np.exp(-n/theta)
		
		# Epsilon
		D1=nu
		mu=1/(D1-1)
		sigma2=2*(2-D1)/(D1-1)
		if mu-2*sigma2>0:
			M2=2*mu-2*sigma2
		else:
			M2=mu**2/(2*sigma2)
		epsilon=-M2+2
		#Interpolate true epsilon
# 		Eps=np.loadtxt('Baker_epsilon.txt')
# 		epsilon=np.interp(nu,Eps[:,0],Eps[:,1])
# 		Var.append(np.sum(n**(-1+epsilon)*hE*np.diff(bin_n))-np.sum(n**(-1)*hE*np.diff(bin_n)))

		# Epsilon : true exponent
		Eps=np.loadtxt('Baker_epsilon.txt')
		epsilon=np.interp(nu,Eps[:,0],Eps[:,1])
		xi=-2+epsilon+1
		# Xi : true exponent
		Xi=np.loadtxt('Baker_xi.txt')
		xi=np.interp(nu,Xi[:,0],Xi[:,1])
		
		
		Var.append(np.sum(n**(-xi)*hE*np.diff(bin_n)))#-np.sum(n**(-1)*hE*np.diff(bin_n)))
		
#		Var.append(np.sum(n**(-xi)*hE*np.diff(bin_n))-np.sum(n**(-1)*hE*np.diff(bin_n)))
		
	Var=np.array(Var)
	plt.plot(TT,np.log(Var),'-+')
	P.append(-np.polyfit(TT,np.float32(np.log(Var)),1)[0])
	print(a,P[-1])
#plt.savefig('/home/joris/Dropbox/Articles/AggregationBatchelor/Figures/Baker_P(N,t)_a{:1.2f}.pdf'.format(a),bbox_inches='tight')

plt.figure()
plt.plot(AA,P)
plt.ylim([0,2*np.log(2)])
np.savetxt('Baker_gamma2_theory2.txt',np.vstack((AA,np.array(P).T)))
#%%%  Value of k for various D1 and sb
fig=plt.figure()
SB=1/np.logspace(0.5,3,20)
D1=np.linspace(1.3,2.0,100)
[plt.plot(D1,-(D1-1)/2/np.log(sb)/(2-D1),color=plt.cm.cool(i/len(SB))) for i,sb in enumerate(SB)]
plt.ylim([0,2])
plt.xlabel('$D_1$')
plt.ylabel('$k_n$')
ax2 = fig.add_axes([0.2, 0.8, 0.3, 0.05])
import matplotlib as mpl
#norm = mpl.colors.Normalize(vmin=T_all[0]*u_pore/d, vmax=T_all[-1]*u_pore/d)
norm = mpl.colors.Normalize(vmin=0.5, vmax=3)
cb1 = mpl.colorbar.ColorbarBase(ax2,cmap=plt.cm.cool,norm=norm,orientation='horizontal')
cb1.set_label(r'$\log(1/s_B)$', color='k',size=8,labelpad=0)
plt.savefig('/home/joris/Dropbox/Articles/AggregationBatchelor/Figures/scaling_k.pdf',bbox_inches='tight')

fig=plt.figure()
sb=1/200
D1=np.linspace(1.3,2.0,100)
mu=1/(D1-1)
sigma2=2*(2-D1)/(D1-1)
M2=2*mu-2*sigma2
M2[mu-2*sigma2<0]=mu[mu-2*sigma2<0]**2/(2*sigma2[mu-2*sigma2<0])
epsilon=-M2+2
k=-(D1-1)/2/np.log(sb)/(2-D1)-1
#epsilon=2-(D1-1)**3/(4*(2-D1))
plt.plot(D1,k+epsilon,'r')
#plt.plot(D1,k,'r:')
#plt.plot(D1,epsilon,'r--')
plt.ylim([-1,2])
plt.plot(D1,np.zeros(D1.shape),'k-')
plt.xlabel('$D_1$')
plt.ylabel(r'$k+\varepsilon-1$')
plt.savefig('/home/joris/Dropbox/Articles/AggregationBatchelor/Figures/scaling_k+eps.pdf',bbox_inches='tight')

#%%% moments of n^-xi

k=0.5
theta=10
xi=1

n=np.logspace(0,5,1000)
pdfn=1/scipy.special.gamma(k)/theta**k*n**(k-1)*np.exp(-n/theta)

plt.plot(n,pdfn,'-',color=plt.cm.cool(a*2))

plt.yscale('log')
plt.xscale('log')
plt.ylim([1e-10,1e2])

#%%% * p(N) for several a
from scipy import spatial
plt.style.use('~/.config/matplotlib/joris.mplstyle')
x0=np.array([0.1])
var_sa=0
nb=15
s0=0.05
D=1e-4
T=20
plt.figure(figsize=(2,2))
AA=np.array([0.1,0.25,0.4])
#TT=np.arange(0,20,5)

M=['d','o','s','*']
for i,a in enumerate(AA):
#	a=0.3
	x,S,wrapped_time=lagrangian(x0,T,a,var_sa)
	sB=1/200
	# Lagrangian
# =============================================================================
# 	tree=spatial.cKDTree(x.reshape(-1,1))
# 	neighboors=tree.query_ball_point(x.reshape(-1,1)[np.abs(x-0.5)<0.5-2*sB], sB/2.)
# 	nagg_lag=np.array([len(n) for n in neighboors])
#	nb=20
#	bin_n=np.unique(np.uint16(np.logspace(np.log10(nagg_lag.min()),
#																				np.log10(nagg_lag.max()+5),nb)))
#	hN,n=np.histogram(nagg_lag,bin_n,density=True)
#	plt.plot(n[:-1],hN,'o',color=plt.cm.jet(T/TT.max()))
# =============================================================================

	#fig,ax=plt.subplots(1,1,figsize=(2,2))
	import matplotlib as mpl
# =================Eulerian =================================================
	he,ne=np.histogram(x,np.arange(0,1+sB,sB),density=False)
	he=he[he>0]
	bin_n=np.unique(np.uint16(np.logspace(np.log10(he.min()),np.log10(he.max()+5),nb)))
	hE,n=np.histogram(he,bin_n,density=True)
	plt.plot(n[:-1],hE,M[i],color=plt.cm.cool(a*2),label='$a={:1.1f}$'.format(a),fillstyle='full')
# =============================================================================
	lyap=-a*np.log(a)-(1-a)*np.log(1-a)
	sigma2=a*np.log(a)**2.+(1-a)*np.log(1-a)**2.-lyap**2.
	dq=1.
	rhoc=1/sB
	nu=(1+dq)*np.log(2)/np.log(1/a**dq+1/(1-a)**dq)+1
	l=(nu-1)*(lyap*T-np.log(rhoc))
	s=sigma2*(nu-1)**2.*T
	n=np.logspace(0,5,200)
#	plt.plot(n,1/np.sqrt(2*np.pi*s)/n*np.exp(-(np.log(n)-l)**2./(2*s)),'-',color=plt.cm.viridis(T/TT.max()))
	import scipy.special
#!!! no dependence on fractal dimension !?
	l=((lyap)*T-np.log(rhoc))
	s=sigma2*T
	n=np.logspace(0,5,200)
#	plt.plot(n,1/np.sqrt(2*np.pi*s)/n*np.exp(-(np.log(n)-l)**2./(2*s)),'--',color=plt.cm.viridis(T/TT.max()))
	dq=2	
	f = lambda Dq : dq*np.log(2)-np.log(1/a**((dq-1)*Dq)+1/(1-a)**((dq-1)*Dq))
	D2=scipy.optimize.broyden1(f, 1, f_tol=1e-14)+1

	# Theory
	mu_n=2**T*sB
	sigma2_n=mu_n**2.*(sB**(D2-2)-1)
	# gamma
	k=mu_n**2./sigma2_n
	theta=sigma2_n/mu_n
	pdfn=1/scipy.special.gamma(k)/theta**k*n**(k-1)*np.exp(-n/theta)
	plt.plot(n,pdfn,'-',color=plt.cm.cool(a*2))
	
	plt.xscale('log')
	plt.yscale('log')
	plt.xlabel('$n$')
	plt.ylabel('$P(n,t)$')
plt.ylim([1e-7,1e-1])
plt.xlim([1e0,1e5])
plt.legend(fontsize=8)
subscript(plt.gca(),1)
plt.savefig('/home/joris/Dropbox/Articles/AggregationBatchelor/Figures/Baker_P(N,t)_a.pdf'.format(a),bbox_inches='tight')


#%%% * p(log N) for several times
from scipy import spatial
plt.style.use('~/.config/matplotlib/joris.mplstyle')
x0=np.array([0.121])
var_sa=0
nb=20
s0=0.05
D=1e-4
TT=np.array([10,15,20,25])
plt.figure(figsize=(2,2))
#TT=np.arange(0,20,5)
for T in TT:
	a=0.4
	x,S,wrapped_time=lagrangian(x0,T,a,var_sa)
	sB=1/50
	# Lagrangian
# =============================================================================
# 	tree=spatial.cKDTree(x.reshape(-1,1))
# 	neighboors=tree.query_ball_point(x.reshape(-1,1)[np.abs(x-0.5)<0.5-2*sB], sB/2.)
# 	nagg_lag=np.array([len(n) for n in neighboors])
#	nb=20
#	bin_n=np.unique(np.uint16(np.logspace(np.log10(nagg_lag.min()),
#																				np.log10(nagg_lag.max()+5),nb)))
#	hN,n=np.histogram(nagg_lag,bin_n,density=True)
#	plt.plot(n[:-1],hN,'o',color=plt.cm.jet(T/TT.max()))
# =============================================================================

	#fig,ax=plt.subplots(1,1,figsize=(2,2))
	import matplotlib as mpl
# =================Eulerian =================================================
	he,ne=np.histogram(x,np.arange(0,1,sB),density=False)
	he=he[he>0]
	hE,n=np.histogram(np.log(he),15,density=True)
	plt.plot(n[:-1],hE,'.',color=plt.cm.viridis(T/TT.max()),label='$t={:1.0f}$'.format(T),fillstyle='full')
# =============================================================================
	lyap=-a*np.log(a)-(1-a)*np.log(1-a)
	sigma2=a*np.log(a)**2.+(1-a)*np.log(1-a)**2.-lyap**2.
	dq=1.
	rhoc=1/sB
	nu=(1+dq)*np.log(2)/np.log(1/a**dq+1/(1-a)**dq)+1
	l=(nu-1)*(lyap*T-np.log(rhoc))
	s=sigma2*(nu-1)**2.*T
	n=np.linspace(0,15,200)
	plt.plot(n,1/np.sqrt(2*np.pi*s)*np.exp(-(n-l)**2./(2*s)),'-',color=plt.cm.viridis(T/TT.max()))

#!!! no dependence on fractal dimension !?
	l=((lyap)*T-np.log(rhoc))
	s=sigma2*T
	n=np.linspace(0,15,200)
	plt.plot(n,1/np.sqrt(2*np.pi*s)*np.exp(-(n-l)**2./(2*s)),'--',color=plt.cm.viridis(T/TT.max()))

	plt.yscale('log')
	plt.xlabel('$\log n$')
	plt.ylabel('$P(n,t)$')
plt.ylim([1e-2,2])
plt.xlim([0,15])
plt.legend(fontsize=8)
plt.savefig('/home/joris/Dropbox/Articles/AggregationBatchelor/Figures/P(logN,t)_a{:1.2f}.pdf'.format(a),bbox_inches='tight')


#%%% * p(log rho) for several times
from scipy import spatial
plt.style.use('~/.config/matplotlib/joris.mplstyle')
x0=np.array([0.121])
var_sa=0
nb=20
s0=0.05
D=1e-4
TT=np.array([15,25])
plt.figure(figsize=(3,2))
#TT=np.arange(0,20,5)
for T in TT:
	a=0.3
	x,S,wrapped_time=lagrangian(x0,T,a,var_sa)
	sB=1/200
	# Lagrangian
# =============================================================================
# 	tree=spatial.cKDTree(x.reshape(-1,1))
# 	neighboors=tree.query_ball_point(x.reshape(-1,1)[np.abs(x-0.5)<0.5-2*sB], sB/2.)
# 	nagg_lag=np.array([len(n) for n in neighboors])
#	nb=20
#	bin_n=np.unique(np.uint16(np.logspace(np.log10(nagg_lag.min()),
#																				np.log10(nagg_lag.max()+5),nb)))
#	hN,n=np.histogram(nagg_lag,bin_n,density=True)
#	plt.plot(n[:-1],hN,'o',color=plt.cm.jet(T/TT.max()))
# =============================================================================

	#fig,ax=plt.subplots(1,1,figsize=(2,2))
	import matplotlib as mpl
# =================Eulerian =================================================
	he,ne=np.histogram(x,np.arange(0,1,sB),density=False)
	logrho,ne=np.histogram(x,np.arange(0,1,sB),density=False,weights=np.log(1/S))
	mlogrho=logrho/he
	mlogrho=mlogrho[he>2]
	hE,n=np.histogram(mlogrho,10,density=True)
	plt.plot(n[:-1],hE,'*',color=plt.cm.viridis(T/TT.max()),label=r'$P_{\langle \log \rho \rangle_B}$',fillstyle='full')
# =============================================================================
	lyap=-a*np.log(a)-(1-a)*np.log(1-a)
	sigma2=a*np.log(a)**2.+(1-a)*np.log(1-a)**2.-lyap**2.
	dq=1.
	rhoc=1/sB
	nu=(1+dq)*np.log(2)/np.log(1/a**dq+1/(1-a)**dq)+1
#!!! no dependence on fractal dimension !?
	l=((lyap)*T)
	s=sigma2*T
	n=np.linspace(0,20,200)
	
	hE,n=np.histogram(np.log(1/S),100,density=True,weights=S)
	plt.plot(n[:-1],hE,'o',color=plt.cm.viridis(T/TT.max()),label=r'$P_{\log \rho}$')
	#plt.plot(n,1/np.sqrt(2*np.pi*s)*np.exp(-(n-l)**2./(2*s)),'-',color=plt.cm.viridis(T/TT.max()))

	plt.yscale('log')
	plt.xlabel(r'$\log \rho $')
plt.ylim([1e-2,1e1])
#plt.xlim([0,25])
plt.legend(fontsize=6,title='$a={:1.2f}$'.format(a))
plt.savefig('/home/joris/Dropbox/Articles/AggregationBatchelor/Figures/P(logrho_B,t)_a{:1.2f}.pdf'.format(a),bbox_inches='tight')


#%%% * var(n) as  a function of D1
import scipy
from scipy import spatial
plt.style.use('~/.config/matplotlib/joris.mplstyle')
x0=np.array([0.56])
var_sa=0
nb=20
s0=0.05
D=1e-4
T=25
M=['d','o','s','*']
TT=np.arange(10,25)
plt.figure(figsize=(4,3))

sB=1/100
T=24
#AA=np.logspace(-3,np.log10(0.49),30)
AA=np.linspace(0.01,0.47,20)
M=['d','o','s','*','.']
Var,Mean=[],[]

for ii,sB in enumerate([1/10,1/50,1/100]):
	Var,Mean=[],[]
	for i,a in enumerate(AA):
		
		x,S,wrapped_time=lagrangian(x0,T,a,var_sa)
		he,ne=np.histogram(x,np.arange(0,1+sB,sB),density=False)
		Mean.append([np.mean(he[he>0]),np.mean((1/S))])
		Var.append([np.var((he[he>0])/2**(T)),np.var(np.log(he[he>0])),np.var(np.log(1/S)),np.var(he[he>0]/np.mean(he))])
	
	Var=np.array(Var)
	Var=np.array(Var)
	Mean=np.array(Mean)
	dq=1.
	nu=(1+dq)*np.log(2)/np.log(1/AA**dq+1/(1-AA)**dq)+1
	ni=-np.log(sB)*sB**2.
	plt.plot(2-nu,Var[:,0]/ni,M[ii]+'--',color=plt.cm.cool(ii/3),label=r'$s_B=1/{:1.0f}$'.format(1/sB))
#	plt.plot(2-nu,Var[:,1]/ni,M[ii]+'-',color=plt.cm.cool(ii/3),label=r'$s_B=1/{:1.0f}$'.format(1/sB))
plt.yscale('log')
plt.xscale('log')
#plt.xscale('log')
#plt.ylim([1e-4,1e2])


plt.ylabel('$\sigma^2_{\log n}/\log(1/s_B)$')
plt.xlabel(r'$2-D_1$')
nu=np.linspace(1.2,2,100)
plt.plot(2-nu,2*(2-nu)/(nu-1),'k--',label=r'$2(2-D_1)/(D_1-1)$')

# plt.plot(nu,0.1+2*(2-nu-0.13)/(nu+0.13-1)**3,'k-',label=r'$2(2-D_1-0.12)$')

plt.legend()

plt.savefig('/home/joris/Dropbox/Articles/AggregationBatchelor/Figures/Baker_Sine_N_lyap.pdf',bbox_inches='tight')

#%%% * var(n) as  a function of sB
import scipy
from scipy import spatial
plt.style.use('~/.config/matplotlib/joris.mplstyle')
x0=np.array([0.56])
var_sa=0
nb=20
s0=0.05
D=1e-4
T=25
M=['d','o','s','*']
TT=np.arange(10,25)
plt.figure(figsize=(4,3))

sB=1/100
T=24
#AA=np.logspace(-3,np.log10(0.49),30)
AA=np.array([0.1,0.2,0.3,0.4])
#AA=np.array([0.2])
M=['d','o','s','*','.']
Var,Mean=[],[]
SB=np.logspace(-3,-1,20)

for i,a in enumerate(AA):
	Var,Mean=[],[]
	for ii,sB in enumerate(SB):
		x,S,wrapped_time=lagrangian(x0,T,a,var_sa)
		he,ne=np.histogram(x,np.arange(0,1+sB,sB),density=False)
		Mean.append([np.mean(he[he>0]),np.mean((1/S))])
		Var.append([np.var((he[he>0])),np.var(np.log(1/S)),np.var(he[he>0]/np.mean(he))])
	
	Var=np.array(Var)
	Var=np.array(Var)
	Mean=np.array(Mean)
	dq=1.
	nu=(1+dq)*np.log(2)/np.log(1/a**dq+1/(1-a)**dq)+1
	plt.plot(SB,Var[:,0]/2**T/(2*(2-nu)/(nu-1)),M[i]+'--',color=plt.cm.cool(i/3),label=r'$a={:1.2f}$'.format(a))

#SB=np.logspace(-6,0,100)
#!!!! Why this do not work perfectly ????? 
plt.plot(SB,SB**2.*np.log(1/SB)*1e8,'k--',label=r'$s_B^2 / \log s_B$')
#plt.plot(SB,SB**2.,'k-',label=r'$s_B^2 \log s_B$')
#plt.plot(SB,-SB**2.*np.log(SB),'r:',label=r'$s_B^2 \log s_B$')
plt.yscale('log')
plt.xscale('log')
#plt.xscale('log')
#plt.ylim([1e-4,1e2])


# plt.ylabel('$\sigma^2_{\log n}/\log(1/s_B)$')
plt.xlabel(r'$s_B$')
# plt.plot(nu,2*(2-nu)/(nu-1)**3,'k--',label=r'$2(2-D_1)/(D_1-1)^3$')

# plt.plot(nu,0.1+2*(2-nu-0.13)/(nu+0.13-1)**3,'k-',label=r'$2(2-D_1-0.12)$')

plt.legend()



#%%% * var(n/<n>) as  a function of D1
import scipy
from scipy import spatial
plt.style.use('~/.config/matplotlib/joris.mplstyle')
x0=np.array([0.56])
var_sa=0
nb=20
s0=0.05
D=1e-4
T=25
M=['d','o','s','*']
TT=np.arange(10,25)
plt.figure(figsize=(4,3))

sB=1/100
T=25
#AA=np.logspace(-3,np.log10(0.49),30)
AA=np.linspace(0.01,0.49,20)
M=['d','o','s','*','.']
Var,Mean=[],[]

for ii,sB in enumerate([1/20,1/50,1/500]):
	Var,Mean=[],[]
	print(sB)
	for i,a in enumerate(AA):
		
		x,S,wrapped_time=lagrangian(x0,T,a,var_sa)
		he,ne=np.histogram(x,np.arange(0,1+sB,sB),density=False)
		Mean.append([np.mean(he[he>0]),np.mean((1/S))])
		Var.append([np.var(np.log(he[he>0])),np.var(np.log(1/S)),np.var(he[he>0]/np.mean(he))])
	
	Var=np.array(Var)
	Var=np.array(Var)
	Mean=np.array(Mean)
	dq=1.
	nu=(1+dq)*np.log(2)/np.log(1/AA**dq+1/(1-AA)**dq)+1
	plt.plot(2-nu,Var[:,0]/np.log(1/sB),M[ii]+'--',color=plt.cm.cool(ii/3),label=r'$s_B=1/{:1.0f}$'.format(1/sB))
plt.yscale('log')
plt.xscale('log')

plt.ylim([1e-4,1e2])
plt.ylabel('$\sigma^2_{\log n}/\log(1/s_B)$')
plt.xlabel(r'$2-D_1$')
plt.plot(2-nu,2*(2-nu)/(nu-1),'k--',label=r'$2(2-D_1)/(D_1-1)$')

#plt.plot(nu,0.1+2*(2-nu-0.13)/(nu+0.13-1)**3,'k-',label=r'$2(2-D_1-0.12)$')

plt.legend()



#Load Sine flow
if False:
	A=np.loadtxt('Sine_Var(logn)_1e7.txt')
	D1=np.loadtxt('Sine_D1.txt')
	
	nsB=len(np.unique(A[:,1]))
	AA=VarsB1=A[::nsB,0]
	D1A=np.interp(AA,D1[:,0],D1[:,1])
	
	for i in range(nsB):
		sB=A[i,1]**0.5
		plt.plot(D1A,A[i::nsB,2]/np.log(1/sB)**2.,M[i]+'-',color=plt.cm.cool(i/5),label=r'$s_B=1/{:1.0f}$'.format(1/sB))


plt.savefig('/home/joris/Dropbox/Articles/AggregationBatchelor/Figures/Baker_Sine_N_a.pdf',bbox_inches='tight')

#%%% * var(n/<n>) as  a function of sigma2/mu
import scipy
from scipy import spatial
plt.style.use('~/.config/matplotlib/joris.mplstyle')
x0=np.array([0.56])
var_sa=0
nb=20
s0=0.05
D=1e-4
T=25
M=['d','o','s','*']
TT=np.arange(10,25)
plt.figure(figsize=(4,3))

sB=1/100
T=23
#AA=np.logspace(-3,np.log10(0.49),30)
AA=np.linspace(0.01,0.49,20)
M=['d','o','s','*','.']
Var,Mean=[],[]

for ii,sB in enumerate([1/20,1/50,1/500]):
	Var,Mean=[],[]
	for i,a in enumerate(AA):
		
		x,S,wrapped_time=lagrangian(x0,T,a,var_sa)
		he,ne=np.histogram(x,np.arange(0,1+sB,sB),density=False)
		Mean.append([np.mean(he[he>0]),np.mean((1/S))])
		Var.append([np.var(np.log(he[he>0])),np.var(np.log(1/S)),np.var(he[he>0]/np.mean(he))])
	
	Var=np.array(Var)
	Var=np.array(Var)
	Mean=np.array(Mean)
	dq=1.
	nu=(1+dq)*np.log(2)/np.log(1/AA**dq+1/(1-AA)**dq)+1
	ly,sig=lyapunov(AA)
	plt.plot(sig/ly,Var[:,0]/np.log(1/sB),M[ii]+'-',color=plt.cm.cool(ii/3),label=r'$s_B=1/{:1.0f}$'.format(1/sB))
plt.yscale('log')
#plt.xscale('log')
plt.ylim([1e-4,1e2])
plt.ylabel('$\sigma^2_{\log n}/\log(1/s_B)$')
plt.xlabel(r'$\sigma^2/\mu$')
#plt.plot(nu,2*(2-nu)/(nu-1)**4,'k--',label=r'$2(2-D_1)/(D_1-1)^4$')

plt.legend()


#Load Sine flow

A=np.loadtxt('Sine_Var(logn)_1e7.txt')
D1=np.loadtxt('Sine_D1.txt')
ly=np.loadtxt('Sine_Lyap.txt')

nsB=len(np.unique(A[:,1]))
AA=VarsB1=A[::nsB,0]
D1A=np.interp(AA,D1[:,0],D1[:,1])

l=np.interp(AA,ly[:,0],ly[:,1])
sig=np.interp(AA,ly[:,0],ly[:,2])
for i in range(nsB):
	sB=A[i,1]**0.5
	plt.plot(sig/l,A[i::nsB,2]/np.log(1/sB)**2*2,M[i]+'-',color=plt.cm.cool(i/5),label=r'$s_B=1/{:1.0f}$'.format(1/sB))


plt.savefig('/home/joris/Dropbox/Articles/AggregationBatchelor/Figures/Baker_Sine_N_lyap.pdf',bbox_inches='tight')

#%%% * var(log n) as a function of time
import scipy
from scipy import spatial
import scipy.special
plt.style.use('~/.config/matplotlib/joris.mplstyle')
x0=np.array([0.56])
var_sa=0
nb=20
s0=0.05
D=1e-4
T=25
M=['d','o','s','*']
L=[':','-','--','-.']
SB=[1/50,1/100,1/500]
TT=np.arange(1,25)
fig=plt.figure(figsize=(4,3))

for j,a in enumerate([0.1,0.2,0.3,0.4]):
	for ii,sB in enumerate(SB):
		Var,Mean=[],[]
		for i,T in enumerate(TT):
			x,S,wrapped_time=lagrangian(x0,T,a,var_sa)
			he,ne=np.histogram(x,np.arange(0,1+sB,sB),density=False)
			Mean.append([np.mean(he[he>0]),np.mean((1/S))])
			Var.append([np.var(np.log(he[he>0])),np.var(np.log(1/S)),np.var(he[he>0]/np.mean(he[he>0]))])
		
		Var=np.array(Var)
		Mean=np.array(Mean)
		dq=1.
		lyap,sigma=lyapunov(a)
		D1=(1+dq)*np.log(2)/np.log(1/a**dq+1/(1-a)**dq)+1
#		ni=np.log(1/sB)*1.2*(2-D1)**2.*(D1-1)**2
		ni=np.log(1/sB)*2*(2-D1)/(D1-1)**4
	#	plt.plot(TT,Var[:,2]/np.log(1/sB),M[ii]+'-',color=plt.cm.cool(ii/3),label=r'$s_B=1/{:1.0f}$'.format(1/sB))
	
		tagg=np.log(1/sB)/(lyap+sigma/2)
		Var[Var==0]=np.nan
		plt.plot((TT-tagg)*(lyap),Var[:,0]/ni,M[ii]+'-',color=plt.cm.cool(j/3))
#		plt.plot(TT/tagg,Var[:,0]/np.log(1/sB),M[ii]+L[j],color=plt.cm.cool(ii/3))
	#plt.yscale('log')
		
	t=np.linspace(-10,20,1000)
	#plt.plot(t,((1+scipy.special.erf((0.08*t*lyap/sigma)))/2)**3,L[j],color=plt.cm.cool(j/3))
#plt.xscale('log')
#plt.ylim([0,1])
plt.xlabel('$(t-t_\mathrm{agg})\mu$')
plt.ylabel(r'$\sigma^2_{\log n} / \log(1/s_B) /[2(2-D_1)/(D_1-1)^4]$')

[plt.plot([],[],M[k]+'-',color='k',label=r'$s_B=1/{:1.0f}$'.format(1/sb)) for k,sb in enumerate(SB)]



t=np.linspace(-3,10,100)
# plt.plot(t,((1+scipy.special.erf(t-2.5))/2)**0.2,'-',color='k',label=r'$\sim \mathrm{erf} (t)^{0.2}$')
# plt.plot(t,((1+scipy.special.erf(t))/2),'-',color='k',label=r'$\sim \mathrm{erf} (t)^{0.2}$')

plt.legend()
plt.xlim(-3,10)


plt.yscale('log')
plt.ylim([1e-3,2e0])
ax2 = fig.add_axes([0.75, 0.6, 0.2, 0.02])
import matplotlib as mpl
#norm = mpl.colors.Normalize(vmin=T_all[0]*u_pore/d, vmax=T_all[-1]*u_pore/d)
norm = mpl.colors.Normalize(vmin=0.1, vmax=0.4)
cb1 = mpl.colorbar.ColorbarBase(ax2,cmap=plt.cm.cool,norm=norm,
								orientation='horizontal')
cb1.set_label(r'$a$', color='k',size=12,labelpad=0)
cb1.set_ticks([0.1,0.2,0.3,0.4])



plt.savefig('/home/joris/Dropbox/Articles/AggregationBatchelor/Figures/Baker_varN_t.pdf',bbox_inches='tight')
#%%% * mean(log n) as a function of time
import scipy
from scipy import spatial
import scipy.special
plt.style.use('~/.config/matplotlib/joris.mplstyle')
x0=np.array([0.56])
var_sa=0
nb=20
s0=0.05
D=1e-4
T=25
M=['d','o','s','*']
L=[':','-','--','-.']
SB=[1/50,1/100,1/500]
TT=np.arange(1,25)
fig=plt.figure(figsize=(4,3))

for j,a in enumerate([0.1,0.2,0.3,0.4]):
	for ii,sB in enumerate(SB):
		Var,Mean=[],[]
		for i,T in enumerate(TT):
			x,S,wrapped_time=lagrangian(x0,T,a,var_sa)
			he,ne=np.histogram(x,np.arange(0,1+sB,sB),density=False)
			Mean.append([np.mean(np.log(he[he>0])),np.mean((1/S))])
			Var.append([np.var(np.log(he[he>0])),np.var(np.log(1/S)),np.var(he[he>0]/np.mean(he[he>0]))])
		
		Var=np.array(Var)
		Mean=np.array(Mean)
		dq=1.
		lyap,sigma=lyapunov(a)
		D1=(1+dq)*np.log(2)/np.log(1/a**dq+1/(1-a)**dq)+1
#		ni=np.log(1/sB)*1.2*(2-D1)**2.*(D1-1)**2
		ni=(D1-1)
	#	plt.plot(TT,Var[:,2]/np.log(1/sB),M[ii]+'-',color=plt.cm.cool(ii/3),label=r'$s_B=1/{:1.0f}$'.format(1/sB))
	
		tagg=np.log(1/sB)/(lyap+sigma/2)
		Var[Var==0]=np.nan
		plt.plot((TT-tagg)*(lyap+sigma/2),Mean[:,0]/ni,M[ii]+'-',color=plt.cm.cool(j/3))
#		plt.plot(TT/tagg,Var[:,0]/np.log(1/sB),M[ii]+L[j],color=plt.cm.cool(ii/3))
	#plt.yscale('log')
		
	t=np.linspace(-10,20,1000)
	#plt.plot(t,((1+scipy.special.erf((0.08*t*lyap/sigma)))/2)**3,L[j],color=plt.cm.cool(j/3))
#plt.xscale('log')
#plt.ylim([0,1])
plt.xlabel('$(t-t_\mathrm{agg})(\mu+\sigma^2/2)$')
plt.ylabel(r'$\mu_{\log n}$')

[plt.plot([],[],M[k]+'-',color='k',label=r'$s_B=1/{:1.0f}$'.format(1/sb)) for k,sb in enumerate(SB)]



t=np.linspace(-3,10,100)
# plt.plot(t,((1+scipy.special.erf(t-2.5))/2)**0.2,'-',color='k',label=r'$\sim \mathrm{erf} (t)^{0.2}$')
# plt.plot(t,((1+scipy.special.erf(t))/2),'-',color='k',label=r'$\sim \mathrm{erf} (t)^{0.2}$')

plt.legend()
plt.xlim(-4,14)

#plt.yscale('log')
#plt.ylim([1e-3,2e0])
ax2 = fig.add_axes([0.75, 0.6, 0.2, 0.02])
import matplotlib as mpl
#norm = mpl.colors.Normalize(vmin=T_all[0]*u_pore/d, vmax=T_all[-1]*u_pore/d)
norm = mpl.colors.Normalize(vmin=0.1, vmax=0.4)
cb1 = mpl.colorbar.ColorbarBase(ax2,cmap=plt.cm.cool,norm=norm,
								orientation='horizontal')
cb1.set_label(r'$a$', color='k',size=12,labelpad=0)
cb1.set_ticks([0.1,0.2,0.3,0.4])



plt.savefig('/home/joris/Dropbox/Articles/AggregationBatchelor/Figures/Baker_meanlogN_t.pdf',bbox_inches='tight')#%%% * mean(log n) as a function of time
#%%% mean(n) as a function of time
import scipy
from scipy import spatial
import scipy.special
plt.style.use('~/.config/matplotlib/joris.mplstyle')
x0=np.array([0.56])
var_sa=0
nb=20
s0=0.05
D=1e-4
T=25
M=['d','o','s','*']
L=[':','-','--','-.']
SB=[1/500,1/100,1/50]
TT=np.arange(1,25)
fig=plt.figure(figsize=(2,2))

# plt.plot(t,((1+scipy.special.erf(t-2.5))/2)**0.2,'-',color='k',label=r'$\sim \mathrm{erf} (t)^{0.2}$')
# plt.plot(t,((1+scipy.special.erf(t))/2),'-',color='k',label=r'$\sim \mathrm{erf} (t)^{0.2}$')

for j,a in enumerate([0.1,0.2,0.3,0.4]):
	
	dq=2.0
	D2=[]
	f = lambda Dq : dq*np.log(2)-np.log(1/a**((dq-1)*Dq)+1/(1-a)**((dq-1)*Dq))
	D2=scipy.optimize.broyden1(f, 1, f_tol=1e-14)+1
	
	for ii,sB in enumerate(SB):
		Var,Mean,Mean0=[],[],[]
		for i,T in enumerate(TT):
			x,S,wrapped_time=lagrangian(x0,T,a,var_sa)
			he,ne=np.histogram(x,np.arange(0,1+sB,sB),density=False)
			Mean0.append([np.mean((he)),np.mean((1/S))])
			Mean.append([np.mean((he[he>0])),np.mean((1/S))])
			Var.append([np.var(he),np.var(np.log(1/S)),np.var(he[he>0]/np.mean(he[he>0]))])
		
		Var=np.array(Var)
		Mean=np.array(Mean)
		Mean0=np.array(Mean0)
		dq=1.
		lyap,sigma=lyapunov(a)
		D1=(1+dq)*np.log(2)/np.log(1/a**dq+1/(1-a)**dq)+1
#		ni=np.log(1/sB)*1.2*(2-D1)**2.*(D1-1)**2
		ni=sB
		#nivar=2*(2-D1)/(D1-1)**1*sB**2*(-np.log(sB))
		nivar=sB**2.*(sB**(D2-1-1)-1)
	#	plt.plot(TT,Var[:,2]/np.log(1/sB),M[ii]+'-',color=plt.cm.cool(ii/3),label=r'$s_B=1/{:1.0f}$'.format(1/sB))
	
		tagg=1#np.log(1/sB)
		Var[Var==0]=np.nan
		plt.plot((TT)*np.log(2),Mean0[:,0]/ni,M[ii]+'-',color=plt.cm.cool(j/3))
		plt.plot((TT)*np.log(2),np.sqrt(Var[:,0]/nivar),M[ii]+'--',color=plt.cm.cool(j/3))
#		plt.plot((TT-tagg),Mean0[:,0]/ni,M[ii]+'-',color=plt.cm.cool(j/3))
#		plt.plot(TT/tagg,Var[:,0]/np.log(1/sB),M[ii]+L[j],color=plt.cm.cool(ii/3))
	#plt.yscale('log')
		
t=np.linspace(0,24,100)
	#plt.plot(t,((1+scipy.special.erf((0.08*t*lyap/sigma)))/2)**3,L[j],color=plt.cm.cool(j/3))
#plt.xscale('log')
#plt.ylim([0,1])
plt.xlabel('$\log L(t) $')
plt.plot([],[],'--',color='k',label=r'$\sigma_n /  s_B / \sqrt{ s_B^{D_2-2}-1}$')
plt.plot([],[],'-',color='k',label=r'$\mu_{ n} /  s_B$')
plt.plot(t,np.exp(t),'k-', label='$L(t)=2^t$',linewidth=2,zorder=-10)
[plt.plot([],[],M[k],color='k',label=r'$s_B=1/{:1.0f}$'.format(1/sb)) for k,sb in enumerate(SB)]

subscript(plt.gca(),1,x=-0.05,y=0.95)


plt.legend(loc=2,frameon=False,fontsize=6)
plt.xlim(0,17)
plt.ylim(1,1e9)

plt.yscale('log')
#plt.ylim([1e-3,2e0])
ax2 = fig.add_axes([0.5, 0.3, 0.3, 0.02])
import matplotlib as mpl
#norm = mpl.colors.Normalize(vmin=T_all[0]*u_pore/d, vmax=T_all[-1]*u_pore/d)
norm = mpl.colors.Normalize(vmin=0.1, vmax=0.4)
cb1 = mpl.colorbar.ColorbarBase(ax2,cmap=plt.cm.cool,norm=norm,
								orientation='horizontal')
cb1.set_label(r'$a$', color='k',size=8,labelpad=0)
cb1.set_ticks([0.1,0.2,0.3,0.4])


plt.savefig('/home/joris/Dropbox/Articles/AggregationBatchelor/Figures/Baker_meanN_t.pdf',bbox_inches='tight')

#%%% * var(log n) as a function of time
import scipy
from scipy import spatial
import scipy.special
plt.style.use('~/.config/matplotlib/joris.mplstyle')
x0=np.array([0.56])
var_sa=0
nb=20
s0=0.05
D=1e-4
T=25
M=['d','o','s','*']
L=[':','-','--','-.']
SB=[1/100]
TT=np.arange(1,25)
fig=plt.figure(figsize=(4,3))

for j,a in enumerate([0.1,0.2,0.3,0.4]):
	for ii,sB in enumerate(SB):
		Var,Mean=[],[]
		for i,T in enumerate(TT):
			x,S,wrapped_time=lagrangian(x0,T,a,var_sa)
			he,ne=np.histogram(x,np.arange(0,1+sB,sB),density=False)
			Mean.append([np.mean(he[he>0]),np.mean((1/S))])
			Var.append([np.var(np.log(he[he>0])),np.var(np.log(1/S)),np.var(he[he>0]/np.mean(he[he>0]))])
		
		Var=np.array(Var)
		Var=np.array(Var)
		Mean=np.array(Mean)
		dq=1.
		lyap,sigma=lyapunov(a)
		D1=(1+dq)*np.log(2)/np.log(1/a**dq+1/(1-a)**dq)+1
		ni=np.log(1/sB)*2*(2-D1)/(D1-1)**2
	#	plt.plot(TT,Var[:,2]/np.log(1/sB),M[ii]+'-',color=plt.cm.cool(ii/3),label=r'$s_B=1/{:1.0f}$'.format(1/sB))
	
		tagg=np.log(1/sB)/(lyap+sigma/2)
#		Var[Var==0]=np.nan
#		plt.plot((TT-tagg)*(lyap),Var[:,0]/ni,M[ii]+'-',color=plt.cm.cool(j/3))
#		plt.plot(TT/tagg,Var[:,0]/np.log(1/sB),M[ii]+L[j],color=plt.cm.cool(ii/3))
		plt.plot(Var[:,0],(D1-1)**2*(sigma*TT-2*(2-D1)*lyap*TT),'o',color=plt.cm.cool(j/3))
	#plt.yscale('log')
		
	t=np.linspace(-10,20,1000)
	#plt.plot(t,((1+scipy.special.erf((0.08*t*lyap/sigma)))/2)**3,L[j],color=plt.cm.cool(j/3))
#plt.xscale('log')
#plt.ylim([0,1])
plt.xlabel('$\sigma^2_{\log n}$')
plt.ylabel(r'$f(\mu_{\log\rho},\sigma^2_{\log\rho},D_1)$')

[plt.plot([],[],M[k]+'-',color='k',label=r'$s_B=1/{:1.0f}$'.format(1/sb)) for k,sb in enumerate(SB)]



t=np.linspace(-3,10,100)
#plt.plot(t,((1+scipy.special.erf(t-2.5))/2)**0.2,'-',color='k',label=r'$\sim \mathrm{erf} (t)^{0.2}$')

#%%
plt.legend()
plt.xlim(-3,10)


plt.yscale('log')
plt.ylim([1e-3,2e1])
ax2 = fig.add_axes([0.75, 0.6, 0.2, 0.02])
import matplotlib as mpl
#norm = mpl.colors.Normalize(vmin=T_all[0]*u_pore/d, vmax=T_all[-1]*u_pore/d)
norm = mpl.colors.Normalize(vmin=0.1, vmax=0.4)
cb1 = mpl.colorbar.ColorbarBase(ax2,cmap=plt.cm.cool,norm=norm,
								orientation='horizontal')
cb1.set_label(r'$a$', color='k',size=12,labelpad=0)
cb1.set_ticks([0.1,0.2,0.3,0.4])



plt.savefig('/home/joris/Dropbox/Articles/AggregationBatchelor/Figures/Baker_N_t.pdf',bbox_inches='tight')





#%%% * var(log n)/var(log rho) as a function of time
import scipy
from scipy import spatial
plt.style.use('~/.config/matplotlib/joris.mplstyle')
x0=np.array([0.56])
var_sa=0
nb=20
s0=0.05
D=1e-4
T=25
M=['d','o','s','*']
TT=np.arange(10,25)
plt.figure(figsize=(4,3))

for ii,sB in enumerate([1/10,1/100,1/1000,1/5000]):
	AA=np.logspace(-3,np.log10(0.49),20)
	M=['d','o','s','*']
	Var,Mean=[],[]
	for i,T in enumerate(TT):
		a=0.15
		x,S,wrapped_time=lagrangian(x0,T,a,var_sa)
		he,ne=np.histogram(x,np.arange(0,1+sB,sB),density=False)
		Mean.append([np.mean(he[he>0]),np.mean((1/S))])
		Var.append([np.var(np.log(he[he>0])),np.var(np.log(1/S))])
	
	Var=np.array(Var)
	Var=np.array(Var)
	Mean=np.array(Mean)
	dq=1.
	nu=(1+dq)*np.log(2)/np.log(1/a**dq+1/(1-a)**dq)+1
	plt.plot(TT,Var[:,0]/Var[:,1]*sB**0.18,M[ii]+'-',color=plt.cm.cool(ii/3),label=r'$s_B=1/{:1.0f}$'.format(1/sB))
plt.yscale('log')
plt.xscale('log')
plt.ylim([1e-2,1e-1])
plt.xlabel('$t$')
plt.ylabel(r'$\sigma^2_{\log n}/\sigma^2_{\log \rho} \cdot {s_B}^{0.18}$')

plt.plot(TT,3/TT,'k--',label=r'$t^{-1}$')
plt.legend()
#%%% * compare var(log n) vs var(log rho) versus a
import scipy
from scipy import spatial
plt.style.use('~/.config/matplotlib/joris.mplstyle')
x0=np.array([0.121])
var_sa=0
nb=20
s0=0.05
D=1e-4
T=20
M=['d','o','s','*']

plt.figure(figsize=(3,2))
for ii,sB in enumerate([1/50,1/500,1/5000]):
	AA=np.logspace(-3,np.log10(0.49),20)
	M=['d','o','s','*']
	Var,Mean=[],[]
	#TT=np.arange(0,20,5)
	for i,a in enumerate(AA):
	#	a=0.3
		x,S,wrapped_time=lagrangian(x0,T,a,var_sa)
		he,ne=np.histogram(x,np.arange(0,1+sB,sB),density=False)
		Mean.append([np.mean(he[he>0]),np.mean((1/S))])
		Var.append([np.var(np.log(he[he>0])),np.var(np.log(1/S))])
	
	Var=np.array(Var)
	Mean=np.array(Mean)
	dq=1.
	nu=(1+dq)*np.log(2)/np.log(1/AA**dq+1/(1-AA)**dq)+1
	plt.plot(nu,Var[:,0]/Var[:,1],M[ii]+'-',color=plt.cm.cool(ii/3),label=r'$s_B=1/{:1.0f}$'.format(1/sB))
	plt.plot(nu,Mean[:,0]/Mean[:,1],M[ii]+'--',color=plt.cm.cool(ii/3),label=r'$s_B=1/{:1.0f}$'.format(1/sB))
plt.yscale('log')
plt.ylim([1e-1,1e0])
plt.xlabel('$D_1$')
plt.ylabel(r'$\sigma^2_{\log n}/\sigma^2_{\log \rho}$')
plt.legend()
#%%% * p(<N>/n) )p(<rho>/rho) function of a 
import scipy
from scipy import spatial
plt.style.use('~/.config/matplotlib/joris.mplstyle')
x0=np.array([0.121])
var_sa=0
nb=20
s0=0.05
D=1e-4
T=25
AA=np.array([0.15,0.3,0.45])
plt.figure(figsize=(3,2))
M=['d','o','s','*']

#TT=np.arange(0,20,5)
for i,a in enumerate(AA):
#	a=0.3
	x,S,wrapped_time=lagrangian(x0,T,a,var_sa)
	sB=1/500
	# Lagrangian
# =============================================================================
# 	tree=spatial.cKDTree(x.reshape(-1,1))
# 	neighboors=tree.query_ball_point(x.reshape(-1,1)[np.abs(x-0.5)<0.5-2*sB], sB/2.)
# 	nagg_lag=np.array([len(n) for n in neighboors])
#	nb=20
#	bin_n=np.unique(np.uint16(np.logspace(np.log10(nagg_lag.min()),
#																				np.log10(nagg_lag.max()+5),nb)))
#	hN,n=np.histogram(nagg_lag,bin_n,density=True)
#	plt.plot(n[:-1],hN,'o',color=plt.cm.jet(T/TT.max()))
# =============================================================================

	#fig,ax=plt.subplots(1,1,figsize=(2,2))
	import matplotlib as mpl
# =================Eulerian =================================================
	he,ne=np.histogram(x,np.arange(0,1+sB,sB),density=False)
	X=np.mean(he)/he[he>0]
	bins=np.logspace(np.log10(X.min()),np.log10(X.max()),20)
	hE,n=np.histogram(X,bins,density=True)
	plt.plot(n[:-1],hE,M[i],color=plt.cm.cool(a/AA.max()),label=r'$a={:1.2f}$'.format(a))
# =============================================================================
	lyap=-a*np.log(a)-(1-a)*np.log(1-a)
	sigma2=a*np.log(a)**2.+(1-a)*np.log(1-a)**2.-lyap**2.
	dq=1.
	rhoc=1/sB
	nu=(1+dq)*np.log(2)/np.log(1/a**dq+1/(1-a)**dq)+1
#!!! no dependence on fractal dimension !?
	l=((lyap)*T)
	s=sigma2*T
	rhom=np.average(1/S)
	X=rhom*S
	bins=np.logspace(np.log10(X.min()),np.log10(X.max()),30)
	hE,n=np.histogram(X,bins,density=True)
#	plt.plot(n[:-1],hE,'-',color=plt.cm.cool(a/AA.max()))
# Theoretical binomial
	N2=np.array([scipy.special.binom(T,i)/(a**i*(1-a)**(T-i)) for i in range(T+1)])
	pN=np.array([(a**i*(1-a)**(T-i)) for i in range(T+1)])
	#plt.bar(pN,N/np.sum(N)*T/(np.log(1-a)-np.log(a)),color='k',width=0.01,label='Binomial distribution')
	plt.plot(pN*rhom,N2/np.sum(N2),'-',color=plt.cm.cool(a/AA.max()))

	#plt.plot(n,1/np.sqrt(2*np.pi*s)*np.exp(-(n-l)**2./(2*s)),'-',color=plt.cm.viridis(T/TT.max()))

	plt.xlabel(r'$\langle n \rangle / n $')
#plt.ylim([1e-2,1e1])
plt.xlim([1e-4,1e5])
plt.ylim([1e-6,1e1])
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.savefig('/home/joris/Dropbox/Articles/AggregationBatchelor/Figures/Baker_N_a.pdf',bbox_inches='tight')


#%%% * p(<N>/n) )p(<rho>/rho) function of sb
import scipy
from scipy import spatial
plt.style.use('~/.config/matplotlib/joris.mplstyle')
x0=np.array([0.121])
var_sa=0
nb=20
s0=0.05
D=1e-4
T=25
AA=np.array([0.15,0.3,0.45])
plt.figure(figsize=(3,2))
M=['d','o','s','*']

SB=1/np.array([50,250,1000])
a=0.3
x,S,wrapped_time=lagrangian(x0,T,a,var_sa)
#TT=np.arange(0,20,5)
for i,sB in enumerate(SB):
#	a=0.3
#	sB=1/500
	# Lagrangian
# =============================================================================
# 	tree=spatial.cKDTree(x.reshape(-1,1))
# 	neighboors=tree.query_ball_point(x.reshape(-1,1)[np.abs(x-0.5)<0.5-2*sB], sB/2.)
# 	nagg_lag=np.array([len(n) for n in neighboors])
#	nb=20
#	bin_n=np.unique(np.uint16(np.logspace(np.log10(nagg_lag.min()),
#																				np.log10(nagg_lag.max()+5),nb)))
#	hN,n=np.histogram(nagg_lag,bin_n,density=True)
#	plt.plot(n[:-1],hN,'o',color=plt.cm.jet(T/TT.max()))
# =============================================================================

	#fig,ax=plt.subplots(1,1,figsize=(2,2))
	import matplotlib as mpl
# =================Eulerian =================================================
	he,ne=np.histogram(x,np.arange(0,1+sB,sB),density=False)
	X=np.mean(he)/he[he>0]
	bins=np.logspace(np.log10(X.min()),np.log10(X.max()),20)
	hE,n=np.histogram(X,bins,density=True)
	plt.plot(n[:-1],hE,M[i],color=plt.cm.cool(i/3),label=r'$s_B=1/{:1.0f}$'.format(1/sB))
# =============================================================================
	lyap=-a*np.log(a)-(1-a)*np.log(1-a)
	sigma2=a*np.log(a)**2.+(1-a)*np.log(1-a)**2.-lyap**2.
	dq=1.
	rhoc=1/sB
	nu=(1+dq)*np.log(2)/np.log(1/a**dq+1/(1-a)**dq)+1
#!!! no dependence on fractal dimension !?
	l=((lyap)*T)
	s=sigma2*T
rhom=np.average(1/S)
X=rhom*S
bins=np.logspace(np.log10(X.min()),np.log10(X.max()),30)
hE,n=np.histogram(X,bins,density=True)
#plt.plot(n[:-1],hE,'ko')

	#plt.plot(n,1/np.sqrt(2*np.pi*s)*np.exp(-(n-l)**2./(2*s)),'-',color=plt.cm.viridis(T/TT.max()))

plt.xlabel(r'$\langle n \rangle / n $')
# Theoretical binomial
N2=np.array([scipy.special.binom(T,i)/(a**i*(1-a)**(T-i)) for i in range(T+1)])
pN=np.array([(a**i*(1-a)**(T-i)) for i in range(T+1)])
#plt.bar(pN,N/np.sum(N)*T/(np.log(1-a)-np.log(a)),color='k',width=0.01,label='Binomial distribution')
plt.plot(pN*rhom,N2/np.sum(N2),'k-')

#plt.ylim([1e-2,1e1])
plt.xlim([1e-4,1e4])
plt.ylim([1e-6,1e1])
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.savefig('/home/joris/Dropbox/Articles/AggregationBatchelor/Figures/Baker_N_sb.pdf',bbox_inches='tight')


#%%% <log rho>  function of log rho 
import matplotlib
from scipy import spatial
plt.style.use('~/.config/matplotlib/joris.mplstyle')
x0=np.array([0.121])
var_sa=0
nb=20
s0=0.05
D=1e-4
T=24
plt.figure(figsize=(3,2))
#TT=np.arange(0,20,5)
#for T in TT:
a=0.2
x,S,wrapped_time=lagrangian(x0,T,a,var_sa)
sB=1/100
# Lagrangian
# =============================================================================
# 	tree=spatial.cKDTree(x.reshape(-1,1))
# 	neighboors=tree.query_ball_point(x.reshape(-1,1)[np.abs(x-0.5)<0.5-2*sB], sB/2.)
# 	nagg_lag=np.array([len(n) for n in neighboors])
#	nb=20
#	bin_n=np.unique(np.uint16(np.logspace(np.log10(nagg_lag.min()),
#																				np.log10(nagg_lag.max()+5),nb)))
#	hN,n=np.histogram(nagg_lag,bin_n,density=True)
#	plt.plot(n[:-1],hN,'o',color=plt.cm.jet(T/TT.max()))
# =============================================================================

#fig,ax=plt.subplots(1,1,figsize=(2,2))
import matplotlib as mpl
# =================Eulerian =================================================
he,ne=np.histogram(x,np.arange(0,1+sB,sB),density=False)
logrho,ne=np.histogram(x,np.arange(0,1+sB,sB),density=False,weights=np.log(1/S))
mlogrho=logrho/he

logrho=np.log(1/S)
logrhom=mlogrho[np.uint16(x/sB-1)]

plt.figure(figsize=(3,3))
plt.hist2d(logrhom[~np.isnan(logrhom)],logrho[~np.isnan(logrhom)],50,norm=matplotlib.colors.LogNorm())
plt.plot([10,30],[10,30],'k--')
plt.xlabel(r'$\langle \log \rho \rangle$')
plt.ylabel(r'$ \log \rho $')
plt.xlim([logrho.min(),logrho.max()])
plt.ylim([logrho.min(),logrho.max()])

plt.legend(fontsize=6,title='$a={:1.2f}$'.format(a))


#%%% log rho  function of log N
import matplotlib
from scipy import spatial
plt.style.use('~/.config/matplotlib/joris.mplstyle')
x0=np.array([0.121])
var_sa=0
nb=20
s0=0.05
D=1e-4
T=24
plt.figure(figsize=(3,2))
#TT=np.arange(0,20,5)
#for T in TT:
a=0.3

x,S,wrapped_time=lagrangian(x0,T,a,var_sa)
sB=1/100

# Lagrangian
# =============================================================================
# 	tree=spatial.cKDTree(x.reshape(-1,1))
# 	neighboors=tree.query_ball_point(x.reshape(-1,1)[np.abs(x-0.5)<0.5-2*sB], sB/2.)
# 	nagg_lag=np.array([len(n) for n in neighboors])
#	nb=20
#	bin_n=np.unique(np.uint16(np.logspace(np.log10(nagg_lag.min()),
#																				np.log10(nagg_lag.max()+5),nb)))
#	hN,n=np.histogram(nagg_lag,bin_n,density=True)
#	plt.plot(n[:-1],hN,'o',color=plt.cm.jet(T/TT.max()))
# =============================================================================

#fig,ax=plt.subplots(1,1,figsize=(2,2))
import matplotlib as mpl
# =================Eulerian =================================================
he,ne=np.histogram(x,np.arange(0,1+sB,sB),density=False)


logrho=np.log(1/S)
N=np.log(he[np.uint16(x/sB-1)])

plt.figure(figsize=(3,3))
#plt.hist2d(N[~np.isnan(N)&(N>0)],logrho[~np.isnan(N)&(N>0)],50,norm=matplotlib.colors.LogNorm())
#h=np.histogram2d(logrho[~np.isnan(N)&(N>0)],N[~np.isnan(N)&(N>0)],50,weights=1/np.exp(logrho[~np.isnan(N)&(N>0)]))
h=np.histogram2d(logrho[~np.isnan(N)&(N>0)],N[~np.isnan(N)&(N>0)],50)
plt.pcolormesh(h[2],h[1],h[0])
plt.xlabel(r'$\log n$')
plt.ylabel(r'$ \log \rho $')
plt.xlim([0,15])
plt.ylim([0,30])

logn=np.linspace(0,15,100)
dq=1.
rhoc=1/sB
logrhoc=np.log(1/sB)
D1=(1+dq)*np.log(2)/np.log(1/a**dq+1/(1-a)**dq)+1
plt.plot(logn, logn/(D1-1)+logrhoc,'k-',label=r'$(D_1-1)^{-1}\log n - \log (s_B/L)$')
plt.plot(logn, logn/(D1-1)+logrhoc+logn*(2*(2-D1)/(D1-1)),'k--',label=r'$+ 2(2-D_1)/(D_1-1)$')
plt.plot(logn, logn/(D1-1)+logrhoc-logn*(2*(2-D1)/(D1-1)),'k--',label=r'$- 2(2-D_1)/(D_1-1)$')
#plt.ylim([logrho.min(),logrho.max()])

plt.legend(fontsize=6,title='$a={:1.2f}$'.format(a))


#%%% log N  function of log rho
import matplotlib
from scipy import spatial
plt.style.use('~/.config/matplotlib/joris.mplstyle')
x0=np.array([0.121])
var_sa=0
nb=20
s0=0.05
D=1e-4
T=24
plt.figure(figsize=(3,2))
#TT=np.arange(0,20,5)
#for T in TT:
a=0.3
x,S,wrapped_time=lagrangian(x0,T,a,var_sa)
sB=1/100

# Lagrangian
# =============================================================================
# 	tree=spatial.cKDTree(x.reshape(-1,1))
# 	neighboors=tree.query_ball_point(x.reshape(-1,1)[np.abs(x-0.5)<0.5-2*sB], sB/2.)
# 	nagg_lag=np.array([len(n) for n in neighboors])
#	nb=20
#	bin_n=np.unique(np.uint16(np.logspace(np.log10(nagg_lag.min()),
#																				np.log10(nagg_lag.max()+5),nb)))
#	hN,n=np.histogram(nagg_lag,bin_n,density=True)
#	plt.plot(n[:-1],hN,'o',color=plt.cm.jet(T/TT.max()))
# =============================================================================

#fig,ax=plt.subplots(1,1,figsize=(2,2))
import matplotlib as mpl
# =================Eulerian =================================================
he,ne=np.histogram(x,np.arange(0,1+sB,sB),density=False)


logrho=np.log(1/S)
N=np.log(he[np.uint16(x/sB-1)])

plt.figure(figsize=(3,3))
plt.hist2d(logrho[~np.isnan(N)&(N>0)],N[~np.isnan(N)&(N>0)],50,norm=matplotlib.colors.LogNorm())
plt.ylabel(r'$\log n$')
plt.xlabel(r'$ \log \rho $')
plt.xlim([0,30])
plt.ylim([0,15])

lr=np.linspace(0,30,100)
dq=1.
rhoc=1/sB
logrhoc=np.log(1/sB)
D1=(1+dq)*np.log(2)/np.log(1/a**dq+1/(1-a)**dq)+1
plt.plot(lr+logrhoc, lr*(D1-1),'k-',label=r'$(D_1-1)^{-1}\log \rho $')
plt.plot(lr+logrhoc, lr*(D1-1)+lr*(2*(2-D1)/(D1-1)),'k--',label=r'$+ 2(2-D_1)/(D_1-1) \log \rho$')
plt.plot(lr+logrhoc, lr*(D1-1)-lr*(2*(2-D1)/(D1-1)),'k--',label=r'$- 2(2-D_1)/(D_1-1) \log \rho$')
#plt.ylim([logrho.min(),logrho.max()])

plt.legend(fontsize=6,title='$a={:1.2f}$'.format(a))



#%%% * p(N) for several times
from scipy import spatial
plt.style.use('~/.config/matplotlib/joris.mplstyle')
x0=np.array([0.1])
var_sa=0
s0=0.05
D=1e-4
aa=np.array([0.1,0.3,0.45])
plt.figure(figsize=(2,2))
#TT=np.arange(0,20,5)
T=15
for a in aa:
#	a=0.45
	x,S,wrapped_time=lagrangian(x0,T,a,var_sa)
	sB=0.005
	# Lagrangian
# =============================================================================
# 	tree=spatial.cKDTree(x.reshape(-1,1))
# 	neighboors=tree.query_ball_point(x.reshape(-1,1)[np.abs(x-0.5)<0.5-2*sB], sB/2.)
# 	nagg_lag=np.array([len(n) for n in neighboors])
#	nb=20
#	bin_n=np.unique(np.uint16(np.logspace(np.log10(nagg_lag.min()),
#																				np.log10(nagg_lag.max()+5),nb)))
#	hN,n=np.histogram(nagg_lag,bin_n,density=True)
#	plt.plot(n[:-1],hN,'o',color=plt.cm.jet(T/TT.max()))
# =============================================================================

	#fig,ax=plt.subplots(1,1,figsize=(2,2))
	import matplotlib as mpl
# =================Eulerian =================================================
	he,ne=np.histogram(x,np.arange(0,1,sB),density=False)
	he=he[he>0]
	bin_n=np.unique(np.uint16(np.logspace(np.log10(he.min()),
																					np.log10(he.max()+5),nb)))
	hE,n=np.histogram(he,bin_n,density=True)
	plt.plot(n[:-1],hE,'.',color=plt.cm.viridis(a/aa.max()),label='$a={:1.2f}$'.format(a),fillstyle='full')
# =============================================================================
	lyap=-a*np.log(a)-(1-a)*np.log(1-a)
	sigma2=a*np.log(a)**2.+(1-a)*np.log(1-a)**2.-lyap**2.
	dq=2*3/4.
	rhoc=1/sB
	nu=(1+dq)*np.log(2)/np.log(1/a**dq+1/(1-a)**dq)+1
	l=(nu-1)*(lyap*T-np.log(rhoc))
	s=sigma2*(nu-1)**2.*T
	n=np.logspace(0,3,200)
	plt.plot(n,1/np.sqrt(2*np.pi*s)/n*np.exp(-(np.log(n)-l)**2./(2*s)),'--',color=plt.cm.viridis(a/aa.max()))

#!!! no dependence on fractal dimension !?
	l=((lyap)*T-np.log(rhoc))
	s=sigma2*T
	n=np.logspace(0,3,200)
	plt.plot(n,1/np.sqrt(2*np.pi*s)/n*np.exp(-(np.log(n)-l)**2./(2*s)),'-',color=plt.cm.viridis(a/aa.max()))
	plt.xscale('log')
	plt.yscale('log')
	plt.xlabel('$n$')
	plt.ylabel(r'$P(n,t={:1.0f})$'.format(T))
plt.ylim([1e-5,2])
plt.xlim([0.5,1e3])
plt.legend(fontsize=8)
plt.savefig('/home/joris/Dropbox/Articles/AggregationBatchelor/Figures/P(N,15)_a.pdf',bbox_inches='tight')




#%% Theory
#%%% Gamma pdf of n, <c^2>

plt.style.use('~/.config/matplotlib/joris.mplstyle')
nmin=0.01
nmax=1e3
t=15
sB=1/100

D11=np.linspace(1.2,1.99,100)
Gamma2=[]
for i,D1 in enumerate(D11):
	mu=1/(D1-1)
	sigma2=2*(2-D1)/(D1-1)
	if mu-2*sigma2>0:
		M2=2*mu-2*sigma2
	else:
		M2=mu**2/(2*sigma2)
	
	
	epsilon=-M2+2
	
	k=-(D1-1)/2/np.log(sB)/(2-D1)
	
	from scipy.special import gamma
	def mean_n_alpha(n,k,theta,alpha):
		# Gamma distribution
		return 1/gamma(k)/theta**k*n**(alpha+k-1)*np.exp(-n/theta)
		#return (n**alpha)/gamma(k)/theta**k*n**(k-1)*np.exp(-n/theta)

	import scipy.integrate
	
	# Empirical
	E=np.loadtxt('Baker_epsilon.txt')
	epsilon=np.interp(D1,E[:,0],E[:,1])
	alpha=-1+epsilon
	
	# Empirical
	xi=np.loadtxt('Baker_xi.txt')
	alpha=-np.interp(D1,xi[:,0],xi[:,1])
	
	
	T=np.arange(5,10)
	C0=np.array([scipy.integrate.quad(mean_n_alpha,nmin,nmax,args=(k,2**t*sB*np.abs(np.log(sB))*2*(2-D1)/(D1-1),0))[0] for t in T])
	C2=np.array([scipy.integrate.quad(mean_n_alpha,nmin,nmax,args=(k,2**t*sB*np.abs(np.log(sB))*2*(2-D1)/(D1-1),alpha))[0] for t in T])
	
	plt.plot(T,C2/C0,'.',color=plt.cm.cool(i/len(D11)))
	plt.yscale('log')
	p=np.polyfit(T,np.log(C2/C0),1)
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
plt.ylim([0,2*np.log(2)])
np.savetxt('gamma2_theory.txt',np.vstack((D11,Gamma2)).T)

#%%% Probability of min(rho)

import scipy.special
T=10
N=np.array([scipy.special.binom(T,i) for i in range(T+1)])
N2=np.array([scipy.special.binom(T,i)*a**i*(1-a)**(T-i) for i in range(T+1)])
pN=np.array([-np.log((a**i*(1-a)**(T-i)))/T for i in range(T+1)])

#%%% Toy model of aggregation
#6/9/2021

n_set=[2,1] # possible class of stretching rates

n=[[[1],[1]]]

for k in range(10):
	n_temp=[]
	for n1 in n:
		K=[[2*n1[1],2*n1[1]],[n1[2]]
		[n1[1],2*n1[2]],[2*n1[2]]
		[2*n1[1]],[2*n1[1],n1[2]]
		[n1[1]],[2*n1[2],2*n1[2]]]
		ntemp.append(K)
		for nn in n_set:
		n_temp.append(n*nn)

#%%% Origin of D2
import numpy as np
import scipy.optimize
a=0.3
q=2
dq=q*0.9519
Dq=(1+dq)*np.log(2)/np.log(1/a**dq+1/(1-a)**dq)
print(np.log(4),np.log(1/a**((q-1)*Dq)+1/(1-a)**((q-1)*Dq)))

a=0.49
f = lambda Dq : np.log(4)-np.log(1/a**((q-1)*Dq)+1/(1-a)**((q-1)*Dq))
Dq = scipy.optimize.broyden1(f, 1, f_tol=1e-14)
print(Dq)


#%%% PDF with sine shape

from scipy import integrate
x=np.linspace(0,2*np.pi,10000)
y=np.sin(x)
h,xb=np.histogram(np.abs(y),np.logspace(-4,0,100),density='True')
plt.plot(xb[1:],h,'.-')
y=np.sin(x)*1e-2
h,xb=np.histogram(np.abs(y),np.logspace(-4,0,100),density='True')
plt.plot(xb[1:],h,'.-')
plt.plot(x,1/np.sqrt(1-x**2)/np.pi*2,'k--')
plt.plot(x,1/np.sqrt(1e-4-x**2)/np.pi*2,'k--')
plt.yscale('log')
plt.xscale('log')

plt.figure()
lyap=6
sigma2=lyap
def PC(c):
	p_cmax_p_c=lambda cmax : 1/np.sqrt(cmax**2. - c**2.)/np.pi*2/(2*np.sqrt(2*sigma2*np.pi)*cmax)*np.exp(-(np.log(cmax)+lyap)**2./2/sigma2)
	return integrate.quad(p_cmax_p_c,c,1)[0]

cmax=np.logspace(-12,0,100)
Pc=np.array([PC(c) for c in cmax])
plt.plot(cmax,Pc)
plt.plot(cmax,1/(2*np.sqrt(2*sigma2*np.pi)*cmax)*np.exp(-(np.log(cmax)+lyap)**2./2/sigma2),'b--')

plt.ylim([1e-4,1e4])
plt.yscale('log')
plt.xscale('log')

# lyap=2
# sigma2=lyap
# c=0.01
# p_cmax_p_c=lambda cmax : 1./np.sqrt(cmax**2. - c**2.)
# plt.plot(cmax,p_cmax_p_c(cmax))
# integrate.quad(p_cmax_p_c,c,)[0]
# plt.yscale('log')
# plt.xscale('log')

# /np.pi*2/(2*np.sqrt(2*sigma2*np.pi)*cmax)*np.exp(-(np.log(cmax)+lyap)**2./2/sigma2)

#%%% Fractal dimension from Ott and ANtonsen

plt.style.use('~/.config/matplotlib/joris.mplstyle')
import scipy
#A=np.logspace(-7,np.log10(0.49),100)
var_sa=0.0
A=np.linspace(0.01,0.49,10)
x0=np.float64(np.random.rand(1))
#x0=np.random.rand(100)
T=15
plt.figure(figsize=(5,5))
Q=np.linspace(2,10,10)
Q=[1.01]
Nu=[]
Dq=[]
Dq_Ott=[]
for a in A:
	x,S,wrapped_time=lagrangian(x0,T,a,var_sa)
	q=1.01
	nu,N,Nf,snu=fractal_best(x,q)
	#nu,N,Nf=fractal(x,q)
	nuS=fractal_weighted(x,S,q)
	print(nu)
	Nu.append([nu/(q-1),snu/(q-1),nuS/(q-1)])
	#f = lambda Dq : 2**q-1/a**((q-1)*Dq)+1/(1-a)**((q-1)*Dq)
	dq=q
	f = lambda Dq : dq*np.log(2)-np.log(1/a**((dq-1)*Dq)+1/(1-a)**((dq-1)*Dq))
	Dq.append(scipy.optimize.broyden1(f, 1, f_tol=1e-14))
	nb=5000
	N=np.histogram(x,np.linspace(0,1,nb))[0]
	rho=np.histogram(x,np.linspace(0,1,nb),weights=1/S)[0]
	rho=rho[N>0]/N[N>0]
	logrho=np.histogram(x,np.linspace(0,1,nb),weights=np.log(1/S))[0]
	logrho=logrho[N>0]/N[N>0]
	sigma=np.linspace(0,4,100)
	q=np.array([np.log(np.sum(N**s))/np.log(np.sum(N)) for s in sigma])
	Dq_sig=2+(sigma-q)/(q-1)
	plt.plot(q,Dq_sig)
b=q
Nu=np.array(Nu)
#	plt.errorbar(A,Nu[:,0],yerr=2*Nu[:,1],label=r'$D_{:1.0f}(1)$'.format(q),color=plt.cm.viridis(q/5.))
plt.plot(A,Nu[:,0]/Nu[:,0].max(),'o',label=r'$D_{:1.0f}(1)$'.format(q),color=plt.cm.viridis(q/10.))
#plt.plot(A,Nu[:,1],label=r'$D_2(s)$')
#plt.plot(A,-np.log(A**2.+(1-A)**2.)/np.log(2),'k--',label='$(\log(a^2+(1-a)^2)/\log 2$')
plt.plot(A,(b+1)*np.log(2)/np.log(1/A**b+1./(1-A)**b),'+',color=plt.cm.viridis(q/10.))
plt.plot(A,Dq,'--',color=plt.cm.viridis(q/10.))

N=10000
x=np.exp(np.random.randn(N))
print(np.sum(x*np.log(x))/N)
print(np.sum(x)*np.sum(np.log(x))/N**2.)
print(np.sum(x)/N*np.log(np.sum(x)/N))

#%% Dispersive  Baker map

plt.style.use('~/.config/matplotlib/joris.mplstyle')
x=np.array([0.2])
sa=np.array([0.1,0.4])
disp=0.2
T=10

V=[]

S=np.ones(x.shape) # we start with a unique lamellae
wrapped_time=np.zeros(x.shape) # we start with a unique lamellae
dt=1.
sa=sa/np.sum(sa)
sai=np.zeros(len(sa)+1)
sai[1:]=sa
sai=np.cumsum(sai)
plt.figure(figsize=(1.5,1.5))
for k in range(T):
	print(k)
	x+=(np.random.randn(1))*disp
	cells=np.arange(np.floor(x.min()),np.ceil(x.max())) # cells
	xtemp=np.array([])
	wtemp=np.array([])
	Stemp=np.array([])
	for c in cells:
		idc=(x>c)&(x<=c+1)
		xtemp=np.hstack((xtemp,
								 c+np.hstack([(x[idc]-c)*sa[i]+sai[i] for i in range(len(sa))])))
		wtemp=np.hstack((wtemp,
								 np.hstack([wrapped_time[idc]+1/(S[idc]*sa[i])**2.*dt for i in range(len(sa))])))
		Stemp=np.hstack((Stemp,np.hstack([S[idc]*sa[i] for i in range(len(sa))])))
	x=xtemp
	S=Stemp
	wrapped_time=wtemp
	V.append(np.var(x))
	plt.plot(np.vstack((x,x)),
					np.vstack((np.zeros(x.shape)+k-1,np.zeros(x.shape)+k))
					,'k-',alpha=0.1)
plt.ylabel('Time')
plt.xlabel('Space')
tt=np.arange(T)
plt.figure()
plt.plot(tt,V)
plt.plot(tt,tt*0.1)
