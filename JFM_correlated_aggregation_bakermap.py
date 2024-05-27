#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 15:53:46 2020

@author: joris
"""

#%% RUN FIRST : Quick Baker Flow simulations
##TODO

#
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, misc
import numpy.fft
from scipy.ndimage import gaussian_filter


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

#	sigma=16
M=10 # number of moments to compute
dt=1/1  # Discretisation of time step

def diffusion_fourier(C,sigma):
	input_ = numpy.fft.fft2(C)
	result = ndimage.fourier_gaussian(input_, sigma=sigma)
	return numpy.fft.ifft2(result).real


def diffusion(C,sigma):
	return gaussian_filter(C, sigma)
# Initial condition
#for sigma in np.linspace(3,50,4):
#for sigma in np.linspace(1,20,3):


def DNS_sigma(x0,T,sa,random,sigma,n=int(2**14)):
	
	X,Y=np.meshgrid(0,np.arange(n))
	np.random.seed(seed=1)
	C=np.zeros((1,n),dtype=np.float64())
	#C[0,:int(n/2)]=1
	s0=0.05*n
	for i in range(x0.shape[0]):
		C[0,:]+=np.exp(-(Y[:,0]-x0[i]*n)**2./s0**2.)
	Cm,VC,K,Theta,H=[],[],[],[],[]
	C0=np.mean(C)
	M=10
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


#%%Figure 5b


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
fig.savefig('baker_c_fractals_a{:1.1f}.jpg'.format(sa),bbox_inches='tight',dpi=600)

#%% Figure 6 D2 = f(sigma2/lambda)


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

plt.savefig('fractal_lyapunov.pdf',bbox_inches='tight')



#%% Figure8a: sigma^2_n for several times Eulerian

import scipy.spatial as spatial
import scipy.optimize

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
	
plt.plot([],[],'k-',label=r'$s_a^{D_2-2}-1$')
plt.plot([],[],'ko',label=r'$ \sigma^2_{n/<n>} $')
plt.plot([],[],'o',label=r'$ s_a=1/20 $',color=plt.cm.cool(0/4))
plt.plot([],[],'o',label=r'$ s_a=1/50 $',color=plt.cm.cool(1/4))
plt.plot([],[],'o',label=r'$ s_a=1/200 $',color=plt.cm.cool(2/4))
plt.plot([],[],'o',label=r'$ s_a=1/500 $',color=plt.cm.cool(3/4))

l,s=lyapunov(a)
plt.xlabel('$a$')
plt.ylabel('Moments of $n$')
plt.legend(loc=3,fontsize=6)
plt.yscale('log')
subscript(plt.gca(),0)
plt.savefig('sigma2_n_baker.pdf',bbox_inches='tight')

#%% Figure 8b mean(n) as a function of time
import scipy
from scipy import spatial
import scipy.special

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
	#	plt.plot(TT,Var[:,2]/np.log(1/sB),M[ii]+'-',color=plt.cm.cool(ii/3),label=r'$s_a=1/{:1.0f}$'.format(1/sB))
	
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
plt.plot([],[],'--',color='k',label=r'$\sigma_n /  s_a / \sqrt{ s_a^{D_2-2}-1}$')
plt.plot([],[],'-',color='k',label=r'$\mu_{ n} /  s_a$')
plt.plot(t,np.exp(t),'k-', label='$L(t)=2^t$',linewidth=2,zorder=-10)
[plt.plot([],[],M[k],color='k',label=r'$s_a=1/{:1.0f}$'.format(1/sb)) for k,sb in enumerate(SB)]

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


plt.savefig('Baker_meanN_t.pdf',bbox_inches='tight')
#%% Figure10a * p(N) for several times
from scipy import spatial,optimize

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
plt.savefig('Baker_P(N,t)_a{:1.2f}.pdf'.format(a),bbox_inches='tight')

#%% Figure 10b p(N) for several a
from scipy import spatial

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
plt.savefig('Baker_P(N,t)_a.pdf'.format(a),bbox_inches='tight')
#%%Figure 11a % Graphical coarsening scale


T=10
sa=0.3
random=0.0
x0=np.array([0.123])
x,S,wrapped_time=lagrangian(x0,T,sa,random)
sB=1/50
s0=0.05

lyap,sigma2=lyapunov(sa)
D=(1/50)**2/2*0.1

n=int(2**14)

C=DNS_sigma(x0,T,sa,random,np.sqrt(2*D)*n,n=n)

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

fig.savefig('baker_N_LAG.pdf',bbox_inches='tight')



#%% Figure 13 * Grid-based statistics: log- rho versus log n

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
#plt.plot(nu-1,np.log(rhoc)+np.zeros(SA.shape),'r--',label=r'$\langle \rho_c \rangle_{s_a}=-\log s_a$',linewidth=1.)
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
plt.savefig('/home/joris/Dropbox/Articles/AggregationBatchelor/Figures/statbundles_aaker.pdf',bbox_inches='tight')

#plt.savefig('statbundles_aaker.pdf',bbox_inches='tight')


#plt.figure(figsize=(2,2))
#plt.xscale('log')
#plt.yscale('log')

#%% Figure 14 * Grid-based statistics of bundles  1/N ~ <1/ RHO> dependence on a

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
ax[1].set_ylabel(r'$ \tilde\omega / 2 \log (\mathcal{A}/(\ell_0 s_a))$')
ax[1].set_xlabel('$D_1$')
subscript(ax[1],1)
plt.subplots_adjust(wspace=0.2)
plt.savefig('/home/joris/Dropbox/Articles/AggregationBatchelor/Figures/gamma2bundles.pdf',bbox_inches='tight')

#%% Figure 15 % * Grid-based statistics of bundles dependence on Pe

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
plt.xlabel('$  \log (\mathcal{A}/(\ell_0 s_a))$')
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
plt.plot(s,2*s,'r-',label='$2 \log (  \mathcal{A} / (\ell_0 s_a))$')
plt.ylim([0,20])
plt.legend()

plt.savefig('gamma2bundles_Peclet.pdf',bbox_inches='tight')

#%% Figure 16 % * Grid-based statistics of bundles  1/N ~ c^2 dependence on a

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
plt.savefig('gamma2_1_rho&c.pdf')

plt.figure()
D1=1+(1+dq)*np.log(2)/np.log(1/SA**dq+1/(1-SA)**dq)
plt.plot(D1,-P2[:,1]/(2*np.log(1/(sB))),label=r'$\omega_{2,c|n}$')
plt.ylim([0,2])
plt.xlabel('$D_1$')
plt.legend()

#%%Figure 20a % * Scalar decay reconstructed Eulerian grid
import scipy.optimize


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
	plt.plot(1+D1,decay_rate[:,2],'ko',label='Simulation',zorder=100)

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

#KSI=np.loadtxt('Baker_xi.txt')
#ksi=np.interp(1+D1,KSI[:,0],KSI[:,1])

# p=np.polyfit(KSI[:-1,0],KSI[:-1,1],1)
# ksi=(1+D1)*p[0]+p[1]

ksi1=np.copy(ksi)
ksi1[kn>ksi]=np.nan
ksi[kn<ksi]=np.nan

idint=np.where(kn>ksi)[0][1]

idint=1

plt.plot(1+D1,-np.log(1-2*A+2*A**2.),'-',color='seagreen',label=r'Isolated lamella',linewidth=1.2)
#plt.plot([],[],'-',color='w',label=r'\textbf{Aggregation models:}',linewidth=1.2)
#plt.plot(1+D1,-np.log(1-3*A+3*A**2.),'--',color='darkorange',label=r'Fully correlated',linewidth=1.2)
plt.plot(1+D1,np.zeros(D1.shape)+np.log(2),':',color='indianred',label=r'Random aggregation',linewidth=1.2)
plt.plot(1+D1,np.log(2)*ksi,'-',color='blueviolet',label=r'Correlated agg. ($\xi<k_n$)',linewidth=1.2) #kn>ksi
plt.plot(1+D1,np.log(2)*ksi1,':',color='blueviolet',label=r'Correlated  agg. ($\xi>k_n$)',linewidth=1.2) #kn>ksi

#plt.plot(1+D1[:idint],np.log(2)*(ksi[:idint]-kn[:idint]),'-',color='blueviolet',linewidth=1.2) # kn<ksi
#plt.yscale('log')
plt.xlabel(r'$D_1$')
plt.ylabel(r'$\gamma_{2,c}$')

# G2=np.loadtxt('Baker_gamma2_theory2.txt').T
# D1=(1+dq)*np.log(2)/np.log(1/G2[:,0]**dq+1/(1-G2[:,0])**dq)
# plt.plot(D1+1,G2[:,1],'k--',label=r'Theory (empirical $P_n$)')
# plt.ylim([1e-2,2e0])

subscript(plt.gca(),1,x=-0.12)
# G2=np.loadtxt('gamma2_theory.txt')
# plt.plot(G2[:,0],G2[:,1],'k-',label=r'Theory (Gamma $P_n$)')
# plt.ylim([1e-2,2e0])

plt.legend(frameon=False,fontsize=6)
#plt.yscale('log')
plt.ylim([1e-2,2])

#plt.savefig('/home/joris/Dropbox/Articles/AggregationBatchelor/Figures/gamma2_baker_log.pdf',bbox_inches='tight')

plt.savefig('gamma2_baker_log.pdf',bbox_inches='tight')

