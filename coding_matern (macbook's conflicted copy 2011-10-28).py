#!/usr/bin/python

import numpy as np
import sys
from videosink import VideoSink, grayscale
import poissonneuron as pn
import gaussianenv as ge
import matplotlib.pyplot as plt

def getMaternSample( gamma = 1.0, eta = 1.0, order = 2, alpha = 0.1, phi = 2.0, dtheta = 0.3, dt = 0.001, repetitions = 100, timewindow = 10000, spacesteps = 400, plot = False):
	zeta = 2
	L = 0.8
	N = 1
	a = np.zeros(N*N)
	sigma = 0.001

	e = ge.GaussianEnv(zeta,gamma,eta,L,N,-2.0,-2.0,4.0,4.0,sigma,order)
	gam = e.getgamma()
	et = e.geteta()
	abar  = np.sqrt(2.0*np.pi)*alpha*phi/dtheta
	code = pn.PoissonCode(np.arange(-4.0,4.0,dtheta),alpha,phi)
	space = np.arange(-4.0,4.0,8.0/spacesteps)
	weight = 1.0/repetitions
	sigmaavg = np.zeros((timewindow,order,order))

	for k in range(repetitions):
		mu = np.zeros((timewindow,order))
		stim = np.zeros((timewindow,order))
		sigma = np.zeros((timewindow,order,order))
		sigma[-1,:,:] = 0.001*np.eye(order)
		sigmanew = np.zeros((order,order))
		P = np.zeros((spacesteps,timewindow))
		spcount = 0
		for i in range(timewindow):
			s = e.samplestep(dt).ravel()
			stim[i,:] = s
			spi = code.spikes(s[0],dt)
			if sum(spi)>=1:
				spcount +=1
				lam = np.linalg.inv(sigma[i-1,:,:])
				while sum(spi)!=1:
					spi = code.spikes(s[0],dt)
				ids = np.where(spi==1)
				thet = np.zeros_like(mu[i,:])
				thet[0] = code.neurons[ids[0]].theta[0]
				sigma[i,:,:] = sigma[i-1,:,:] - np.dot(np.array([sigma[i-1,:,0]]).T,np.array([sigma[i-1,:,0]]))/(alpha**2+sigma[i-1,0,0])
				mu[i,:] = np.dot(sigma[i,:,:],np.dot(lam,mu[i-1,:])+thet/alpha**2)	
			else:
				mu[i,:] = mu[i-1,:] - dt*np.dot(gam,mu[i-1,:])
				sigma[i,:,:] = sigma[i-1,:,:] - dt*(np.dot(gam,sigma[i-1,:,:])+np.dot(sigma[i-1,:,:],gam.T)-et)
		sigmaavg = sigmaavg + sigma*weight
		print "Run", k, "Firing rate was ", np.float(spcount)/(timewindow*dt), "abar is ", abar
	
	for i in range(timewindow):
		P[:,i] = np.exp(-(space-mu[i,0])**2/(2.0*sigma[i,0,0]))/(np.sqrt(2.0*np.pi*sigma[i,0,0]))		
	sigmamf = np.zeros((timewindow,order,order))
	sigmamf[-1,:,:] = 0.001*np.eye(order)
	for i in range(timewindow):
		sigmamf[i,:,:] = sigmamf[i-1,:,:] - dt*(np.dot(gam,sigmamf[i-1,:,:])+np.dot(sigmamf[i-1,:,:],gam.T)-et) - dt*abar*np.dot(np.array([sigmamf[i-1,:,0]]).T,np.array([sigmamf[i-1,:,0]]))/(alpha**2+sigmamf[i-1,0,0])

	if plot == True:
		fig = plt.figure()
		ax = fig.add_subplot(2,1,1)
		ax2 = fig.add_subplot(2,1,2)
		ts = np.arange(0.0,dt*timewindow,dt)
		ax.plot(ts,mu[:,0],'b:',ts,stim[:,0],'r')
		ax.imshow(P,extent=[0,ts[-1],4.0,-4.0],aspect='auto',cmap='gist_yarg')
		ax2.plot(ts,sigma[:,0,0],'r:',ts,sigmamf[:,0,0],'b',ts,sigmaavg[:,0,0],'k')
	return [sigmaavg, sigma, sigmamf]

def getMaternEqVariance( gamma = 1.0, eta = 1.0, order = 2, alpha = 0.2, phi = 1.3, dtheta = 0.3, dt = 0.001, samples = 100, timewindow = 10000, spacesteps = 400, plot = False, Trelax = 1, histmax = 0.8 ):
	zeta = 2
	L = 0.8
	N = 1
	a = np.zeros(N*N)
	sigma = 0.001
	e = ge.GaussianEnv(zeta,gamma,eta,L,N,-2.0,-2.0,4.0,4.0,sigma,order)
	gam = e.getgamma()
	et = e.geteta()
	abar  = np.sqrt(2.0*np.pi)*alpha*phi/dtheta
	code = pn.PoissonCode(np.arange(-4.0,4.0,dtheta),alpha,phi)
	space = np.arange(-4.0,4.0,8.0/spacesteps)
	sigmaavg = np.zeros((samples,order,order))
	

	mu = np.zeros((order))
	sigma = np.zeros((order,order))
	sigmamf = np.zeros((order,order))
	sigma[:,:] = 0.001*np.eye(order)
	sigmamf[:,:] = 0.001*np.eye(order)
	sigmanew = np.zeros((order,order))
	for i in range(timewindow):
		s = e.samplestep(dt).ravel()
		spi = code.spikes(s[0],dt)
		if sum(spi)>=1:
			#lam = np.linalg.inv(sigma[:,:])
			#ids = np.where(spi==1)
			#thet = np.zeros_like(mu[:])
			#thet[0] = code.neurons[ids[0]].theta[0]
			sigma = sigma - np.dot(np.array([sigma[:,0]]).T,np.array([sigma[:,0]]))/(alpha**2+sigma[0,0])
			#mu = np.dot(sigma,np.dot(lam,mu)+thet/alpha**2)	
		else:
			#mu = mu - dt*np.dot(gam,mu)
			sigma = sigma - dt*(np.dot(gam,sigma)+np.dot(sigma,gam.T)-et)
	
	sample_interval = 3
	for i in range(sample_interval*samples):
		s = e.samplestep(dt).ravel()
		spi = code.spikes(s[0],dt)
		if sum(spi)>=1:
			#lam = np.linalg.inv(sigma[:,:])
			#ids = np.where(spi==1)
			#thet = np.zeros_like(mu[:])
			#thet[0] = code.neurons[ids[0]].theta[0]
			#mu = np.dot(sigma,np.dot(lam,mu)+thet/alpha**2)	
			sigma = sigma - np.dot(np.array([sigma[:,0]]).T,np.array([sigma[:,0]]))/(alpha**2+sigma[0,0])
		else:
			#mu = mu - dt*np.dot(gam,mu)
			sigma = sigma - dt*(np.dot(gam,sigma)+np.dot(sigma,gam.T)-et)
		if i%sample_interval == 0:
			sigmaavg[i/sample_interval,:,:] = sigma[:,:]

	(vers,subvers,_,_,_) = sys.version_info
	if subvers>=7:
		[freqs,xs] = np.histogram(sigmaavg[:,0,0], bins = np.arange(0.0,histmax,histmax/80),normed = True)#, new = True)
	else:
		[freqs,xs] = np.histogram(sigmaavg[:,0,0], bins = np.arange(0.0,histmax,histmax/80),normed = True, new = True)
	xs = 0.5*(xs[0:-1]+xs[1:])
	sigma = np.average(sigmaavg,axis=0)

	for i in range(timewindow):
                sigmamf = sigmamf + dt*(et - np.dot(gam,sigmamf) - np.dot(sigmamf,gam.T)) -dt*abar*np.dot(np.array([sigmamf[:,0]]).T,np.array([sigmamf[:,0]]))/(alpha**2+sigmamf[0,0])
	for i in range(timewindow):
		sigmanew = sigmanew + dt*(et - np.dot(gam,sigmanew) - np.dot(sigmanew,gam.T))
	
	return [sigma, sigmamf, sigmanew,  xs, freqs]
