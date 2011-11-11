#!/usr/bin/python

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time as t
import matplotlib.cm as cm
import subprocess
import os
import sys


class neuron(object):
	"""Implements a poisson neuron with a gaussian tuning function"""
	def __init__(X,A,phi):
		"""Generates the neuron object,
		X are the coordinates of the stimulus points
		A is the correlation of the gaussian tuning function
		phi is the maximal firing rate
		inva the inverse of a is calculated once for reuse"""
		self.X=X
		self.A=A
		self.inva = np.array(np.matrix(A).I)
		self.phi=phi
		(M,D) = self.X.shape
		self.theta = np.random.normal(0.0,1.0,M)
	def rate(S):
		"""Gives the rate of the poisson spiking process given the stimulus S"""
		exponent = np.dot(S.transpose()-self.theta.transpose(),np.dot(self.inva,S-self.theta))
		return phi*np.exp(-0.5*exponent)
	def spike(S,dt):
		"""Generates a spike with probability rate*dt"""
		r = dt*self.rate(S)
		if np.random.uniform()<r:
			return 1
		return 0


class GaussianEnv(object):
	def __init__(self,zeta,gamma,eta,L,N,x0,y0,Lx,Ly,sigma):
		self.xs = np.arange(x0,x0+Lx,Lx/N)
		self.ys = np.arange(y0,y0+Ly,Ly/N)
		self.zeta = zeta
		self.gamma = gamma
		self.L = L
		self.N = N
		self.k = np.zeros((N*N,N*N))
		self.S = np.random.normal(0.0,1.0,N*N)
		for i in range(0,N*N):
			for j in range(0,N*N):
				(ix,iy) =divmod(i,N)
				(jx,jy) =divmod(j,N)
				dist = np.sqrt((self.xs[ix]-self.xs[jx])**2 + (self.ys[iy]-self.ys[jy])**2) 
				self.k[i,j] = np.exp(-(dist/L)**zeta)+sigma**2*(i==j)
		self.khalf = np.linalg.cholesky(self.k)
		
	def sample(self):
		s = np.random.normal(0.0,1.0,N*N)
		s = np.dot(self.khalf,s)
		return np.reshape(s,(N,N))
		
	def samplestep(self,dt):
		self.S = (1-gamma*dt)*self.S + np.sqrt(dt)*eta*np.random.normal(0.0,1.0,N*N)
		sample = np.dot(self.khalf,self.S)
		return np.reshape(sample,(N,N))
		
		
			
nframes = 50
N = 20
zeta = 2
L = 5
a = np.zeros(N*N)
sigma = 0.001
gamma = 1.0
eta = 1.0
dt = 0.01

e = GaussianEnv(zeta,gamma,eta,L,N,-2.0,-2.0,4.0,4.0,sigma)

prefix = sys.argv[1]

# for i in range(0,N*N):
# 	for j in range(0,N*N):
# 		(ix,iy) =divmod(i,N)
# 		(jx,jy) =divmod(j,N)
# 		dist = np.sqrt((ix-jx)**2 + (iy-jy)**2) 
# 		k[i,j] = np.exp(-(dist/L)**zeta)+sigma**2*(i==j)

# 
# L = np.linalg.cholesky(k)
# id = np.identity(N*N)
# sample = np.random.multivariate_normal(a,id)
# sample1 = np.random.multivariate_normal(a,id)

print "Now sampling and plotting...\n"

delete_them = []

for i in range(0,nframes):
# 	sample1 = (1-2*gamma*dt)*sample1-gamma**2*sample+eta*np.random.multivariate_normal(a,id)
# 	sample = sample + dt*sample1
# 	ss = np.dot(L,sample)
# 	plot = np.reshape(ss,(N,N))
	plot = e.samplestep(dt)
	f = plt.figure()
	plt.imshow(plot,cmap = cm.Greys_r, vmin=-3, vmax = 3)
	filename = prefix+str('%05d' % i) + '.png'
	delete_them.append(filename)
 	plt.savefig(filename, dpi = 100)
	print 'Wrote plot to ', filename
	plt.close(f)
	#plt.colorbar()

outfile = prefix+'.avi'

command = ('mencoder',
           'mf://*%s*.png' %prefix,
           '-mf',
           'type=png:w=800:h=600:fps=25',
           '-ovc',
           'lavc',
           '-lavcopts',
           'vcodec=mpeg4',
           '-oac',
           'copy',
           '-o',
           outfile)
           
print "\n\nabout to execute:\n%s\n\n" % ' '.join(command)
subprocess.check_call(command)
print "\n\n The movie was written to 'output.avi'"

for i in delete_them:
	command = 'rm '+i
	print 'running ',command
	os.system(command)