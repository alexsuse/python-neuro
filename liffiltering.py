import gaussianenv as ge
import poissonneuron as pn
import numpy as np
import matplotlib.pyplot as plt

order = 1
eta = 1
gamma = 0.5
zeta = 1.0
L = 1
N = 1
sigma = 0.0001
alpha = 0.000001

lif = pn.IFNeuron(tau=2.0,k = np.random.normal(1.0,1.0,order),stochastic = False, alpha=alpha)
e = ge.GaussianEnv(zeta,gamma,eta,L,N,-2.0,-2.0,4.0,4.0,sigma,order)


time = 500000
spike = np.zeros((time,))
stim = np.zeros((time,order))
V = np.zeros((time,))
dt = 0.0001

mu = np.zeros((time,order))
covar = np.zeros((time,time,order,order))

for i in range(20000):
	e.samplestep(dt)

G = e.getgamma()
E = e.geteta()

Gamma = np.zeros((order+1,order+1))
Eta = np.zeros((order+1,order+1))

Eta[1:,1:] = E
Gamma[1:,1:] = G

Gamma[0,0] = 1.0/lif.tau
Gamma[0,1:] = -lif.k

mu = np.zeros((time,order+1))
sigma = np.zeros((time,order+1,order+1))
sigma[-1,:,:] = np.eye(order+1)
Vthresh = np.zeros((order+1,))
Vthresh[0] = lif.averagethresh
i=0
while True:
	t_old = i
	while True:
		stim[i,:] = e.samplestep(dt)
		spike[i] = lif.spike(stim[i,:],dt)
		V[i] = lif.V
		if spike[i]==1:
			break
		i = i+1
	times = arange(0.0,dt*(i-t_old,dt),dt)
	K = grammatrix(times,lif.tau)	
	L = makelikelihood(times,lif.tau,lif.k)
	G = np.linalg.inv(np.linalg.inv(K)+L)	

def grammatrix(times,tau):
	M = times.repeat(times.size).reshape(times.size,times.size)
	K = np.exp(-np.abs(M-M.transpose())/tau)
	return K

def makelikelihood(times,tau,k,alpha,tdash):
	M = times.repeat(times.size).reshape(times.size,times.size)
	kktranspose = np.dot(k.reshape(k.size,1),k.reshape(1,k.size))
	expterm = np.exp(-(2tdash-M-M.transpose())/tau)
	L = expterm*kktranspose/alpha**2


plt.plot(np.arange(0.0,dt*time,dt),mu[:,0],'r:',np.arange(0.0,dt*time,dt),mu[:,1],'b:')
plt.plot(np.arange(0.0,dt*time,dt),V[:],'r',np.arange(0.0,dt*time,dt),stim[:,0],'b')
