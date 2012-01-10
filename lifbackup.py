import gaussianenv as ge
import poissonneuron as pn
import numpy as np
import matplotlib.pyplot as plt

order = 3
eta = 1
gamma = 0.5
zeta = 1.0
L = 1
N = 1
sigma = 0.0001
alpha = 0.000001

lif = pn.IFNeuron(tau=2.0,k = np.random.normal(1.0,1.0,order),stochastic = True, alpha=alpha)
e = ge.GaussianEnv(zeta,gamma,eta,L,N,-2.0,-2.0,4.0,4.0,sigma,order)


time = 500000
spike = np.zeros((time,))
stim = np.zeros((time,order))
V = np.zeros((time,))
dt = 0.0001

for i in range(20000):
	e.samplestep(dt)

mu = np.zeros((time,order+1))
sigma = np.zeros((time,order+1,order+1))

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

for i in range(time):
	stim[i,:] = e.samplestep(dt)
	spike[i] = lif.spike(stim[i,:],dt)
	V[i] = lif.V
	if spike[i] == 1:
		sigma[i,:,:] = sigma[i-1,:,:] - np.dot(np.array([sigma[i-1,:,0]]).T,np.array([sigma[i-1,:,0]]))/(alpha**2+sigma[i-1,0,0])
		theta = Vthresh
		theta[0] -= mu[i-1,0]
		mu[i,:] = mu[i-1,:] + np.dot(sigma[i,:,:],theta)/alpha**2
		mu[i,0] = 0
	else:
		mu[i,:] = mu[i-1,:]-dt*np.dot(Gamma,mu[i-1,:])	
		sigma[i,:,:] = sigma[i-1,:,:] - dt*(np.dot(Gamma,sigma[i-1,:,:])+np.dot(sigma[i-1,:,:],Gamma.T)-Eta)
#plt.plot(np.arange(0.0,dt*time,dt),V,np.arange(0.0,dt*time,dt),spike,np.arange(0.0,dt*time,dt),stim)

plt.plot(np.arange(0.0,dt*time,dt),mu[:,0],'r:',np.arange(0.0,dt*time,dt),mu[:,1],'b:')
plt.plot(np.arange(0.0,dt*time,dt),V[:],'r',np.arange(0.0,dt*time,dt),stim[:,0],'b')
