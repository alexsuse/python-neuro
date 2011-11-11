#!/usr/bin/python
import numpy as np
import gaussianenv as ge
import poissonneuron as neu
from matplotlib.pyplot import *


emat = ge.GaussianEnv(1.0,1.0,1.0,1.0,1,0.0,0.0,1.0,1.0,0.00001,2)
eou = ge.GaussianEnv(1.0,1.0,1.0,1.0,1,0.0,0.0,1.0,1.0,0.00001,1)

code = neu.PoissonPlasticCode(phi = 10.0,alpha = 0.2)

dt = 0.001
ts = np.arange(0.0,10.0,dt)

smat1 = []
sou1 = []
spiketimes = []
spikeids = []

for i,time in enumerate(ts):
	a = emat.samplestep(dt)
	smat1.append(a)
	sp = code.spikes(a,dt)
	spikers = (sp==1).nonzero()[0].astype('int')
	for spike in spikers:
		spiketimes.append(time)
		spikeids.append(code.neurons[spike].theta)	
			
fig = figure()
ax = fig.add_subplot(1,1,1)
smat1 = np.array(smat1).ravel()
sou1 = np.array(sou1).ravel()

ax.plot(ts,smat1,'b:')
ax.plot(spiketimes,spikeids,',')
fig.show()
emat.reset()
eou.reset()
	
smat2 = []
sou2 = []

for i in ts:
	smat2.append(emat.samplestep(0.01))
	sou2.append(eou.samplestep(0.01))

