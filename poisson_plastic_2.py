#!/usr/bin/python
import numpy as np
import poissonneuron as neu
from matplotlib.pyplot import *



code = neu.PoissonRessonantNeuron(np.array([0.0]),np.array([1.0]), 50.0,5.0,5.0)

dt = 0.001
ts = np.arange(0.0,10.0,dt)
spiketimes = []
resource = np.zeros_like(ts)
a=np.array([0.0])


for i,time in enumerate(ts):
	sp = code.spike(a,dt)
	if sp==1:
		spiketimes.append(time)
	resource[i] = code.mu1
			
fig = figure()
ax = fig.add_subplot(1,1,1)

ax.plot(ts,resource,'b:')
ax.plot(spiketimes,np.ones_like(spiketimes)*np.average(resource) ,',')
fig.show()

