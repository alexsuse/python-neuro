#!/usr/bin/python

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#import matplotlib.cm as cm
import sys
from neuron import IFNeuron, PoissonCode
from gaussianenv import GaussianEnv


prefix = 'trash' if (len(sys.argv)<2) else sys.argv[1]
N = 20 if (len(sys.argv)<3) else int(sys.argv[2]) 
nframes = 1000 if (len(sys.argv)<4) else int(sys.argv[3])
zeta = 2
L = 0.1
a = np.zeros(N*N)
sigma = 0.001
gamma = 1.0
eta = 1.0
dt = 0.01
order = 2
alpha = 100
phi = 10

e = GaussianEnv(zeta,gamma,eta,L,N,-2.0,-2.0,4.0,4.0,sigma,order)
poisscode = PoissonCode(np.random.normal(0.0,1.0,(N*N,N*N)),alpha*np.identity(N*N),phi,N)
lifneuron = IFNeuron(np.zeros(N),dt,1)

print "Now sampling and plotting...\n"

rates = []

for i in range(0,nframes):
	plot = e.samplestep(dt)
	rates.append(poisscode.totalrate(plot))
 	
f = plt.figure()
filename = prefix+ '.png'
plt.plot(rates)
plt.savefig(filename, dpi = 100)
print 'Wrote plot to ', filename
plt.close(f)


# 
# command = ('mencoder',
#            'mf://*%s*.png' %prefix,
#            '-mf',
#            'type=png:w=800:h=600:fps=25',
#            '-ovc',
#            'lavc',
#            '-lavcopts',
#            'vcodec=mpeg4',
#            '-oac',
#            'copy',
#            '-o',
#            outfile)
#            
# print "\n\nabout to execute:\n%s\n\n" % ' '.join(command)
# subprocess.check_call(command)
# print "\n\n The movie was written to 'output.avi'"
# 
# for i in delete_them:
# 	command = 'rm '+i
# 	print 'running ',command
# 	os.system(command)
