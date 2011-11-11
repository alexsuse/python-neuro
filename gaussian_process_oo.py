#!/usr/bin/python

import numpy as np
import matplotlib
matplotlib.use('Agg')
#import matplotlib.pyplot as plt
#import matplotlib.cm as cm
import sys
from videosink import VideoSink, grayscale
from neuron import PoissonNeuron, IFNeuron, PoissonPlasticCode, PoissonPlasticNeuron
from gaussianenv import GaussianEnv


filename = 'trash' if (len(sys.argv)<2) else sys.argv[1]
N = 20 if (len(sys.argv)<3) else int(sys.argv[2]) 
nframes = 1000 if (len(sys.argv)<4) else int(sys.argv[3])
zeta = 2
L = 0.8
a = np.zeros(N*N)
sigma = 0.001
gamma = 1.0
eta = 10.0
dt = 0.01
order = 2
alpha = 0.1
phi = 10

e = GaussianEnv(zeta,gamma,eta,L,N,-2.0,-2.0,4.0,4.0,sigma,order)
neuron = PoissonNeuron(np.random.normal(0.0,1.0,N*N),alpha*np.identity(N*N),phi,N)
lifneuron = IFNeuron(np.zeros(N),dt,1)

code = PoissonPlasticCode()

video = VideoSink((N,N),filename,rate = 25,byteorder = "Y8")

print "Now sampling and plotting...\n"

delete_them = []

gr = lambda x : (grayscale(x,-6.0,6.0))
vgr = np.vectorize(gr)
for i in range(0,nframes):
	plot = e.samplestep(dt)
	p =	vgr(plot)
	spi = neuron.spike(plot,dt)
	video.run(p.astype(np.uint8))
	print spi
# 	f = plt.figure()
# 	plt.imshow(plot,cmap = cm.Greys_r,extent=[-2,2,-2,2], vmin=-3, vmax = 3)
# 	plt.imshow(p.astype(np.uint8),cmap=cm.Greys_r, vmin = 0, vmax = 255)
# 	filename = prefix+str('%05d' % i) + '.png'
# 	delete_them.append(filename)
#  	plt.savefig(filename, dpi = 100)
# 	print 'Wrote plot to ', filename
# 	plt.close(f)
	#plt.colorbar()

video.close()

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
