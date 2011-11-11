import gaussianenv as ge
from numpy import *
from matplotlib.pyplot import *

emat = ge.GaussianEnv(1.0,1.3,10.0,1.0,1,0.0,0.0,1.0,1.0,0.0000,5)
eou = ge.GaussianEnv(1.0,1.0,1.0,1.0,1,0.0,0.0,1.0,1.0,0.0000,1)

dt = 0.01

steps = 4000

ts = arange(0.0,dt*steps,dt)

smat1 = []
sou1 = []

ou = 0.0
mat = 0.0

smat1 = emat.samplestep(dt,steps)
sou1 = eou.samplestep(dt,steps)

#smat1 = array(smat1)
#sou1 = array(sou1)

emat.reset()
eou.reset()
	
smat2 = []
sou2 = []

smat2 =emat.samplestep(dt,steps)
sou2 =eou.samplestep(dt,steps)

emat.reset()
eou.reset()

smat3 = []
sou3 = []

smat3 =emat.samplestep(dt,steps)
sou3 =eou.samplestep(dt,steps)


emat.reset()
eou.reset()

smat4 = []
sou4 = []

smat4 =emat.samplestep(dt,steps)
sou4 =eou.samplestep(dt,steps)
	
f = figure()
axou = f.add_subplot(2,1,1)
axmat = f.add_subplot(2,1,2)

axou.plot(ts,sou1,'b',ts,sou2,'r',ts,sou3,'g',ts,sou4,'k')
axmat.plot(ts,smat1[:,-1],'b',ts,smat2[:,-1],'r',ts,smat3[:,-1],'g',ts,smat4[:,-1],'k')
axmat.set_xlabel('Time [s]')
axou.set_xlabel('Time [s]')
axou.set_ylabel('Space [cm]')
axou.set_title('Ornstein-Uhlenbeck Process')
axmat.set_title('Gaussian Process with Matern Kernel of higher order')
