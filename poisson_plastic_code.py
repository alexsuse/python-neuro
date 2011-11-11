from gaussianenv import GaussianEnv

emat = GaussianEnv(1.0,1.0,1.0,1.0,1,0.0,0.0,1.0,1.0,0.00001,2)
eou = GaussianEnv(1.0,1.0,1.0,1.0,1,0.0,0.0,1.0,1.0,0.00001,1)

ts = arange(0.0,4.0,0.01)

smat1 = []
sou1 = []

for i in ts:
	smat1.append(emat.samplestep(0.01))
	sou1.append(eou.samplestep(0.01))

smat1 = array(smat1).ravel()
sou1 = array(sou1).ravel()

emat.reset()
eou.reset()
	
smat2 = []
sou2 = []

for i in ts:
	smat2.append(emat.samplestep(0.01))
	sou2.append(eou.samplestep(0.01))

smat2 = array(smat2).ravel()
sou2 = array(sou2).ravel()

emat.reset()
eou.reset()

smat3 = []
sou3 = []

for i in ts:
	smat3.append(emat.samplestep(0.01))
	sou3.append(eou.samplestep(0.01))

smat3 = array(smat3).ravel()
sou3 = array(sou3).ravel()

emat.reset()
eou.reset()

smat4 = []
sou4 = []

for i in ts:
	smat4.append(emat.samplestep(0.01))
	sou4.append(eou.samplestep(0.01))

smat4 = array(smat4).ravel()
sou4 = array(sou4).ravel()
	
f = figure()
axou = f.add_subplot(1,2,1)
axmat = f.add_subplot(1,2,2)

axou.plot(ts,sou1,'b',ts,sou2,'r',ts,sou3,'g',ts,sou4,'k')
axmat.plot(ts,smat1,'b',ts,smat2,'r',ts,smat3,'g',ts,smat4,'k')
axmat.set_xlabel('Time [s]')
axou.set_xlabel('Time [s]')
axou.set_ylabel('Space [cm]')
axou.set_title('Ornstein-Uhlenbeck Process')
axmat.set_title('Gaussian Process with Matern Kernel of higher order')
