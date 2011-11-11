import cPickle as pic
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

matplotlib.rc('text',usetex = True)
#matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
#matplotlib.rc('mathtext',fontset='stixsans')
#matplotlib.rc('mathtext',default='sf')
matplotlib.rc('font',size=8.0)

try:
	picklename
	filename=picklename
	print "all good"
except NameError:
	print "not so good"
	filename = "pickle_order_3"
#filename = "pickle_order_3"
try:
	outprefix
	prefix = outprefix
except NameError:
	print "not so good"
	prefix = "pickle_any_"

fi = open(filename,"r")

a = pic.load(fi)

fi.close()

phis = sorted(np.unique([j for (i,j) in a.keys()]))
alphas = sorted(np.unique([i for (i,j) in a.keys()]))
sigmas = np.array([[a[(i,j)][0][0,0] for j in phis] for i in alphas])
sigmasmf = np.array([[a[(i,j)][1][0,0] for j in phis] for i in alphas])
sigmaseq = np.array([[a[(i,j)][2][0,0] for j in phis] for i in alphas])
hists = np.array([[a[(i,j)][4] for j in phis] for i in alphas])
xs = a[(alphas[0],phis[0])][3]

sigmaeq = np.average(sigmaseq)

normsigma = sigmas/sigmaeq
normmf = sigmasmf/sigmaeq

figmap = plt.figure()

axmap = figmap.add_subplot(1,2,1)

immap= axmap.imshow(normsigma.T,extent=[alphas[0],alphas[-1],phis[-1],phis[0]],vmin=0.0,vmax=1.0, interpolation='nearest',aspect='auto')
axmap.set_xlabel(r'$\alpha$')
axmap.set_ylabel(r'$\phi$')
axmap.set_title(r'$MMSE/\sigma^2_x$ (simulated)')
cb = plt.colorbar(immap)
cb.set_label(r'$MMSE/\sigma^2_x$')


axplot = figmap.add_subplot(1,2,2)
axplot.set_xlabel(r'$\alpha$')
#axplot.set_ylabel(r'$MMSE/\sigma^2_x$')
axplot.set_title(r'$MMSE/\sigma^2_x$ (simulated vs. mean-field)')
axplot.plot(alphas,normsigma[:,5],'g',alphas,normsigma[:,10],'b',alphas,normsigma[:,15],'r',alphas,normmf[:,5],'g:',alphas,normmf[:,10],'b:',alphas,normmf[:,15],'r:')
axplot.legend((r'$\phi=0.5$',r'$\phi=1.0$',r'$\phi=1.5$'))
min5 = np.array(np.where(normmf[:,5]==np.min(normmf[:,5]))).ravel()
min10 = np.array(np.where(normmf[:,10]==np.min(normmf[:,10]))).ravel()
min15 = np.array(np.where(normmf[:,15]==np.min(normmf[:,15]))).ravel()
axplot.plot(alphas[min5],normmf[min5,5],'go',alphas[min10],normmf[min10,10],'bo',alphas[min15],normmf[min15,15],'ro')
figmap.savefig(prefix+'_simulated.png',dpi =600)
plt.close(figmap)

figmap = plt.figure()

axmap = figmap.add_subplot(1,1,1)

immap= axmap.imshow(normmf.T,extent=[alphas[0],alphas[-1],phis[-1],phis[0]],vmin=0.0,vmax=1.0, interpolation='nearest')
axmap.set_xlabel(r'$\alpha$')
axmap.set_ylabel(r'$\phi$')
axmap.set_title(r'$MMSE/\sigma^2_x$')
cb = plt.colorbar(immap)
cb.set_label(r'$MMSE/\sigma^2_x$')
figmap.savefig(prefix+'_mf.png',dpi =600)
plt.close(figmap)

err = abs(sigmas-sigmasmf)/sigmas

figmap = plt.figure()

axmap = figmap.add_subplot(1,1,1)

immap= axmap.imshow(err.T,extent=[alphas[0],alphas[-1],phis[-1],phis[0]],vmin=0.0,vmax=1.0, interpolation='nearest')
axmap.set_xlabel(r'$\alpha$')
axmap.set_ylabel(r'$\phi$')
axmap.set_title(r'$MMSE/\sigma^2_x$')
cb = plt.colorbar(immap)
cb.set_label(r'$MMSE/\sigma^2_x$')
figmap.savefig(prefix+'_error.png',dpi =600)
plt.close(figmap)

fighists = plt.figure()

axhists = fighists.add_subplot(1,1,1)

for hs in hists:
	for h in hs:
		axhists.plot(xs,h)

fighists.savefig(prefix+'hists.png',dpi=300)
plt.close(fighists)
