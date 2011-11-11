#!/usr/bin/python
import numpy as np
import coding_matern as coding
import cPickle
import multiprocessing as mp
from multiprocessing import Queue
from Queue import Empty

order = 2

nsample = 100


def runm(i,o):
	while True:
		try:
			[a,p] = i.get(block=False)
			sigmas = np.zeros((order,order))
			sigmasmf = np.zeros((order,order))
			sigmaseq = np.zeros((order,order))
			xs = np.array([])
			freqs = np.array([])
			[sigmas[:,:], sigmasmf[:,:], sigmaseq[:,:], xs, freqs] = coding.getMaternEqVariance(samples = nsample, phi = p, alpha = a)
			o.put([(a,p),sigmas[:,:],sigmasmf[:,:],sigmaseq[:,:],xs,freqs],block=False)
		except Empty:
			break

if __name__=='__main__':
	inp = mp.Queue()
	outp = mp.Queue()

	phis = np.arange(0.0,3.0,1.0)
	alphas = np.arange(0.001,3.0,1.0)

	ncpus = mp.cpu_count()
	out = dict()
	for p in phis:
		for a in alphas:
			inp.put([a,p])
	processes =[mp.Process(target=runm,args=(inp,outp)) for i in range(ncpus)]

	for p in processes:
		p.start()
	for p in processes:
		p.join()

	while True:
		try:
			sigmas = np.zeros((order,order))
			sigmasmf = np.zeros((order,order))
			sigmaseq = np.zeros((order,order))
			[(a,p),sigmas[:,:],sigmasmf[:,:],sigmaseq[:,:],xs,freqs] = outp.get(block=False)
			out[(a,p)]=[sigmas,sigmasmf,sigmaseq,xs,freqs]
		except Empty:
			break
	
	fileout = open("phis_alphas_histograms.test","w+")
	cPickle.dump(out,fileout)
	fileout.close()
