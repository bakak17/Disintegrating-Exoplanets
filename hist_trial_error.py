# coding: utf-8
plt.hist(flux1, bins = 'auto')
from astropy.table import Table
import matplotlib.pyplot as plt 
import numpy as np
dat = Table.read('Kep1520folded_3.fits')
pts = (dat['TIME'] < -0.46812330415978130036517942754554542301650447803823163392)
flux1 = dat['FLUX'][pts]
HistData = np.histogram(flux1)
plt.hist(HistData)
plt.hist(flux1)
plt.hist(HistData)
dat['TIME'][pts]
time1 = dat['TIME'][pts]
time1
flux1
plt.plot(time1, flux1)
plt.show()
HistData = np.histogram(flux1)
HistData
plt.hist(flux1, bins = time1)
plt.hist(flux1, bins = 'auto')
plt.savefig('plot_from_slice_1'.format(1))
pts = (-0.46812330415978130036517942754554542301650447803823163392 < dat['TIME'] < -0.43624660831956260073035885509109084603300895607646326784)
pts = (dat['TIME'] > -0.46812330415978130036517942754554542301650447803823163392) & (dat['TIME'] < -0.43624660831956260073035885509109084603300895607646326784)
pts
dat['TIME'][pts]
flux2 = dat['FLUX'][pts]
plt.hist(flux2, bins = 'auto')
plt.hist(flux2, bins = 'auto')
plt.savefig('plot_from_slice_2'.format(2))
pts = (dat['TIME'] > 0.43624660831956260073035885509109084603300895607646326784) & (dat['TIME'] < -0.40436991247934390109553828263663626904951343411469490176)
flux3 = dat['FLUX'][pts]
plt.hist(flux3, bins = 'auto')
flux3
pts
dat['TIME'][pts]
dat
pts
dat = Table.read('Kep1520folded_3.fits')
dat
pts = (dat['TIME'] > -0.43624660831956260073035885509109084603300895607646326784) & (dat['TIME'] < -0.40436991247934390109553828263663626904951343411469490176) 
pts
dat['TIME'][pts]
flux3 = dat['FLUX'][pts]
flux3
plt.hist(flux3, bins = 'auto')
plt.hist(flux3, bins = 'auto')
plt.savefig('plot_from_slice_3'.format(3))
pts = (dat['TIME'] > -0.40436991247934390109553828263663626904951343411469490176) & (dat['TIME'] < -0.37249321663912520146071771018218169206601791215292653568)
pts
dat['TIME'][pts]
flux4 = dat['FLUX'][pts]
flux4
plt.hist(flux4, bins = 'auto')
plt.savefig('plot_from_slice_4'.format(4))
plt.savefig('plot_from_slice_{}.pdf'.format(4))
plt.hist(flux4, bins = 'auto')
plt.savefig('plot_from_slice_{}.pdf'.format(4))
pts = (dat['TIME'] > -0.37249321663912520146071771018218169206601791215292653568) & (dat['TIME'] > -0.34061652079890650182589713772772711508252239019115816960)
dat['TIME'][pts]
pts = (dat['TIME'] > -0.37249321663912520146071771018218169206601791215292653568) & (dat['TIME'] < -0.34061652079890650182589713772772711508252239019115816960)
dat['TIME'][pts]
flux5 = dat['FLUX'][pts]
plt.hist(flux5, bins = 'auto')
plt.hist(flux5, bins = 'auto')
plt.savefig('plot_from_slice_{}.pdf'.format(5))
counts, bin_edges, junk = plt.hist(flux5, bins = 'auto')
counts
np.arange(0.98, 1.05, 0.0005)
counts.shape
bin_edges.shape
np.arange(0.98025, 1.04975, .0005)
np.arange(0.98, 1.05, 0.0005).shape
np.arange(0.98025, 1.04975, .0005).shape
np.arange(0.98025, 1.05025, .0005)
np.arange(0.98025, 1.05025, .0005).shape
bin_mid = np.arange(0.98025, 1.05025, .0005)
nBin = bin_mid.shape[0]
nBin
nSlices = 32
bigTable = np.zeroes([nSlices, nBin])
bigTable = np.zeros([nSlices, nBin])
bigTable[0,:]
bigTable[0,:].shape
histResults = {}
histResults['table of Histograms'] = bigTable
histResults['bin middles'] = bin_mid
bin_edges = np.arange(0.98, 1.05, 0.0005)
histResults['bin edges'] = bin_edges
histResults['slice number'] = np.arange(32) + 1
histResults
np.savez(open('hist_results.npz', 'w'), histResults)
np.savez('hist_results.npz', 'w', histResults)
np.savez('hist_results.npz', histResults)
