##THIS DOCUMENT IS PRACTICE/SETUP FOR ASSEMBLING LIGHTCURVE HISTOGRAMS AUTOMATICALLY##

from astropy.table import Table
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

##VARIABLES. LOTS OF VARIABLES##
file = 'Kep1520folded_3.fits'
t = -0.5
dt = 0.03187669
binsize = 0.0005
bin_edges = np.arange(0.98, 1.05, binsize)
bin_mid = np.arange(0.98025, 1.05025, binsize)
slice_phase = np.arange((t+0.5*dt), (-t+0.5*dt), dt)
nBin = bin_mid.shape[0]
nSlices = 32
bigTable = np.zeros([nSlices, nBin])
i = 0
##########

##Potential automated loop for compiling
dat = Table.read(file)
while i < nSlices:
    pts = (dat['TIME'] > t) & (dat['TIME'] < (t + dt))
    flux = dat['FLUX'][pts]
    counts, junk1, junk2 = plt.hist(flux, bins = bin_edges)
    bigTable[i,:] = counts
    i += 1
    t += dt
#Git test

##New Compiling Method?
HDUList = fits.PrimaryHDU(bigTable)
primHDU = fits.PrimaryHDU(bigTable)
bin_midHDU = fits.ImageHDU(bin_mid)
bin_midHDU.name = 'Bin Middles'
sliceHDU = fits.ImageHDU(np.arange(32)+1)
sliceHDU.name = 'Slice Numbers'
phaseHDU = fits.ImageHDU(slice_phase)
phaseHDU.name = 'TIME'
HDUList = fits.HDUList([primHDU,bin_midHDU,sliceHDU,phaseHDU])
HDUList.writeto('total_hist.fits',overwrite=True)


