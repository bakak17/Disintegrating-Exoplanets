# coding: utf-8
from astropy.table import Table
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
file = 'Kep1520folded_3.fits'
t = -0.5
dt = 0.03187669
binsize = 0.0005
bin_edges = np.arange(0.98, 1.05, binsize)
bin_mid = np.arange(0.98025, 1.05025, binsize)
nBin = bin_mid.shape[0]
nSlices = 32
bigTable = np.zeros([nSlices, nBin])
i = 0
dat = Table.read(file)
while i < nSlices:
    pts = (dat['TIME'] > t) & (dat['TIME'] < (t + dt))
    flux = dat['FLUX'][pts]
    counts, junk1, junk2 = plt.hist(flux, bins = bin_edges)
    bigTable[i,:] = counts
    i += 1
    t += dt

HDUList = fits.PrimaryHDU(bigTable)
primHDU = fits.PrimaryHDU(bigTable)
bin_midHDU = fits.ImageHDU(bin_mid)
bin_midHDU.name = 'Bin Middles'
sliceHDU = fits.ImageHDU(np.arange(32)+1)
sliceHDU.name = 'Slice Numbers'
HDUList = fits.HDUList([primHDU,bin_midHDU,sliceHDU])
HDUList.writeto('my_results.fits',overwrite=True)
