##THIS DOCUMENT ASSEMBLES LIGHTCURVE HISTOGRAMS AUTOMATICALLY##

from astropy.table import Table
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import planet_params
plt.close('all')

##VARIABLES. LOTS OF VARIABLES##
def compiler(planet, binsize=0.0005, scale=1.0, shift=0):
    period = planet_params.period(planet)
    t = -0.5
    if planet == 'Kep1520':
        dt = 0.03187669/(float(scale))
        dat = Table.read("{}folded.fits".format(planet))
        phase = dat['TIME'] / 0.653553800
    elif planet == 'K2d22':
        dt = 0.05466947/(float(scale))
        dat = Table.read("{}folded.fits".format(planet))
        phase = dat['TIME'] / 0.381078
    nSlices = (int(1 / dt) + 1)
    bin_edges = np.arange(0.98, 1.05, binsize)
    bin_mid = np.arange((0.98 + (binsize*0.5)), (1.05), binsize)
    nBin = bin_mid.shape[0]
    slide = shift/nBin
    slice_phase = np.arange(((t+0.5*dt)-slide), ((-t+0.5*dt) - slide), dt)
    #print(slice_phase)
    bigTable = np.zeros([nSlices, nBin])
    i = 0
    ##########

    ##Potential automated loop for compiling
    
    while i < nSlices:
        pts = (phase > t) & (phase < (t + dt))
        flux = dat['FLUX'][pts]
        counts, junk1, junk2 = plt.hist(flux, bins = bin_edges)
        bigTable[i,:] = counts
        i += 1
        t += dt

    #Normalizing
    for i,row in enumerate(bigTable):
        bigTable[i] = row / (np.sum(row) * binsize)

    ##New Compiling Method?
    primHDU = fits.PrimaryHDU(bigTable)
    bin_midHDU = fits.ImageHDU(bin_mid)
    bin_midHDU.name = 'Bin Middles'
    sliceHDU = fits.ImageHDU(np.arange(nSlices)+1)
    sliceHDU.name = 'Slice Numbers'
    phaseHDU = fits.ImageHDU(slice_phase)
    phaseHDU.name = 'TIME'
    HDUList = fits.HDUList([primHDU,bin_midHDU,sliceHDU,phaseHDU])
    HDUList.writeto("{}total_hist.fits".format(planet),overwrite=True)
    plt.close('all')
