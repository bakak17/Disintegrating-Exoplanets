from astropy.table import Table
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
plt.close('all')

def slice_flux(planet, binsize=0.0005, scale=1.0, shift=0):
    t = -0.5
    if planet == 'Kep1520':
        dt = 0.03187669/(float(scale))
        dat = Table.read("{}folded.fits".format(planet))
    elif planet == 'K2d22':
        dt = 0.05466947/(float(scale))
        dat = Table.read("{}folded.fits".format(planet))
    nSlices = (int(1 / dt) + 1)
    bin_edges = np.arange(0.98, 1.05, binsize)
    bin_mid = np.arange((0.98 + (binsize*0.5)), (1.05), binsize)
    nBin = bin_mid.shape[0]
    slide = shift/nBin
    slice_phase = np.arange(((t+0.5*dt)-slide), ((-t+0.5*dt) - slide), dt)
    bigTable = np.zeros([nSlices, nBin])
    i = 0

    while i < nSlices:
        pts = (dat['TIME'] > t) & (dat['TIME'] < (t + dt))
        flux = dat['FLUX'][pts]
        error = dat['FLUX_ERR'][pts]
        i += 1
        t += dt

        hdr = fits.Header()
        hdr['Slice'] = i
        empty_primary = fits.PrimaryHDU(header=hdr)
        FluxHDU = fits.ImageHDU(flux)
        FluxHDU.name = 'Flux'
        ErrorHDU = fits.ImageHDU(error)
        ErrorHDU.name = 'Error'
        HDUList = fits.HDUList([empty_primary, FluxHDU, ErrorHDU])
        flnm = 'FluxFiles/{plnt}_slice{{nmbr}}.fits'.format(plnt = planet)
        filename= flnm.format(nmbr = i)
        HDUList.writeto(filename, overwrite = True)

def out_flux(planet, binsize=0.0005, scale=1.0, shift=0):
    t = -0.5
    if planet == 'Kep1520':
        dt = 0.03187669/(float(scale))
        dat = Table.read("{}folded.fits".format(planet))
    elif planet == 'K2d22':
        dt = 0.05466947/(float(scale))
        dat = Table.read("{}folded.fits".format(planet))
    nSlices = (int(1 / dt) + 1)
    bin_edges = np.arange(0.98, 1.05, binsize)
    bin_mid = np.arange((0.98 + (binsize*0.5)), (1.05), binsize)
    nBin = bin_mid.shape[0]
    slide = shift/nBin
    slice_phase = np.arange(((t+0.5*dt)-slide), ((-t+0.5*dt) - slide), dt)
    bigTable = np.zeros([nSlices, nBin])
    i = 0

    pts = (dat['TIME'] > t) & (dat['TIME'] < (t + dt))
    flux = dat['FLUX'][pts]
    error = dat['FLUX_ERR'][pts]
    i += 1
    t += dt

    while i < nSlices:
        if planet == 'Kep1520':
            if ((t + dt) < -0.128) or (t > 0.2):
                pts = (dat['TIME'] > t) & (dat['TIME'] < (t + dt))
                flux = np.append(flux, dat['FLUX'][pts])
                error = np.append(error, dat['FLUX_ERR'][pts])
        elif planet == 'K2d22':
            if ((t + dt) < -0.2) or (t > 0.22):
                pts = (dat['TIME'] > t) & (dat['TIME'] < (t + dt))
                flux = np.append(flux, dat['FLUX'][pts])
                error = np.append(error, dat['FLUX_ERR'][pts])
        i += 1
        t += dt

    hdr = fits.Header()
    hdr['Info'] = 'All the out of transit slices for the planet'
    empty_primary = fits.PrimaryHDU(header=hdr)
    FluxHDU = fits.ImageHDU(flux)
    FluxHDU.name = 'Flux'
    ErrorHDU = fits.ImageHDU(error)
    ErrorHDU.name = 'Error'
    HDUList = fits.HDUList([empty_primary, FluxHDU, ErrorHDU])
    filename = 'FluxFiles/{}_sliceFullOut.fits'.format(planet)
    HDUList.writeto(filename, overwrite = True)
