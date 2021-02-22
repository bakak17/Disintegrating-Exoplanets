from lightkurve import search_lightcurvefile
import matplotlib.pyplot as plt
import lightkurve as lk
import numpy as np
from astropy.table import Table
from astropy.io import fits
import pdb
from scipy.optimize import curve_fit
import probability_funcs
import pymc3 as pm
import arviz as az
plt.close('all')

'''LIGHTCURVE_DOWNLOAD.PY'''

def lightcurve_22():
    import lightkurve
    target = "K2-22"
    K222 = search_lightcurvefile(target, mission='K2', campaign=1).download().PDCSAP_FLUX.normalize().flatten(window_length=101)
    K222.plot()
    K222.to_fits('K2d22curve.fits')


def lightcurve_1520():
    import lightkurve
    target = "Kepler-1520"
    lcKep1520 = search_lightcurvefile(target, quarter=1).download().PDCSAP_FLUX.normalize().flatten(window_length=101)
    for q in range(2, 18):
        lcKep1520 = lcKep1520.append(search_lightcurvefile(target, quarter=q).download().PDCSAP_FLUX.normalize().flatten(window_length=101))
    lcKep1520.plot()
    lcKep1520.to_fits('Kep1520curve.fits')

lightcurve_22()
lightcurve_1520()

plt.close('all')


'''PERIODOGRAM_BIN.PY'''

def recoverplanet(filename='Kep1520', periodstart=0.64, periodend=0.66, t0 = 2454968.982):
    """tpf = lk.search_targetpixelfile("Kepler-1520", quarter = q).download()
    tpf.plot(frame=100, scale='log', show_colorbar=True)
    lc = tpf.to_lightcurve(aperture_mask=tpf.pipeline_mask)
    lc.plot()
    plt.show()
    flat, trend = lc.flatten(window_length=301, return_trend=True)"""
    lcf = lk.lightcurvefile.KeplerLightCurveFile("{}curve.fits".format(filename))
    flat = lcf.get_lightcurve('FLUX')
    #ax = lc.errorbar(label="Kepler-1520")
    #trend.plot(ax=ax, color='red', lw=2, label='Trend')
    plt.show()
    flat.errorbar()
    flat = flat.remove_outliers(sigma_lower=1000, sigma_upper=5)
    flat.errorbar()
    plt.show()
    periodogram = flat.to_periodogram(method="bls", period=np.arange(periodstart, periodend, 0.00001))
    periodogram.plot()
    plt.show()
    if filename == 'K2d22':
        best_fit_period = 0.381078
    else:
        best_fit_period = periodogram.period_at_max_power
    print('Best fit period: {:.9f}'.format(best_fit_period))
    folded = flat.fold(period=best_fit_period, t0=(t0 - 2454833.0))#.errorbar()
    folded_binned = folded.bin(100, 'mean')
    #foldedlc = folded.to_lightcurve()
    #folded_binned = foldedlc.bin(100, 'mean')
    folded.errorbar().plot()
    plt.show()
    folded.to_fits('{}folded.fits'.format(filename), overwrite=True)
    folded_binned.plot()
    plt.show()
    plt.savefig("{}plots/folded_binned.pdf".format(filename))
    folded_binned.to_fits('{}foldedbinned.fits'.format(filename), overwrite=True)

def k2d22():
    recoverplanet('K2d22', .38, .39, 2456811.1208)

recoverplanet()
k2d22()

plt.close('all')


'''HISTOGRAM_COMPILER.PY'''

def compiler(planet, binsize=0.0005, scale=1.0, shift=0):
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
    ##########

    ##Potential automated loop for compiling
    
    while i < nSlices:
        pts = (dat['TIME'] > t) & (dat['TIME'] < (t + dt))
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

compiler(planet = 'Kep1520')
compiler(planet = 'K2d22')


'''FLUXSAVER.PY'''
os.mkdir('FluxFiles')

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

slice_flux(planet = 'Kep1520')
slice_flux(planet = 'K2d22')

out_flux(planet = 'Kep1520')
out_flux(planet = 'K2d22')

plt.close('all')


'''CURVE_FITTER.PY'''
os.mkdir('Kep1520plots')
os.mkdir('K2d22plots')

def fitting(planet):
    HDUList = fits.open("{}total_hist.fits".format(planet))
    bigTable = HDUList[0].data
    bin_mid = HDUList[1].data
    nSlices = HDUList[2].data
    slices = len(nSlices[0:])
    Time = HDUList[3].data
    out_of_transit = np.zeros_like(bigTable[0])
    time_index = np.arange(slices)
    if planet == 'Kep1520':
        j = 0.0
        for i in time_index:
            if (Time[i] < -0.128) or (Time[i] > 0.2):
                out_of_transit = out_of_transit + bigTable[i]
                j += 1.0
        out_of_transit = out_of_transit/j
        p0 = [0.000717105679, 1]
    elif planet == 'K2d22':
        j = 0.0
        for i in time_index:
            if (Time[i] < -0.2) or (Time[i] > 0.22):
                out_of_transit = out_of_transit + bigTable[i]
                j += 1.0
        out_of_transit = out_of_transit/j
        p0 = [0.000572959250, 1]
    plt.plot(bin_mid, out_of_transit, label = 'Out of Transit')
    
    xdata = bin_mid
    ydata = out_of_transit
    popt1, pcov1 = curve_fit(probability_funcs.gaussian, xdata, ydata, p0=p0)
    print(popt1)
    print(popt1[0])
    probability_funcs.sigma_g = popt1[0]
    plt.plot(xdata, probability_funcs.gaussian(xdata, *popt1), 'r-', label = 'Out of Transit Model')
    plt.xlabel('Flux')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig("{}plots/out_of_transit_avg.pdf".format(planet))
    plt.close()
    ListOfMaxima = []
    ListOfSigmaR = []
    MaximaError = []
    SigmaRError = []
    for i in range(0, slices):
        in_transit = bigTable[i]
        plt.plot(bin_mid, in_transit, label = 'Slice {}'.format(i))
        ydata = in_transit
        popt2, pcov2 = curve_fit(probability_funcs.joint_func, xdata, ydata, p0=[0.0000418, 1.0], bounds=([0.00004, 0.98], [0.07, 1.05]))
        ListOfMaxima.append(popt2[1])
        ListOfSigmaR.append(popt2[0])
        MaximaError.append((pcov2[1,1])**0.5)
        SigmaRError.append((pcov2[0,0])**0.5)
        print(popt2, i)
        plt.plot(xdata, probability_funcs.joint_func(xdata, *popt2), 'g-', linewidth=0.5, label = 'Slice {} Model'.format(i))
        plt.plot(xdata, probability_funcs.raleigh(xdata, sigma = popt2[0], mu = popt2[1]), linewidth=0.5, label = 'Slice {} Raleigh'.format(i))
        #plt.xlim(0.9997, 1.000)
        #plt.ylim(0, 50)
        #plt.rcParams.update({'font.size': 6})
        plt.xlabel("Flux")
        plt.ylabel("Counts")
        plt.tight_layout()
        plt.legend()
        plt.savefig("{}plots/slice_fit_{:03d}.pdf".format(planet, i))
        plt.close()
    plt.errorbar(Time, ListOfMaxima, MaximaError, xerr=None, fmt='r', ecolor='b', elinewidth = 1)
    if planet == 'Kep1520':
        plt.title('Maximum Time Series for Kepler 1520b')
    elif planet == 'K2d22':
        plt.title('Maximum Time Series for K2-22b')
    plt.xlabel('Transit Phase')
    plt.ylabel('Flux')
    plt.tight_layout()
    plt.show()
    plt.savefig("{}plots/Max_time_series.pdf".format(planet))
    plt.close('all')

fitting(planet = 'Kep1520')
fitting(planet = 'K2d22')


'''TRACE_MAKE.PY'''
def rapido(planet):
    t = -0.5
    if planet == 'Kep1520':
        dt = 0.03187669
    elif planet == 'K2d22':
        dt = 0.05466947
    nSlices = (int(1 / dt) + 1)
    i = 0

    while i < nSlices:
        i += 1
        flnm = 'FluxFiles/{plnt}_slice{{nmbr}}.fits'.format(plnt = planet)
        filename= flnm.format(nmbr = i)
        HDUList = fits.open(filename)
        flux = HDUList[1].data
        error = HDUList[2].data
        with pm.Model() as slice_model:
            sigma = pm.Normal('sigma', 0.00065, 0.0026)
            mu = pm.Normal('mu', 1, 0.00065)
            slice_mdl = pm.Normal('model', sigma = sigma, mu = mu, observed = flux)
            trace1 = pm.sample(1000, random_seed = 123)
        
            direc = '/Users/keithbaka/Documents/0Research_Schlawin/Traces/{plnt}/Slice{{nmbr}}'.format(plnt = planet)
            direct = direc.format(nmbr = i)
            try:
                pm.save_trace(trace1, directory = direct, overwrite = True)
            except IOError:
                os.mkdir(direct)
            pm.save_trace(trace1, directory = direct, overwrite = True)

def single_slice(planet, slice_num, load = 0):
    flnm = 'FluxFiles/{plnt}_slice{{nmbr}}.fits'.format(plnt = planet)
    filename= flnm.format(nmbr = slice_num)
    HDUList = fits.open(filename)
    flux = HDUList[1].data
    error = HDUList[2].data
    with pm.Model() as slice_model:
        sigma = pm.Normal('sigma', 0.00065, 0.0026)
        if slice_num == 'FullOut':
            mu = pm.Normal('mu', 1, 0.0065)
        else:
            mu = pm.Normal('mu', 1, 0.00065)
        slice_mdl = pm.Normal('model', sigma = sigma, mu = mu, observed = flux)
        
        direc = '/Users/keithbaka/Documents/0Research_Schlawin/Traces/{plnt}/Slice{{nmbr}}'.format(plnt = planet)
        direct = direc.format(nmbr = slice_num)
        if load == 0:
            trace1 = pm.sample(1000, random_seed = 123)
            try:
                pm.save_trace(trace1, directory = direct, overwrite = True)
            except IOError:
                os.mkdir(direct)
                pm.save_trace(trace1, directory = direct, overwrite = True)
        elif load == 1:
            trace1 = pm.load_trace(directory = direct)
            return trace1
        
def load(planet, slice_num): #no longer needed
    flnm = flnm = 'FluxFiles/{plnt}_slice{{nmbr}}.fits'.format(plnt = planet)
    filename= flnm.format(nmbr = slice_num)
    HDUList = fits.open(filename)
    flux = HDUList[1].data
    error = HDUList[2].data
    with pm.Model() as slice_model:
        sigma = pm.Normal('sigma', 0.00065, 0.0026)
        mu = pm.Normal('mu', 1, 0.00065)
        slice_mdl = pm.Normal('model', sigma = sigma, mu = mu, observed = flux)
        
        direc = '/Users/keithbaka/Documents/0Research_Schlawin/Traces/{plnt}/Slice{{nmbr}}'.format(plnt = planet)
        direct = direc.format(nmbr = slice_num)
        trace1 = pm.load_trace(directory = direct)
        return trace1

rapido(planet = 'Kep1520')
rapido(planet = 'K2d22')

single_slice(planet = 'Kep1520', slice_num = 'FullOut')
single_slice(planet = 'K2d22', slice_num = 'FullOut')

plt.close('all')
