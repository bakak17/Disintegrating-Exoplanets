# coding: utf-8
import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np
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
    plt.savefig("{}plots2/folded_binned.pdf".format(filename))
    folded_binned.to_fits('{}foldedbinned.fits'.format(filename), overwrite=True)

def k2d22():
    recoverplanet('K2d22', .38, .39, 2456811.1208)
