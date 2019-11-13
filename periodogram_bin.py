# coding: utf-8
import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np
def recoverplanet():
    """tpf = lk.search_targetpixelfile("Kepler-1520", quarter = q).download()
    tpf.plot(frame=100, scale='log', show_colorbar=True)
    lc = tpf.to_lightcurve(aperture_mask=tpf.pipeline_mask)
    lc.plot()
    plt.show()
    flat, trend = lc.flatten(window_length=301, return_trend=True)"""
    lcf = lk.lightcurvefile.KeplerLightCurveFile('Kep1520curve.fits')
    flat = lcf.get_lightcurve('FLUX')
    #ax = lc.errorbar(label="Kepler-1520")
    #trend.plot(ax=ax, color='red', lw=2, label='Trend')
    plt.show()
    flat.errorbar()
    plt.show()
    periodogram = flat.to_periodogram(method="bls", period=np.arange(0.64, 0.66, 0.00001))
    periodogram.plot()
    plt.show()
    best_fit_period = periodogram.period_at_max_power
    print('Best fit period: {:.9f}'.format(best_fit_period))
    folded = flat.fold(period=best_fit_period, t0=periodogram.transit_time_at_max_power)#.errorbar()
    folded_binned = folded.bin(100, 'mean')
    #foldedlc = folded.to_lightcurve()
    #folded_binned = foldedlc.bin(100, 'mean')
    folded.errorbar().plot()
    plt.show()
    #folded.to_fits('Kep1520folded.fits')
    folded_binned.plot()
    plt.show()
