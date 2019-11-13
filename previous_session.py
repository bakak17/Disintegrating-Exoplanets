# coding: utf-8
import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np
def recoverplanet():
    tpf = lk.search_targetpixelfile("Kepler-1520", quarter = 3).download()
    tpf.plot(frame=100, scale='log', show_colorbar=True)
    lc = tpf.to_lightcurve(aperture_mask=tpf.pipeline_mask)
    lc.plot()
    plt.show()
    flat, trend = lc.flatten(window_length=301, return_trend=True)
    ax = lc.errorbar(label="Kepler-10")
    ax = lc.errorbar(label="Kepler-1520")
    trend.plot(ax=ax, color='red', lw=2, label='Trend')
    plt.show()
    flat.errorbar(label="Kepler-1520")
    plt.show()
    periodogram = flat.to_periodogram(method="bls", period=np.arange(0.3, 1.5, 0.001))
    periodogram.plot()
    plt.show()
    best_fit_period = periodogram.period_at_max_power
    print('Best fit period: {:.3f}'.format(best_fit_period))
    flat.fold(period=best_fit_period, t0=periodogram.transit_time_at_max_power).errorbar()
    plt.show()
