# coding: utf-8
import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
plt.close('all')
def recoverplanet(filename='Kep1520', periodstart=0.64, periodend=0.66, t0 = 2454968.982):
    """tpf = lk.search_targetpixelfile("Kepler-1520", quarter = q).download()
    tpf.plot(frame=100, scale='log', show_colorbar=True)
    lc = tpf.to_lightcurve(aperture_mask=tpf.pipeline_mask)
    lc.plot()
    plt.show()
    flat, trend = lc.flatten(window_length=301, return_trend=True)"""
    lcf = lk.lightcurvefile.KeplerLightCurveFile("{}curve.fits".format(filename))
    #lcf.plot()
    lcf_clean = lcf.remove_outliers(sigma_lower=100, sigma_upper=3)
    flat = lcf_clean
    flat.plot()
    plt.savefig("{}plots/clean.pdf".format(filename), bbox_inches='tight', overwrite = True)
    #ax = lc.errorbar(label="Kepler-1520")
    #trend.plot(ax=ax, color='red', lw=2, label='Trend')
    #flat = flat.remove_outliers(sigma_lower=1000, sigma_upper=5)
    #flat.errorbar()
    periodogram = flat.to_periodogram(method="bls", period=np.arange(periodstart, periodend, 0.00001))
    periodogram.plot()
    plt.savefig("{}plots/periodogram.pdf".format(filename), bbox_inches='tight', overwrite = True)
    #plt.show()
    #plt.close('all')
    if filename == 'K2d22':
        best_fit_period = 0.381078
    else:
        #best_fit_period = periodogram.period_at_max_power
        best_fit_period = 0.6535538
    print('Best fit period: {:.9f}'.format(best_fit_period))
    folded = flat.fold(period=best_fit_period, epoch_time = (t0 - 2454833.0))#.errorbar()
    folded.plot()
    plt.savefig("{}plots/folded.pdf".format(filename),  bbox_inches='tight',overwrite = True)
    #t0=(t0 - 2454833.0)
    folded_binned = folded.bin(time_bin_size = (10. * u.min))
    #foldedlc = folded.to_lightcurve()
    #folded_binned = foldedlc.bin(100, 'mean')
    #plt.close('all')
    #folded.errorbar().plot()
    folded.to_fits('{}folded.fits'.format(filename), overwrite=True)
    #plt.close('all')
    folded_binned.plot()
    #plt.show()
    plt.savefig("{}plots/folded_binned.pdf".format(filename), bbox_inches='tight', overwrite = True)
    plt.close('all')
    folded_binned.to_fits('{}foldedbinned.fits'.format(filename), overwrite=True)

def k2d22():
    recoverplanet('K2d22', .38, .39, 2456811.1208)
