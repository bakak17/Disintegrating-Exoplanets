# coding: utf-8
from lightkurve import search_lightcurvefile
import matplotlib.pyplot as plt
def lightcurve_1520(lcKep1520):
    import lightkurve
    target = "Kepler-1520"
    lcKep1520 = search_lightcurvefile(target, quarter=1).download().PDCSAP_FLUX.normalize().flatten(window_length=101)
    for q in range(2, 18):
        lcKep1520 = lcKep1520.append(search_lightcurvefile(target, quarter=q).download().PDCSAP_FLUX.normalize().flatten(window_length=101))

    
    lcKep1520.plot()
    lcKep1520.to_fits('Kep1520curve.fits')
    #Kep1520lc2 = lightkurve.lightcurvefile.KeplerLightCurveFile('Kep1520curve.fits')
    #return Kep1520lc2
