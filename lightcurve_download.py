# coding: utf-8
import lightkurve as lk
import matplotlib.pyplot as plt
def lightcurve_22():
    import lightkurve
    target = "K2-22"
    K222 = lk.search_lightcurve(target, mission='K2', campaign=1).download().PDCSAP_FLUX.normalize().flatten(window_length=101)
    K222.plot()
    K222.to_fits('K2d22curve.fits', overwrite = True)


def lightcurve_1520():
    import lightkurve
    target = "Kepler-1520"
    lcKep1520 = lk.search_lightcurve(target, quarter=1).download().PDCSAP_FLUX.normalize().flatten(window_length=101)
    for q in range(2, 18):
        lcKep1520 = lcKep1520.append(lk.search_lightcurve(target, quarter=q).download().PDCSAP_FLUX.normalize().flatten(window_length=101))
    lcKep1520.plot()
    lcKep1520.to_fits('Kep1520curve.fits', overwrite = True)
