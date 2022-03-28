# coding: utf-8
import lightkurve as lk
import matplotlib.pyplot as plt

def lightcurve_1520():
    target = "Kepler-1520"
    search_result = lk.search_lightcurve(target, mission='Kepler', exptime = 'long')
    collection = search_result.download_all()
    lcKep1520 = collection.stitch(corrector_func=lambda x: x.normalize().flatten(window_length = 101))
    lcKep1520.to_fits('Kep1520curve.fits', overwrite = True)


def lightcurve_22():
    target = "K2-22"
    search_result = lk.search_lightcurve(target, mission='K2', campaign=1)
    collection = search_result.download()
    lc_trunc = collection.truncate(before = 1979.5, after = 2057)
    #lc_stitch = collection.stitch()
    lc_norm = lc_trunc.normalize()
    lc_flat = lc_norm.flatten(window_length = 101)
    K222 = lc_flat#.remove_nans()
    K222.to_fits('K2d22curve.fits', overwrite = True)
    #1979.5, 2057
