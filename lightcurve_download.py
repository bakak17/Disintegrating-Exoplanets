# coding: utf-8
import lightkurve as lk
import matplotlib.pyplot as plt
def lightcurve_22():
    #import lightkurve
    target = "K2-22"
    search_result = lk.search_lightcurve(target, mission='K2', campaign=1)
    collection = search_result.download()
    print(collection)
    #lc_stitch = collection.stitch()
    lc_norm = collection.normalize()
    lc_flat = lc_norm.flatten(window_length = 101)
    K222 = lc_flat#.remove_nans()
    #.download().PDCSAP_FLUX.normalize().flatten(window_length=101)#.remove_nans()
    #K222.plot()
    K222.to_fits('K2d22curve.fits', overwrite = True)


def lightcurve_1520():
    #import lightkurve
    target = "Kepler-1520"
    search_result = lk.search_lightcurve(target, mission='Kepler', exptime = 'long')#.download().PDCSAP_FLUX.normalize().flatten(window_length=101)
    collection = search_result.download_all()
    print(collection)
    lc_stitch = collection.stitch()
    lc_norm = lc_stitch.normalize()
    lc_flat = lc_norm.flatten(window_length = 101)
    lcKep1520 = lc_flat#.remove_nans()
    '''for q in range(2, 18):
        lcKep1520 = lcKep1520.append(lk.search_lightcurve(target, mission = 'Kepler', quarter=q).download().PDCSAP_FLUX.normalize().flatten(window_length=101))
    #lcKep1520.plot()'''
    lcKep1520.to_fits('Kep1520curve.fits', overwrite = True)
