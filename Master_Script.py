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
