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
import os
plt.close('all')

def script():
    
    '''LIGHTCURVE_DOWNLOAD.PY'''
    
    import lightcurve_download
    
    lightcurve_download.lightcurve_22()
    lightcurve_download.lightcurve_1520()
    
    '''PERIODOGRAM_BIN.PY'''
    try:
        os.mkdir('Kep1520plots')
    except OSError as error:
        print(error)
    try:
        os.mkdir('K2d22plots')
    except OSError as error:
        print(error)
    
    import periodogram_bin
    
    periodogram_bin.recoverplanet()
    periodogram_bin.k2d22()
    
    
    '''HISTOGRAM_COMPILER.PY'''
    
    import Histogram_Compiler
    
    Histogram_Compiler.compiler(planet = 'Kep1520')
    Histogram_Compiler.compiler(planet = 'K2d22')
    
    
    '''FLUXSAVER.PY'''
    try:
        os.mkdir('FluxFiles')
    except OSError as error:
        print(error)
    
    import FluxSaver
    
    FluxSaver.slice_flux(planet = 'Kep1520')
    FluxSaver.slice_flux(planet = 'K2d22')
    
    FluxSaver.out_flux(planet = 'Kep1520')
    FluxSaver.out_flux(planet = 'K2d22')
    
    
    '''CURVE_FITTER.PY
    
    import Curve_Fitter

    Curve_Fitter.fitting(planet = 'Kep1520')
    Curve_Fitter.fitting(planet = 'K2d22')'''
    
    
    '''TRACE_MAKE.PY'''
    try:
        os.mkdir('Kep1520plots/norm_trace')
    except OSError as error:
        print(error)
    try:
        os.mkdir('K2d22plots/norm_trace')
    except OSError as error:
        print(error)
    
    import trace_make
    
    trace_make.rapido(planet = 'Kep1520')
    trace_make.rapido(planet = 'K2d22')
    
    trace_make.single_slice(planet = 'Kep1520', slice_num = 'FullOut')
    trace_make.single_slice(planet = 'K2d22', slice_num = 'FullOut')
    
    trace_make.hist_maker('Kep1520')
    trace_make.hist_maker('K2d22')
