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
i = plt.close('all')
i

def script():
    i = plt.close('all')
    
    '''LIGHTCURVE_DOWNLOAD.PY'''
    
    import lightcurve_download
    
    lightcurve_download.lightcurve_22()
    lightcurve_download.lightcurve_1520()
    i
    
    
    '''PERIODOGRAM_BIN.PY'''
    
    import periodogram_bin
    
    periodogram_bin.recoverplanet()
    periodogram_bin.k2d22()
    i
    
    
    '''HISTOGRAM_COMPILER.PY'''
    
    import histogram_compiler
    
    histogram_compiler.compiler(planet = 'Kep1520')
    histogram_compiler.compiler(planet = 'K2d22')
    
    
    '''FLUXSAVER.PY'''
    os.mkdir('FluxFiles')
    
    import FluxSaver
    
    FluxSaver.slice_flux(planet = 'Kep1520')
    FluxSaver.slice_flux(planet = 'K2d22')
    i
    
    FluxSaver.out_flux(planet = 'Kep1520')
    FluxSaver.out_flux(planet = 'K2d22')
    i
    
    
    '''CURVE_FITTER.PY'''
    os.mkdir('Kep1520plots')
    os.mkdir('K2d22plots')
    
    import curve_fitter
    
    curve_fitter.fitting(planet = 'Kep1520')
    curve_fitter.fitting(planet = 'K2d22')
    
    
    '''TRACE_MAKE.PY'''
    os.mkdir('Kep1520plots/0trace_results')
    os.mkdir('K2d22plots/0trace_results')
    
    import trace_make
    
    trace_make.rapido(planet = 'Kep1520')
    trace_make.rapido(planet = 'K2d22')
    i
    
    trace_make.single_slice(planet = 'Kep1520', slice_num = 'FullOut')
    trace_make.single_slice(planet = 'K2d22', slice_num = 'FullOut')
    i
    
    trace_make.hist_maker('Kep1520')
    trace_make.hist_maker('K2d22')
    i
