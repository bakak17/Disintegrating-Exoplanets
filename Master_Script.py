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

def script(scale = 1):
    
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
    ## HAVE TO CONVERT TIME TO PHASE FOR VARIABLE "pts"
    
    import Histogram_Compiler
    
    Histogram_Compiler.compiler(planet = 'Kep1520', scale = scale)
    Histogram_Compiler.compiler(planet = 'K2d22', scale = scale)
    
    
    '''FLUXSAVER.PY'''
    ## HAVE TO CONVERT TIME TO PHASE FOR VARIABLE "pts"
    try:
        os.mkdir('FluxFiles')
    except OSError as error:
        print(error)
    
    import FluxSaver
    ##Regular dips in flux in folded.fits, cadence of Kepler, could alter results
    
    FluxSaver.slice_flux(planet = 'Kep1520', scale = scale)
    FluxSaver.slice_flux(planet = 'K2d22', scale = scale)
    
    FluxSaver.out_flux(planet = 'Kep1520', scale = scale)
    FluxSaver.out_flux(planet = 'K2d22', scale = scale)
    
    
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

    ##May have to edit "i"s and "j"s so each is increased by one to keep notation
    ##and filenames consistent
    
    trace_make.rapido(planet = 'Kep1520', scale = scale)
    trace_make.rapido(planet = 'K2d22', scale = scale)
    
    trace_make.single_slice(planet = 'Kep1520', slice_num = 'FullOut')
    trace_make.single_slice(planet = 'K2d22', slice_num = 'FullOut')
    
    #trace_make.hist_maker('Kep1520', scale = scale)
    #trace_make.hist_maker('K2d22', scale = scale)

    '''JOINT_FUNCTION_TRACE'''
    
    import joint_function_trace

    joint_function_trace.joint_trace('Kep1520', scale = scale)
    joint_function_trace.joint_trace('K2d22', scale = scale)

    joint_function_trace.single_slice('Kep1520', 'FullOut')
    joint_function_trace.single_slice('K2d22', 'FullOut')

    '''PLOTTING'''
    
    import violin_maker

    ##Have to make an OS.Mkdir for violin_plots

    violin_maker.chunk_violin('Kep1520', scale = scale)
    violin_maker.chunk_violin('K2d22', scale = scale)

    violin_maker.median_pull('Kep1520', scale = scale)
    violin_maker.median_pull('K2d22', scale = scale)

    
