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

def script(scale = 1, shift = 0):
    
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
    
    
    '''FLUXSAVER.PY'''
    ## HAVE TO CONVERT TIME TO PHASE FOR VARIABLE "pts"
    try:
        os.mkdir('FluxFiles')
    except OSError as error:
        print(error)
    
    import FluxSaver
    ##Regular dips in flux in folded.fits, cadence of Kepler, could alter results
    
    FluxSaver.slice_flux(planet = 'Kep1520', scale = scale, shift = shift)
    FluxSaver.slice_flux(planet = 'K2d22', scale = scale, shift = shift)
    
    FluxSaver.out_flux(planet = 'Kep1520', scale = scale, shift = shift)
    FluxSaver.out_flux(planet = 'K2d22', scale = scale, shift = shift)
    
    '''HISTOGRAM_COMPILER.PY'''
    ## HAVE TO CONVERT TIME TO PHASE FOR VARIABLE "pts"
    
    import Histogram_Compiler
    
    Histogram_Compiler.compiler(planet = 'Kep1520', scale = scale, shift = shift)
    Histogram_Compiler.compiler(planet = 'K2d22', scale = scale, shift = shift)

    
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
    
    trace_make.single_slice(planet = 'Kep1520', slice_num = 'FullOut', scale = scale)
    trace_make.single_slice(planet = 'K2d22', slice_num = 'FullOut', scale = scale)
    
    #trace_make.hist_maker('Kep1520', scale = scale)
    #trace_make.hist_maker('K2d22', scale = scale)

    '''JOINT_FUNCTION_TRACE'''
    
    import joint_function_trace

    joint_function_trace.joint_trace('Kep1520', scale = scale)
    joint_function_trace.joint_trace('K2d22', scale = scale)

    joint_function_trace.single_slice('Kep1520', 'FullOut', scale = scale)
    joint_function_trace.single_slice('K2d22', 'FullOut', scale = scale)

    '''PLOTTING'''
    
    import violin_maker

    try:
        os.mkdir('Kep1520plots/violin_plots')
    except OSError as error:
        print(error)
    try:
        os.mkdir('K2d22plots/violin_plots')
    except OSError as error:
        print(error)

    ##Have to make an OS.Mkdir for violin_plots

    violin_maker.chunk_violin('Kep1520', scale = scale)
    violin_maker.chunk_violin('K2d22', scale = scale)

    violin_maker.median_pull('Kep1520', scale = scale)
    violin_maker.median_pull('K2d22', scale = scale)

    
def script2(planet = 'Kep1520', scale = 1, shift = 1):
    
    import lightcurve_download
    import periodogram_bin
    if planet == 'Kep1520':
        lightcurve_download.lightcurve_1520()
        periodogram_bin.recoverplanet()
    elif planet == 'K2d22':
        lightcurve_download.lightcurve_22()
        periodogram_bin.k2d22()
    else:
        print('ur bad, kid')
        exit()
    
    import FluxSaver
    FluxSaver.slice_flux(planet = planet, scale = scale, shift = shift)
    FluxSaver.out_flux(planet = planet, scale = scale, shift = shift)
    
    import Histogram_Compiler
    Histogram_Compiler.compiler(planet = planet, scale = scale, shift = shift)

    import trace_make
    trace_make.rapido(planet = planet, scale = scale)
    trace_make.single_slice(planet = planet, slice_num = 'FullOut', scale = scale)

    import joint_function_trace
    joint_function_trace.joint_trace(planet, scale = scale)
    joint_function_trace.single_slice(planet, 'FullOut', scale = scale)

    import violin_maker
    violin_maker.chunk_violin(planet, scale = scale)
    violin_maker.median_pull(planet, scale = scale)
