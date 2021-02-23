import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import pymc3 as pm
import arviz as az
plt.close('all')

def rapido(planet):
    t = -0.5
    if planet == 'Kep1520':
        dt = 0.03187669
    elif planet == 'K2d22':
        dt = 0.05466947
    nSlices = (int(1 / dt) + 1)
    i = 0

    while i < nSlices:
        i += 1
        flnm = 'FluxFiles/{plnt}_slice{{nmbr}}.fits'.format(plnt = planet)
        filename= flnm.format(nmbr = i)
        HDUList = fits.open(filename)
        flux = HDUList[1].data
        error = HDUList[2].data
        with pm.Model() as slice_model:
            sigma = pm.Normal('sigma', 0.00065, 0.0026)
            mu = pm.Normal('mu', 1, 0.00065)
            slice_mdl = pm.Normal('model', sigma = sigma, mu = mu, observed = flux)
            trace1 = pm.sample(1000, random_seed = 123)
        
            direc = '/Users/keithbaka/Documents/0Research_Schlawin/Traces/{plnt}/Slice{{nmbr}}'.format(plnt = planet)
            direct = direc.format(nmbr = i)
            try:
                pm.save_trace(trace1, directory = direct, overwrite = True)
            except IOError:
                os.mkdir(direct)
            pm.save_trace(trace1, directory = direct, overwrite = True)

def single_slice(planet, slice_num, load = 0):
    flnm = 'FluxFiles/{plnt}_slice{{nmbr}}.fits'.format(plnt = planet)
    filename= flnm.format(nmbr = slice_num)
    HDUList = fits.open(filename)
    flux = HDUList[1].data
    error = HDUList[2].data
    with pm.Model() as slice_model:
        sigma = pm.Normal('sigma', 0.00065, 0.0026)
        if slice_num == 'FullOut':
            mu = pm.Normal('mu', 1, 0.0065)
        else:
            mu = pm.Normal('mu', 1, 0.00065)
        slice_mdl = pm.Normal('model', sigma = sigma, mu = mu, observed = flux)
        
        direc = '/Users/keithbaka/Documents/0Research_Schlawin/Traces/{plnt}/Slice{{nmbr}}'.format(plnt = planet)
        direct = direc.format(nmbr = slice_num)
        if load == 0:
            trace1 = pm.sample(1000, random_seed = 123)
            try:
                pm.save_trace(trace1, directory = direct, overwrite = True)
            except IOError:
                os.mkdir(direct)
                pm.save_trace(trace1, directory = direct, overwrite = True)
        elif load == 1:
            trace1 = pm.load_trace(directory = direct)
            return trace1
        
def load(planet, slice_num): #no longer needed
    flnm = flnm = 'FluxFiles/{plnt}_slice{{nmbr}}.fits'.format(plnt = planet)
    filename= flnm.format(nmbr = slice_num)
    HDUList = fits.open(filename)
    flux = HDUList[1].data
    error = HDUList[2].data
    with pm.Model() as slice_model:
        sigma = pm.Normal('sigma', 0.00065, 0.0026)
        mu = pm.Normal('mu', 1, 0.00065)
        slice_mdl = pm.Normal('model', sigma = sigma, mu = mu, observed = flux)
        
        direc = '/Users/keithbaka/Documents/0Research_Schlawin/Traces/{plnt}/Slice{{nmbr}}'.format(plnt = planet)
        direct = direc.format(nmbr = slice_num)
        trace1 = pm.load_trace(directory = direct)
        return trace1

def hist_maker(planet):
    if planet == 'Kep1520':
        j = 33
    if planet == 'K2d22':
        j = 20
    i = 1
    while i < j:
        trace1 = single_slice(planet = planet, slice_num = i, load = 1)
        plt.hist(trace1['sigma'])
        plt.savefig('{}plots/0trace_results/slice{:02d}_sigma.pdf'.format(planet, i))
        plt.close()
        plt.hist(trace1['mu'])
        plt.savefig('{}plots/0trace_results/slice{:02d}_mu.pdf'.format(planet, i))
        plt.close()
        i += 1
    trace1 = single_slice(planet = planet, slice_num = 'FullOut', load = 1)
    plt.hist(trace1['sigma'])
    plt.savefig('{}plots/0trace_results/slice{}_sigma.pdf'.format(planet, 'FullOut'))
    plt.close()
    plt.hist(trace1['mu'])
    plt.savefig('{}plots/0trace_results/slice{}_mu.pdf'.format(planet, 'FullOut'))
    plt.close()

