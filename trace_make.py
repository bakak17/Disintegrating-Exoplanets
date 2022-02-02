import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import pymc3 as pm
import pymc3_ext as pmx
import arviz as az
import os
import probability_funcs
from sklearn import preprocessing
import joint_function_trace
import pdb
homedir = os.getcwd()
plt.close('all')



def rapido(planet, scale = 1):
    
    t = -0.5
    if planet == 'Kep1520':
        dt = 0.03187669/(float(scale))
        #This may cause an error!! Causes j = 32 not j = 33
    elif planet == 'K2d22':
        dt = 0.05466947/(float(scale))
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
            sigma = pm.Normal('sigma', 0.00065, 0.00026)
            mu = pm.Normal('mu', 1, 0.00065)
            slice_mdl = pm.Normal('model', sigma = sigma, mu = mu, observed = flux)
            #map_params = pmx.optimize()
            #pdb.set_trace()
            trace1 = pm.sample(1000, random_seed = 123)
        
            direc = homedir + '/Traces/Gauss/{plnt}/Slice{{nmbr}}'.format(plnt = planet)
            direct = direc.format(nmbr = i)
            try:
                pm.save_trace(trace1, directory = direct, overwrite = True)
            except IOError:
                os.mkdir(direct)
            pm.save_trace(trace1, directory = direct, overwrite = True)
            #return map_params

def single_slice(planet, slice_num, load = 0):
    flnm = 'FluxFiles/{plnt}_slice{{nmbr}}.fits'.format(plnt = planet)
    filename= flnm.format(nmbr = slice_num)
    HDUList = fits.open(filename)
    flux = HDUList[1].data
    error = HDUList[2].data
    with pm.Model() as slice_model:

        sigma_g = pm.Normal('sigma_g', 0.00065, 0.026)
        if slice_num == 'FullOut':
            mu_g = pm.Normal('mu_g', 1, 0.065)
        else:
            mu_g = pm.Normal('mu_g', 1, 0.00065)
        slice_mdl = pm.Normal('model', sigma = sigma_g, mu = mu_g, observed = flux)
        direc = homedir + '/Traces/Gauss/{plnt}/Slice{{nmbr}}'.format(plnt = planet)
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
        elif load == 2:
            map_params = pmx.optimize()
            return map_params

def hist_maker(planet, scale = 1):
    if planet == 'Kep1520':
        dt = 0.03187669/(float(scale))
        #This may cause an error!! Causes j = 32 not j = 33
    elif planet == 'K2d22':
        dt = 0.05466947/(float(scale))
    j = (int(1 / dt) + 1)
    i = 0
    while i < j:
        i += 1
        trace1 = single_slice(planet = planet, slice_num = i, load = 1)
        plt.hist(trace1['sigma'])
        plt.savefig('{}plots/norm_trace/slice{:02d}_sigma.pdf'.format(planet, i))
        plt.close()
        plt.hist(trace1['mu'])
        plt.savefig('{}plots/norm_trace/slice{:02d}_mu.pdf'.format(planet, i))
        plt.close()
    trace1 = single_slice(planet = planet, slice_num = 'FullOut', load = 1)
    plt.hist(trace1['sigma'])
    plt.savefig('{}plots/norm_trace/slice{}_sigma.pdf'.format(planet, 'FullOut'))
    plt.close()
    plt.hist(trace1['mu'])
    plt.savefig('{}plots/norm_trace/slice{}_mu.pdf'.format(planet, 'FullOut'))
    plt.close()

def optimize_pull(planet1, scale):
    plt.close('all')
    if planet == 'Kep1520':
        dt = 0.03187669/(float(scale))
        x = np.linspace(0.97, 1.005, 1000)
    if planet1 == 'K2d22':
        dt = 0.05466947/(float(scale))
        x = np.linspace(0.97, 1.01, 1000)
    j = (int(1 / dt) + 1)
    i = 0
    while i < j:
            i += 1
            map_params = joint_function_trace.single_slice(planet = planet1, slice_num = i, load = 2)
            mu_r = map_params["mu_r"]
            sigma_r = map_params["sigma_r"]
            sigma_g = map_params["sigma_gauss"]
            #Update this with out of transit trace value after checking
            mu_g_only = 1.00007
            print(mu_r, sigma_r, mu_g_only, sigma_g)
            y_j_tens = probability_funcs.joint_func(x, sigma_r = sigma_r, mu = mu_r, sigma_g = sigma_g)
            y_r = probability_funcs.raleigh(x, sigma = sigma_r, mu = mu_r)
            y_g = probability_funcs.gaussian(x, sigma = sigma_g, mu = mu_g_only)
            y_j = y_j_tens.eval()
            #y_r = y_r_tens.eval()
            yjmin, yjmax = min(y_j), max(y_j)
            for k, val in enumerate(y_j):
                y_j[k] = (val-yjmin) / (yjmax-yjmin)
            yrmin, yrmax = min(y_r), max(y_r)
            for k, val in enumerate(y_r):
                y_r[k] = (val-yrmin) / (yrmax-yrmin)
            ygmin, ygmax = min(y_g), max(y_g)
            for k, val in enumerate(y_g):
                y_g[k] = (val-ygmin) / (ygmax-ygmin)
            #plt.plot((y_j/(2.2*(j/2)))+(2.2*(i-j/2))/(2.2*(j/2)), x, 'r-')
            #plt.plot(-(y_j/(2.2*(j/2)))+(2.2*(i-j/2))/(2.2*(j/2)), x, 'r-')
            plt.plot((y_r/(2.2*(j/2)))+(2.2*(i-j/2))/(2.2*(j/2)), x, 'b-')
            plt.plot(-(y_r/(2.2*(j/2)))+(2.2*(i-j/2))/(2.2*(j/2)), x, 'b-')
            plt.plot((y_g/(2.2*(j/2)))+(2.2*(i-j/2))/(2.2*(j/2)), x, 'g-')
            plt.plot(-(y_g/(2.2*(j/2)))+(2.2*(i-j/2))/(2.2*(j/2)), x, 'g-')
            plt.fill_betweenx(x,(y_j/(2.2*(j/2)))+(2.2*(i-j/2))/(2.2*(j/2)),-(y_j/(2.2*(j/2)))+(2.2*(i-j/2))/(2.2*(j/2)), alpha = 0.4, color = 'red')
    plt.xlabel('Phase, Counts')
    plt.ylabel('Flux')
    plt.title('Maximum A Priori Solution, Flux vs Orbital Phase')
    plt.savefig('{}plots/{}_joint_trace_optimize.pdf'.format(planet1, planet1), overwrite = True)
    plt.close('all')

def median_pull(planet):
    plt.close('all')
    if planet == 'Kep1520':
        j = 33
        x = np.linspace(0.985, 1.005, 1000)
    if planet == 'K2d22':
        j = 19
        x = np.linspace(0.985, 1.01, 1000)
    i = 1
    while i < (2*j/3):
            i +=1
            map_params = single_slice(planet = planet, slice_num = i, load = 2)
            mu1 = map_params["mu"]
            sigma1 = map_params["sigma"]
            y = probability_funcs.gaussian(x, sigma = sigma1, mu = mu1)
            ymin, ymax = min(y), max(y)
            for k, val in enumerate(y):
                y[k] = (val-ymin) / (ymax-ymin)
            plt.plot((y/(2.2*(j/2)))+(2.2*(i-j/2))/(2.2*(j/2)), x, 'r-')
            plt.plot(-(y/(2.2*(j/2)))+(2.2*(i-j/2))/(2.2*(j/2)), x, 'r-')
            plt.fill_betweenx(x,(y/(2.2*(j/2)))+(2.2*(i-j/2))/(2.2*(j/2)),-(y/(2.2*(j/2)))+(2.2*(i-j/2))/(2.2*(j/2)), alpha = 0.5, color = 'red')
            print((2.2*(i-j/2))/(2.2*(j/2)))
    plt.savefig('{}plots/{}_gauss_trace_optimize.pdf'.format(planet, planet), overwrite = True)
