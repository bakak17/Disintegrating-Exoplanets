import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import pymc3 as pm
import pymc3_ext as pmx
import arviz as az
import os
import probability_funcs
from sklearn import preprocessing
plt.close('all')

class joint_function(pm.Continuous):
    def __init__(self,sigma_r=1,sigma_g=1,mu_r=0.0,*args,**kwargs):
        """
        Combined Raleigh and Gaussian
        Parameters
        ----------
        sigma_r: float
            Sigma of Raleigh distribution
        mu_r: float
            The maximum for the raleigh distribution
        """
        
        super().__init__(*args, **kwargs)
        self.sigma_r = sigma_r
        self.mu_r = mu_r
        self.sigma_g = sigma_g
    
    def logp(self,x):
        p = probability_funcs.joint_func(x,sigma_r=self.sigma_r,
                                        mu=self.mu_r, sigma_g = self.sigma_g)
        return np.log(p)

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
            #map_params = pmx.optimize()
            trace1 = pm.sample(1000, random_seed = 123)
        
            direc = '/Users/keithbaka/Documents/0Research_Schlawin/Traces/Gauss/{plnt}/Slice{{nmbr}}'.format(plnt = planet)
            direct = direc.format(nmbr = i)
            try:
                pm.save_trace(trace1, directory = direct, overwrite = True)
            except IOError:
                os.mkdir(direct)
            pm.save_trace(trace1, directory = direct, overwrite = True)
            #return map_params

def single_slice(planet, slice_num, load = 0, m = 2):
    flnm = 'FluxFiles/{plnt}_slice{{nmbr}}.fits'.format(plnt = planet)
    filename= flnm.format(nmbr = slice_num)
    HDUList = fits.open(filename)
    flux = HDUList[1].data
    error = HDUList[2].data
    with pm.Model() as slice_model:
        if m == 2:
            sigma_g = pm.Normal('sigma_g', 0.00065, 0.0026)
            if slice_num == 'FullOut':
                mu_g = pm.Normal('mu_g', 1, 0.065)
            else:
                mu_g = pm.Normal('mu_g', 1, 0.00065)
            slice_mdl = pm.Normal('model', sigma = sigma_g, mu = mu_g, observed = flux)
            direc = '/Users/keithbaka/Documents/0Research_Schlawin/Traces/Gauss/{plnt}/Slice{{nmbr}}'.format(plnt = planet)
            direct = direc.format(nmbr = slice_num)
        elif m == 1:
            mu_r = pm.Normal('mu_r', mu=1.0,sigma=0.01, testval = 1)
            sigma_r = pm.Normal('sigma_r', mu=0.01,sigma=0.01)
            slice_mdl = pm.Normal('model', sigma = sigma_r, mu = mu_r, observed = flux)
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

def hist_maker(planet):
    if planet == 'Kep1520':
        j = 32
    if planet == 'K2d22':
        j = 19
    i = 1
    while i < j:
        trace1 = single_slice(planet = planet, slice_num = i, load = 1)
        plt.hist(trace1['sigma'])
        plt.savefig('{}plots/norm_trace/slice{:02d}_sigma.pdf'.format(planet, i))
        plt.close()
        plt.hist(trace1['mu'])
        plt.savefig('{}plots/norm_trace/slice{:02d}_mu.pdf'.format(planet, i))
        plt.close()
        i += 1
    trace1 = single_slice(planet = planet, slice_num = 'FullOut', load = 1)
    plt.hist(trace1['sigma'])
    plt.savefig('{}plots/norm_trace/slice{}_sigma.pdf'.format(planet, 'FullOut'))
    plt.close()
    plt.hist(trace1['mu'])
    plt.savefig('{}plots/norm_trace/slice{}_mu.pdf'.format(planet, 'FullOut'))
    plt.close()

def optimize_pull(planet1):
    plt.close('all')
    m1 = 2
    if planet1 == 'Kep1520':
        j = 33
        x = np.linspace(0.985, 1.005, 1000)
    if planet1 == 'K2d22':
        j = 19
        x = np.linspace(0.985, 1.01, 1000)
    while m1 > (-1):
        i = 1
        while i < j:
            if m1 == 2:
                map_params = single_slice(planet = planet1, slice_num = i, m = m1, load = 2)
                mu_g = map_params["mu_g"]
                sigma_g = map_params["sigma_g"]
                y = probability_funcs.gaussian(x, sigma = sigma_g, mu = mu_g)
            elif m1 == 1:
                map_params = single_slice(planet = planet1, slice_num = i, m = m1, load = 2)
                mu_r = map_params["mu_r"]
                sigma_r = map_params["sigma_r"]
                y = probability_funcs.raleigh(x, sigma = sigma_r, mu = mu_r)
            elif m1 == 0:
                y = joint_function(x, sigma_r = sigma_r, mu = mu_r, sigma_g = sigma_g)
                print(y)
            ymin, ymax = min(y), max(y)
            for k, val in enumerate(y):
                y[k] = (val-ymin) / (ymax-ymin)
            plt.plot((y/(2.2*(j/2)))+(2.2*(i-j/2))/(2.2*(j/2)), x, 'r-')
            plt.plot(-(y/(2.2*(j/2)))+(2.2*(i-j/2))/(2.2*(j/2)), x, 'r-')
            plt.fill_betweenx(x,(y/(2.2*(j/2)))+(2.2*(i-j/2))/(2.2*(j/2)),-(y/(2.2*(j/2)))+(2.2*(i-j/2))/(2.2*(j/2)), alpha = 0.5, color = 'red')
            i +=1
        if m1 == 2:
            plt.xlabel('Phase, Counts')
            plt.ylabel('Flux')
            plt.title('Maximum A Priori Solution, Flux vs Orbital Phase')
            plt.savefig('{}plots/{}_gauss_trace_optimize.pdf'.format(planet1, planet1), overwrite = True)
            plt.close('all')
        elif m1 == 1:
            plt.xlabel('Phase, Counts')
            plt.ylabel('Flux')
            plt.title('Maximum A Priori Solution, Flux vs Orbital Phase')
            plt.savefig('{}plots/{}_raleigh_trace_optimize.pdf'.format(planet1, planet1), overwrite = True)
            plt.close('all')
        elif m1 == 0:
            plt.xlabel('Phase, Counts')
            plt.ylabel('Flux')
            plt.title('Maximum A Priori Solution, Flux vs Orbital Phase')
            plt.savefig('{}plots/{}_joint_trace_optimize.pdf'.format(planet1, planet1), overwrite = True)
            plt.close('all')
        m1-=1

def optimize_pull1(planet):
    plt.close('all')
    if planet == 'Kep1520':
        j = 33
        x = np.linspace(0.985, 1.005, 1000)
    if planet == 'K2d22':
        j = 19
        x = np.linspace(0.985, 1.01, 1000)
    i = 1
    while i < (2*j/3):
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
            i +=1
    plt.savefig('{}plots/{}_gauss_trace_optimize.pdf'.format(planet, planet), overwrite = True)
