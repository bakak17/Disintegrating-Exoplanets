import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import pymc3 as pm
import probability_funcs
import pdb
import os
import pymc3_ext as pmx
import trace_make
homedir = os.getcwd()
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


def joint_trace(planet, scale = 1, load = 0):
    t = -0.5
    if planet == 'Kep1520':
        dt = 0.03187669/(float(scale))
    elif planet == 'K2d22':
        dt = 0.05466947/(float(scale))
    nSlices = (int(1 / dt) + 1)
    i = 0

    while i < nSlices:
        i += 1
        flnm = 'FluxFiles/{plnt}_slice{nmbr}_s{scale1}.fits'.format(plnt = planet, nmbr = i, scale1 = int(scale))
        HDUList = fits.open(flnm)
        flux = HDUList[1].data
        error = HDUList[2].data
        trace2 = trace_make.single_slice(planet, 'FullOut', load = 1)
        dat_sigg = []
        dat_sigg.append(trace2['sigma_g'])
        sigma_g_mu = np.median(dat_sigg)
        sigma_g_sig = np.std(dat_sigg)
        
        with pm.Model() as model:
            
            ## These will come from out-of-transit posterior
            ## This encapsulates the photon, erorr and stellar noises
            sigma_gauss = pm.Normal('sigma_gauss', mu=sigma_g_mu,sigma=sigma_g_sig, testval = 0.00069)#mean(trace(sigma)), np.std(trace(sigma))
            
            '''
            mu_r = pm.Normal('mu_r', mu=1.0,sigma=0.01, testval = 1)
            sigma_r = pm.Normal('sigma_r', mu=0.01,sigma=0.01)
            '''
            mu_r = pm.Normal('mu_r', mu = 1.0, sigma = 0.05, testval = 1)
            sigma_r = pm.TruncatedNormal('sigma_r', mu = 0.01, sigma = 0.005,lower=0.0)
            
            y_obs = joint_function('joint_function',sigma_r=sigma_r,sigma_g=sigma_gauss,mu_r=mu_r,observed=flux)
            
            #pdb.set_trace()

            direc = homedir + '/Traces/Joint/{plnt}/Slice{nmbr}_s{scale1}'.format(plnt = planet, nmbr = i, scale1 = int(scale))
            dirct = direc.format(nmbr = i)
            direct = dirct.format(scale1 = int(scale))
            print(direct)
            if load == 0:
                #trace1 = pm.sample(1000, random_seed = 123)
                trace1 = pm.sample(init="adapt_diag")
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

def single_slice(planet, slice_num, load = 0, scale = 1):
    flnm = 'FluxFiles/{plnt}_slice{nmbr}_s{scale1}.fits'.format(plnt = planet, nmbr = slice_num, scale1 = int(scale))
    HDUList = fits.open(flnm)
    flux = HDUList[1].data
    error = HDUList[2].data
    trace2 = trace_make.single_slice(planet, 'FullOut', load = 1, scale = scale)
    dat_sigg = []
    dat_sigg.append(trace2['sigma_g'])
    sigma_g_mu = np.median(dat_sigg)
    sigma_g_sig = np.std(dat_sigg)
        
    with pm.Model() as model:
            
            ## These will come from out-of-transit posterior
            ## This encapsulates the photon, erorr and stellar noises
        sigma_gauss = pm.Normal('sigma_gauss', mu=sigma_g_mu,sigma=sigma_g_sig, testval = 0.00069)#mean(trace(sigma)), np.std(trace(sigma))
            
        mu_r = pm.Normal('mu_r', mu=1.0,sigma=0.01, testval = 1)
        sigma_r = pm.TruncatedNormal('sigma_r', mu = 0.01, sigma = 0.005,lower=0.0)
        
        y_obs = joint_function('joint_function',sigma_r=sigma_r,sigma_g=sigma_gauss,mu_r=mu_r,observed=flux)
        
        #pdb.set_trace()
        direct = homedir + '/Traces/Joint/{plnt}/Slice{nmbr}_s{scale1}'.format(plnt = planet, nmbr = slice_num, scale1 = int(scale))
        if load == 0:
            #trace1 = pm.sample(1000, random_seed = 123)
            trace1 = pm.sample(init="adapt_diag")
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
        dt = 0.03187669/(float(scale))
    elif planet == 'K2d22':
        dt = 0.05466947/(float(scale))
    nSlices = (int(1 / dt) + 1)
    i = 0
    while i < j:
        i += 1
        trace1 = single_slice(planet = planet, slice_num = i, load = 1)
        plt.xlim(0.998, 1.0025)
        plt.hist(trace1['mu_r'])
        plt.savefig('{}plots/mix_trace/slice{:02d}_muR.pdf'.format(planet, i))
        plt.close()
        


