import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import arviz as az
import pymc3 as pm
import probability_funcs
import pdb
import os
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


def joint_trace(planet, load = 0):
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
        
        with pm.Model() as model:
            
            ## These will come from out-of-transit posterior
            ## This encapsulates the photon, erorr and stellar noises
            mu_gauss = pm.Normal('mu_gauss', mu=1.00007,sigma=0.00001, testval = flux.mean(axis=0))#mean(trace(mu)), np.std(trace(mu))
            sigma_gauss = pm.Normal('sigma_gauss', mu=0.0007,sigma=0.0001, testval = 0.00069)#mean(trace(sigma)), np.std(trace(sigma))
            
            gauss = pm.Normal('model_gauss', mu=mu_gauss,sigma=sigma_gauss)

            #flux_maximum = pm.Normal('flux_maximum', mu=1.0,sigma=0.01)
            mu_r = pm.Normal('mu_r', mu=1.0,sigma=0.01, testval = 1)
            sigma_r = pm.Normal('sigma_r', mu=0.01,sigma=0.01)
            
            y_obs = joint_function('joint_function',sigma_r=sigma_r,sigma_g=sigma_gauss,mu_r=mu_r,observed=flux)
            
            #pdb.set_trace()

            direc = '/Users/keithbaka/Documents/0Research_Schlawin/Traces/Joint/{plnt}/Slice{{nmbr}}'.format(plnt = planet)
            direct = direc.format(nmbr = i)
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

