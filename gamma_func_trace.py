import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import pymc3 as pm
import prob_funcs_gamma
import pdb
import os
import pymc3_ext as pmx
import trace_make
import theano.tensor as tt
homedir = os.getcwd()
plt.close('all')

class joint_gamma(pm.Continuous):
    def __init__(self,k=0.02,s=0.7,m=3,a=5,b=0.3,*args,**kwargs):
        """
        Combined Gamma and Gaussian
        Parameters
        ----------
        k: float
            Scale parameter
        s: float
            Sigma of Gaussian distribution
        m: float
            The maximum for the Gamma distribution
        a: float
            The power of the Gamma distribution
        b: float
            The effective "spread" of the Gamma distribution
        """
        
        super().__init__(*args, **kwargs)
        self.k = k
        self.s = s
        self.m = m
        self.a = a
        self.b = b

    def logp(self, x):
        p = prob_funcs_gamma.joint_func(x, k = self.k, m = self.m, s = self.s, a = self.a, b = self.b)
        return np.log(p)

def gamma_trace(planet, scale = 1, load = 0):
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
        '''trace2 = trace_make.single_slice(planet, 'FullOut', load = 1)
        dat_sigg = []
        dat_sigg.append(trace2['sigma_g'])
        sigma_g_mu = np.median(dat_sigg)
        sigma_g_sig = np.std(dat_sigg)'''

        with pm.Model() as model:
            k = pm.TruncatedNormal('k', mu = 0.1, sigma = 0.5, lower = 0.0001, testval = 0.02)
            ## These will come from out-of-transit posterior
            ## This encapsulates the photon, erorr and stellar noises
            sig = pm.TruncatedNormal('sig', mu=0.7,sigma=0.5, lower = 0.25, testval = 0.7)

            mu = pm.Normal('mu', mu = 1, sigma = 0.5, testval = 1)
            alpha = pm.Normal('alpha', mu = 5, sigma = 1, testval = 5)
            beta = pm.TruncatedNormal('beta', mu = 2, sigma = 2, lower = 0.0, testval = 0.3)

            y_obs = joint_gamma('joint_gamma', k = k, s = sig, m = mu, a = alpha, b = beta, observed = flux)

            direct = homedir + '/Traces/Gamma/{plnt}/Slice{nmbr}_s{scale1}'.format(plnt = planet, nmbr = i, scale1 = int(scale))

            if load == 0:
                #trace1 = pm.sample(1000, random_seed = 123)
                trace1 = pm.sample(init="adapt_diag")
                if os.path.exists(direct) == False:
                    os.makedirs(direct)
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

        sig = pm.Normal('sig', mu=sigma_g_mu,sigma=sigma_g_sig, testval = 0.00069)

        mu = pm.Normal('mu', mu = 1, sigma = 0.05, testval = 1)
        beta = pm.TruncatedNormal('beta', mu = 4, sigma = 2, lower = 0.0)

        y_obs = joint_gamma('joint_gamma', s = sig, m = mu, b = beta, observed = flux)

        direct = homedir + '/Traces/Gamma/{plnt}/Slice{nmbr}_s{scale1}'.format(plnt = planet, nmbr = i, scale1 = int(scale))

        if load == 0:
                #trace1 = pm.sample(1000, random_seed = 123)
                trace1 = pm.sample(init="adapt_diag")
                if os.path.exists(direct) == False:
                    os.makedirs(direct)
                pm.save_trace(trace1, directory = direct, overwrite = True)
        elif load == 1:
                trace1 = pm.load_trace(directory = direct)
                return trace1
        elif load == 2:
                map_params = pmx.optimize()
                return map_params
