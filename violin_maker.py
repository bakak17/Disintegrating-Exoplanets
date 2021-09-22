import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
from astropy.io import fits
import probability_funcs
import pdb

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

def single_slice(planet, slice_num):
    flnm = 'FluxFiles/{plnt}_slice{{nmbr}}.fits'.format(plnt = planet)
    filename= flnm.format(nmbr = slice_num)
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
        direct = direc.format(nmbr = slice_num)
        trace1 = pm.load_trace(directory = direct)
        return trace1

def chunk_violin(planet):
    if planet == 'Kep1520':
        j = 33
        plan = 'Kepler 1520b'
    if planet == 'K2d22':
        j = 20
        plan = 'K2-22b'
    i = 1
    HDUList = fits.open("{}total_hist.fits".format(planet))
    Time = HDUList[3].data
    dat = []
    while i < j:
        trace1 = single_slice(planet = planet, slice_num = i)
        dat.append(trace1['mu_r'])
        i += 1
    plt.violinplot(dat, Time, widths = 1/(j), showmeans = True, showextrema = False,
                   showmedians = True, bw_method='silverman')
    plt.title('Posterior Distributions of ' + r'$\mu_R$' + ' for {}'.format(plan))
    plt.xlabel('Time (Phase)')
    plt.ylabel('Normalized Flux')
    plt.savefig("{}plots/violin_plots/trace_violin.pdf".format(planet), overwrite = True)
    plt.close('all')
    dat = []
    i = 1
    while i<j:
        flnm = 'FluxFiles/{plnt}_slice{{nmbr}}.fits'.format(plnt = planet)
        filename = flnm.format(nmbr = i)
        HDUList = fits.open(filename)
        flux = HDUList[1].data
        dat.append(flux)
        i += 1
        #pdb.set_trace()
    plt.violinplot(dat, Time, widths = 1/(j), showmeans = True, showextrema = False,
                   showmedians = True, bw_method='silverman')
    plt.title('Distribution of Fluxes for Observed Transits for {}'.format(plan))
    plt.xlabel('Time (Phase)')
    plt.ylabel('Normalized Flux')
    #plt.show()
    plt.savefig("{}plots/violin_plots/flux_violin.pdf".format(planet), overwrite = True)
    plt.close('all')

def slice_violin(planet):
    if planet == 'Kep1520':
        j = 33
    if planet == 'K2d22':
        j = 20
    i = 1
    HDUList = fits.open("{}total_hist.fits".format(planet))
    Time = HDUList[3].data
    while i < j:
        dat = []
        trace1 = single_slice(planet = planet, slice_num = i)
        dat.append(trace1['mu_r'])
        flnm = 'FluxFiles/{plnt}_slice{{nmbr}}.fits'.format(plnt = planet)
        filename = flnm.format(nmbr = i)
        HDUList = fits.open(filename)
        flux = HDUList[1].data
        dat.append(flux)
        direc = "{plnt}plots/violin_plots/slice{{nmbr}}_violin.pdf".format(plnt = planet)
        plt.violinplot(dat, positions = (Time[i-1], Time[i-1]), showmeans = True,
                       showextrema = True, showmedians = True, bw_method='silverman')
        plt.savefig(direc.format(nmbr = i), overwrite = True)
        plt.close('all')
        i += 1



    
