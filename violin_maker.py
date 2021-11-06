import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
from astropy.io import fits
import probability_funcs
import pdb
import joint_function_trace
import trace_make

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
        sigma_r = pm.TruncatedNormal('sigma_r', mu = 0.01, sigma = 0.005,lower=0.0)
        
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
        trace1 = joint_function_trace.single_slice(planet = planet, slice_num = i, load = 1)
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
        trace1 = joint_function_trace.single_slice(planet = planet, slice_num = i, load = 1)
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

def median_pull(planet):
    plt.close('all')
    if planet == 'Kep1520':
        j = 33
        x = np.linspace(0.985, 1.005, 1000)
    if planet == 'K2d22':
        j = 19
        x = np.linspace(0.985, 1.01, 1000)
    i = 1
    trace2 = trace_make.single_slice(planet = planet, slice_num = 'FullOut', load = 1)
    dat_mug = []
    dat_sigg = []
    dat_mug.append(trace2['mu_gauss'])
    dat_sigg.append(trace2['sigma_gauss'])
    mu_g = np.median(dat_mug)
    sigma_g = np.median(dat_sigg)
    while i < j:
        dat_mur = []
        dat_sigr = []
        trace1 = joint_function_trace.single_slice(planet = planet, slice_num = i, load = 1)
        dat_mur.append(trace1['mu_r'])
        dat_sigr.append(trace1['sigma_r'])
        mu_r = np.median(dat_mur)
        sigma_r = np.median(dat_sigr)
        print(sigma_r, mu_r, sigma_g)
        y_j_tens = probability_funcs.joint_func(x, sigma_r = sigma_r, mu = mu_r, sigma_g = sigma_g)
        y_r = probability_funcs.raleigh(x, sigma = sigma_r, mu = mu_r)
        y_g = probability_funcs.gaussian(x, sigma = sigma_g, mu = mu_g)
        y_j = y_j_tens.eval()
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
        plt.fill_betweenx(x,(y_j/(2.2*(j/2)))+(2.2*(i-j/2))/(2.2*(j/2)),-(y_j/(2.2*(j/2)))+(2.2*(i-j/2))/(2.2*(j/2)), alpha = 0.7, color = 'red')
        i +=1
    plt.xlabel('Phase, Counts')
    plt.ylabel('Flux')
    plt.title('Median Posterior Solution, Flux vs Orbital Phase')
    plt.savefig('{}plots/{}_joint_trace_optimize.pdf'.format(planet, planet), overwrite = True)
    plt.close('all')
        
    
