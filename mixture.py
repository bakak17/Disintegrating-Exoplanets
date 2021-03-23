import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import pymc3 as pm
import arviz as az
plt.close('all')

def mixture(planet):

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

        # 2-Mixture Poisson using iterable of distributions.
        with pm.Model() as model:
            
            ## These will come from out-of-transit posterior
            ## This encapsulates the photon, erorr and stellar noises
            mu_gauss = pm.Normal('mu_gauss', mu=1.00007,sigma=0.00001)#mean(trace(mu)), np.std(trace(mu))
            sigma_gauss = pm.Normal('sigma_gauss', mu=0.0007,sigma=0.0001)#mean(trace(sigma)), np.std(trace(sigma))
            
            gauss = pm.Normal('model_gauss', mu=mu_gauss,sigma=sigma_gauss)
            
            ## This component will be the astrophysical signal (ie. the dust)
            ## Probably have to fix the syntax
            flux_maximum = pm.Normal('flux_maximum', mu=1.0,sigma=0.01)
            nu_rice = pm.HalfNormal('mu_rice', sigma=5.0)
            sigma_rice = pm.Normal('sigma_rice', mu=0.01,sigma=0.01)
            rice = pm.Rice('rice_model', nu=nu_rice,sigma=sigma_rice,testval=0.01)# + flux_maximum
            
            w = pm.Dirichlet('w', a=np.array([1, 1]))
            
            like = pm.Mixture('like', w=w, comp_dists = [gauss, rice], observed=flux)
