import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc, erf
import pdb
import warnings
import theano.tensor as tt
#sigma_g = 1

def gaussian(x,sigma=1,mu=0):
    return np.exp(-0.5 * ((x - mu)/sigma)**2)/(np.sqrt(2. * np.pi) * sigma)
    
def raleigh(inX,sigma=1,mu=0):
    x = np.array(inX)
    pts = (x - mu) > 0
    yout = np.zeros(len(x))
    yout = -(x - mu) * np.exp(-0.5 * ((x - mu)/sigma)**2) / sigma**2
    yout[pts] = 0.
    return yout

def joint_func(x,sigma_r=1,mu=0, sigma_g = 1):
    #global sigma_g
    sigma_2 = np.sqrt(sigma_r**2 + sigma_g**2)
    
    term1 = gaussian((x - mu),sigma=sigma_g) / (1. + sigma_r**2/sigma_g**2)
    
    mult1 = -(x - mu) * np.exp(-0.5 * (x - mu)**2/sigma_2**2)
    mult2 = sigma_r / (2. * sigma_2**3)

    #pdb.set_trace()
    #mult3 = erfc((x - mu) * sigma_r / (np.sqrt(2.) * sigma_2 * sigma_g))
    mult3 = 1.0 - tt.erf((x - mu) * sigma_r / (np.sqrt(2.) * sigma_2 * sigma_g))
    
    term2 = mult1 * mult2 * mult3
    #pdb.set_trace()
    return term1 + term2

def joint_func_eval(x, sigma_r=1, mu = 0):
    global sigma_g
    #try copying and pasting and switching mult3 to get it to run faster
    return joint_func(x, sigma_r, mu, sigma_g).eval()
    
def plot_joint_func(sigma_r=1,sigma_g=0.3,mu=0):
    x = np.linspace(-8,3,1024)
    y = joint_func(x,sigma_r=sigma_r,sigma_g=sigma_g,mu=mu)
    dx = np.median(np.diff(x))
    
    
    print(np.sum(y * dx))
    
    fig, ax = plt.subplots()
    ax.plot(x,gaussian(x,sigma_g,mu),label='Gaussian, $\sigma_G$={}'.format(sigma_g))
    ax.plot(x,raleigh(x,sigma_r,mu),label='Raleigh, $\sigma_R$={}'.format(sigma_r))
    ax.plot(x,y,label='joint')
    ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('P(x)')
    fig.savefig('plots/joint_func.pdf')
    plt.close(fig)
    #plt.show()
    
