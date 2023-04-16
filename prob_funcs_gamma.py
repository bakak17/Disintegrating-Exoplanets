import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma, gammainc, gammaincc
import pdb
import os
import theano.tensor as tt



def gaussian(x,sigma=1):
    return np.exp(-0.5 * (x/sigma)**2)/(np.sqrt(2. * np.pi) * sigma)
    
def gamma_p(inX,m=0,alpha = 2., beta=10.):
    x = np.array(inX)
    pts = x < m
    yout = np.zeros(len(x))
    xsub = m - x[pts]
    yout[pts] = xsub**(alpha-1.0) * np.exp(-beta * xsub) * beta**alpha/ gamma(alpha)
    
    return yout


def joint_func(x_orig, k = 0.02, s = 0.7, m = 3, a = 5, b = 0.3):
    x = x_orig - m
    y1 = -k * (1/s) * x * np.exp(-x/(2*s**2)) * tt.gamma(a+1)
    y2 = (b / (b + x**2))**(a+1)
    yout = y1*y2

    yout2 = tt.where(x < m,
                     yout,
                     0*yout)
    
    return yout2

    
def plot_joint_func(beta=10.,sigma_g=0.3,m=1, alpha = 2.):
    x = np.linspace(-3,5,1024)
    #x = np.linspace(-3,0.1,1024)
    y = joint_func(x)
    dx = np.median(np.diff(x))
    
    ## Check the normalizations
#    print(np.sum(y * dx))
#    print(np.sum(gamma_p(x) * dx))
    
    fig, ax = plt.subplots()
    ax.plot(x,gaussian(x,sigma_g),label='Gaussian, $\sigma_G$={}'.format(sigma_g))
    gam_label = r'Gamma, $\alpha$={}, $\beta$={}'.format(alpha,beta)
    ax.plot(x,gamma_p(x,m=m,alpha = alpha, beta=beta),label=gam_label)
    ax.plot(x,y,label='joint')
    ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('P(x)')
    fig.savefig('plots/joint_func_gamma.pdf')
    plt.close(fig)
    #plt.show()

if __name__ == "__main__":
    if os.path.exists('plots') == False:
        os.mkdirs('plots')
    plot_joint_func()
