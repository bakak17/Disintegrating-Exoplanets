import pdb
from astropy.table import Table
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.optimize import curve_fit
import probability_funcs
plt.close('all')

def fitting(planet):
    HDUList = fits.open("{}total_hist.fits".format(planet))
    bigTable = HDUList[0].data
    bin_mid = HDUList[1].data
    nSlices = HDUList[2].data
    slices = len(nSlices[0:])
    Time = HDUList[3].data
    out_of_transit = np.zeros_like(bigTable[0])
    time_index = np.arange(slices)
    if planet == 'Kep1520':
        j = 0.0
        for i in time_index:
            if (Time[i] < -0.128) or (Time[i] > 0.2):
                out_of_transit = out_of_transit + bigTable[i]
                j += 1.0
        out_of_transit = out_of_transit/j
        p0 = [0.000717105679, 1]
    elif planet == 'K2d22':
        j = 0.0
        for i in time_index:
            if (Time[i] < -0.2) or (Time[i] > 0.22):
                out_of_transit = out_of_transit + bigTable[i]
                j += 1.0
        out_of_transit = out_of_transit/j
        p0 = [0.000572959250, 1]
    plt.plot(bin_mid, out_of_transit, label = 'Out of Transit')
    
    xdata = bin_mid
    ydata = out_of_transit
    popt1, pcov1 = curve_fit(probability_funcs.gaussian, xdata, ydata, p0=p0)
    print(popt1)
    print(popt1[0])
    probability_funcs.sigma_g = popt1[0]
    plt.plot(xdata, probability_funcs.gaussian(xdata, *popt1), 'r-', label = 'Out of Transit Model')
    plt.xlabel('Flux')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig("{}plots/out_of_transit_avg.pdf".format(planet))
    plt.close()
    ListOfMaxima = []
    ListOfSigmaR = []
    MaximaError = []
    SigmaRError = []
    for i in range(0, slices):
        in_transit = bigTable[i]
        plt.plot(bin_mid, in_transit, label = 'Slice {}'.format(i))
        ydata = in_transit
        popt2, pcov2 = curve_fit(probability_funcs.joint_func, xdata, ydata, p0=[0.0000418, 1.0], bounds=([0.00004, 0.98], [0.07, 1.05]))
        ListOfMaxima.append(popt2[1])
        ListOfSigmaR.append(popt2[0])
        MaximaError.append((pcov2[1,1])**0.5)
        SigmaRError.append((pcov2[0,0])**0.5)
        print(popt2, i)
        plt.plot(xdata, probability_funcs.joint_func(xdata, *popt2), 'g-', linewidth=0.5, label = 'Slice {} Model'.format(i))
        plt.plot(xdata, probability_funcs.raleigh(xdata, sigma = popt2[0], mu = popt2[1]), linewidth=0.5, label = 'Slice {} Raleigh'.format(i))
        #plt.xlim(0.9997, 1.000)
        #plt.ylim(0, 50)
        #plt.rcParams.update({'font.size': 6})
        plt.xlabel("Flux")
        plt.ylabel("Counts")
        plt.tight_layout()
        plt.legend()
        plt.savefig("{}plots/slice_fit_{:03d}.pdf".format(planet, i))
        plt.close()
    plt.errorbar(Time, ListOfMaxima, MaximaError, xerr=None, fmt='r', ecolor='b', elinewidth = 1)
    if planet == 'Kep1520':
        plt.title('Maximum Time Series for Kepler 1520b')
    elif planet == 'K2d22':
        plt.title('Maximum Time Series for K2-22b')
    plt.xlabel('Transit Phase')
    plt.ylabel('Flux')
    plt.tight_layout()
    plt.show()
    plt.savefig("{}plots/Max_time_series.pdf".format(planet))
    plt.close('all')
