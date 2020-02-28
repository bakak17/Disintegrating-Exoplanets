import pdb
from astropy.table import Table
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.optimize import curve_fit
import probability_funcs
import probability_funcs_2
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
        #out_of_transit = ((bigTable[4] + bigTable[5] + bigTable[6] + bigTable[7] + bigTable[8] + bigTable[9] + bigTable[10] + bigTable[11] + bigTable[12] + bigTable[13] + bigTable[14] + bigTable[15] + bigTable[16] + bigTable[17] + bigTable[18])/15)
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
    plt.savefig("{}plots2/out_of_transit_avg.pdf".format(planet))
    plt.close()
    ListOfMaxima = []
    ListOfSigmaR = []
    for i in range(0, slices):
        in_transit = bigTable[i]
        plt.plot(bin_mid, in_transit, label = 'Slice {}'.format(i))
        ydata = in_transit
        popt2, pcov2 = curve_fit(probability_funcs.joint_func, xdata, ydata, p0=[0.0000418, 1.0])
        ListOfMaxima.append(popt2[1])
        ListOfSigmaR.append(popt2[0])
        print(popt2)
        plt.plot(xdata, probability_funcs.joint_func(xdata, *popt2), 'g-', label = 'Slice {} Model'.format(i))
        plt.plot(xdata, probability_funcs.raleigh(xdata, sigma = popt2[0], mu = popt2[1]), label = 'Slice {} Raleigh'.format(i))
        plt.legend()
        plt.savefig("{}plots2/slice_fit_{:03d}.pdf".format(planet, i))
        plt.close()
    plt.plot(Time, ListOfMaxima)
    plt.savefig("{}plots2/maximum_time_series.pdf".format(planet))
    plt.close('all')
