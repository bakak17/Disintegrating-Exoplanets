from astropy.table import Table
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.optimize import curve_fit
import probability_funcs
import probability_funcs_2

HDUList = fits.open('total_hist.fits')
bigTable = HDUList[0].data
bin_mid = HDUList[1].data
Time = HDUList[3].data
out_of_transit = (bigTable[0] + bigTable[1] + bigTable[2] + bigTable[3] + bigTable[4] + bigTable[5] + bigTable[6] + bigTable[7] + bigTable[23] + bigTable[24] + bigTable[25] + bigTable[26] + bigTable[27] + bigTable[28] + bigTable[29] + bigTable[30] + bigTable[31])/17
plt.plot(bin_mid, out_of_transit, label = 'Out of Transit')

#plt.plot(bin_mid, in_transit, label = 'In Transit')

xdata = bin_mid
ydata = out_of_transit
popt1, pcov1 = curve_fit(probability_funcs.gaussian, xdata, ydata)
print(popt1)
plt.plot(xdata, probability_funcs.gaussian(xdata, *popt1), 'r-', label = 'Out of Transit Model')
plt.xlabel('Flux')
plt.ylabel('Frequency')
plt.legend()
plt.savefig("plots/out_of_transit_avg.pdf")
plt.close()
ListOfMaxima = []
ListOfSigmaR = []
for i in range(0, 32):
    in_transit = bigTable[i]
    plt.plot(bin_mid, in_transit, label = 'Slice {}'.format(i))
    ydata = in_transit
    popt2, pcov2 = curve_fit(probability_funcs_2.joint_func, xdata, ydata, p0=[0.0000418, 1.0])
    ListOfMaxima.append(popt2[1])
    ListOfSigmaR.append(popt2[0])
    print(popt2)
    plt.plot(xdata, probability_funcs_2.joint_func(xdata, *popt2), 'g-', label = 'Slice {} Model'.format(i))
    plt.legend()
    plt.savefig("plots/slice_fit_{:03d}.pdf".format(i))
    plt.close()
plt.plot(Time, ListOfMaxima)
plt.savefig("plots/maximum_time_series.pdf")



