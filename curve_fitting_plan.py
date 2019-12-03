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
out_of_transit = (bigTable[0] + bigTable[1] + bigTable[2] + bigTable[3] + bigTable[4] + bigTable[5] + bigTable[6] + bigTable[7] + bigTable[23] + bigTable[24] + bigTable[25] + bigTable[26] + bigTable[27] + bigTable[28] + bigTable[29] + bigTable[30] + bigTable[31])/17
in_transit = bigTable[11]
plt.plot(bin_mid, out_of_transit, label = 'Out of Transit')

plt.plot(bin_mid, in_transit, label = 'In Transit')

xdata = bin_mid
ydata = out_of_transit
popt1, pcov1 = curve_fit(probability_funcs.gaussian, xdata, ydata)
print(popt1)
plt.plot(xdata, probability_funcs.gaussian(xdata, *popt1), 'r-', label = 'Out of Transit Model')
plt.xlabel('Flux')
plt.ylabel('Frequency')


ydata = in_transit
popt2, pcov2 = curve_fit(probability_funcs_2.joint_func, xdata, ydata)
print(popt2)
plt.plot(xdata, probability_funcs_2.joint_func(xdata, *popt2), 'g-', label = 'In Transit Model')
plt.legend()
plt.show()
