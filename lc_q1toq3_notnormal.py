# coding: utf-8
from lightkurve import search_lightcurvefile
import matplotlib.pyplot as plt
target = "Kepler-1520"
lc_q1 = search_lightcurvefile(target, quarter=1).download().PDCSAP_FLUX
lc_q1.plot()
lc_q2 = search_lightcurvefile(target, quarter=2).download().PDCSAP_FLUX
lc_q2.plot()
lc_q2_plot()
lc_q2.plot()
plt.show()
lc_q1q2 = lc_q1.append(lc_q2)
lc_q1q2.plot()
plt.show()
lc_q3 = search_lightcurvefile(target, quarter=3).download().PDCSAP_FLUX
lc_q3.plot()
plt.show()
lc_q1_q3 = lc_q1q2.append(lc_q3)
lc_q1_q3.plot()
plt.show()
