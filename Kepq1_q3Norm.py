# coding: utf-8
from lightkurve import search_lightcurvefile
import matplotlib.pyplot as plt
target = "Kepler-1520"
lcq1 = search_lightcurvefile(target, quarter=1).download().PDCSAP_FLUX.normalize()
lcq2 = search_lightcurvefile(target, quarter=2).download().PDCSAP_FLUX.normalize()
lcq3 = search_lightcurvefile(target, quarter=3).download().PDCSAP_FLUX.normalize()
lcq1_q2 = lcq1.append(lcq2)
lcq1_q3 = lcq1_q2.append(lcq3)
lcq1_q3.plot()
plt.show()
