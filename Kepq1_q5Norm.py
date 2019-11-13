# coding: utf-8
from lightkurve import search_lightcurvefile
import matplotlib.pyplot as plt
target = "Kepler-1520"
lcq1 = search_lightcurvefile(target, quarter=1).download().PDCSAP_FLUX.normalize()
lcq2 = search_lightcurvefile(target, quarter=2).download().PDCSAP_FLUX.normalize()
lcq3 = search_lightcurvefile(target, quarter=3).download().PDCSAP_FLUX.normalize()
lcq1_q2 = lcq1.append(lcq2)
lcq1_q3 = lcq1_q2.append(lcq3)
lcq4 = search_lightcurvefile(target, quarter=4).download().PDCSAP_FLUX.normalize()
lcq1_q4 = lcq1_q3.append(lcq4)
lcq1_q4.plot()
plt.show()
lcq5 = search_lightcurvefile(target, quarter=5).download().PDCSAP_FLUX.normalize()
lcq1_q5 = lcq1_q4.append(lcq5)
lcq1_q5.plot()
plt.show()

#TEST/HYPOTHESIZED CODE. DONE OUTSIDE OF iPYTHON

from lightkurve import search_lightcurvefile
import matplotlib.pyplot as plt
target = "Kepler-1520"
q = 2
lc_q1 = search_lightcurvefile(target, quarter=1).download().PDCSAP_FLUX.normalize()
while q < 18:
    lcKep1520 = lcq1.append(lcq1 = search_lightcurvefile(target, quarter=q).download().PDCSAP_FLUX.normalize())
    q += 1
lcKep1520.plot()
plt.show()
