# coding: utf-8
from lightkurve import search_lightcurvefile
lcf = search_lightcurvefile("Kepler-1520")
import matplotlib.pyplot as plt
lcf = search_lightcurvefile("Kepler-1520", quarter=1)
Kep1520_q1 = lcf.download()
Kep1520_q1.plot()
plt.show()
lcf = search_lightcurvefile("Kepler-1520", quarter=2)
Kep1520_q2 = lcf.download()
Kep1520_q2.plot()
