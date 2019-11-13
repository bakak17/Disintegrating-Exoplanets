# coding: utf-8
from lightkurve import search_lightcurvefile
import matplotlib.pyplot as plt
target = "Kepler-1520"
q = 2
lcKep1520 = search_lightcurvefile(target, quarter=1).download().PDCSAP_FLUX.normalize()
while q < 18:
    lcKep1520 = lcKep1520.append(search_lightcurvefile(target, quarter=q).download().PDCSAP_FLUX.normalize())
    q += 1
    
lcKep1520.plot()
