import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
from astropy.io import fits
import probability_funcs
import pdb
import joint_function_trace
import trace_make
import os
homedir = os.getcwd()

def chunk_violin(planet, scale = 1):
    if planet == 'Kep1520':
        dt = 0.03187669/(float(scale))
        plan = 'Kepler 1520b'
    if planet == 'K2d22':
        dt = 0.05466947/(float(scale))
        plan = 'K2-22b'
    j = (int(1 / dt) + 1)
    i = 0
    t = -0.5
    flnm = "{plnt}total_hist_s{scale1}.fits".format(plnt = planet, scale1 = int(scale))
    HDUList = fits.open(flnm)
    Time = HDUList[3].data
    dat = []
    dat1 = []
    while i < j:
        i += 1
        trace1 = joint_function_trace.single_slice(planet = planet, slice_num = i, load = 1, scale = scale)
        dat.append(trace1['mu_r'])
        if (-0.2 < t) or (t > 0.2):
            dat1.append(trace1['mu_r'])
        t += dt
    avgs = []
    for thing in dat:
        print(np.median(thing))
    for thing in dat1:
        avgs.append(np.mean(thing))
    print(avgs)
    print(np.mean(avgs))
    return

##SCALE = 1:
#out of transit avg = 1.000571680903853
#transit min = 0.9960056138012859
#transit mean = 1.0001103395140487
#transit median = 1.0001994762435746
#transit max = 1.001858340239003
#diff_avg = 0.000372204660278
#diff_med1 = 0.00121174953
#diff_med2 = 0.00073767232
#avg_diff_med = 0.00097471092


##SCALE = 2
#out of transit avg = 1.0004563545680702
#transit min = 0.9945859694441783
#transit mean = 0.9988386141199027
#transit median = 0.9990109115203851
#transit max = 1.0011071143642654
#diff = 0.001445443047685
