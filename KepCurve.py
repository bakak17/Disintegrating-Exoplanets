# coding: utf-8
import lightkurve
import matplotlib.pyplot as plt
def Quickcurve():
    KepCurve = lightkurve.lightcurvefile.KeplerLightCurveFile('Kep1520curve.fits')
    KepCurve.plot()
