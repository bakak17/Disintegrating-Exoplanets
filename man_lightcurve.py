# coding: utf-8
from astropy.io import ascii,fits
HDUList = fits.open('Kep1520folded.fits')
HDUList.info()
from astropy.table import Table
t = Table(HDUList['LIGHTCURVE'].data)
t
plt.plot(t['TIME'],t['FLUX'],'.'); plt.show()
lc = lk.LightCurve(time = t['TIME'], flux = t['FLUX'], flux_err = t['FLUX_ERR'])
lc.plot()
plt.show()
