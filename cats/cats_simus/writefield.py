from astropy.io import fits

def writefield(filename, field):

    fits.writeto(filename + '_r.fits', field.real, header=None, overwrite=True)
    fits.writeto(filename + '_i.fits', field.imag, header=None, overwrite=True)

    return



