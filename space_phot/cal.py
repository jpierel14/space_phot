import numpy as np
import scipy
import astropy
from astropy import units as u
from astropy.wcs.utils import proj_plane_pixel_scales

import stpsf
import matplotlib.pyplot as plt
from poppy.utils import radial_profile

from .util import simple_aperture_sum
from .wfc3_photometry.psf_tools.PSFUtils import make_models
from .wfc3_photometry.psf_tools.PSFPhot import get_standard_psf



def hst_get_zp(filt,zpsys='ab'):
    if zpsys.lower()=='ab':
        return {'F098M':25.666,'F105W':26.264,'F110W':26.819,'F125W':26.232,'F140W':26.450,'F160W':25.936}[filt]
    elif zpsys.lower()=='vega':
        return {'F098M':25.090,'F105W':25.603,'F110W':26.042,'F125W':25.312,'F140W':25.353,'F160W':24.662}[filt]
    else:
        print('unknown zpsys')
        return

def calibrate_JWST_flux(
    flux,
    fluxerr,
    imwcs,
    flux_units=None,
    units=astropy.units.MJy,
):
    """
    Calibrate JWST image-plane fluxes to a chosen flux unit and AB magnitudes.

    Parameters
    ----------
    flux : array_like
        Measured source flux in the native image units (usually MJy/sr).
    fluxerr : array_like
        1-sigma uncertainties on `flux` in the same native units.
    imwcs : astropy.wcs.WCS
        WCS of the image, used to compute the pixel area on the sky.
    flux_units : astropy.units.Unit or None, optional
        Unit of `flux` and `fluxerr`. If None or equal to MJy/sr, we assume
        MJy/sr and multiply by the pixel solid angle to get MJy.
    units : astropy.units.Unit, optional
        Desired output flux units (default: MJy).

    Returns
    -------
    flux_out : ndarray or float
        Calibrated flux in `units`.
    fluxerr_out : ndarray or float
        Calibrated flux uncertainty in `units`.
    mag : ndarray or float
        AB magnitudes corresponding to `flux_out`.
    magerr : ndarray or float
        Magnitude uncertainties (uses 2.5*log10(1 + fluxerr/flux)).
    zp : float or ndarray
        Zero point defined as mag + 2.5*log10(flux).
    """
    # Ensure array-like for internal computations; this safely handles scalars too.
    flux = np.asanyarray(flux, dtype=float)
    fluxerr = np.asanyarray(fluxerr, dtype=float)

    # Exact expression from the original code
    magerr = 2.5 * np.log10(1.0 + (fluxerr / flux))

    # Default: treat input as surface brightness (MJy/sr) and convert to MJy.
    if flux_units is None or flux_units == astropy.units.MJy / astropy.units.sr:
        pixel_scale = (
            astropy.wcs.utils.proj_plane_pixel_scales(imwcs)[0]
            * imwcs.wcs.cunit[0].to("arcsec")
        )
        flux_units = (
            astropy.units.MJy / astropy.units.sr
            * (pixel_scale * astropy.units.arcsec) ** 2
        )

    flux_q = flux * flux_units
    fluxerr_q = fluxerr * flux_units

    flux_q = flux_q.to(units)
    fluxerr_q = fluxerr_q.to(units)

    mag_q = flux_q.to(astropy.units.ABmag)
    zp = mag_q.value + 2.5 * np.log10(flux_q.value)

    try:
        return (
            flux_q.value,
            fluxerr_q.value,
            mag_q.value,
            magerr,
            float(zp),
        )
    except Exception:
        # If zp is array-like, keep it that way.
        return flux_q.value, fluxerr_q.value, mag_q.value, magerr, zp


def JWST_mag_to_flux(mag, imwcs, zpsys="ab", density=True):
    """
    Convert AB/Vega magnitudes back to JWST fluxes.

    Parameters
    ----------
    mag : array_like
        AB or Vega magnitudes.
    imwcs : astropy.wcs.WCS
        WCS used to determine the pixel scale, if `density=True`.
    zpsys : {'ab', 'vega'}, optional
        Magnitude system of the input magnitudes.
    density : bool, optional
        If True, return surface-brightness units (MJy/sr).
        If False, return integrated flux per pixel (MJy).

    Returns
    -------
    flux : ndarray
        Flux in MJy/sr (if density=True) or MJy (if density=False).
    """
    mag = np.asanyarray(mag, dtype=float)

    if zpsys.lower() == "ab":
        flux_q = (mag * astropy.units.ABmag).to(astropy.units.MJy)
    elif zpsys.lower() == "vega":
        flux_q = (mag * astropy.units.Vegamag).to(astropy.units.MJy)
    else:
        raise RuntimeError("Do not recognize zpsys")

    if density:
        # Return MJy/sr
        pixel_scale = (
            astropy.wcs.utils.proj_plane_pixel_scales(imwcs)[0]
            * imwcs.wcs.cunit[0].to("arcsec")
        )
        # Convert MJy (per pixel) to MJy/sr by dividing by pixel solid angle
        pixel_area_sr = ((pixel_scale * astropy.units.arcsec) ** 2).to(
            astropy.units.sr
        )
        flux_q = flux_q / pixel_area_sr
    # else: flux is already MJy

    return flux_q.value



def calibrate_HST_flux(flux, fluxerr, primary_header, sci_header):
    """
    Calibrate HST fluxes to magnitudes using PHOTFLAM/PHOTPLAM.

    Parameters
    ----------
    flux : array_like
        Flux values in the same units used by HST PHOTFLAM/PHOTPLAM
        (i.e. already in flux-density units, not raw counts).
    fluxerr : array_like
        1-sigma uncertainties on `flux`.
    primary_header : fits.Header
        Primary header (must contain DETECTOR, and possibly PHOTFLAM/PHOTPLAM).
    sci_header : fits.Header
        SCI extension header (preferred place for PHOTFLAM/PHOTPLAM).

    Returns
    -------
    flux_out : float or ndarray
        Input flux (unchanged), but cast to float/ndarray consistently.
    fluxerr_out : float or ndarray
        Input flux uncertainty (unchanged).
    mag : float or ndarray
        AB magnitudes derived from `flux`.
    magerr : float or ndarray
        Magnitude uncertainties (1.086 * fluxerr/flux).
    zp : float or ndarray
        Zero point used in the conversion.
    """
    # import pdb
    # pdb.set_trace()
    flux = np.asanyarray(flux, dtype=float)
    fluxerr = np.asanyarray(fluxerr, dtype=float)

    magerr = 1.086 * fluxerr / flux
    
    if 'PHOTFLAM' in sci_header.keys():
        photflam = sci_header["PHOTFLAM"]
    elif 'PHOTFLAM' in primary_header.keys():
        photflam = primary_header["PHOTFLAM"]
    else:
        raise RuntimeError('No PHOTFLAM in headers.')
    if 'PHOTPLAM' in sci_header.keys():
        photplam = sci_header["PHOTPLAM"]
    elif 'PHOTPLAM' in primary_header.keys():
        photplam = primary_header["PHOTPLAM"]
    else:
        raise RuntimeError('No PHOTPLAM in headers.')
    zp = -2.5 * np.log10(photflam) - 5 * np.log10(photplam) - 2.408

    mag = -2.5 * np.log10(flux) + zp

    try:
        # scalar case
        return float(flux), float(fluxerr), float(mag), float(magerr), float(zp)
    except Exception:
        # array case
        return (
            np.array(flux),
            np.array(fluxerr),
            np.array(mag),
            np.array(magerr),
            np.array(zp),
        )




def HST_mag_to_flux(mag, primary_header, sci_header, zpsys="ab"):
    """
    Convert HST magnitudes to flux density using PHOTFLAM/PHOTPLAM
    (or the IR zeropoint helper if that branch is ever enabled).

    Parameters
    ----------
    mag : array_like
        Magnitudes in the system defined by `zpsys` (currently 'ab' expected).
    primary_header : fits.Header
        Primary header (DETECTOR, FILTER, etc.).
    sci_header : fits.Header
        SCI header with PHOTFLAM/PHOTPLAM where available.
    zpsys : str, optional
        Magnitude system. Currently only 'ab' is actually supported here.

    Returns
    -------
    flux : ndarray
        Flux density corresponding to the input magnitudes.
    """
    mag = np.asanyarray(mag, dtype=float)

    if 'PHOTFLAM' in sci_header.keys():
        photflam = sci_header["PHOTFLAM"]
    elif 'PHOTFLAM' in primary_header.keys():
        photflam = primary_header["PHOTFLAM"]
    else:
        raise RuntimeError('No PHOTFLAM in headers.')
    if 'PHOTPLAM' in sci_header.keys():
        photplam = sci_header["PHOTPLAM"]
    elif 'PHOTPLAM' in primary_header.keys():
        photplam = primary_header["PHOTPLAM"]
    else:
        raise RuntimeError('No PHOTPLAM in headers.')
    zp = -2.5 * np.log10(photflam) - 5 * np.log10(photplam) - 2.408

    # Inverse of mag = -2.5 log10(flux) + zp → flux = 10^(-0.4 * (mag - zp))
    flux = 10 ** (-0.4 * (mag - zp))
    return flux




def calc_jwst_psf_corr(ap_rad,instrument,band,imwcs,oversample=4,show_plot=False,psf=None):
    if psf is None:
        inst = stpsf.instrument(instrument)
        inst.filter = band
        psf = inst.calc_psf(oversample=oversample)

    if show_plot:
        stpsf.display_ee(psf)
        plt.show()
    radius, profile, ee = radial_profile(psf, ee=True, ext=0)
    ee_func = scipy.interpolate.interp1d(radius,ee)
    pixel_scale = astropy.wcs.utils.proj_plane_pixel_scales(imwcs)[0]  * imwcs.wcs.cunit[0].to('arcsec')
    #print(ap_rad*pixel_scale,ee_func(ap_rad*pixel_scale))
    return(1/ee_func(ap_rad*pixel_scale),psf)

    
def calc_hst_psf_corr(ap_rad,instrument,band,pos,psf=None,sci_ext=1):

    if psf is None:
        psf = make_models(get_standard_psf('/Users/jpierel/DataBase/HST/psfs',band,instrument))[sci_ext-1]

    elif isinstance(psf,str):
            
        psf = make_models(get_standard_psf(psf,band,instrument))[sci_ext-1]


    psf_width =500
    psf.x_0 = 250#pos[0]
    psf.y_0 = 250#pos[1]
    x,y=(250,250)

    yg, xg = np.mgrid[-1*(psf_width-1)/2:(psf_width+1)/2,-1*(psf_width-1)/2:(psf_width+1)/2].astype(int)
    yf, xf = yg+int(y+.5), xg+int(x+.5)
    psf = np.array(psf(xf,yf)).astype(float)
    big_psf = np.sum(psf)

    psf_width =ap_rad*2
    psf.x_0 = psf_width/2#pos[0]
    psf.y_0 = psf_width/2#pos[1]
    x,y=(psf.x_0,psf.y_0)

    yg, xg = np.mgrid[-1*(psf_width-1)/2:(psf_width+1)/2,-1*(psf_width-1)/2:(psf_width+1)/2].astype(int)
    yf, xf = yg+int(y+.5), xg+int(x+.5)
    psf = np.array(psf(xf,yf)).astype(float)
    return(big_psf/np.sum(psf))
    #print('tot',float(simple_aperture_sum(psf,[250,250],100)))
    #return(np.sum(psf)/simple_aperture_sum(psf,[250,250],100)/simple_aperture_sum(psf,[250,250],ap_rad))


    


