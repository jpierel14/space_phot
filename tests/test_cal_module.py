# tests/test_cal_module.py

import numpy as np
import astropy.units as u
from astropy.wcs import WCS
from astropy.io import fits

from space_phot.cal import (
    calibrate_JWST_flux,
    JWST_mag_to_flux,
    calibrate_HST_flux,
    HST_mag_to_flux,
)


def _make_simple_wcs(pixel_scale_arcsec=1.0):
    """
    Create a trivial TAN WCS with a given pixel scale in arcsec/pixel.
    """
    w = WCS(naxis=2)
    # 1 pixel = pixel_scale_arcsec arcsec = pixel_scale_arcsec/3600 deg
    cdelt = pixel_scale_arcsec / 3600.0
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [150.0, 2.0]
    w.wcs.crpix = [10.5, 10.5]
    w.wcs.cd = np.array([[-cdelt, 0.0], [0.0, cdelt]])
    return w


def test_calibrate_jwst_flux_magerr_formula():
    """
    Check that calibrate_JWST_flux computes magerr using the analytic
    2.5*log10(1+fluxerr/flux) expression and returns finite values.
    """
    imwcs = _make_simple_wcs(pixel_scale_arcsec=1.0)
    flux_native = np.array([1.0, 3.0])
    fluxerr_native = np.array([0.1, 0.3])

    flux, fluxerr, mag, magerr, zp = calibrate_JWST_flux(
        flux_native, fluxerr_native, imwcs
    )

    # Same formula as in cal.py
    expected_magerr = 2.5 * np.log10(1.0 + (fluxerr_native / flux_native))

    np.testing.assert_allclose(
        magerr, expected_magerr, rtol=1e-10, atol=0.0
    )

    # Basic sanity
    assert mag.shape == flux.shape == fluxerr.shape == magerr.shape
    assert np.all(np.isfinite(mag))
    assert np.all(np.isfinite(magerr))
    assert np.all(magerr > 0)



def test_jwst_mag_to_flux_density_scaling():
    """
    For density=True vs False, the flux should differ by the pixel solid angle.
    """
    imwcs = _make_simple_wcs(pixel_scale_arcsec=1.0)
    mag = np.array([25.0])  # arbitrary AB mag

    flux_density = JWST_mag_to_flux(mag, imwcs, zpsys="ab", density=True)
    flux_nondens = JWST_mag_to_flux(mag, imwcs, zpsys="ab", density=False)

    # density=True returns MJy/sr; density=False returns MJy
    # The relation is: flux_nondens ≈ flux_density * pixel_area
    pixel_scale = (
        u.deg.to(u.arcsec)
        * np.abs(imwcs.wcs.cd[0, 0])
    )  # arcsec/pixel
    pixel_area_sr = (pixel_scale * u.arcsec) ** 2
    pixel_area_sr = pixel_area_sr.to(u.sr).value

    np.testing.assert_allclose(
        flux_nondens, flux_density * pixel_area_sr, rtol=1e-6, atol=0.0
    )


def test_calibrate_hst_flux_roundtrip():
    """
    Use synthetic PHOTFLAM/PHOTPLAM to test that HST_mag_to_flux and
    calibrate_HST_flux are consistent.
    """
    prim = fits.Header()
    sci = fits.Header()
    prim["DETECTOR"] = "UVIS"
    photflam = 1e-19
    photplam = 5500.0
    sci["PHOTFLAM"] = photflam
    sci["PHOTPLAM"] = photplam

    # Pick some magnitudes, convert to flux using HST_mag_to_flux
    mags = np.array([24.0, 26.0])
    flux = HST_mag_to_flux(mags, prim, sci)
    fluxerr = flux * 0.05  # 5% errors

    flux_out, fluxerr_out, mag_out, magerr, zp = calibrate_HST_flux(
        flux, fluxerr, prim, sci
    )

    # Check that mags are recovered
    np.testing.assert_allclose(mag_out, mags, rtol=1e-6, atol=1e-6)

    # Compare zp to analytic formula
    expected_zp = -2.5 * np.log10(photflam) - 5 * np.log10(photplam) - 2.408
    assert np.allclose(zp, expected_zp)

    # magerr ≈ 1.086 * fluxerr/flux (by construction in cal.py)
    approx_magerr = 1.086 * (fluxerr / flux)
    np.testing.assert_allclose(magerr, approx_magerr, rtol=1e-6, atol=1e-6)
