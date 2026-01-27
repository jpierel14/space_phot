import numpy as np
import pytest
from astropy.wcs import WCS
from astropy.io.fits import Header
from photutils.psf import IntegratedGaussianPRF

import space_phot.photometry as photmod


def test_fast_psf_sets_psf_result_and_calibrated_columns(monkeypatch):
    # Patch calibration to be deterministic and independent of header details
    def fake_calibrate_hst_flux(flux, fluxerr, prim_header, sci_header):
        flux = np.array(flux, dtype=float)
        fluxerr = np.array(fluxerr, dtype=float)
        zp = 25.0
        # Avoid log10 issues if flux is tiny/negative in edge cases
        safe_flux = np.where(flux > 0, flux, np.nan)
        mag = -2.5 * np.log10(safe_flux) + zp
        magerr = 1.0857362047581294 * fluxerr / np.where(safe_flux > 0, safe_flux, np.nan)
        return flux, fluxerr, mag, magerr, zp

    monkeypatch.setattr(photmod, "calibrate_HST_flux", fake_calibrate_hst_flux, raising=True)

    # Create a dummy observation-like object without calling the heavy __init__
    obs = photmod.observation.__new__(photmod.observation)

    # Minimal attributes used by fast_psf
    obs.pipeline_level = 3
    obs.telescope = "hst"

    # Synthetic image
    ny, nx = 25, 25
    yy, xx = np.mgrid[:ny, :nx]

    true_x, true_y, true_flux = 12.3, 11.7, 1000.0
    psf = IntegratedGaussianPRF(sigma=1.2)
    psf.x_0 = true_x
    psf.y_0 = true_y
    psf.flux = true_flux

    data = psf(xx, yy)
    err = np.ones_like(data) * 1.0

    obs.data = data
    obs.err = err

    # WCS + headers (not used by fake calibrator, but passed through)
    obs.wcs = WCS(naxis=2)
    obs.prim_header = Header()
    obs.sci_header = Header()

    # Run fast_psf near the truth
    centers = [(true_y, true_x)]
    phot = obs.fast_psf(psf_model=IntegratedGaussianPRF(sigma=1.2), centers=centers, psf_width=9)

    # Verify result tables exist
    assert hasattr(obs, "psf_result")
    assert hasattr(obs.psf_result, "phot_table")
    assert hasattr(obs.psf_result, "phot_cal_table")

    # Verify expected columns exist
    for col in ["x_fit", "y_fit", "flux_fit", "flux_err"]:
        assert col in phot.colnames

    for col in ["flux", "fluxerr", "mag", "magerr", "zp"]:
        assert col in phot.colnames
        assert np.all(np.isfinite(phot[col])), f"{col} should be finite"

    # Sanity: fitted flux should be in the right ballpark
    assert np.isfinite(phot["flux_fit"][0])
    assert phot["flux_fit"][0] > 0
    assert abs(phot["x_fit"][0] - true_x) < 2.0
    assert abs(phot["y_fit"][0] - true_y) < 2.0
