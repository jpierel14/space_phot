# tests/test_photometry_examples.py

from pathlib import Path
import glob

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
import astropy.units as u
from photutils.psf import IntegratedGaussianPRF

from space_phot.photometry import observation2, observation3

TEST_DATA_DIR = Path(__file__).resolve().parent / "data"


# ---------------------------------------------------------------------
# Helpers to find data files
# ---------------------------------------------------------------------

def _get_level2_files(telescope: str):
    """
    Return a sorted list of level 2 exposure filenames for a given telescope.

    You should place your real data here and/or adjust the patterns below.
    """
    if telescope.lower() == "hst":
        pattern = TEST_DATA_DIR / "*flt.fits"
    elif telescope.lower() == "jwst":
        pattern = TEST_DATA_DIR / "*cal.fits"
    else:
        raise ValueError(f"Unknown telescope {telescope}")

    files = sorted(glob.glob(str(pattern)))
    return files


def _get_level3_file(telescope: str):
    """
    Return the level 3 (drizzled) filename for a given telescope.
    """
    if telescope.lower() == "hst":
        fname = TEST_DATA_DIR / "hst_level3.fits"
    elif telescope.lower() == "jwst":
        fname = TEST_DATA_DIR / "jwst_level3.fits"
    else:
        raise ValueError(f"Unknown telescope {telescope}")

    return fname if fname.exists() else None


# ---------------------------------------------------------------------
# Coordinates for the test source
#
# These are currently set to the SN position used in your docs examples:
#   RA  = 21:29:40.2103
#   Dec = +00:05:24.158
#
# If you decide to use a different target in your test data, update this.
# ---------------------------------------------------------------------

SN_COORD = SkyCoord("21:29:40.2103", "+0:05:24.158", unit=(u.hourangle, u.deg))


# ---------------------------------------------------------------------
# Level 2 APERTURE PHOTOMETRY (HST & JWST)
# ---------------------------------------------------------------------

def test_level2_aperture_hst():
    files = _get_level2_files("hst")
    if not files:
        pytest.skip("No HST level 2 files found in tests/data (hst_level2_*.fits).")

    obs = observation2(files)

    # For HST, radius is required; sky annulus recommended (same style as docs)
    obs.aperture_photometry(
        sky_location=SN_COORD,
        radius=3.0,
        skyan_in=5.0,
        skyan_out=7.0,
    )

    tab = obs.aperture_result.phot_cal_table

    # One row per exposure is the usual behavior
    assert len(tab) == len(files)

    for col in ("flux", "fluxerr", "mag", "magerr"):
        assert col in tab.colnames
        vals = np.array(tab[col])
        assert np.all(np.isfinite(vals)), f"{col} has non-finite values"

    # Basic sanity: flux and errors should be > 0 for a real source
    assert np.all(tab["fluxerr"] > 0)


def test_level2_aperture_jwst():
    files = _get_level2_files("jwst")
    if not files:
        pytest.skip("No JWST level 2 files found in tests/data (jwst_level2_*.fits).")

    obs = observation2(files)

    # For JWST, you can either specify radius or an encircled_energy string.
    # This mirrors your docs example (encircled_energy="70").
    obs.aperture_photometry(
        sky_location=SN_COORD,
        encircled_energy="70",
    )

    tab = obs.aperture_result.phot_cal_table

    assert len(tab) == len(files)

    for col in ("flux", "fluxerr", "mag", "magerr"):
        assert col in tab.colnames
        vals = np.array(tab[col])
        assert np.all(np.isfinite(vals)), f"{col} has non-finite values"

    assert np.all(tab["flux"] > 0)
    assert np.all(tab["fluxerr"] > 0)


# ---------------------------------------------------------------------
# Level 2 PSF PHOTOMETRY (HST & JWST)
# ---------------------------------------------------------------------

def _make_simple_psf_model(sigma_pix: float = 1.0):
    """
    Construct a simple analytic PSF model.

    This is *not* intended to be an accurate HST/JWST PSF — it's just a
    convenient, smooth kernel that lets us verify that the PSF machinery
    runs end-to-end and produces finite fluxes.
    """
    return IntegratedGaussianPRF(sigma=sigma_pix)


def test_level2_psf_hst():
    files = _get_level2_files("hst")
    if not files:
        pytest.skip("No HST level 2 files found in tests/data (hst_level2_*.fits).")

    obs = observation2(files)
    psf_model = _make_simple_psf_model(sigma_pix=1.0)

    # Bounds are defined *relative* to initial guesses by default.
    bounds = {
        "flux": np.array([-5e4, 5e4]),      # broad flux range
        "centroid": np.array([-2.0, 2.0]),  # +/- 2 pixels
        "bkg": np.array([-1e2, 1e2]),       # modest background range
    }

    obs.psf_photometry(
        psf_model=psf_model,
        sky_location=SN_COORD,
        fit_width=9,
        background=0.0,
        fit_flux="single",
        fit_centroid="pixel",
        fit_bkg=True,
        bounds=bounds,
        npoints=50,
        use_MLE=True,
    )

    tab = obs.psf_result.phot_cal_table

    # Single flux across all exposures -> still one row per exposure
    assert len(tab) == len(files)

    for col in ("flux", "fluxerr", "mag", "magerr"):
        assert col in tab.colnames
        vals = np.array(tab[col])
        assert np.all(np.isfinite(vals)), f"{col} has non-finite values"

    assert np.all(tab["fluxerr"] > 0)


def test_level2_psf_jwst():
    files = _get_level2_files("jwst")
    if not files:
        pytest.skip("No JWST level 2 files found in tests/data (jwst_level2_*.fits).")

    obs = observation2(files)

    # observation2 doesn't currently define flux_units in __init__, but the
    # level 2 PSF calibration path uses it for JWST. Set it explicitly here.
    #
    # If your JWST BUNIT is MJy/sr (usual for CAL images), this is appropriate.
    import astropy.units as u
    obs.flux_units = u.MJy / u.sr

    psf_model = _make_simple_psf_model(sigma_pix=1.5)

    bounds = {
        "flux": np.array([-5e4, 5e4]),
        "centroid": np.array([-2.0, 2.0]),
        "bkg": np.array([-1e2, 1e2]),
    }

    obs.psf_photometry(
        psf_model=psf_model,
        sky_location=SN_COORD,
        fit_width=9,
        background=0.0,
        fit_flux="single",
        fit_centroid="pixel",
        fit_bkg=True,
        bounds=bounds,
        npoints=50,
        use_MLE=True,
    )

    tab = obs.psf_result.phot_cal_table

    assert len(tab) == len(files)

    for col in ("flux", "fluxerr", "mag", "magerr"):
        assert col in tab.colnames
        vals = np.array(tab[col])
        assert np.all(np.isfinite(vals)), f"{col} has non-finite values"

    assert np.all(tab["flux"] > 0)
    assert np.all(tab["fluxerr"] > 0)


# ---------------------------------------------------------------------
# Level 3 APERTURE PHOTOMETRY (HST & JWST)
# ---------------------------------------------------------------------

def test_level3_aperture_hst():
    fname = _get_level3_file("hst")
    if fname is None:
        pytest.skip("No HST level 3 file found in tests/data (hst_level3.fits).")

    obs = observation3(str(fname))

    # Same aperture choices as level 2 HST example
    obs.aperture_photometry(
        sky_location=SN_COORD,
        radius=3.0,
        skyan_in=5.0,
        skyan_out=7.0,
    )

    tab = obs.aperture_result.phot_cal_table

    # Single drizzle image -> expect a single row
    assert len(tab) == 1

    for col in ("flux", "fluxerr", "mag", "magerr"):
        assert col in tab.colnames
        vals = np.array(tab[col])
        assert np.all(np.isfinite(vals)), f"{col} has non-finite values"

    assert tab["fluxerr"][0] > 0


def test_level3_aperture_jwst():
    fname = _get_level3_file("jwst")
    if fname is None:
        pytest.skip("No JWST level 3 file found in tests/data (jwst_level3.fits).")

    obs = observation3(str(fname))

    # For JWST level 3, you can use either a radius or encircled_energy;
    # here we mirror the level 2 JWST example as much as possible.
    obs.aperture_photometry(
        sky_location=SN_COORD,
        encircled_energy="70",
    )

    tab = obs.aperture_result.phot_cal_table

    assert len(tab) == 1

    for col in ("flux", "fluxerr", "mag", "magerr"):
        assert col in tab.colnames
        vals = np.array(tab[col])
        assert np.all(np.isfinite(vals)), f"{col} has non-finite values"

    assert tab["flux"][0] > 0
    assert tab["fluxerr"][0] > 0
