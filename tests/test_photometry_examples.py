# tests/test_photometry_examples.py

from pathlib import Path
import glob,pdb

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
import astropy.units as u
from photutils.psf import IntegratedGaussianPRF

import space_phot
from space_phot.photometry import observation2, observation3

import warnings
warnings.simplefilter('ignore')

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


def _assert_psf_table_ok(tab):
    for col in ["flux", "fluxerr", "mag", "magerr"]:
        assert col in tab.colnames
        assert np.all(np.isfinite(tab[col])), f"{col} has non-finite values"

    assert np.all(tab["fluxerr"] > 0), "fluxerr must be > 0"
    assert np.all(tab["magerr"] >= 0), "magerr must be >= 0"


def _magdiff(m1, m2):
    return np.abs(np.array(m1, dtype=float) - np.array(m2, dtype=float))

@pytest.mark.parametrize("telescope", ["hst", "jwst"])
def test_level2_psf_smoke_and_consistency_with_aperture(telescope):
    # You already have fixtures for real data + skycoord from the aperture tests.
    # example_files_level2 should return the list of filenames for that telescope.
    files = _get_level2_files(telescope)
    if not files:
        pytest.skip("No %s level 2 files found in tests/data."%telescope)

    obs = observation2(files)

    if telescope=='jwst':
        SN_COORD = SkyCoord("21:29:40.0650", "+0:05:25.202", unit=(u.hourangle, u.deg))
        psf = space_phot.get_jwst_psf(obs,SN_COORD)
    else:
        SN_COORD = SkyCoord("21:29:40.0628", "+0:05:25.218", unit=(u.hourangle, u.deg))
        psf = space_phot.get_hst_psf(obs,SN_COORD)

    # Run aperture (baseline)
    ap = obs.aperture_photometry(SN_COORD, radius=3, skyan_in=9, skyan_out=12,encircled_energy=70)  # match your docs/test style
    ap_tab = ap.phot_cal_table

    
    

    # Keep PSF fit simple/robust for CI: fixed centroid, single flux.
    res = obs.psf_photometry(
        psf,
        sky_location=SN_COORD,
        bounds={'flux':(-1000,1000),'centroid':(-3,3),'bkg':(-100,100)},
        fit_width=11,
        fit_flux="single",
        fit_centroid="pixel",
        background=[np.nanmedian(obs.data_arr_pam[i]) for i in range(obs.n_exposures)],
        fit_bkg=False,
        npoints=100,
        maxiter=None,
    )
    psf_tab = res.phot_cal_table


    _assert_psf_table_ok(psf_tab)
    
    # Loose agreement with aperture (per exposure)
    # If your psf_photometry returns one row per exposure, this compares directly.
    assert len(psf_tab) == len(ap_tab)

    dm = _magdiff(psf_tab["mag"], ap_tab["mag"])
    #import pdb
    #pdb.set_trace()
    assert np.nanmax(dm) < 1.0, f"PSF vs aperture disagree too much (max Δmag={np.nanmax(dm):.3f})"


@pytest.mark.parametrize("telescope", ["hst", "jwst"])
@pytest.mark.slow
def test_level3_psf_smoke_and_consistency_with_aperture(telescope):
    # level3: probably a single drizzled file for each telescope
    fname = _get_level3_file(telescope)
    if fname is None:
        pytest.skip("No %s level 3 file found in tests/data."%telescope)
    obs = observation3(fname)
    files = _get_level2_files(telescope)
    obs2 = observation2(files)
    if telescope=='jwst':
        SN_COORD = SkyCoord("21:29:40.0650", "+0:05:25.202", unit=(u.hourangle, u.deg))
        psf = space_phot.get_jwst3_psf(obs2,obs,SN_COORD)
    else:
        SN_COORD = SkyCoord("21:29:40.0628", "+0:05:25.218", unit=(u.hourangle, u.deg))
        psf = space_phot.get_hst3_psf(obs2,obs,SN_COORD)

    ap = obs.aperture_photometry(SN_COORD, radius=3, skyan_in=9, skyan_out=12,encircled_energy=70)
    ap_tab = ap.phot_cal_table
    

    res = obs.psf_photometry(
        psf,
        sky_location=SN_COORD,
        fit_width=9,
        bounds={'flux':(-1000,1000),'centroid':(-3,3),'bkg':(-100,100)},
        fit_flux=True,
        fit_centroid=True,
        fit_bkg=True,
        npoints=100,
        maxiter=None,
    )
    psf_tab = res.phot_cal_table
    _assert_psf_table_ok(psf_tab)

    # level3 likely produces a single row
    assert len(psf_tab) == len(ap_tab) == 1
    assert float(_magdiff(psf_tab["mag"][0], ap_tab["mag"][0])) < 1.0

# ---------------------------------------------------------------------
# Level 2 APERTURE PHOTOMETRY (HST & JWST)
# ---------------------------------------------------------------------

def test_level2_aperture_hst():

    SN_COORD = SkyCoord("21:29:40.0628", "+0:05:25.218", unit=(u.hourangle, u.deg))
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
    SN_COORD = SkyCoord("21:29:40.0650", "+0:05:25.202", unit=(u.hourangle, u.deg))
    

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




def test_level2_psf_hst():
    files = _get_level2_files("hst")
    if not files:
        pytest.skip("No HST level 2 files found in tests/data (hst_level2_*.fits).")

    obs = observation2(files)
    

    SN_COORD = SkyCoord("21:29:40.0694", "+0:05:25.240", unit=(u.hourangle, u.deg))
    psf_model = space_phot.get_hst_psf(obs,SN_COORD)
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
    SN_COORD = SkyCoord("21:29:40.0650", "+0:05:25.202", unit=(u.hourangle, u.deg))
    
    # observation2 doesn't currently define flux_units in __init__, but the
    # level 2 PSF calibration path uses it for JWST. Set it explicitly here.
    #
    # If your JWST BUNIT is MJy/sr (usual for CAL images), this is appropriate.
    
    obs.flux_units = u.MJy / u.sr

    psf_model = space_phot.get_jwst_psf(obs,SN_COORD)

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
        use_MLE=False,
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
    SN_COORD = SkyCoord("21:29:40.0694", "+0:05:25.240", unit=(u.hourangle, u.deg))
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
    SN_COORD = SkyCoord("21:29:40.0650", "+0:05:25.202", unit=(u.hourangle, u.deg))
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

if __name__=='__main__':
    #test_level2_aperture_hst()
    test_level2_psf_smoke_and_consistency_with_aperture('hst')
    #test_level3_psf_smoke_and_consistency_with_aperture('hst')