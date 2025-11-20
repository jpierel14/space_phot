# tests/test_util_core.py

from pathlib import Path
import glob

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u

from space_phot.util import (
    mjd_dict_from_list,
    filter_dict_from_list,
    simple_aperture_sum,
    generic_aperture_phot,
)


def test_mjd_dict_from_list_groups_by_rounded_mjd(tmp_path):
    """
    mjd_dict_from_list should group filenames by rounded MJD-AVG / EXPSTART.
    """
    f1 = tmp_path / "f1.fits"
    f2 = tmp_path / "f2.fits"
    f3 = tmp_path / "f3.fits"

    for fname, mjd in ((f1, 60000.1234), (f2, 60000.1267), (f3, 60001.5)):
        hdu = fits.PrimaryHDU()
        hdu.header["MJD-AVG"] = mjd
        hdu.writeto(fname, overwrite=True)

    filelist = [str(f1), str(f2), str(f3)]

    # Round to 1 decimal so f1 and f2 share the same key, f3 remains separate.
    mjd_dict = mjd_dict_from_list(filelist, tolerance=1)

    assert len(mjd_dict) == 2
    sizes = [len(v) for v in mjd_dict.values()]
    assert sorted(sizes) == [1, 2]



def test_filter_dict_from_list_groups_by_filter(tmp_path):
    """
    filter_dict_from_list should group files by their FILTER keyword.
    """
    f1 = tmp_path / "f1.fits"
    f2 = tmp_path / "f2.fits"
    f3 = tmp_path / "f3.fits"

    for fname, filt in ((f1, "F150W"), (f2, "F150W"), (f3, "F200W")):
        phdu = fits.PrimaryHDU()
        phdu.header["FILTER"] = filt
        phdu.writeto(fname, overwrite=True)

    filelist = [str(f1), str(f2), str(f3)]

    filt_dict = filter_dict_from_list(filelist, sky_location=None, ext=0)

    assert set(filt_dict.keys()) == {"F150W", "F200W"}
    assert len(filt_dict["F150W"]) == 2
    assert len(filt_dict["F200W"]) == 1


def test_simple_aperture_sum_picks_central_source():
    """
    With a single bright pixel at the center and radius>0, the sum should
    equal that pixel regardless of the exact circle area.
    """
    ny, nx = 11, 11
    data = np.zeros((ny, nx), dtype=float)
    y0, x0 = ny // 2, nx // 2
    data[y0, x0] = 42.0

    positions = [[x0, y0]]  # note util.simple_aperture_sum uses [x, y] ordering internally
    radius = 3.0

    aper_sum = simple_aperture_sum(data, positions, radius)
    assert np.isclose(aper_sum[0], 42.0)


def test_generic_aperture_phot_background_and_flux():
    """
    Check that generic_aperture_phot recovers the background median and
    a roughly correct background-subtracted flux.
    """
    ny, nx = 41, 41
    data = np.zeros((ny, nx), dtype=float) + 5.0  # uniform background
    y0, x0 = ny // 2, nx // 2
    data[y0, x0] += 100.0  # central 'source' pixel

    positions = [(x0, y0)]  # (x, y) order
    sky = {"sky_in": 8.0, "sky_out": 12.0}
    radius = 3.0

    phot = generic_aperture_phot(
        data,
        positions,
        radius=radius,
        sky=sky,
        epadu=1,
        error=None,
    )

    # annulus_median ≈ 5.0 everywhere
    assert "annulus_median" in phot.colnames
    assert np.allclose(phot["annulus_median"], 5.0, atol=1e-3)

    # Background-subtracted flux should be close to 100 (all in center pixel)
    assert "aper_sum_bkgsub" in phot.colnames
    flux = phot["aper_sum_bkgsub"][0]
    assert flux > 0
    # We're not expecting exact 100 due to exact aperture geometry, but it
    # should be in the same ballpark.
    assert abs(flux - 100.0) < 5.0

    # Error should be finite and > 0
    assert "aperture_sum_err" in phot.colnames
    assert phot["aperture_sum_err"][0] > 0
    assert np.isfinite(phot["aperture_sum_err"][0])
