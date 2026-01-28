"""
PSF Photometry on Public HST and JWST Data
=========================================

This tutorial demonstrates PSF photometry with ``space_phot`` on public HST and JWST data.

The page is generated with Sphinx-Gallery, so code is executed top-to-bottom and figures
and printed outputs appear inline.

Notes
-----
- These examples may download data from MAST. If you want to avoid network access
  during documentation builds, set ``SPACE_PHOT_DOCS_NETWORK=0`` (default) and
  pre-download the files locally.
"""

import os
import glob

import matplotlib.pyplot as plt
import numpy as np

from astropy.coordinates import SkyCoord
import astropy.units as u

# Optional: network download
RUN_NETWORK = os.environ.get("SPACE_PHOT_DOCS_NETWORK", "1") == "1"
RUN_LEVEL3 = os.environ.get("SPACE_PHOT_DOCS_LEVEL3", "1") == "1"

import space_phot


# %%
# HST: locate files
# -----------------
#
# If files are not present and network is enabled, we download from MAST.
hst_obs_id = "hst_16264_12_wfc3_ir_f110w_iebc12"
sn_hst = SkyCoord("21:29:40.2110", "+0:05:24.154", unit=(u.hourangle, u.deg))


if RUN_NETWORK:
    from astroquery.mast import Observations

    obs_table = Observations.query_criteria(obs_id=hst_obs_id)
    obs_table = obs_table[obs_table["filters"] == "F110W"]

    prods = Observations.get_product_list(obs_table)

    prods = prods[prods["calib_level"] == 2]
    prods = prods[prods["productSubGroupDescription"] == "FLT"]

    Observations.download_products(prods, extension="fits")
    hst_files = sorted(space_phot.util.filter_dict_from_list(glob.glob("mastDownload/HST/*/*flt.fits"),
                            sn_hst)['F110W'])
    hst_files = [x for x in hst_files if 'skycell' not in x]
else:
    hst_files = sorted(space_phot.util.filter_dict_from_list(glob.glob("mastDownload/HST/*/*flt.fits"),
                            sn_hst)['F110W'])
    hst_files = [x for x in hst_files if 'skycell' not in x]

if len(hst_files) == 0:
    raise RuntimeError(
        "No HST files found. Either pre-download into mastDownload/ "
        "or set SPACE_PHOT_DOCS_NETWORK=1 for docs builds."
    )

print(f"HST files: {len(hst_files)}")


# %%
# Run HST PSF photometry
# ----------------------
#
# Build an observation object and a PSF model, then run PSF photometry.
obs_hst = space_phot.observation2(hst_files)

psfs_hst = space_phot.get_hst_psf(obs_hst, sn_hst)
plt.figure()
plt.imshow(psfs_hst[0].data, origin="lower")
plt.title("Example HST PSF model")
plt.colorbar()
plt.tight_layout()

obs_hst.psf_photometry(
    psfs_hst,
    sn_hst,
    bounds={"flux": [-3000, 100], "centroid": [-0.5, 0.5], "bkg": [0, 10]},
    fit_width=5,
    fit_bkg=True,
    fit_flux="single",
)

# Show diagnostics (these should create figures in your updated code)
obs_hst.plot_psf_fit()
plt.show()

obs_hst.plot_psf_posterior(minweight=0.0005)
plt.show()

print("HST calibrated PSF photometry:")
print(obs_hst.psf_result.phot_cal_table)


# %%
# JWST: locate files
# ------------------
jwst_obs_id = "jw02767-o002_t001_nircam_clear-f150w"
sn_jwst = SkyCoord("21:29:40.2103", "+0:05:24.158", unit=(u.hourangle, u.deg))


if RUN_NETWORK:
    from astroquery.mast import Observations

    obs_table = Observations.query_criteria(obs_id=jwst_obs_id)
    prods = Observations.get_product_list(obs_table)

    prods3 = prods[prods["calib_level"] == 3]
    prods3 = prods3[prods3["productSubGroupDescription"] == "I2D"]

    prods = prods[prods["calib_level"] == 2]
    prods = prods[prods["productSubGroupDescription"] == "CAL"]

    Observations.download_products(prods, extension="fits")
    Observations.download_products(prods3, extension="fits")
    jwst_files = sorted(space_phot.util.filter_dict_from_list(glob.glob("mastDownload/JWST/*/*cal.fits"),
                            sn_jwst)['F150W'])
else:
    jwst_files = sorted(space_phot.util.filter_dict_from_list(glob.glob("mastDownload/JWST/*/*cal.fits"),
                            sn_jwst)['F150W'])
if len(jwst_files) == 0:
    raise RuntimeError(
        "No JWST files found. Either pre-download into mastDownload/ "
        "or set SPACE_PHOT_DOCS_NETWORK=1 for docs builds."
    )

print(f"JWST files: {len(jwst_files)}")


# %%
# Run JWST PSF photometry
# -----------------------
obs_jwst = space_phot.observation2(jwst_files)

psfs_jwst = space_phot.get_jwst_psf(obs_jwst, sn_jwst)
plt.figure()
plt.imshow(psfs_jwst[0].data, origin="lower")
plt.title("Example JWST PSF model")
plt.colorbar()
plt.tight_layout()

obs_jwst.psf_photometry(
    psfs_jwst,
    sn_jwst,
    bounds={"flux": [-3000, 1000], "centroid": [-1.0, 1.0], "bkg": [0, 50]},
    fit_width=5,
    fit_bkg=True,
    fit_flux="single",
)

obs_jwst.plot_psf_fit()
plt.show()

obs_jwst.plot_psf_posterior(minweight=0.0005)
plt.show()

print("JWST calibrated PSF photometry:")
print(obs_jwst.psf_result.phot_cal_table)


# %%
# Level 3 JWST PSF photometry (i2d)
# ---------------------------------
#
# This section demonstrates PSF photometry on Level 3 JWST products (e.g. ``*_i2d.fits``).
# We gate it behind an env var so docs builds remain stable if Level 3 products
# are not available locally.
if RUN_LEVEL3:
    jwst_i2d_files = sorted(glob.glob("mastDownload/JWST/*/*i2d.fits"))
    if len(jwst_i2d_files) == 0:
        raise RuntimeError(
            "No JWST *_i2d.fits found for Level 3 example. "
            "Pre-download Level 3 products or disable with SPACE_PHOT_DOCS_LEVEL3=0."
        )

    print(f"JWST Level 3 files: {len(jwst_i2d_files)}")

    obs3_jwst = space_phot.observation3(jwst_i2d_files[0])

    # Prefer the level-3 PSF helper if your package has it
    psfs3_jwst = space_phot.get_jwst3_psf(obs_jwst,obs3_jwst, sn_jwst, num_psfs=4)
    
    plt.figure()
    plt.imshow(psfs3_jwst.data, origin="lower")
    plt.title("JWST Level 3 PSF model")
    plt.colorbar()
    plt.tight_layout()

    obs3_jwst.psf_photometry(
        psfs3_jwst,
        sn_jwst,
        bounds={"flux": [-5000, 5000], "centroid": [-1.0, 1.0], "bkg": [0, 200]},
        fit_width=7,
        fit_bkg=True,
        fit_flux="single",
    )

    obs3_jwst.plot_psf_fit()
    plt.show()

    obs3_jwst.plot_psf_posterior(minweight=0.0005)
    plt.show()

    print("JWST Level 3 calibrated PSF photometry:")
    print(obs3_jwst.psf_result.phot_cal_table)
else:
    print("Skipping Level 3 JWST PSF example (set SPACE_PHOT_DOCS_LEVEL3=1 to enable).")

