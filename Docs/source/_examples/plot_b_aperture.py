"""
Aperture Photometry on Public HST and JWST Data
===============================================

This tutorial demonstrates aperture photometry with ``space_phot`` on public HST and JWST data.

The JWST path uses the distortion-corrected (PAM-applied) data products in level-2
processing (as used by ``space_phot``).
"""

import os
import glob

import matplotlib.pyplot as plt

from astropy.coordinates import SkyCoord
import astropy.units as u

RUN_NETWORK = os.environ.get("SPACE_PHOT_DOCS_NETWORK", "0") == "1"
RUN_LEVEL3 = os.environ.get("SPACE_PHOT_DOCS_LEVEL3", "0") == "1"

import space_phot


# %%
# HST: download or locate files
hst_obs_id = "hst_16264_12_wfc3_ir_f110w_iebc12"
sn_hst = SkyCoord("21:29:40.2110", "+0:05:24.154", unit=(u.hourangle, u.deg))

hst_files = sorted(glob.glob("mastDownload/HST/*/*flt.fits"))
if (len(hst_files) == 0) and RUN_NETWORK:
    from astroquery.mast import Observations

    obs_table = Observations.query_criteria(obs_id=hst_obs_id)
    obs_table = obs_table[obs_table["filters"] == "F110W"]

    prods = Observations.get_product_list(obs_table)
    prods = prods[prods["calib_level"] == 2]
    prods = prods[prods["productSubGroupDescription"] == "FLT"]

    Observations.download_products(prods[:3], extension="fits")
    hst_files = sorted(glob.glob("mastDownload/HST/*/*flt.fits"))

if len(hst_files) == 0:
    raise RuntimeError(
        "No HST files found. Pre-download or set SPACE_PHOT_DOCS_NETWORK=1."
    )

print(f"HST files: {len(hst_files)}")


# %%
# HST aperture photometry
obs_hst = space_phot.observation2(hst_files)

# Example: fixed pixel aperture + sky annulus
obs_hst.aperture_photometry(
    sn_hst,
    radius=3,
    skyan_in=5,
    skyan_out=7,
)

print("HST calibrated aperture photometry:")
print(obs_hst.aperture_result.phot_cal_table)


# %%
# JWST: download or locate files
jwst_obs_id = "jw02767-o002_t001_nircam_clear-f150w"
sn_jwst = SkyCoord("21:29:40.2103", "+0:05:24.158", unit=(u.hourangle, u.deg))

jwst_files = sorted(glob.glob("mastDownload/JWST/*/*cal.fits"))
if (len(jwst_files) == 0) and RUN_NETWORK:
    from astroquery.mast import Observations

    obs_table = Observations.query_criteria(obs_id=jwst_obs_id)
    prods = Observations.get_product_list(obs_table)
    prods = prods[prods["calib_level"] == 2]
    prods = prods[prods["productSubGroupDescription"] == "CAL"]

    Observations.download_products(prods[:4], extension="fits")
    jwst_files = sorted(glob.glob("mastDownload/JWST/*/*cal.fits"))

if len(jwst_files) == 0:
    raise RuntimeError(
        "No JWST files found. Pre-download or set SPACE_PHOT_DOCS_NETWORK=1."
    )

print(f"JWST files: {len(jwst_files)}")


# %%
# JWST aperture photometry
obs_jwst = space_phot.observation2(jwst_files)

# Example: use JWST aperture correction by EE (preferred)
obs_jwst.aperture_photometry(
    sn_jwst,
    encircled_energy="70",
)

print("JWST calibrated aperture photometry:")
print(obs_jwst.aperture_result.phot_cal_table)

# %%
# Level 3 HST aperture photometry (drz/drc)
# -----------------------------------------
#
# Level 3 HST products are typically drizzled: ``*_drz.fits`` or ``*_drc.fits``.

hst_lvl3_files = sorted(glob.glob("mastDownload/HST/*/*dr?.fits"))
# glob pattern *dr?.fits matches *drz.fits and *drc.fits

if len(hst_lvl3_files) == 0:
    raise RuntimeError(
        "No HST Level 3 (*_drz.fits or *_drc.fits) found. "
        "Pre-download Level 3 products or disable with SPACE_PHOT_DOCS_LEVEL3=0."
    )

print(f"HST Level 3 files: {len(hst_lvl3_files)}")

obs3_hst = space_phot.observation3(hst_lvl3_files[0])

obs3_hst.aperture_photometry(
    sn_hst,
    radius=3,
    skyan_in=5,
    skyan_out=7,
)

print("HST Level 3 calibrated aperture photometry:")
print(obs3_hst.aperture_result.phot_cal_table)

# %%
# Level 3 JWST aperture photometry (i2d)
# --------------------------------------

jwst_i2d_files = sorted(glob.glob("mastDownload/JWST/*/*i2d.fits"))

if len(jwst_i2d_files) == 0:
    raise RuntimeError(
        "No JWST Level 3 (*_i2d.fits) found. "
        "Pre-download Level 3 products or disable with SPACE_PHOT_DOCS_LEVEL3=0."
    )

print(f"JWST Level 3 files: {len(jwst_i2d_files)}")

obs3_jwst = space_phot.observation3(jwst_i2d_files[0])

# Keep the interface consistent: use EE-based aperture correction
obs3_jwst.aperture_photometry(
    sn_jwst,
    encircled_energy="70",
)

print("JWST Level 3 calibrated aperture photometry:")
print(obs3_jwst.aperture_result.phot_cal_table)

