"""
Planting and Recovering a PSF (JWST Injection Test)
==================================================

This tutorial plants a synthetic point source (PSF injection) into real JWST data
and then recovers it with PSF and aperture photometry.

This is a practical end-to-end sanity check for:
- PSF model behavior
- flux calibration
- agreement between methods
"""

import os
import glob

import matplotlib.pyplot as plt

from astropy.coordinates import SkyCoord
import astropy.units as u

RUN_NETWORK = os.environ.get("SPACE_PHOT_DOCS_NETWORK", "0") == "1"

import space_phot


# %%
# Download or locate JWST data
jwst_obs_id = "jw02767-o002_t001_nircam_clear-f150w"

plant_location = SkyCoord("21:29:42.4104", "+0:04:53.253", unit=(u.hourangle, u.deg))

jwst_files = sorted(glob.glob("mastDownload/JWST/*/*cal.fits"))
if (len(jwst_files) == 0) and RUN_NETWORK:
    from astroquery.mast import Observations

    obs_table = Observations.query_criteria(obs_id=jwst_obs_id)
    prods = Observations.get_product_list(obs_table)
    prods = prods[prods["calib_level"] == 2]
    prods = prods[prods["productSubGroupDescription"] == "CAL"]

    Observations.download_products(prods[:1], extension="fits")
    jwst_files = sorted(glob.glob("mastDownload/JWST/*/*cal.fits"))

if len(jwst_files) == 0:
    raise RuntimeError(
        "No JWST files found. Pre-download or set SPACE_PHOT_DOCS_NETWORK=1."
    )

print(f"JWST files: {len(jwst_files)}")


# %%
# Build PSF models
obs = space_phot.observation2(jwst_files)
psfs = space_phot.get_jwst_psf(obs, plant_location)

plt.figure()
plt.imshow(psfs[0].data, origin="lower")
plt.title("JWST PSF model")
plt.colorbar()
plt.tight_layout()


# %%
# Plant a source and recover it
#
# Plant a magnitude-26 source (adjust as desired).
obs.plant_psf([psfs], [plant_location], 26)

# `plant_psf` should write new files or update an internal list depending on your implementation.
# Here we assume it writes "*plant.fits" products in the same download tree.
planted_files = sorted(glob.glob("mastDownload/JWST/*/*plant.fits"))
if len(planted_files) == 0:
    raise RuntimeError("No planted files found. Check plant_psf output behavior/paths.")

obs2 = space_phot.observation2(planted_files)

# PSF photometry at the planted position
obs2.psf_photometry(
    psfs,
    plant_location,
    bounds={"flux": [-3000, 1000], "centroid": [-1, 1], "bkg": [0, 50]},
    fit_width=5,
    fit_bkg=True,
    fit_flux="single",
)

obs2.plot_psf_fit()
plt.show()

print("Recovered PSF photometry:")
print(obs2.psf_result.phot_cal_table)

# Aperture photometry at the same location for comparison
obs2.aperture_photometry(
    plant_location,
    encircled_energy="50",
)

print("Recovered aperture photometry:")
print(obs2.aperture_result.phot_cal_table)
