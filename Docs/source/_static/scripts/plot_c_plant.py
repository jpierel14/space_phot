"""
Plant a PSF and Recover Photometry
==================================

Example: download one JWST CAL image, plant a PSF at a blank location,
then measure PSF + aperture photometry on the planted source.
"""

"""
Planting PSFs

Example: download one JWST CAL image, plant a PSF at a blank location,
then measure PSF + aperture photometry on the planted source.
"""

from __future__ import annotations

from pathlib import Path
import glob

import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.nddata import extract_array
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel

from astroquery.mast import Observations

import space_phot


def main():
    outdir = Path("mastDownload")

    jwst_obs_id = "jw02767-o002_t001_nircam_clear-f150w"
    # Match your chosen blank-ish region :contentReference[oaicite:9]{index=9}
    plant_location = SkyCoord("21:29:42.4104", "+0:04:53.253", unit=(u.hourangle, u.deg))

    # Download one CAL exposure (same selection as your script) :contentReference[oaicite:10]{index=10}
    files = glob.glob(str(outdir / "JWST" / "jw02767002001_02103_00001_nrcb3" / "*cal.fits"))
    if len(files) == 0:
        obs_table = Observations.query_criteria(obs_id=jwst_obs_id)
        prods = Observations.get_product_list(obs_table)
        prods = prods[prods["obs_id"] == "jw02767002001_02103_00001_nrcb3"]
        prods = prods[prods["calib_level"] == 2]
        prods = prods[prods["productSubGroupDescription"] == "CAL"]
        Observations.download_products(prods, extension="fits")

    files = sorted(glob.glob(str(outdir / "JWST" / "jw02767002001_02103_00001_nrcb3" / "*cal.fits")))
    print(files)

    jwst_obs = space_phot.observation2(files)

    # Build PSF model(s) at the location
    psfs = space_phot.get_jwst_psf(jwst_obs, plant_location)
    psf_stamp = extract_array(psfs[0].data, (9, 9), (psfs[0].data.shape[1] / 2, psfs[0].data.shape[0] / 2))

    plt.figure()
    plt.imshow(psf_stamp, origin="lower")
    plt.title("JWST PSF stamp")
    plt.show()

    # Examine the image at the plant location
    plant_image = files[0]
    with fits.open(plant_image) as hdul:
        data = hdul["SCI", 1].data
        w = WCS(hdul["SCI", 1].header, hdul)

        y, x = skycoord_to_pixel(plant_location, w)
        cut = extract_array(data, (9, 9), (x, y))

    plt.figure()
    plt.imshow(cut, origin="lower")
    plt.title("Pre-plant cutout")
    plt.gca().tick_params(labelcolor="none", axis="both", color="none")
    plt.show()

    # Plant the PSF (same call as your script) :contentReference[oaicite:11]{index=11}
    jwst_obs.plant_psf([psfs], [[x, y]], 26)

    planted_image = plant_image.replace(".fits", "_plant.fits")
    with fits.open(planted_image) as hdul:
        planted_data = hdul["SCI", 1].data
        planted_cut = extract_array(planted_data, (9, 9), (x, y))

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(cut, origin="lower")
    axes[0].set_title("Pre-Plant")
    axes[1].imshow(planted_cut, origin="lower")
    axes[1].set_title("Post-Plant")
    for ax in axes:
        ax.tick_params(labelcolor="none", axis="both", color="none")
    plt.tight_layout()
    plt.show()

    # Measure PSF photometry + aperture photometry on planted source
    jwst_obs2 = space_phot.observation2(
        glob.glob(str(outdir / "JWST" / "jw02767002001_02103_00001_nrcb3" / "*plant.fits"))
    )

    jwst_obs2.psf_photometry(
        psfs,
        plant_location,
        bounds={"flux": [-3000, 1000], "centroid": [-1, 1], "bkg": [0, 50]},
        fit_width=5,
        fit_bkg=True,
        fit_flux="single",
    )
    jwst_obs2.plot_psf_fit()
    plt.show()

    jwst_obs2.plot_psf_posterior(minweight=0.0005)
    plt.show()

    print("PSF Mag:", float(jwst_obs2.psf_result.phot_cal_table["mag"]))

    jwst_obs2.aperture_photometry(plant_location, encircled_energy="50")
    print("Aperture Mag:", float(jwst_obs2.aperture_result.phot_cal_table["mag"]))


if __name__ == "__main__":
    main()
