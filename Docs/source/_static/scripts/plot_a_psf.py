"""
PSF Photometry on Public HST and JWST Data
=========================================

"""


"""
PSF Photometry

Example: download public HST + JWST data from MAST and measure PSF photometry with space_phot.
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
from astropy.visualization import simple_norm

from astroquery.mast import Observations

import space_phot


def download_products(obs_table, calib_level: int, subgroup: str, nmax: int | None = None):
    prods = Observations.get_product_list(obs_table)
    prods = prods[prods["calib_level"] == calib_level]
    prods = prods[prods["productSubGroupDescription"] == subgroup]
    if nmax is not None:
        prods = prods[:nmax]
    Observations.download_products(prods, extension="fits")


def show_image_and_zoom(fname: str, sky: SkyCoord, title: str, cutout_size: int = 11):
    with fits.open(fname) as hdul:
        data = hdul["SCI", 1].data
        w = WCS(hdul["SCI", 1].header, hdul)

        norm = simple_norm(data, stretch="linear", min_cut=-1, max_cut=10)
        plt.figure()
        plt.imshow(data, origin="lower", norm=norm, cmap="gray")
        plt.gca().tick_params(labelcolor="none", axis="both", color="none")
        plt.title(title)
        plt.show()

        y, x = skycoord_to_pixel(sky, w)
        cut = extract_array(data, (cutout_size, cutout_size), (x, y))
        norm2 = simple_norm(cut, stretch="linear", min_cut=-1, max_cut=10)

        plt.figure()
        plt.imshow(cut, origin="lower", norm=norm2, cmap="gray")
        plt.gca().tick_params(labelcolor="none", axis="both", color="none")
        plt.title(f"{title} (zoom)")
        plt.show()


def main():
    outdir = Path("mastDownload")

    # ----------------
    # HST PSF example
    # ----------------
    # Matches your current demo (SN 2022riv, 3 FLT images) :contentReference[oaicite:3]{index=3}
    hst_obs_id = "hst_16264_12_wfc3_ir_f110w_iebc12"
    sn_hst = SkyCoord("21:29:40.2110", "+0:05:24.154", unit=(u.hourangle, u.deg))

    hst_files = glob.glob(str(outdir / "HST" / "*" / "*flt.fits"))
    if len(hst_files) == 0:
        obs_table = Observations.query_criteria(obs_id=hst_obs_id)
        obs_table = obs_table[obs_table["filters"] == "F110W"]
        download_products(obs_table, calib_level=2, subgroup="FLT", nmax=3)

    hst_files = sorted(glob.glob(str(outdir / "HST" / "*" / "*flt.fits")))
    show_image_and_zoom(hst_files[0], sn_hst, title="HST FLT (SN2022riv)")

    hst_obs = space_phot.observation2(hst_files)
    psfs = space_phot.get_hst_psf(hst_obs, sn_hst)

    plt.figure()
    plt.imshow(psfs[0].data, origin="lower")
    plt.title("HST PSF model")
    plt.show()

    hst_obs.psf_photometry(
        psfs,
        sn_hst,
        bounds={"flux": [-3000, 100], "centroid": [-0.5, 0.5], "bkg": [0, 10]},
        fit_width=5,
        fit_bkg=True,
        fit_flux="single",
    )
    hst_obs.plot_psf_fit()
    plt.show()

    hst_obs.plot_psf_posterior(minweight=0.0005)
    plt.show()

    print(hst_obs.psf_result.phot_cal_table)

    # Optional: flux per exposure (kept from your script) :contentReference[oaicite:4]{index=4}
    hst_obs.psf_photometry(
        psfs,
        sn_hst,
        bounds={"flux": [-3000, 100], "centroid": [-0.5, 0.5], "bkg": [0, 10]},
        fit_width=5,
        fit_bkg=True,
        fit_flux="multi",
    )
    hst_obs.plot_psf_fit()
    plt.show()

    hst_obs.plot_psf_posterior(minweight=0.0005)
    plt.show()

    print(hst_obs.psf_result.phot_cal_table)

    # -----------------
    # JWST PSF example
    # -----------------
    jwst_obs_id = "jw02767-o002_t001_nircam_clear-f150w"
    sn_jwst = SkyCoord("21:29:40.2103", "+0:05:24.158", unit=(u.hourangle, u.deg))

    jwst_files = glob.glob(str(outdir / "JWST" / "*" / "*cal.fits"))
    if len(jwst_files) == 0:
        obs_table = Observations.query_criteria(obs_id=jwst_obs_id)
        prods = Observations.get_product_list(obs_table)
        prods = prods[prods["calib_level"] == 2]
        prods = prods[prods["productSubGroupDescription"] == "CAL"]

        # Keep your original selection logic: only NRCB3 exposures :contentReference[oaicite:5]{index=5}
        keep = [i for i in range(len(prods)) if str(prods[i]["obs_id"]).endswith("nrcb3")]
        prods = prods[keep]
        Observations.download_products(prods[:4], extension="fits")

    jwst_files = sorted(glob.glob(str(outdir / "JWST" / "*" / "*cal.fits")))
    show_image_and_zoom(jwst_files[0], sn_jwst, title="JWST CAL (SN2022riv)")

    jwst_obs = space_phot.observation2(jwst_files)
    psfs = space_phot.get_jwst_psf(jwst_obs, sn_jwst)

    plt.figure()
    plt.imshow(psfs[0].data, origin="lower")
    plt.title("JWST PSF model")
    plt.show()

    jwst_obs.psf_photometry(
        psfs,
        sn_jwst,
        bounds={"flux": [-1000, 1000], "centroid": [-2, 2], "bkg": [0, 50]},
        fit_width=5,
        fit_bkg=True,
        fit_flux="single",
    )
    jwst_obs.plot_psf_fit()
    plt.show()

    jwst_obs.plot_psf_posterior(minweight=0.0005)
    plt.show()

    print(jwst_obs.psf_result.phot_cal_table)

    # -----------------
    # Level 3 JWST PSF
    # -----------------
    i2d_files = glob.glob(str(outdir / "JWST" / "*" / "*i2d.fits"))
    if len(i2d_files) == 0:
        obs_table = Observations.query_criteria(obs_id=jwst_obs_id)
        prods = Observations.get_product_list(obs_table)
        prods = prods[prods["calib_level"] == 3]
        prods = prods[prods["productSubGroupDescription"] == "I2D"]
        Observations.download_products(prods[0], extension="fits")

    i2d_files = sorted(glob.glob(str(outdir / "JWST" / "*" / "*i2d.fits")))
    show_image_and_zoom(i2d_files[0], sn_jwst, title="JWST I2D (SN2022riv level 3)")

    jwst3_obs = space_phot.observation3(i2d_files[0])
    psf3 = space_phot.get_jwst3_psf(jwst_obs, jwst3_obs, sn_jwst, num_psfs=4)

    plt.figure()
    plt.imshow(psf3.data, origin="lower")
    plt.title("JWST Level 3 PSF model")
    plt.show()

    jwst3_obs.psf_photometry(
        psf3,
        sn_jwst,
        bounds={"flux": [-1000, 1000], "centroid": [-2, 2], "bkg": [0, 50]},
        fit_width=5,
        fit_bkg=True,
        fit_flux=True,
    )
    jwst3_obs.plot_psf_fit()
    plt.show()

    jwst3_obs.plot_psf_posterior(minweight=0.0005)
    plt.show()


if __name__ == "__main__":
    main()
