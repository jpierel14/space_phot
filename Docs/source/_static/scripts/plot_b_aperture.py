"""
Aperture Photometry on Public HST and JWST Data
==============================================

"""


"""
Aperture Photometry

Example: download public HST + JWST data from MAST and measure aperture photometry with space_phot.
"""

from __future__ import annotations

from pathlib import Path
import glob

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

    # HST (same dataset as your current example) :contentReference[oaicite:6]{index=6}
    hst_obs_id = "hst_16264_12_wfc3_ir_f110w_iebc12"
    sn_hst = SkyCoord("21:29:40.2110", "+0:05:24.154", unit=(u.hourangle, u.deg))

    hst_files = glob.glob(str(outdir / "HST" / "*" / "*flt.fits"))
    if len(hst_files) == 0:
        obs_table = Observations.query_criteria(obs_id=hst_obs_id)
        obs_table = obs_table[obs_table["filters"] == "F110W"]
        prods = Observations.get_product_list(obs_table)
        prods = prods[prods["calib_level"] == 2]
        prods = prods[prods["productSubGroupDescription"] == "FLT"][:3]
        Observations.download_products(prods, extension="fits")

    hst_files = sorted(glob.glob(str(outdir / "HST" / "*" / "*flt.fits")))
    show_image_and_zoom(hst_files[0], sn_hst, title="HST FLT (SN2022riv)")

    hst_obs = space_phot.observation2(hst_files)
    hst_obs.aperture_photometry(sn_hst, radius=3, skyan_in=5, skyan_out=7)
    print(hst_obs.aperture_result.phot_cal_table)

    # JWST (same dataset as your current example) :contentReference[oaicite:7]{index=7}
    jwst_obs_id = "jw02767-o002_t001_nircam_clear-f150w"
    sn_jwst = SkyCoord("21:29:40.2103", "+0:05:24.158", unit=(u.hourangle, u.deg))

    jwst_files = glob.glob(str(outdir / "JWST" / "*" / "*cal.fits"))
    if len(jwst_files) == 0:
        obs_table = Observations.query_criteria(obs_id=jwst_obs_id)
        prods = Observations.get_product_list(obs_table)
        prods = prods[prods["calib_level"] == 2]
        prods = prods[prods["productSubGroupDescription"] == "CAL"]

        # Keep your NRCB3-only selection :contentReference[oaicite:8]{index=8}
        keep = [i for i in range(len(prods)) if str(prods[i]["obs_id"]).endswith("nrcb3")]
        prods = prods[keep]
        Observations.download_products(prods[:4], extension="fits")

    jwst_files = sorted(glob.glob(str(outdir / "JWST" / "*" / "*cal.fits")))
    show_image_and_zoom(jwst_files[0], sn_jwst, title="JWST CAL (SN2022riv)")

    jwst_obs = space_phot.observation2(jwst_files)
    jwst_obs.aperture_photometry(sn_jwst, encircled_energy="70")
    print(jwst_obs.aperture_result.phot_cal_table)


if __name__ == "__main__":
    main()
