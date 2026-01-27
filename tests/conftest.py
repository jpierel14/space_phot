# tests/conftest.py
import pytest

@pytest.fixture(autouse=False)
def patch_calibration(monkeypatch):
    """
    Patch the calibration functions to something simple and deterministic.

    This way, we can test the photometry flow without depending on real
    JWST/HST calibration files / headers.
    """
    import space_phot.photometry as photmod

    def fake_jwst_flux(flux, fluxerr, wcs, flux_units=None):
        # Return input flux directly, and a fixed zp
        mag = -2.5 * (flux if hasattr(flux, "__len__") else flux / 1.0)  # dummy
        magerr = fluxerr * 0 + 0.01
        zp = 25.0
        return flux, fluxerr, mag, magerr, zp

    def fake_hst_flux(flux, fluxerr, prim_header, sci_header):
        mag = -2.5 * (flux if hasattr(flux, "__len__") else flux / 1.0)  # dummy
        magerr = fluxerr * 0 + 0.01
        zp = 25.0
        return flux, fluxerr, mag, magerr, zp

    monkeypatch.setattr(photmod, "calibrate_JWST_flux", fake_jwst_flux, raising=False)
    monkeypatch.setattr(photmod, "calibrate_HST_flux", fake_hst_flux, raising=False)

    yield
