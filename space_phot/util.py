from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

try:
    import crds
except Exception as e:
    crds = None

import warnings
import os
import glob
import shutil
import urllib.request

import numpy as np
import scipy
import stpsf
import sncosmo

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import astropy
from astropy import units as u
from astropy import wcs
from astropy.io import fits
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clipped_stats
from astropy.time import Time
from astropy.nddata import extract_array
from astropy.wcs.utils import skycoord_to_pixel, pixel_to_skycoord

import gwcs
from gwcs.utils import make_fitswcs_transform
from gwcs import coordinate_frames as cf
from astropy import coordinates as coord
from asdf import AsdfFile

import photutils
from photutils.aperture import (
    CircularAperture,
    CircularAnnulus,
    aperture_photometry,
)
from photutils.psf import EPSFModel

import jwst
#from jwst import datamodels
from jwst.pipeline import Detector1Pipeline, Image2Pipeline, Image3Pipeline
from jwst.associations import asn_from_list
from jwst.associations.lib.rules_level2_base import DMSLevel2bBase
from jwst.associations.lib.rules_level3_base import DMS_Level3_Base
from jwst import datamodels

#from jwst.pipeline import Detector1Pipeline, Image2Pipeline, Image3Pipeline
#from jwst.associations import asn_from_list
#from jwst.associations.lib.rules_level3_base import DMS_Level3_Base

from .wfc3_photometry.psf_tools.PSFUtils import make_models
from .wfc3_photometry.psf_tools.PSFPhot import get_standard_psf

__all__ = ['get_jwst_psf','get_hst_psf','get_jwst3_psf','get_hst3_psf','get_jwst_psf_grid',
            'get_jwst_psf_from_grid']


def mjd_dict_from_list(filelist, tolerance=0):
    """
    Group a list of FITS files by their observation date.

    Files are grouped by rounding their MJD keyword (MJD-AVG or EXPSTART)
    to a given number of decimal places.

    Parameters
    ----------
    filelist : list of str
        List of FITS filenames.
    tolerance : int, optional
        Number of decimal places to keep when rounding the MJD. Files whose
        rounded MJD match are grouped together.

    Returns
    -------
    mjd_dict : dict
        Dictionary mapping rounded MJD values to lists of filenames.
    """
    mjd_dict = {}
    for fname in filelist:
        with fits.open(fname) as dat:
            try:
                mjd = dat[0].header["MJD-AVG"]
            except Exception:
                mjd = dat[0].header["EXPSTART"]

        mjd_key = np.round(mjd, tolerance)
        mjd_dict.setdefault(mjd_key, []).append(fname)

    return mjd_dict


def filter_dict_from_list(
    filelist,
    sky_location=None,
    ext=1,
    buffer=0.0,
):
    """
    Group FITS files by FILTER keyword, with an optional check that a
    given sky position lies on the detector and is at least `buffer`
    pixels away from its edges.

    Parameters
    ----------
    filelist : list of str
        FITS filenames.
    sky_location : astropy.coordinates.SkyCoord or None, optional
        If provided, only include files in which this sky position can
        be transformed to valid detector pixel coordinates.
    ext : int, optional
        FITS extension to read the WCS / FILTER keyword from.
    buffer : float, optional
        Required minimum distance (in pixels) between the derived pixel
        coordinate and all image edges. Default: 0 (no edge exclusion).

        A file is only included if:
            buffer <= x <= nx - 1 - buffer
            buffer <= y <= ny - 1 - buffer

    Returns
    -------
    filt_dict : dict
        Dictionary mapping filter name → list of filenames.
    """
    filt_dict = {}

    for fname in filelist:
        try:
            with fits.open(fname) as hdul:
                header = hdul[ext].header
                filt = header.get("FILTER")
                if filt is None and ext==1:
                    filt = hdul[0].header.get('FILTER')    
                    if filt is None:
                        continue

                # If sky check requested…
                if sky_location is not None:
                    try:
                        w = wcs.WCS(header)
                        x, y = w.world_to_pixel(sky_location)
                    except Exception:
                        # WCS transform failed → exclude file
                        continue
                    # Reject NaNs or infs
                    if not np.isfinite(x) or not np.isfinite(y):
                        continue

                    # Image shape for bounds check
                    ny, nx = hdul[ext].data.shape

                    # Pixel must lie inside detector and outside edge buffer
                    if (
                        x < buffer
                        or y < buffer
                        or x > nx - 1 - buffer
                        or y > ny - 1 - buffer
                    ):
                        continue

                # Passed all checks → add file
                filt_dict.setdefault(filt, []).append(fname)

        except Exception:
            # unreadable FITS, bad header, etc — ignore
            continue

    return filt_dict



def stpsf_setup_sim_to_match_file(filename_or_HDUList, verbose=True, plot=False,dateobs=None):
    """ Setup a stpsf Instrument instance matched to a given
    """
    if isinstance(filename_or_HDUList,str):
        if verbose:
            print(f"Setting up sim to match {filename_or_HDUList}")
        header = fits.getheader(filename_or_HDUList)
    else:
        header = filename_or_HDUList[0].header
        if verbose:
            print(f"Setting up sim to match provided FITS HDUList object")

    inst = stpsf.instrument(header['INSTRUME'])

    if inst.name=='MIRI' and header['FILTER']=='P750L':
        # stpsf doesn't model the MIRI LRS prism spectral response
        print("Please note, stpsf does not currently model the LRS spectral response. Setting filter to F770W instead.")
        inst.filter='F770W'
    else:
        inst.filter=header['filter']
    inst.set_position_from_aperture_name(header['APERNAME'])

    if dateobs is None:
        dateobs = astropy.time.Time(header['DATE-OBS']+"T"+header['TIME-OBS'])
    inst.load_wss_opd_by_date(dateobs, verbose=verbose, plot=plot)


    # per-instrument specializations
    if inst.name == 'NIRCam':
        if header['PUPIL'].startswith('MASK'):
            inst.pupil_mask = header['PUPIL']
            inst.image_mask = header['CORONMSK'].replace('MASKA', 'MASK')  # note, have to modify the value slightly for
                                                                           # consistency with the labels used in stpsf
    elif inst.name == 'MIRI':
        if inst.filter in ['F1065C', 'F1140C', 'F1550C']:
            inst.image_mask = 'FQPM'+inst.filter[1:5]
        elif inst.filter == 'F2300C':
            inst.image_mask = 'LYOT2300'
        elif header['FILTER'] == 'P750L':
            inst.pupil_mask = 'P750L'
            if header['APERNAME'] == 'MIRIM_SLIT':
                inst.image_mask = 'LRS slit'

    # TODO add other per-instrument keyword checks

    if verbose:
        print(f"""
Configured simulation instrument for:
    Instrument: {inst.name}
    Filter: {inst.filter}
    Detector: {inst.detector}
    Apername: {inst.aperturename}
    Det. Pos.: {inst.detector_position} {'in subarray' if "FULL" not in inst.aperturename else ""}
    Image plane mask: {inst.image_mask}
    Pupil plane mask: {inst.pupil_mask}
    """)

    return inst

def get_jwst_psf_grid(st_obs,num_psfs=16,fname=None,dateobs=None):
    if fname is None:
        inst = stpsf_setup_sim_to_match_file(st_obs.exposure_fnames[0],dateobs=dateobs,verbose=False)
    else:
        inst = stpsf_setup_sim_to_match_file(fname,dateobs=dateobs,verbose=False)

    grid = inst.psf_grid(num_psfs=num_psfs, all_detectors=False, verbose=False)

    return grid

def get_jwst_psf_from_grid(st_obs,sky_location,grid,psf_width=101):

    grid.oversampling = (1,1)
    psf_list = []
    for i in range(st_obs.n_exposures):
        imwcs = st_obs.wcs_list[i]
        x,y = astropy.wcs.utils.skycoord_to_pixel(sky_location,imwcs)
        grid.x_0 = x
        grid.y_0 = y

        xf, yf = np.meshgrid(np.arange(-4*psf_width/2,psf_width/2*4+1,1).astype(int)+int(x+.5),
                            np.arange(-4*psf_width/2,psf_width/2*4+1,1).astype(int)+int(y+.5))

        psf = np.array(grid(xf,yf)).astype(float)
        psf/=np.sum(psf)
        psf*=16
        epsf_model = photutils.psf.FittableImageModel(psf,normalize=False,oversampling=4)
        psf_list.append(epsf_model)
    return psf_list

def get_jwst_psf(st_obs,sky_location,psf_width=61,pipeline_level=2,fname=None,dateobs=None):

    #inst = stpsf.instrument(st_obs.instrument)
    #inst.filter = st_obs.filter
    #inst.detector=st_obs.detector
    if fname is None:
        inst = stpsf_setup_sim_to_match_file(st_obs.exposure_fnames[0],dateobs=dateobs,verbose=False)
    else:
        inst = stpsf_setup_sim_to_match_file(fname,dateobs=dateobs,verbose=False)

    if pipeline_level == 3:

        oversampling = 1
    else:
        oversampling = 4

    psf_list = []

    #kernel = astropy.convolution.Box2DKernel(width=4)
    for i in range(st_obs.n_exposures):



        #inst.pixelscale = st_obs.pixel_scale[i]
        imwcs = st_obs.wcs_list[i]
        x,y = astropy.wcs.utils.skycoord_to_pixel(sky_location,imwcs)
        #inst.detector_position = (x,y)
        c = stpsf.gridded_library.CreatePSFLibrary(inst,inst.filter,  num_psfs = 1, psf_location = (x,y), fov_pixels = psf_width,
                                                                        detectors=st_obs.detector,save=False,verbose=False,
                                                                        use_detsampled_psf=True if oversampling==1 else False)
        #psf = inst.calc_psf(oversample=4,normalize='last')
        grid = c.create_grid()

        #psf[0].data = astropy.convolution.convolve(psf[0].data, kernel)
        #stpsf.detectors.apply_detector_ipc(psf, extname=0)

        #epsf_model = photutils.psf.FittableImageModel(psf[0].data*16,normalize=False,oversampling=oversampling)
        #epsf_model = photutils.psf.FittableImageModel(grid.data[0,:,:]/np.sum(grid.data[0,:,:])*16,normalize=False,oversampling=oversampling)
        epsf_model = photutils.psf.FittableImageModel(grid.data[0,:,:],normalize=False,oversampling=oversampling)
        psf_list.append(epsf_model)
    return psf_list

def get_jwst3_psf_spike(st_obs,st_obs3,sky_location,temp_outdir='.',verbose=True,psf_width=31):
    try:
        import spike
    except:
        raise RuntimeError('Must have spike-psf for level 3 psfs.')
    psf_drz = spike.psf.jwst(os.path.basename(st_obs.exposure_fnames[0]), '%f %f'%(sky_location.ra.value,sky_location.dec.value), 
        st_obs3.instrument, img_type = os.path.splitext(st_obs.exposure_fnames[0])[1], camera = None, method = 'WebbPSF', usermethod = None, 
                overwrite=True, savedir = temp_outdir, drizzleimgs = False, objonly = True, pretweaked = True, usecrds = True,
                keeporig = False, plot = False, verbose = verbose, parallel = False, out = 'fits', returnpsf = 'crop', cutout_fov = psf_width, savecutout = False, 
                finalonly = True, removedir = temp_outdir, tweakparams = {}, 
                drizzleparams = {'pixel_scale':st_obs3.pixel_scale, 'output_wcs': st_obs3.wcs})
    return(psf_drz)

def get_jwst3_psf(st_obs,st_obs3,sky_location,num_psfs=4,psf_width=31,temp_outdir='.'):
    with open('./stpipe-log.cfg','w') as f:
        s = '[*]\nhandler = file:/dev/null\nlevel = INFO\n'
        f.write(s)
    #sys.exit()
    psfs = get_jwst_psf(st_obs,sky_location,psf_width=psf_width,pipeline_level=3)

    #grid = get_jwst_psf_grid(st_obs,num_psfs=num_psfs)
    #grid.oversampling = 1
    # kernel = astropy.convolution.Box2DKernel(width=4)
    # psfs = []
    # for i in range(st_obs.n_exposures):
    #     imwcs = st_obs.wcs_list[i]
    #     x,y = astropy.wcs.utils.skycoord_to_pixel(sky_location,imwcs)
    #     psf = inst.calc_psf(oversample=4,normalize='last')
    #     psf[0].data = astropy.convolution.convolve(psf[0].data, kernel)
    # #    grid.x_0 = x
    # #    grid.y_0 = y
    # #
    # #    xf, yf = np.meshgrid(np.arange(-4*psf_width/2,psf_width/2*4+1,1).astype(int)+int(x+.5),
    # #                        np.arange(-4*psf_width/2,psf_width/2*4+1,1).astype(int)+int(y+.5))

    # #    psf = np.array(grid(xf,yf)).astype(float)
    #     epsf_model = photutils.psf.FittableImageModel(psf,normalize=True,oversampling=1)
    #     psfs.append(epsf_model)

    outdir = os.path.join(temp_outdir,'temp_psf_dir')#%np.random.randint(0,1000))
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    #print(outdir)
    level2_sums = []
    try:
        out_fnames = []
        for i,f in enumerate(st_obs.exposure_fnames):
            #print(f)
            dat = fits.open(f)

            imwcs = wcs.WCS(dat['SCI',1])
            #print(imwcs)
            y,x = skycoord_to_pixel(sky_location,imwcs)

            #xf, yf = np.mgrid[0:dat['SCI',1].data.shape[0]+int(psf_width*8),0:dat['SCI',1].data.shape[1]+int(psf_width*8)].astype(int)
            xf, yf = np.mgrid[0:dat['SCI',1].data.shape[0],0:dat['SCI',1].data.shape[1]].astype(int)
            #psfs[i].x_0 = x+psf_width*4
            #psfs[i].y_0 = y+psf_width*4
            psfs[i].x_0 = x#int(x)+.5
            psfs[i].y_0 = y#int(y)+.5
            #import pdb
            #pdb.set_trace()
            #dat['SCI',1].data = psfs[i].data#
            dat['SCI',1].data = psfs[i](xf,yf)
            level2_sums.append(np.sum(dat['SCI',1].data))
            dat.writeto(os.path.join(outdir,os.path.basename(f)),overwrite=True)
            #out_fnames.append(os.path.join(outdir,os.path.basename(f)))
            out_fnames.append(os.path.basename(f))
        #sys.exit()
        asn = asn_from_list.asn_from_list(out_fnames, rule=DMS_Level3_Base,
            product_name='temp_psf_cals')

        with open(os.path.join(outdir,'cal_data_asn.json'),"w") as outfile:
            name, serialized = asn.dump(format='json')
            outfile.write(serialized)

        ref_image = fits.open(st_obs3.fname)['SCI',1]
        ref_dm = datamodels.open(st_obs3.fname)
        ref_wcs = wcs.WCS(ref_image)
        transform = make_fitswcs_transform(ref_image.header) 
        # 3. Define frames
        detector_frame = cf.Frame2D(
            name="detector",
            axes_names=("x", "y"),
            unit=(u.pix, u.pix),
        )

        sky_frame = cf.CelestialFrame(
            name="icrs",
            reference_frame=coord.ICRS(),
            unit=(u.deg, u.deg),
        )
        # 4. Build the GWCS pipeline
        pipeline = [(detector_frame, transform), (sky_frame, None)]
        gw = gwcs.WCS(pipeline)

        ny,nx = ref_image.data.shape

        # 5. Set bounding box
        # NOTE: GWCS uses "F" order: ((xmin, xmax), (ymin, ymax)) for (x, y) axes
        gw.bounding_box = ((0, nx), (0, ny))

        #ref_wcs = #gwcs.wcs.WCS(ref_image)
        tree = {"wcs": gw}
        wcs_file = AsdfFile(tree)
        wcs_file.write_to(os.path.join(temp_outdir,'ref_wcs.asdf'))
        
        pipe3 = Image3Pipeline()
        pipe3.resample.output_wcs = os.path.join(temp_outdir,'ref_wcs.asdf')
        pipe3.output_dir = outdir
        pipe3.save_results = True
        pipe3.tweakreg.skip = True
        pipe3.outlier_detection.skip = True
        pipe3.skymatch.skip = True
        pipe3.source_catalog.skip = True
        pipe3.resample.output_shape = st_obs3.data.shape
        pipe3.outlier_detection.save_results = False
        #pipe3.resample.output_shape = (dat['SCI',1].data.shape)
        pipe3.resample.pixel_scale = st_obs3.pixel_scale#/4#[0]#/4

        #pipe3.resample.pixel_scale_ratio = st_obs3.pixel_scale/st_obs.pixel_scale[0]
        pipe3.run(os.path.join(outdir,'cal_data_asn.json'))

        #imwcs = None
        #level3 = None
        with fits.open(os.path.join(outdir,'temp_psf_cals_i2d.fits')) as dat:
            imwcs = wcs.WCS(dat['SCI',1])
            level3 = dat[1].data

        level3[np.isnan(level3)] = 0
        level3[level3<0] = 0
        #print(np.max(level3))
        #sys.exit()

        #kernel = astropy.convolution.Box2DKernel(width=4)
        #level3 = astropy.convolution.convolve(level3, kernel)
        y,x = astropy.wcs.utils.skycoord_to_pixel(sky_location,imwcs)
        # mx,my = np.meshgrid(np.arange(-4*psf_width/2,psf_width/2*4+1,1).astype(int)+int(x+.5+psf_width*4),
        #                     np.arange(-4*psf_width/2,psf_width/2*4+1,1).astype(int)+int(y+.5+psf_width*4))
        mx,my = np.meshgrid(np.arange(-4*psf_width/2,psf_width/2*4+1,1).astype(int)+int(x+.5),
                            np.arange(-4*psf_width/2,psf_width/2*4+1,1).astype(int)+int(y+.5))



        level3_psf = photutils.psf.FittableImageModel(level3[mx,my],normalize=False,
                                                      oversampling=1)

        #import pdb
        #pdb.set_trace()
        temp_fnames = glob.glob(os.path.join(outdir,'*'))
        for f in temp_fnames:
            os.remove(f)
        shutil.rmtree(outdir, ignore_errors=True)
        #os.rmdir(outdir)
        os.remove('stpipe-log.cfg')
    except RuntimeError:#Exception as e:
        print('Failed to create PSF model')
        print(e)
        temp_fnames = glob.glob(os.path.join(outdir,'*'))
        for f in temp_fnames:
            os.remove(f)
        shutil.rmtree(outdir, ignore_errors=True)
        os.remove('stpipe-log.cfg')


    return level3_psf

def get_hst_psf_grid(st_obs):
    grid = make_models(get_standard_psf(os.path.join(os.path.abspath(os.path.dirname(__file__)),
            'wfc3_photometry/psfs'),st_obs.filter,st_obs.detector))[0]
    return grid

def get_hst_psf(st_obs,sky_location,psf_width=25,pipeline_level=2):
    grid = make_models(get_standard_psf(os.path.join(os.path.abspath(os.path.dirname(__file__)),
            'wfc3_photometry/psfs'),st_obs.filter,st_obs.detector))[0]
    psf_list = []
    _, oversamp = np.array(grid.oversampling, dtype=float)

    for i in range(st_obs.n_exposures):
        imwcs = st_obs.wcs_list[i]
        y,x = astropy.wcs.utils.skycoord_to_pixel(sky_location,imwcs)

        size_os = psf_width * oversamp

        # Make size_os odd so we have a clean center
        if size_os % 2 == 0:
            size_os += 1

        half_os = size_os // 2

        # Grid in oversampled pixel indices around the center
        # (j, i) ~ (y, x)
        jj, ii = np.mgrid[-half_os:half_os+1, -half_os:half_os+1]

        # Convert oversampled pixel offsets to detector coordinates
        # 1 oversampled pixel = 1 / oversamp detector pixels
        x_coords = x + ii / oversamp
        y_coords = y + jj / oversamp

        # GriddedPSFModel expects x, y arrays (broadcastable)
        #stamp = psf_model(x_coords, y_coords)

        # Photutils wants positions as (y, x) stacked into an array
        

        # Evaluate PSF
        vals = grid.evaluate(x_coords,y_coords,1,float(x),float(y))
        _psf_interp = vals.reshape((int(size_os), int(size_os)))
        #grid_idx, _ = grid._find_bounding_points(x,y)
        #print(grid_idx)
        #psfinterp = grid._calc_interpolator(int(x), int(y))
        #psfinterp = grid._calc_interpolator(grid_idx)
        #_psf_interp = psfinterp(grid._xidx, grid._yidx)
        _psf_interp/=simple_aperture_sum(_psf_interp,[[_psf_interp.shape[0]/2,_psf_interp.shape[0]/2]],5.6*4)
        _psf_interp*=16
        _psf_interp*=(hst_apcorr(5.6*st_obs.px_scale,st_obs.filter,st_obs.instrument))

        if pipeline_level==2:
            psfmodel = photutils.psf.FittableImageModel(_psf_interp,
                                      oversampling=grid.oversampling)
        else:
            psfmodel = photutils.psf.FittableImageModel(_psf_interp,
                                      oversampling=1)
        psfmodel.x_0 = x#int(x)
        psfmodel.y_0 = y#int(y)
        psf_list.append(psfmodel)

        #yg, xg = np.mgrid[-1*(psf_width-1)/2:(psf_width+1)/2,-1*(psf_width-1)/2:(psf_width+1)/2].astype(int)
        #yf, xf = yg+int(y+.5), xg+int(x+.5)
        #yf, xf = yg+int(np.round(y)), xg+int(np.round(x))
        #psf = np.array(psfmodel(xf,yf)).astype(float)
        #plt.imshow(psf)
        #plt.show()
        #continue
        #print(x,y)

        #epsf_model = EPSFModel(psf)
        #psf_list.append(epsf_model)
    return psf_list

def get_hst3_psf(st_obs,st_obs3,sky_location,psf_width=25):
    from drizzlepac import astrodrizzle
    psfs = get_hst_psf(st_obs,sky_location,psf_width=psf_width,pipeline_level=3)

    outdir = os.path.join(os.path.abspath(os.path.dirname(__file__)),'temp_%i'%np.random.randint(0,1000))
    os.mkdir(outdir)
    level2_sums = []
    try:
        out_fnames = []
        for i,f in enumerate(st_obs.exposure_fnames):
            dat = fits.open(f)
            if False:



                newx = dat[1].header['NAXIS1']*4
                newy = dat[1].header['NAXIS2']*4

                old_wcs = wcs.WCS(dat[1],dat)
                new_wcs = old_wcs[::.25,::.25].to_header()
                for k in ['PC1_1', 'PC1_2','PC2_1','PC2_2']:
                    new_wcs[k]/=4


                for key in new_wcs.keys():
                       if len(key)>0:
                           #dm_fits[i].header[key+'A'] = dm_fits[i].header[key]
                           #if not (self.do_driz or ('CRPIX' in key or 'CTYPE' in key)):
                            if 'CTYPE' not in key:
                                if key.startswith('PC') and key not in dat[1].header.keys():
                                    dat[1].header.set(key.replace('PC','CD'),value=new_wcs[key])
                                elif key in dat[1].header:
                                    dat[1].header.set(key,value=new_wcs[key])
                                #else:
                                #    dm_fits[i].header.set(key,value='TWEAK')

                dat[1].header['IDCSCALE'] = dat[1].header['IDCSCALE']/4
            else:
                newx = dat['SCI',st_obs.sci_ext].data.shape[0]#header['NAXIS1']
                newy = dat['SCI',st_obs.sci_ext].data.shape[1]#header['NAXIS2']
            if True:
                #dat['SCI',1].data = np.zeros((newy,newx))
                imwcs = wcs.WCS(dat['SCI',st_obs.sci_ext],dat)
                y,x = skycoord_to_pixel(sky_location,imwcs)

                #xf, yf = np.mgrid[0:dat['SCI',1].data.shape[0]+int(psf_width*8),0:dat['SCI',1].data.shape[1]+int(psf_width*8)].astype(int)
                xf, yf = np.mgrid[0:dat['SCI',st_obs.sci_ext].data.shape[0],0:dat['SCI',st_obs.sci_ext].data.shape[1]].astype(int)
                #psfs[i].x_0 = x+psf_width*4
                #psfs[i].y_0 = y+psf_width*4
                psfs[i].x_0 = int(x)+.5
                psfs[i].y_0 = int(y)+.5

                dat['SCI',st_obs.sci_ext].data = psfs[i](xf,yf)

                #x,y = astropy.wcs.utils.skycoord_to_pixel(sky_location,wcs.WCS(dat[1],dat))
                #psf2 = photutils.psf.FittableImageModel(psfs[i].data,normalize=False,
                #                                          oversampling=1)
                #psf2.x_0 = x
                #psf2.y_0 = y
                #x = int(x+.5)
                #y = int(y+.5)

                #gx, gy = np.mgrid[0:newx,0:newy].astype(int)
                #dat[1].data = psf2.evaluate(gx,gy,psf2.flux.value,psf2.x_0.value,psf2.y_0.value,
                #                            use_oversampling=False)
                dat['SCI',st_obs.sci_ext].data[dat['SCI',st_obs.sci_ext].data<0] = 0

                #dat[1].data/=scipy.ndimage.zoom(st_obs.pams[0].T,4)
                #dat[1].data/=st_obs.pams[0]
                #if st_obs.detector in ["ACS","UVIS"]:
                #    dat['D2IMARR',st_obs.sci_ext].data = scipy.ndimage.zoom(dat['D2IMARR',st_obs.sci_ext].data,4)

                dat['DQ',st_obs.sci_ext].data = np.zeros((newx,newy)).astype(int)
                dat['ERR',st_obs.sci_ext].data = np.ones((newx,newy))
                #dat = dat[:4]
                level2_sums.append(simple_aperture_sum(dat['SCI',st_obs.sci_ext].data,[[y,x]],5.6*4))

                dat.writeto(os.path.join(outdir,os.path.basename(f)),overwrite=True)
                out_fnames.append(os.path.join(outdir,os.path.basename(f)))



        astrodrizzle.AstroDrizzle(','.join(out_fnames),output=os.path.join(outdir,'temp_psf'),
                            build=True,median=False,skysub=False,sky_bits=None,
                            driz_cr_corr=False,final_wht_type='ERR',driz_separate=False,
                            driz_cr=False,blot=False,clean=True,group='sci,'+str(st_obs.sci_ext),
                            final_scale=st_obs3.pixel_scale
                            )
        try:
            dat = fits.open(glob.glob(os.path.join(outdir,'temp_psf_drz.fits'))[0])
        except:
            dat = fits.open(glob.glob(os.path.join(outdir,'temp_psf_drc.fits'))[0])
        #sys.exit()
        imwcs = wcs.WCS(dat[1],dat)
        y,x = skycoord_to_pixel(sky_location,imwcs)
        level3 = dat[1].data
        level3[np.isnan(level3)] = 0
        level3[level3<0] = 0
        y,x = astropy.wcs.utils.skycoord_to_pixel(sky_location,imwcs)
        mx,my = np.meshgrid(np.arange(-4*psf_width/2,psf_width/2*4+1,1).astype(int)+int(x+.5),
                            np.arange(-4*psf_width/2,psf_width/2*4+1,1).astype(int)+int(y+.5))

        mx2,my2 = np.meshgrid(np.arange(-1*psf_width/2,psf_width/2*1+1,1).astype(int)+int(x+.5),
                            np.arange(-1*psf_width/2,psf_width/2*1+1,1).astype(int)+int(y+.5))

        

        level3_sum = simple_aperture_sum(level3,[y,x],5.6*4)


        level3[mx,my]/=level3_sum
        #level3[mx,my]*=np.median(level2_sums)

        level3[mx,my]*=16
        level3[mx,my]*=(hst_apcorr(5.6*st_obs3.px_scale,st_obs3.filter,st_obs3.instrument))
        #level3[mx,my]*=(np.median(level2_sums)/simple_aperture_sum(level3[mx,my],[[level3[mx,my].shape[0]/2,
        #                                    level3[mx,my].shape[1]/2]],5.6*4))
        #level3[mx,my]*=16
        level3_psf = photutils.psf.FittableImageModel(level3[mx,my],normalize=False,
                                                      oversampling=4)
        #kernel = astropy.convolution.Box2DKernel(width=4)
        #level3_psf = photutils.psf.FittableImageModel(astropy.convolution.convolve(level3[mx,my], kernel),normalize=False,
        #                                              oversampling=4)
        #sys.exit()
        shutil.rmtree(outdir)
    except RuntimeError:
        print('Failed to create PSF model')
        shutil.rmtree(outdir)
    return level3_psf

def jwst_apcorr_interp(fname, radius, alternate_ref: Optional[str] = None):
    """
    Interpolate JWST imaging aperture correction as a function of aperture radius (pixels),
    bypassing JWST pipeline step machinery.

    Parameters
    ----------
    fname : str
        JWST image (e.g. *_cal.fits, *_i2d.fits) used to select the correct APCORR ref.
    radius : float
        Aperture radius in PIXELS.
    alternate_ref : str, optional
        If provided, use this file to select the APCORR reference, or pass an APCORR file
        directly (convenience) if its name/path contains 'apcorr'.

    Returns
    -------
    tuple
        (ee_percent, apcorr, skyin_pix, skyout_pix)

        - ee_percent: interpolated EE in percent (0–100)
        - apcorr: interpolated aperture correction factor
        - skyin_pix, skyout_pix: recommended background annulus radii in pixels
    """
    radius = float(radius)
    if radius <= 0:
        raise ValueError(f"radius must be positive (pixels), got {radius}")

    with datamodels.open(alternate_ref or fname) as model:

        # Determine APCORR reference file
        if alternate_ref and alternate_ref.lower().endswith((".asdf", ".fits")) and "apcorr" in alternate_ref.lower():
            apcorr_path = alternate_ref
        else:
            apcorr_path = _get_best_apcorr_reffile(model)

        with datamodels.open(apcorr_path) as apm:
            tab = apm.apcorr_table  # often a numpy recarray
            names = tab.dtype.names

            # Instrument selection keys
            filt = (model.meta.instrument.filter or "").upper()
            pup = (model.meta.instrument.pupil or "").upper()

            # Build mask for this filter/pupil
            filt_col = np.array([str(x).upper() for x in tab["filter"]])
            m = (filt_col == filt)

            if "pupil" in names and pup:
                pup_col = np.array([str(x).upper() for x in tab["pupil"]])
                m &= (pup_col == pup)

            if not np.any(m):
                # relax pupil if needed
                m2 = (filt_col == filt)
                if np.any(m2):
                    m = m2
                else:
                    raise ValueError(
                        f"No APCORR rows match FILTER={filt!r}, PUPIL={pup!r}. Ref: {apcorr_path}"
                    )

            sub = tab[m]

            # Collect arrays
            r = np.array(sub["radius"], dtype=float)        # pixels
            c = np.array(sub["apcorr"], dtype=float)
            ee_frac = np.array(sub["eefraction"], dtype=float)

            # Get background annulus values.
            # Many ref tables keep skyin/skyout constant for the whole filter/pupil,
            # but we safely pick the first.
            skyin = float(np.array(sub["skyin"], dtype=float)[0])
            skyout = float(np.array(sub["skyout"], dtype=float)[0])

            # Sort by radius (interp1d requires monotonic x)
            idx = np.argsort(r)
            r = r[idx]
            c = c[idx]
            ee_frac = ee_frac[idx]

            # Radius bounds
            rmin, rmax = float(np.nanmin(r)), float(np.nanmax(r))
            if radius < rmin or radius > rmax:
                raise ValueError(
                    f"radius={radius} px is outside APCORR bounds [{rmin}, {rmax}] px "
                    f"for FILTER={filt}, PUPIL={pup}. Ref: {apcorr_path}"
                )

            # Interpolate (linear; matches your prior behavior)
            apcorr = float(scipy.interpolate.interp1d(r, c)(radius))
            ee_percent = float(scipy.interpolate.interp1d(r, ee_frac)(radius) * 100.0)

            return ee_percent, apcorr, skyin, skyout



def _get_best_apcorr_reffile(model) -> str:
    hdr = model.to_flat_dict()
    ctx = crds.get_context_name("jwst")
    return crds.getreferences(hdr, reftypes=["apcorr"], context=ctx)["apcorr"]


def jwst_apcorr(
    fname: str,
    ee: float = 70,
    alternate_ref: Optional[str] = None,
):
    """
    Lookup JWST imaging aperture correction directly from the APCORR reference file,
    bypassing the JWST pipeline machinery.

    Parameters
    ----------
    fname : str
        JWST image (e.g. *_cal.fits, *_i2d.fits).
    ee : float
        Encircled energy percentage (e.g. 70 for 70% EE).
    alternate_ref : str, optional
        If provided, use this file to select the APCORR reference.

    Returns
    -------
    list
        [radius_pix, apcorr, skyin_pix, skyout_pix]
    """
    ee_fraction = float(ee) / 100.0

    with datamodels.open(alternate_ref or fname) as model:

        if alternate_ref and alternate_ref.lower().endswith((".asdf", ".fits")) and "apcorr" in alternate_ref.lower():
            apcorr_path = alternate_ref
        else:
            apcorr_path = _get_best_apcorr_reffile(model)

        with datamodels.open(apcorr_path) as apm:
            tab = apm.apcorr_table
            names = tab.dtype.names

            filt = (model.meta.instrument.filter or "").upper()
            pup = (model.meta.instrument.pupil or "").upper()

            filt_col = np.array([str(x).upper() for x in tab["filter"]])
            ee_col = np.array(tab["eefraction"], dtype=float)

            m = (filt_col == filt) & np.isclose(ee_col, ee_fraction)

            if "pupil" in names and pup:
                pup_col = np.array([str(x).upper() for x in tab["pupil"]])
                m &= (pup_col == pup)

            if not np.any(m):
                raise ValueError(
                    f"No APCORR match for FILTER={filt}, PUPIL={pup}, EE={ee}% "
                    f"in ref file {apcorr_path}"
                )

            row = tab[m][0]

            return [
                float(row["radius"]),
                float(row["apcorr"]),
                float(row["skyin"]),
                float(row["skyout"]),
            ]


def estimate_bkg(data,position,inner, outer,model_psf=None,corr=None):
    assert model_psf is not None or corr is not None, 'Must supply model_psf or corr'
    assert inner<outer

    annulus_aperture = CircularAnnulus(np.flip(position), r_in=inner, r_out=outer)
    annulus_mask = annulus_aperture.to_mask(method='center')

    annulus_data = annulus_mask.multiply(data)
    import matplotlib.pyplot as plt
    model_psf.x_0 = position[1]
    model_psf.y_0 = position[0]
    yf, xf = np.mgrid[0:data.shape[0],0:data.shape[1]].astype(int)
    psf = np.array(model_psf(xf,yf)).astype(float)
    annulus_psf = annulus_mask.multiply(psf)
    print(np.sum(annulus_psf)/np.sum(psf))
    plt.imshow(annulus_data)
    plt.show()
    plt.imshow(annulus_psf)
    plt.show()
    sys.exit()

def generic_aperture_phot(data, positions, radius, sky, epadu=1, error=None):
    """
    Perform circular aperture photometry with a local sky annulus.

    Parameters
    ----------
    data : ndarray
        2D image array.
    positions : array_like
        Position(s) in pixel coordinates (x, y).
    radius : float
        Aperture radius in pixels.
    sky : dict
        Dictionary with keys 'sky_in' and 'sky_out' giving inner and outer
        radii of the sky annulus in pixels.
    epadu : float, optional
        Electrons per ADU for Poisson error estimation.
    error : ndarray or None, optional
        Per-pixel 1-sigma uncertainties. If provided, photutils will use this
        for error propagation; otherwise we estimate errors from Poisson +
        sky scatter.

    Returns
    -------
    phot : astropy.table.Table
        Photometry table with at least the following columns:
        'aperture_sum', 'annulus_median', 'aper_bkg',
        'aper_sum_bkgsub', and 'aperture_sum_err' if error is None.
    """
    aperture = CircularAperture(positions, r=radius)
    annulus_aperture = CircularAnnulus(
        positions, r_in=sky["sky_in"], r_out=sky["sky_out"]
    )
    annulus_mask = annulus_aperture.to_mask(method="center")

    bkg_median = []
    bkg_stdev = []

    for mask in annulus_mask:
        try:
            annulus_data = mask.multiply(data)
            annulus_data_1d = annulus_data[mask.data > 0]
            _, median_sigclip, stdev_sigclip = sigma_clipped_stats(annulus_data_1d)
        except Exception:
            median_sigclip = np.nan
            stdev_sigclip = np.nan
        bkg_median.append(median_sigclip)
        bkg_stdev.append(stdev_sigclip)

    bkg_median = np.array(bkg_median)
    bkg_stdev = np.array(bkg_stdev)

    phot = aperture_photometry(data, aperture, method="exact", error=error)
    phot["annulus_median"] = bkg_median
    phot["aper_bkg"] = bkg_median * aperture.area
    phot["aper_sum_bkgsub"] = phot["aperture_sum"] - phot["aper_bkg"]

    if error is None:
        # Poisson error on background-subtracted counts
        error_poisson = np.sqrt(phot["aper_sum_bkgsub"])
        # Scatter inside the sky annulus
        error_scatter_sky = aperture.area * bkg_stdev**2
        # Error on the mean sky level
        error_mean_sky = (
            bkg_stdev**2 * aperture.area**2 / annulus_aperture.area
        )

        fluxerr = np.sqrt(
            (error_poisson**2) / epadu
            + error_scatter_sky
            + error_mean_sky
        )
        phot["aperture_sum_err"] = fluxerr

    return phot



def jwst_aperture_phot(fname,ra,dec,
                    filt,ee='r70'):
    try:
        force_ra = float(ra)
        force_dec = float(dec)
        unit = u.deg
    except:
        unit = (u.hourangle, u.deg)

    if isinstance(ee,str):
        radius,apcorr,skyan_in,skyan_out = get_apcorr_params(fname,int(ee[1:]))
    else:
        radius,apcorr,skyan_in,skyan_out = ee,1,ee+1,ee+3
    #radius =1.8335238
    #apcorr = aper_func(radius)
    #radius,apcorr = 1.83,1
    image = fits.open(fname)

    data = image['SCI',1].data#*image['AREA',1].data
    err = image['ERR',1].data
    imwcs = wcs.WCS(image[1])
    #positions = np.atleast_2d(np.flip([582.80256776,819.78997553]))#
    positions = np.atleast_2d(astropy.wcs.utils.skycoord_to_pixel(SkyCoord(ra, dec,unit=unit),imwcs))

    imh = image['SCI',1].header
    area = image[1].header['PIXAR_SR']
    aa = np.argwhere(data < 0)

    for i in np.arange(0, len(aa), 1):
        data[aa[i][0], aa[i][1]] = 0
    sky = {'sky_in':skyan_in,'sky_out':skyan_out}
    #with datamodels.open(fname) as model:
    #    dat = model.data
    #    err = model.err

    #phot = generic_aperture_phot(data,positions,radius,sky,error=image['ERR',1].data)
    phot = generic_aperture_phot(data,positions,radius,sky,error=err)

    phot['aper_sum_corrected'] = phot['aper_sum_bkgsub'] * apcorr
    phot['aperture_sum_err']*=apcorr
    phot['magerr'] = 2.5 * np.log10(1.0 + (phot['aperture_sum_err']/phot['aper_sum_bkgsub']))

    pixel_scale = wcs.utils.proj_plane_pixel_scales(imwcs)[0]  * imwcs.wcs.cunit[0].to('arcsec')
    flux_units = u.MJy / u.sr * (pixel_scale * u.arcsec)**2
    flux = phot['aper_sum_corrected']*flux_units
    phot['mag'] = flux.to(u.ABmag).value

    return phot


def hst_apcorr(ap,filt,inst):
    if inst=='ir':
        if not os.path.exists('ir_ee_corrections.csv'):
            urllib.request.urlretrieve('https://www.stsci.edu/files/live/sites/www/files/home/hst/'+\
                                       'instrumentation/wfc3/data-analysis/photometric-calibration/'+\
                                       'ir-encircled-energy/_documents/ir_ee_corrections.csv',
                                       'ir_ee_corrections.csv')

        ee = Table.read('ir_ee_corrections.csv',format='ascii')
        ee.remove_column('FILTER')
        waves = ee['PIVOT']
        ee.remove_column('PIVOT')
    else:
        if not os.path.exists('wfc3uvis2_aper_007_syn.csv'):

            urllib.request.urlretrieve('https://www.stsci.edu/files/live/sites/www/files/home/hst/'+\
                                    'instrumentation/wfc3/data-analysis/photometric-calibration/'+\
                                    'uvis-encircled-energy/_documents/wfc3uvis2_aper_007_syn.csv','wfc3uvis2_aper_007_syn.csv')
        ee = Table.read('wfc3uvis2_aper_007_syn.csv',format='ascii')

        ee.remove_column('FILTER')
        waves = ee['WAVELENGTH']
        ee.remove_column('WAVELENGTH')
    ee_arr = np.array([ee[col] for col in ee.colnames])
    apps = [float(x.split('#')[1]) for x in ee.colnames]

    interp = scipy.interpolate.RectBivariateSpline(waves,apps,ee_arr.T)

    try:
        filt_wave = sncosmo.get_bandpass(filt).wave_eff
    except:
        filt_wave = sncosmo.get_bandpass('uv'+filt).wave_eff
    return(interp(filt_wave,ap))

def hst_get_zp(filt,zpsys='ab'):
    if zpsys.lower()=='ab':
        return {'F098M':25.666,'F105W':26.264,'F110W':26.819,'F125W':26.232,'F140W':26.450,'F160W':25.936}[filt]
    elif zpsys.lower()=='vega':
        return {'F098M':25.090,'F105W':25.603,'F110W':26.042,'F125W':25.312,'F140W':25.353,'F160W':24.662}[filt]
    else:
        print('unknown zpsys')
        return

def hst_aperture_phot(fname,force_ra,force_dec,filt,radius=3,
                      skyan_in=4,skyan_out=8):
    data_file = fits.open(fname)
    drc_dat = data_file['SCI',1]
    if data_file[1].header['BUNIT']=='ELECTRON':
        epadu = 1
    else:
        epadu = data_file[0].header['EXPTIME']
    try:
        force_ra = float(force_ra)
        force_dec = float(force_dec)
        unit = u.deg
    except:
        unit = (u.hourangle, u.deg)
    sky_location = SkyCoord(force_ra,force_dec,unit=unit)
    imwcs = wcs.WCS(drc_dat.header,data_file)
    x,y = astropy.wcs.utils.skycoord_to_pixel(sky_location,imwcs)
    px_scale = wcs.utils.proj_plane_pixel_scales(imwcs)[0] * imwcs.wcs.cunit[0].to('arcsec')
    try:
        zp = hst_get_zp(filt,'ab')
        inst = 'ir'
    except:
        inst = 'uvis'
    phot = generic_aperture_phot(drc_dat.data,np.atleast_2d([x,y]),
                                       radius,{'sky_in':skyan_in,'sky_out':skyan_out},epadu=epadu)
    phot['magerr'] = 1.086 * phot['aperture_sum_err']/phot['aper_sum_bkgsub']

    apcorr = hst_get_ee_corr(radius*px_scale,filt,inst)
    if inst=='ir':
        ee_corr = 2.5*np.log10(apcorr)
        zp = hst_get_zp(filt,'ab')
        phot['aper_sum_corrected'] = phot['aper_sum_bkgsub']/apcorr
        phot['mag'] = -2.5*np.log10(phot['aper_sum_corrected'])+zp
    else:
        try:
            hdr = drc_dat.header
            photflam = hdr['PHOTFLAM']
        except:
            hdr = fits.open(data_file)[0].header
            photflam = hdr['PHOTFLAM']
        photplam = drc_dat.header['PHOTPLAM']

        ee_corr = 2.5*np.log10(apcorr)
        zp = -2.5*np.log10(photflam)-5*np.log10(photplam)-2.408
    phot['aper_sum_corrected'] = phot['aper_sum_bkgsub'] / apcorr
    phot['aperture_sum_err']/=apcorr
    phot['mag'] = -2.5*np.log10(phot['aper_sum_corrected']) + zp
    return(phot)

def simple_aperture_sum(data, positions, radius):
    """
    Compute a simple circular-aperture sum at one or more positions.

    Parameters
    ----------
    data : ndarray
        2D image array.
    positions : array_like
        Position or list of positions in pixel coordinates. Follows the
        photutils convention (x, y).
    radius : float
        Aperture radius in pixels.

    Returns
    -------
    aperture_sum : astropy.table.Column
        The 'aperture_sum' column from photutils.aperture_photometry.
    """
    aperture = CircularAperture(positions, r=radius)
    phot = aperture_photometry(data, aperture, method="exact")
    return phot["aperture_sum"]

