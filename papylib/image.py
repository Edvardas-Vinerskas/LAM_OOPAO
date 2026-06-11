"""
Functions related to the scientific camera image processing
"""

import numpy as np
from numpy.fft import fftshift, fft2
from papylib.constant import TELESCOPE


def clean_data(img, bkg, nx=None):
    """
    Clean an image: filter dead pixels, remove background.
    Crop the image to (nx,nx) pixels if the `nx` keyword is set.
    """
    dead_pix_map = dead_pixel_map(bkg)
    bkg_dp = filter_dead_pixel(bkg, dead_pix_map)
    img_dp = filter_dead_pixel(img, dead_pix_map)
    if nx is not None:
        img_dp, bkg_dp = center_cog(img_dp, bkg_dp, nx)
    return img_dp - bkg_dp


def structured_bkg(img):
    """Compute a synthetic structured background"""
    a_bg = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            a_bg[i,j] = np.median(np.concatenate((img[i,:],img[:,j])))
    return a_bg


def dead_pixel_map(bkg, maxi=0.2):
    """Compute the map of dead pixels"""
    bkg = bkg - np.median(bkg)
    return bkg > (bkg.max()*maxi)


def filter_dead_pixel(img, dead_pix_map):
    """Filter dead pixels"""
    img[np.where(dead_pix_map==1)] = np.median(img)
    return img


def compute_cog(img, bkg, integer=False, low=0.1):
    """
    Compute the center of gravity of an image.
    Threshold pixels lower than `low*np.max(img)`
    """
    x = np.arange(0,img.shape[1])
    y = np.arange(0,img.shape[0])
    X,Y = np.meshgrid(x,y)
    img_filtered = (img - bkg) - np.median(img - bkg)
    img_filtered = np.clip(img_filtered - low*np.max(img_filtered), 0, None)
    cx = np.sum(img_filtered*X)/np.sum(img_filtered)
    cy = np.sum(img_filtered*Y)/np.sum(img_filtered)
    if integer:
        cx = int(np.round(cx))
        cy = int(np.round(cy))
    return cx, cy


def center_cog(img, bkg, nx):
    """Center an image on its CoG"""
    cx,cy = compute_cog(img, bkg, integer=True)
    m_center = img[cy-nx//2:cy+nx//2,cx-nx//2:cx+nx//2]
    bkg_center = bkg[cy-nx//2:cy+nx//2,cx-nx//2:cx+nx//2]
    return m_center, bkg_center


def psf_diffraction(nx, samp, sky=True):
    """Papyrus PSF at diffraction limit"""
    xx,yy = np.mgrid[0:nx,0:nx] - nx//2
    rr = np.sqrt(xx**2+yy**2)/(nx/2)*samp
    pupil = rr<1
    if sky:
        pupil *= rr>TELESCOPE.OBSTRUCTION
    return np.abs(fftshift(fft2(fftshift(pupil))))**2 / np.sum(pupil)/nx**2


def otf_diffraction(*args, **kwargs):
    """Papyrus OTF at diffraction limit"""
    return fftshift(fft2(fftshift(psf_diffraction(*args, **kwargs))))


def strehl_ratio(psf, sampling, sky=True):
    """Compute the Strehl ratio from a PSF"""
    otf_diff = otf_diffraction(psf.shape[0], sampling, sky=sky)
    otf_exp = fftshift(fft2(fftshift(psf)))
    noise_filtering = (otf_diff>1e-3)
    sr = np.sum(np.abs(otf_exp*noise_filtering)) / np.sum(np.abs(otf_diff))
    return sr
    