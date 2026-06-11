"""
Perform fitting of a science image.
Retrieve Strehl ratio and seeing.
"""

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
from matplotlib.patches import Circle, Ellipse
import numpy as np
from astropy.io import fits
from numpy.fft import fft2, fftshift
import os
from datetime import datetime
import re

from maoppy.utils import circavg
from maoppy.instrument import papyrus
from maoppy.psfmodel import Psfao, Turbulent
from maoppy.psffit import psffit

from papylib.image import strehl_ratio, center_cog, compute_cog


"""try:
    from maoppy.psfmodel import PsfaoPolychromatic
except:
    print('Warning: PsfaoPolychromatic is not available from MAOPPY, create an alias here!')
    class PsfaoPolychromatic(Psfao):
        def __init__(self, *args, samp=None, **kwargs):
            super().__init__(*args, samp=np.mean(samp), **kwargs)
"""
try:
    from aoerror.variance import var_fitting, var_temporal
except:
    print('Warning: AOERROR library not available')

#%% PARAMETERS
# path = r'C:\Users\rfetick\Documents\papyrus\AIT OHP\2025-03-04\\'
# bkg_name = 'bkg.npy'
# psf_name = 'psf_calib_1550.npy'

# path = r'C:\Users\rfetick\Documents\papyrus\AIT OHP\2025-03-05\\'
# bkg_name = 'cred3Avg6.npy'
# psf_name = 'cred3Avg15.npy'
save_name = 'OL'
iters = 1
path = 'bench_sky_04_16/onsky_arcturus_1st200_2nd400_v8_linear_20260417-040839/'
process_dir = os.path.join(path, 'process')
os.makedirs(process_dir, exist_ok=True)
psf_name = 'HD124897-CL-2026-04-17T04_17_14-Cube_pythint.fits'
#path = r'C:\Users\rfetick\Downloads\2025-10-28\\'

#all_psf = [f for f in os.listdir(path) if f.endswith('.fits')]

all_psf = [psf_name]

is_close_loop = True
save_png = True

nx = 100 # number of pixels for cropping
nb_mode_control = 50
elevation_deg = 70 # star elevation
pix_mas = 63.3 #70.6 # pixel size [milli-arcsec]75.9

### FIXED PARAMETERS
wvl = 1310e-9 # central wavelength

DM_D_CALIB = 37.5 # mm
DM_D_SKY = 35.5 # mm
DM_PITCH = 3.75 # mm
DM_NACT = 97 #241
save_results = False


TELESCOPE_OBSTRUCTION = 0.27
TELESCOPE_DIAMETER = 1.52 # m

NCPA_ASTIG_NM = 0 #80
NCPA_DEFOC_NM = 0 #-50
NCPA_TREFOIL_NM = 0

#%% DEFINITIONS
all_cog_maj = []
all_cog_min = []
all_theta = []
all_sr = []
all_seeing = []
all_po4ao = []
all_date= []


def load(pth):
    if pth.endswith('.npy'):
        return np.load(pth)
    elif pth.endswith('.fits'):
        f = fits.open(pth)
        d = f[0].data * 1.0 # make float
        f.close()
        return d
    else:
        raise ValueError('Unknown file format')


def get_otf(img):
    return np.abs(fftshift(fft2(fftshift(img))))


rad2arcsec = 180/np.pi * 3600
sampling = rad2arcsec * wvl/TELESCOPE_DIAMETER /(pix_mas*1e-3) # camera sampling at the observation wavelength ()
papyrus.occ = TELESCOPE_OBSTRUCTION
nb_act_lin = 1 + DM_D_CALIB/DM_PITCH
papyrus.Nact = round(nb_act_lin * DM_D_SKY/DM_D_CALIB * np.sqrt(nb_mode_control/DM_NACT))
ron = 10

samp_list = np.array([1, 1.1/1.4, 1.2/1.4, 1.3/1.4]) * sampling

if is_close_loop:
    #psfmodel = PsfaoPolychromatic((nx,nx), system=papyrus, samp=samp_list)
    psfmodel = Psfao((nx, nx), system=papyrus, samp=sampling,)
    psfparam_guess = [0.09, 1e-4, 0.4, 0.5, 1, 0, 1.5]
    fixed = [False, ]*7
    psfmodel.bounds[1][0] = 0.4 # min seeing set to 0.72"
else:
    psfmodel = Turbulent((nx,nx), system=papyrus, samp=sampling)
    psfparam_guess = [0.09, 30]
    fixed = [False, ]*2
    
psfmodel.zernike = np.array([0, NCPA_DEFOC_NM, NCPA_ASTIG_NM, 0, 0, 0, NCPA_TREFOIL_NM]) * 2*np.pi/(wvl*1e9)

#%% LOOP ON FILES
cube = load(path+psf_name)
#cube = cube[:,150:-150,150:-150]
bkg = cube[-1,...]
cube_wth_bg = cube[:-2,...]


sr_psfao_list = []
seeing_list = []
cx_list = []
cy_list = []
for i in range(iters):
    #img = np.mean(cube_wth_bg[100 * i:100 * (i+1),...], axis=0)
    img = np.mean(cube_wth_bg, axis=0)
    bkg = bkg + np.median(img-bkg)

    ### PROCESS PSF
    
    cx, cy = compute_cog(img, bkg, integer=True)
    cube_c = cube[:-2, cy-nx//2:cy+nx//2, cx-nx//2:cx+nx//2]
    psf_c, bkg_c = center_cog(img, bkg, nx)
    psf_c = psf_c - bkg_c

    if psf_c.shape == (nx,nx):

        ### FIT with MODEL
        weights = None # 1/(gaussian_filter(psf_c, 2)+ron**2)
        out = psffit(psf_c, psfmodel, psfparam_guess, weights=weights, fixed=fixed, max_nfev=30)
        otf_fit_avg = circavg(get_otf(out.psf), center=(nx//2,nx//2))
        
        ### COMPUTE OTF and STREHL
        psf_norm = (psf_c-out.flux_bck[1])/out.flux_bck[0]
        
        otf_avg = circavg(get_otf(psf_norm), center=(nx//2,nx//2))
        sr = strehl_ratio(psf_norm, sampling)
        otf_diff = get_otf(psfmodel.psfDiffraction)
        otf_diff_avg = circavg(otf_diff, center=(nx//2,nx//2))
        
        ### PRINT
        r0_zenith = out.x[0]/np.cos(np.pi/2-elevation_deg*np.pi/180)**(3/5)
        seeing = rad2arcsec*wvl/r0_zenith
        seeing_550 = seeing * (550e-9/wvl)**(-1/5)
        sr_psfao = np.array(psfmodel.strehlOTF(out.x))
        if is_close_loop:
            sr_psfao_list.append(sr_psfao) #sr_psfao[0]
        else:
            sr_psfao_list.append(sr_psfao)
        seeing_list.append(seeing_550)
        
        print('Strehl : %.1f %% (from OTF)'%(100*sr))
        print('Strehl : %s %% (from fit)'%(100*sr_psfao))
        print('Seeing : %.1f \" (zenith @ %u nm)'%(seeing_550,550))
        
        ### PLOT
        axis = np.linspace(-nx//2, nx//2, nx) * pix_mas * 1e-3
        
        def setplt(tab, cmap = 'Spectral_r', norm = LogNorm(vmin=1e-3, vmax=1)):
            dx,dy = out.dxdy * pix_mas * 1e-3
            maxi = np.max(psf_norm)
            im1 = plt.imshow(tab/maxi, norm=norm, cmap=cmap, extent=[axis[0],axis[-1],axis[0],axis[-1]])
            plt.colorbar(im1, fraction=0.046, pad=0.04)
            corr_zone = Circle([dx,-dy], wvl/TELESCOPE_DIAMETER*papyrus.Nact/2*rad2arcsec, fc='none', ec='k', ls=':')
            plt.gca().add_artist(corr_zone)
            plt.xlabel('[arcsec]')
            hfov = 2
            plt.xlim(-hfov+dx, hfov+dx)
            plt.ylim(-hfov-dy, hfov-dy)
            
        
        plt.figure(1, figsize=(5,5), dpi=300)

        plt.title(f'Linear | Strehl: {100 * sr_psfao:.1f}%', fontsize=20)
        dx,dy = out.dxdy * pix_mas * 1e-3
        maxi = np.max(psf_norm)
        im1 = plt.imshow(psf_norm/maxi, norm=LogNorm(vmin=1e-3, vmax=1), cmap='Spectral_r', extent=[axis[0],axis[-1],axis[0],axis[-1]])
        cbar = plt.colorbar(im1, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=12)
        corr_zone = Circle([dx,-dy], wvl/TELESCOPE_DIAMETER*papyrus.Nact/2*rad2arcsec, fc='none', ec='k', ls=':')
        #plt.gca().add_artist(corr_zone)
        plt.xlabel('[arcsec]', fontsize=16)
        plt.ylabel('Arcturus, $m_H$ = -2.81', fontsize=16)
        hfov = 1.2
        plt.xlim(-hfov+dx, hfov+dx)
        plt.ylim(-hfov-dy, hfov-dy)
        plt.tick_params(axis='both', labelsize=14)
        plt.tight_layout()
        plt.savefig('psf_linear.png', dpi=300, bbox_inches='tight')
        plt.show()


        '''plt.clf()
        
        plt.subplot(231)
        plt.title(f'Linear | Sterhl: {sr_psfao}')
        setplt(psf_norm)
        
        plt.subplot(232)
        plt.title('fit')
        setplt(out.psf)
        
        plt.subplot(233)
        plt.title('fit - data')
        setplt(out.psf-psf_norm, cmap='RdBu', norm=SymLogNorm(1e-3, vmin=-1, vmax=1))
        
        plt.subplot(234)
        plt.title('PSF')
        cxcy = (nx//2 + out.dxdy[1], nx//2 + out.dxdy[0])
        maxi = psfmodel.psfDiffraction.max()
        plt.semilogy(circavg(psfmodel.psfDiffraction/maxi, center=(nx//2,nx//2)), label='diffrac.', c='k')
        plt.semilogy(circavg(psf_norm/maxi, center=cxcy), label='data')
        plt.semilogy(circavg(out.psf/maxi, center=cxcy), label='fit')
        plt.axhline(out.flux_bck[1]/out.flux_bck[0], c='C1', ls='--', label='bck fit')
        plt.axvline(papyrus.Nact/2*sampling, c='k', ls=':', label='AO')
        plt.grid()
        plt.xlim(0, 70)
        plt.ylim(1e-5, 1)
        plt.xlabel('Position [pix]')
        plt.legend()
        
        plt.subplot(235)
        plt.title('OTF')
        plt.loglog(otf_diff_avg, label='diffrac.', c='k')
        plt.loglog(otf_avg, label='data')
        plt.loglog(otf_fit_avg, label='fit')
        plt.xlabel('Frequency [1/pix]')
        plt.ylim(1e-4,2)
        plt.xlim(right=nx//2)
        plt.grid()
        plt.legend()
        
        plt.subplot(236)
        plt.text(0, 0.1, 'Strehl(max) : %.1f %%\n\nWVL : %s um \n\nStrehl(fit) : %s %%\n\nSeeing : %.1f \"'%(100*sr, wvl*samp_list/sampling*1e6, np.round(1000*sr_psfao)/10, seeing_550), size=20)
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        if save_png:
            plt.savefig(path+r'process\\fitting_'+psf_name[:-5]+'.png')
        '''
        ### PLOT INSTANTANEOUS
        
        if i == iters - 1:
            cx_raw = np.zeros(cube_c.shape[0])
            cy_raw = np.zeros(cube_c.shape[0])
            maxi = np.zeros(cube_c.shape[0])
            flux = np.zeros(cube_c.shape[0])
            
            for j in range(cube_c.shape[0]):
                cx_raw[j],cy_raw[j] = compute_cog(cube_c[j,...], bkg_c, low=0.4)
                maxi[j] = np.max(cube_c[j,...]-bkg_c)
                flux[j] = np.sum(cube_c[j,...]-bkg_c)
            
            cx_avg = int(round(np.mean(cx_raw)))
            cy_avg = int(round(np.mean(cy_raw)))
            
            sr_instant = cube_c[:,cx_avg,cy_avg] - bkg_c[cx_avg,cy_avg]
            
            cx = cx_raw/sampling
            cy = cy_raw/sampling


            cx -= np.mean(cx)
            cy -= np.mean(cy)
            
            maxi = maxi/flux
            maxi /= np.mean(maxi)
            
            var_x = np.var(cx)
            var_y = np.var(cy)
            var_xy = np.sum(cx*cy)/len(cx)
            corr = np.array([[var_x,var_xy],[var_xy,var_y]])
            
            eigval, eigvec = np.linalg.eigh(corr)
            theta = np.arctan2(eigvec[0,1], eigvec[0,0]) * 180/np.pi

            cx_list.append(cx)
            cy_list.append(cy)
        
        #Center of gravity inside the camera TODO you can use the cx and cy for jitter frequency calculations
        '''plt.figure(2)
        plt.clf()
        plt.title('CoG    std=(%.1f,%.1f) $\lambda/D$'%(np.sqrt(eigval[1]),np.sqrt(eigval[0])))
        plt.scatter(cx, cy, alpha=0.2)
        plt.gca().add_patch(Ellipse((0,0), 2*np.sqrt(eigval[0]), 2*np.sqrt(eigval[1]), angle=theta, ec='r', fc='none', lw=2))
        plt.xlabel('Position [$\lambda/D$]')
        plt.ylabel('Position [$\lambda/D$]')
        

        if save_png:
            plt.savefig(path+r'process\\jitter_'+psf_name[:-5]+'.png')
            
        
        plt.figure(3, figsize=(22,5))
        plt.clf()
        mini = np.argmin(maxi)
        nf = 7
        
        for i in range(nf):
            plt.subplot(1,nf,i+1)
            i_f = mini+i-nf//2
            plt.title('Frame %u'%i_f)
            try:
                plt.imshow(cube_c[i_f,...]-bkg_c, vmin=0, vmax=psf_c.max(), cmap='Spectral_r')
                plt.colorbar(orientation='horizontal')
                plt.axvline(cx_raw[i_f], c='r', ls=':')
                plt.axhline(cy_raw[i_f], c='r', ls=':')
                plt.xlim(nx//4, 3*nx//4)
                plt.ylim(nx//4, 3*nx//4)
            except IndexError:
                pass
        
        plt.tight_layout()
            
        if save_png:
            plt.savefig(path+r'process\\fading_'+psf_name[:-5]+'.png')
            
            
        plt.figure(4, figsize=(22,5))
        plt.clf()
        mini = np.argmin(maxi)
        nf = 7
        
        for i in range(nf):
            plt.subplot(1,nf,i+1)
            i_f = (i*(cube_c.shape[0]-1)) // nf
            plt.title('Frame %u'%i_f)
            plt.imshow(cube_c[i_f,...]-bkg_c, vmin=0, vmax=psf_c.max(), cmap='Spectral_r')
            plt.colorbar(orientation='horizontal')
            plt.axvline(cx_raw[i_f], c='r', ls=':')
            plt.axhline(cy_raw[i_f], c='r', ls=':')
            plt.xlim(nx//4, 3*nx//4)
            plt.ylim(nx//4, 3*nx//4)
            
        plt.tight_layout()
            
        if save_png:
            plt.savefig(path+r'process\\cog_follow_'+psf_name[:-5]+'.png')'''
            
        ### SAVE DATA
        '''all_cog_maj += [np.sqrt(eigval[0])]
        all_cog_min += [np.sqrt(eigval[1])]
        all_theta += [theta]
        all_sr += [sr_psfao[0]]
        all_seeing += [seeing_550]
        all_po4ao += ['PO4AO' in psf_name.upper()]
        dt = re.findall('[0-9][0-9-]+T[0-9_]+',psf_name)[0].replace('_',':')
        all_date += [datetime.fromisoformat(dt)]'''
        
        
    else:
        print('Issue with data: %s'%psf_name)
  

savedir = path
os.makedirs(savedir, exist_ok=True)

seeing_array        = np.asarray(seeing_list)
strehl_array        = np.asarray(sr_psfao_list)
cx_array            = np.asarray(cx_list).reshape(np.asarray(cx_list).shape[1])
cy_array            = np.asarray(cy_list).reshape(np.asarray(cy_list).shape[1])


if save_results == True:
    np.savez(
        os.path.join(savedir, f'results_{save_name}.npz'),
        seeing        = seeing_array,       # [arcsec @ 550nm]
        strehl        = strehl_array,           # from PSF fit
        cx            = cx_array,
        cy            = cy_array
    )
    print(f"Results saved to {savedir}/results_{save_name}.npz")

#%% PLOT STATS
'''plt.close('all')

plt.figure(1, figsize=(25,6))
plt.clf()

plt.subplot(141)
plt.scatter(all_seeing, 100*np.array(all_sr), label='integrator', s=60)'''
try:
    x_seeing_550 = np.arange(1, 6, 0.1)
    x_seeing = x_seeing_550 * (wvl/550e-9)**(-1/5)
    r0 = wvl/x_seeing * rad2arcsec
    freq_cutoff = papyrus.Nact / (2*papyrus.D)
    var_fit = var_fitting(r0, freq_cutoff)
    windspeed = 12
    nmode = np.pi * (papyrus.Nact/2)**2
    nradial = round(np.sqrt(2*nmode))
    bandwidth = 500 / 20
    var_temp = var_temporal(papyrus.D, r0, nradial, windspeed, bandwidth)
    ncpa_nm = 100
    var_ncpa = (2*np.pi*ncpa_nm/(wvl*1e9))**2
    sr_marechal = 100 * np.exp(-(var_fit+var_temp+var_ncpa))
    plt.plot(x_seeing_550, sr_marechal, c='k', ls='--', label='fitting + temporal(%u m/s)'%round(windspeed), zorder=0)
except:
    pass
'''plt.xlabel('Seeing [arcsec] @ 550 nm')
plt.ylabel('Strehl [%%] @ %u nm'%(wvl*1e9))
plt.xlim(1, 4)
plt.yticks(np.arange(0,100,5))
plt.ylim(0, 60)
plt.grid()
plt.legend(loc='upper right')


plt.subplot(142)
plt.scatter(all_cog_maj, all_cog_min, s=60, label='integrator')
# plt.scatter(np.array(all_cog_maj)[all_po4ao], np.array(all_cog_min)[all_po4ao], s=60, label='PO4AO')
plt.plot([0,5], [0,5], c='k', alpha=0.2)
plt.xlabel('CoG std minor [lambda/D]')
plt.ylabel('CoG std major [lambda/D]')
plt.xlim(0, 0.7)
plt.ylim(0, 1.2)
plt.grid(which='both')
plt.legend()

plt.subplot(143)
plt.hist(np.array(all_theta)%180, bins=8)
plt.ylabel('counts')
plt.xlabel('CoG major/minor angle [°]')
plt.grid()

plt.subplot(144)
plt.hist(all_cog_maj, bins=np.linspace(0, 2, num=20), label='minor')
plt.hist(all_cog_min, bins=np.linspace(0, 2, num=20), histtype='step', fc='none', ec='C1', lw=3, label='major')
plt.ylabel('counts')
plt.xlabel('CoG std [lambda/D]')
plt.grid()
plt.legend()
plt.xlim(0, 1.2)

plt.tight_layout()


plt.figure(2)
plt.clf()
vld_cog = np.where(np.array(all_cog_min)<1)
p = np.polyfit(np.array(all_seeing)[vld_cog], np.array(all_cog_min)[vld_cog], 1)
p2 = np.polyfit(np.array(all_seeing)[vld_cog], np.array(all_cog_maj)[vld_cog], 1)
plt.scatter(np.array(all_seeing)[vld_cog], np.array(all_cog_min)[vld_cog])
plt.scatter(np.array(all_seeing)[vld_cog], np.array(all_cog_maj)[vld_cog])
plt.plot(x_seeing_550, p[0]*x_seeing_550+p[1], c='C0', ls='--', lw=2, label='y = %.2f x %.2f'%(p[0],p[1]))
plt.plot(x_seeing_550, p2[0]*x_seeing_550+p2[1], c='C1', ls='--', lw=2, label='y = %.2f x %.2f'%(p2[0],p2[1]))
plt.xlabel('Seeing [arcsec]')
plt.ylabel('CoG std [lambda/D]')
plt.xlim(2, 5)
plt.ylim(0, 0.8)
plt.legend()


import matplotlib.dates as mdates
xformatter = mdates.DateFormatter('%H:%M')

plt.figure(3, figsize=(12,6))
plt.clf()

plt.subplot(211)
plt.plot(all_date, all_seeing)
plt.scatter(all_date, all_seeing)
plt.ylabel('Seeing [arcsec]')
plt.grid()
plt.gca().xaxis.set_major_formatter(xformatter)

plt.subplot(212)
plt.plot(all_date, 100*np.array(all_sr))
plt.scatter(all_date, 100*np.array(all_sr))
plt.ylabel('Strehl [%%] @ %u nm'%(wvl*1e9))
plt.ylim(bottom=0)
plt.grid()
plt.gca().xaxis.set_major_formatter(xformatter)

plt.tight_layout()'''
