import numpy as np
from ozitelemetry.PSFTelemetry import PSFTele
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm


savedir = "bench_sky_04_16/onsky_HD98262_1st200_2nd400_v7_linear20260417-015851"
filename = "HD98262-CL-2026-04-17T02_12_34-Cube_OL"
#v10 for both innit



psf_telemetry = PSFTele(
    tele_path=f"{savedir}/{filename}.fits",
    is_onsky= True,
    is_cl= False,
    crop_img=100
)

psf_telemetry.psf_analysis(
    elevation_deg=59
)

print("out:", psf_telemetry.out)
print("psf_fit:", psf_telemetry.psf_fit.shape)
print("psf:", psf_telemetry.psf.shape)
print("otf_fit_avg:", psf_telemetry.otf_fit_avg.shape)
print("otf_avg:", psf_telemetry.otf_avg.shape)
print("SR:", psf_telemetry.SR)
print("otf_diff:", psf_telemetry.otf_diff.shape)
print("r0_zenith:", psf_telemetry.r0_zenith)
print("seeing:", psf_telemetry.seeing)
print("seeing_550:", psf_telemetry.seeing_550)
print("SR_otf:", psf_telemetry.SR_otf)


rad2arcsec = 180/np.pi * 3600
sampling = 2.86 * 37.5 / 35.5 * 1.31e-06 / 1.55e-06
pix_mas = rad2arcsec * 1.31e-06/1.52 / (sampling*1e-3)

nx = psf_telemetry.psf.shape[0]
dx, dy = psf_telemetry.out.dxdy * pix_mas * 1e-3
axis = np.linspace(-nx//2, nx//2, nx) * pix_mas * 1e-3
extent = [axis[0], axis[-1], axis[0], axis[-1]]
hfov = 1.2

psf = psf_telemetry.psf
psf_fit = psf_telemetry.psf_fit
psf_norm = psf / np.max(psf)
psf_fit_norm = psf_fit / np.max(psf_fit)
residual = psf_norm - psf_fit_norm

SR_otf_val = psf_telemetry.SR_otf
if np.isscalar(SR_otf_val):
    SR_otf_str = f'{SR_otf_val*100:.1f}%'
else:
    SR_otf_str = ' / '.join(f'{s*100:.1f}%' for s in SR_otf_val)


#PLOTS THE DATA + DATA_FIT + (DATA - DATA_FIT)
fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)
norm_log = LogNorm(vmin=1e-3, vmax=1)

im0 = axes[0].imshow(psf_norm, norm=norm_log, cmap='Spectral_r', extent=extent)
axes[0].set_title('Measured PSF', fontsize=13)
axes[0].set_xlabel('[arcsec]', fontsize=11)
axes[0].set_ylabel('Arcturus, $m_H$ = -2.81', fontsize=11)
axes[0].set_xlim(-hfov+dx, hfov+dx)
axes[0].set_ylim(-hfov-dy, hfov-dy)
axes[0].tick_params(labelsize=10)
plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

im1 = axes[1].imshow(psf_fit_norm, norm=norm_log, cmap='Spectral_r', extent=extent)
axes[1].set_title('Model PSF fit', fontsize=13)
axes[1].set_xlabel('[arcsec]', fontsize=11)
axes[1].set_xlim(-hfov+dx, hfov+dx)
axes[1].set_ylim(-hfov-dy, hfov-dy)
axes[1].tick_params(labelsize=10)
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

res_max = np.max(np.abs(residual))
im2 = axes[2].imshow(residual, vmin=-res_max, vmax=res_max, cmap='RdBu_r', extent=extent)
axes[2].set_title('Residual (measured − fit)', fontsize=13)
axes[2].set_xlabel('[arcsec]', fontsize=11)
axes[2].set_xlim(-hfov+dx, hfov+dx)
axes[2].set_ylim(-hfov-dy, hfov-dy)
axes[2].tick_params(labelsize=10)
plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

fig.suptitle(
    f'SR (image) = {psf_telemetry.SR:.1f}%   |   SR (OTF) = {SR_otf_str}   |   '
    f'Seeing @ obs λ = {psf_telemetry.seeing:.2f}"   |   Seeing @ 550 nm = {psf_telemetry.seeing_550:.2f}"   |   '
    f'r₀ zenith = {psf_telemetry.r0_zenith*100:.1f} cm',
    fontsize=11
)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(f'{savedir}/{filename}.png', dpi=150, bbox_inches='tight')
plt.show()


# #PLOTS a single data PSF image
# plt.figure(figsize=(6, 5))
# plt.title(f'CNN | Strehl = {psf_telemetry.SR:.1f}%', fontsize=20)
# im1 = plt.imshow(psf_norm, norm=LogNorm(vmin=1e-3, vmax=1), cmap='Spectral_r', extent=extent)
# cbar = plt.colorbar(im1, fraction=0.046, pad=0.04)
# cbar.ax.tick_params(labelsize=12)
# plt.xlabel('[arcsec]', fontsize=16)
# plt.ylabel('Arcturus, $m_H$ = -2.81', fontsize=16)
# plt.xlim(-hfov+dx, hfov+dx)
# plt.ylim(-hfov-dy, hfov-dy)
# plt.tick_params(axis='both', labelsize=14)
# plt.tight_layout()
# # plt.savefig(f'{savedir}/CNN.png', dpi=300, bbox_inches='tight')
# plt.show()