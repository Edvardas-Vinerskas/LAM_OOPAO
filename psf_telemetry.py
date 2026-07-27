import numpy as np
from ozitelemetry.PSFTelemetry import PSFTele
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
from astropy.io import fits


savedir = "D:/bench_sky_04_14/bench_1st200_2nd400_v2_atm_15_3520260414-193033"
filename = "RL v2-CL-2026-04-14T19_39_26-Cube"
#v10 for both innit



psf_telemetry = PSFTele(
    tele_path=f"{savedir}/{filename}.fits",
    is_onsky= False,
    is_cl= True,
    crop_img=100
)

psf_telemetry.psf_analysis(
    elevation_deg=63,
    polychromatic = False #False for on bench since you are using a monochromatic source
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


# THIS WAS USED TO PLOT THE STREHL RATIO GAIN WHEN USING PO4AO vs integrator for 2nd stage
# THIS ONLY HAS DATA FROM 15 and 16 days
# # Strehl ratio [%] per target / reconstructor.
# # 'rl' = reinforcement-learning controller, 'pythint' = python integrator.
# # Arcturus has both a CNN and a linear reconstructor; Dubhe merges its two nights.
# strehl_data = {
#     'Arcturus (CNN)': {
#         'rl':      [42.6, 45.2, 47.5, 49.4],
#         'pythint': [28.6, 34.3, 43.5, 44.7],
#     },
#     'Arcturus (linear)': {
#         'rl':      [37.3, 33.9, 37.2, 30.6],
#         'pythint': [35.7, 29.2, 37.9, 27.1],
#     },
#     'Dubhe': {
#         'rl':      [32.0, 34.1, 31.1, 33.6, 31.0, 30.5, 36.3, 33.1],
#         'pythint': [25.1, 31.2, 29.9, 31.3, 14.2, 23.1, 25.5, 28.3],
#     },
#     'HD134943': {
#         'rl':      [30.4, 17.2, 10.8],
#         'pythint': [30.2, 13.5, 11.2],
#     },
#     'HD98262': {
#         'rl':      [43.2, 41.3, 41.3],
#         'pythint': [40.1, 37.6, 39.7],
#     },
# }


# # Paired comparison: each point is one (integrator, PO4AO) measurement pair.
# # Points above the y = x line are pairs where PO4AO beats the integrator.
# # (color, marker, filled) — Arcturus linear is the open marker of the same hue.
# target_style = {
#     'Arcturus (CNN)':    ('#1f77b4', 'o', True),
#     'Arcturus (linear)': ('#1f77b4', 'o', False),
#     'Dubhe':             ('#ff7f0e', 's', True),
#     'HD134943':          ('#2ca02c', '^', True),
#     'HD98262':           ('#d62728', 'D', True),
# }
# lo, hi = 5, 55

# fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
# ax.plot([lo, hi], [lo, hi], color='0.4', ls='--', lw=1, zorder=1)


# for star, d in strehl_data.items():
#     x, y = d['pythint'], d['rl']          # x: integrator, y: PO4AO
#     color, marker, filled = target_style[star]
#     if filled:
#         ax.scatter(x, y, c=color, marker=marker, s=55,
#                    edgecolor='white', linewidth=0.5, zorder=3, label=star)
#     else:
#         ax.scatter(x, y, facecolors='none', edgecolors=color, marker=marker, s=55,
#                    linewidth=1.3, zorder=3, label=star)

# ax.set_xlim(lo, hi)
# ax.set_ylim(lo, hi)
# ax.set_aspect('equal')
# ax.set_xlabel('Python integrator Strehl [%]', fontsize=12)
# ax.set_ylabel('PO4AO Strehl [%]', fontsize=12)
# ax.set_title(f'Paired comparison', fontsize=13)
# ax.legend(fontsize=10, loc='lower right')
# plt.tight_layout()
# plt.savefig('strehl_PO4AO_vs_pythint.png', dpi=150, bbox_inches='tight')
# plt.show()


