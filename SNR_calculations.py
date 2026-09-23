import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
"""
SNR calculations

masks are custom

The calculations in theory should be as follows (but adapt them to your particular case):

average_frame = np.mean(vzwfs_frame_cube, axis = 0) # assumes frame_cube already background subtracted
valid_pix_map = average_frame > np.max(average_frame) * 0.1 #select this value to get the pupils or get the SHM with the valid pixels

print(vzwfs_frame_cube[:, 0, 0])

signal = average_frame[valid_pix_map]
background = average_frame[~valid_pix_map]

flux_per_pixel = np.mean(signal) - np.median(background)
snr = flux_per_pixel / np.sqrt(flux_per_pixel + np.var(background) * N_frames)



So the problem is that after so much averaging you get no random background (averages to almost 0).
Instead you get left with some offset fixed pattern. This means the multiplying by N_frames does not really recover the background

Since you have the data sets, you can calculate the background from a single raw frame and estimate the SNR that way
"""


def annulus_mask(shape, center, r_in, r_out):
    """Boolean annulus mask.

    shape  : (ny, nx) of the frame
    center : (x, y) in pixel coords, i.e. (column, row) as displayed by imshow
    r_in   : inner radius in pixels, included (r_in = 0 gives a filled disk)
    r_out  : outer radius in pixels, included
    """
    ny, nx = shape
    cx, cy = center
    y, x = np.ogrid[:ny, :nx]
    r2 = (x - cx) ** 2 + (y - cy) ** 2
    return (r2 >= r_in ** 2) & (r2 <= r_out ** 2)


directory = "bench_sky_04_16/onsky_HD98262_1st200_2nd400_v7_linear20260417-015851"
filename = "2026-04-17T02_05_32_telemetry_2nd_data_RLiter100"
xxxx = np.load(f"{directory}/{filename}.npy", allow_pickle=True)



wfs_frame_dark_subtracted = xxxx.item()['CRED2Cube'] - xxxx.item()['credDark']
wfs_frame_dark_subtracted = wfs_frame_dark_subtracted[:, 1:]


centers = [(60.5, 82.5), (59.5, 174.0)]
r_in = 14
r_out = 39

r_in_bg = 0
r_out_bg = 70

shape = wfs_frame_dark_subtracted.shape[-2:]
masks = [annulus_mask(shape, c, r_in, r_out) for c in centers]
background_mask = [annulus_mask(shape, (60.5, 82.5), r_in_bg, r_out_bg), annulus_mask(shape, (59.5, 174.0), r_in_bg, 70)] #custom

mask = sum(masks) > 0
background_mask = sum(background_mask) > 0

wfs_frame_dark_subtracted_avg = np.mean(wfs_frame_dark_subtracted, axis = 0)
vmin, vmax = np.percentile(wfs_frame_dark_subtracted[::500], (1, 99.5))


signal = wfs_frame_dark_subtracted_avg[mask]
background = wfs_frame_dark_subtracted_avg[~background_mask]
background_single_frame = wfs_frame_dark_subtracted[1000][~background_mask]


flux_per_pixel = np.mean(signal) - np.median(background) #here you just remove the background in the average frame=
#here you try to calculate the snr per frame and so you need background from single frame for accurate noise estimation
#since if you used just background you would get the static pattern as raw frame noise is averaged out
#this is also the reason that you don't need to multiply by N the background variance
snr_per_frame = flux_per_pixel / np.sqrt(flux_per_pixel + np.var(background_single_frame))


print(f"SNR is: {snr_per_frame}")
print(f"median background: {np.median(background)}")
print(f"variance background (avg): {np.var(background)}")
print(f"variance background (single): {np.var(background_single_frame)}")
print(f"number of frames: {wfs_frame_dark_subtracted.shape[0]}")


#calculated SNR of all targets
arcturus_SNR = [18.26, 15.30, 17.61, 17.88]
dubhe_SNR = [3.88, 3.76, 3.77, 3.87, 3.53]
HD98262_SNR = [1.55, 1.54, 1.52, 1.48]
HD134943_SNR = [1.12, 1.16, 1.29]

print(f'SNR arcturus average: {np.mean(arcturus_SNR)}')
print(f'SNR dubhe average: {np.mean(dubhe_SNR)}')
print(f'SNR HD98262 average: {np.mean(HD98262_SNR)}')
print(f'SNR HD134943 average: {np.mean(HD134943_SNR)}')
errr



stdd = np.std(wfs_frame_dark_subtracted, axis = 0) * mask
sqrtt = np.sqrt(wfs_frame_dark_subtracted_avg)

plt.figure()
plt.imshow(wfs_frame_dark_subtracted[1000], vmin = 0)

plt.figure()
plt.imshow(wfs_frame_dark_subtracted_avg, vmin = 0)

plt.figure()
plt.imshow(mask)
plt.title(f"annuli r_in={r_in} r_out={r_out}")

plt.show()

