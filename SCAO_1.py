"""
NGS/SRC -> atmosphere -> telescope -> dm -> wfs -> camera
split somewhere for src/ngs

figure out what is cam and cam.binned (one is science and on is wfs?)

FROM 11/10/2025 THIS SHOULD ONLY BE USED FOR FORMULA SCAVENGING
"""

import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import numpy as np
from matplotlib.patches import Circle

from OOPAO.Zernike import Zernike
from functions import *
from numpy.fft import fft, ifft, fft2, fftshift


import OOPAO
from OOPAO.Source import Source
from OOPAO.Telescope import Telescope
from OOPAO.DeformableMirror import DeformableMirror
from OOPAO.Atmosphere import Atmosphere
from OOPAO.Pyramid import Pyramid
from OOPAO.ShackHartmann import ShackHartmann
from OOPAO.calibration.InteractionMatrix import InteractionMatrix
from OOPAO.calibration.compute_KL_modal_basis import compute_KL_basis

##############################################################################################################
#ALL GLOBALS HERE
n_subaperture = 20 #basically the number of actuators og20
#the resolution is the telescope resolution? (you can check by changing it and seeing what the pupil and other masks output)
resolution    = n_subaperture * 8 #n_subaperture * 4 (og) (4 represents the number of pixels per subaperture)
loop_frequency = 1000
sampling_time = 1/loop_frequency #1/1000 og
diameter = 1.52
central_obstruction = 0.1
r_0 = 0.15
mechanical_coupling = 0.4
modulation_ratio = 5 #full cycle needs to happen in one frame of the pyramid camera to successfully average it out (#in terms of lambda/D)
light_ratio = 0.1 #flux criterion for subaperture pixel consideration (below the threshlod the dm does not react?)
post_process = "slopesMaps"
L_0 = 20
wind_speed = [5, 50] #og 5, 50
fractional_R0 = [0.85, 0.15]
wind_direction = [30, 60] #in deg 30, 60 og
alt = [0, 10000]

#always check whether there are enough pixels for r_0
pixel_size = diameter / resolution

if 3 * pixel_size <= r_0:
    print(f"PIXELS ARE SMALL ENOUGH, number of pixels per r_0 {r_0 / pixel_size}")
else:
    print("YOUR PIXELS ARE TOO BIG!!!")

#the factor increases your image size by zeroPaddingFactor, allowing for finer frequency bins in the fft
#so for 6, you image size is increased 6 times
zeroPaddingFactor = 2
Z_coefs = 300

rad2arcsec = 180 * 60 * 60 / np.pi

##############################################################################################################

#SOURCE
#usefull function
#src.print_properties()

##############################################################################################################
src = Source(optBand   = "I",
             magnitude = 2)

wvl = 550e-9
r_0_src = r_0 * (src.wavelength / wvl) ** (6/5)


##############################################################################################################

#TELESCOPE
#tel.print_optical_path()
#resolution - describes the resolution of the simulation

"""
- tel.OPD       - optical path difference
- tel.src.phase - 2D map of the phase difference (tel.OPD scaled to src wavelength)  
- fun fact: tel.src.phase is a 3D matrix before doing the tel+atm and just stores actuator influence functions for each actuator

zeroPaddingFactor refers to the ratio between original and padded signal in FFT
i.e. we just add zeros to the original signal
this helps in doing the numerical FFT - smoother and less choppy?
prevents aliasing

In fact when calculating the PSF, the zeroPaddingFactor acts as the number of pixels per diffraction limited angular resolution
so the tel.PSF is then ffted onto a camera that has a resolution of zeroPaddingFactor number of pixels per diffraction limited angular resolution

NOT TO BE CONFUSED WITH THE telescope resolution parameter which is the resolution of the pupil
In conclusion the pupil resolution is increased by zeroPaddingFactor number of times for the PSF camera

Fun fact: since we are increasing the pixel density, we get less photons per pixel and basically waste them
that is why optimal sampling is key here
"""



##############################################################################################################


tel =  Telescope(resolution           = resolution,
                 diameter             = diameter,
                 samplingTime         = sampling_time,
                 centralObstruction   = central_obstruction
                 )
src * tel

tel.print_optical_path()

tel.computePSF(zeroPaddingFactor)
#we just rescale to the resolution size to zoom into the PSF
#the arcsec_per_pixel code assumes that we are sampling the diffraction limit per zeroPaddingFactor number of pixels (so 6 pixels per diffraction limited angular resolution)
#zeroPaddingFactor effectively acts as a camera with denser/more pixels in the focal plane
arcsec_per_pixel  = 206265 * (tel.src.wavelength/tel.D) / zeroPaddingFactor
N                 = 50
zoomed_PSF        = tel.PSF[N:-N, N: -N]
zoomed_PSF        = zoomed_PSF/np.sum(zoomed_PSF)
fov               = zoomed_PSF.shape[0] * arcsec_per_pixel


#OTF calculation (DON'T FORGET TO NORMALISE THE PSF)
OTF_dl = fftshift(fft2(fftshift(tel.PSF / np.sum(tel.PSF))))

x_axis, OTF_dl_averaged = circular_average((np.abs(OTF_dl)).shape, np.abs(OTF_dl))

plt.figure()
plt.imshow(zoomed_PSF, norm = SymLogNorm(1e-7), extent = [ -fov/2, fov/2, -fov/2, fov/2])
plt.title("Source PSF")
plt.xlabel("arcsec")
plt.ylabel("arcsec")
plt.colorbar()

##############################################################################################################

#ATMOSPHERE
#how does the atmosphere look for different object? (surely it is different turbulence or do you make multiple atmosphere object?)
#always initialize atm.initializeAtmosphere(telescope = tel)
#atm.display_atm_layers() #just a print out (NOT just a print out, just need to plt.show())
#combining atm + telescope (always compare diffraction limited case with the aosystem corrected case)
#tel+atm

##############################################################################################################

atm = Atmosphere(telescope     = tel,
                 r0            = r_0, #defined at 500nm
                 L0            = L_0,
                 windSpeed     = wind_speed,
                 windDirection = wind_direction,
                 fractionalR0  = fractional_R0,
                 altitude      = alt)


atm.initializeAtmosphere(telescope = tel)


##############################################################################################################

#DEFORMABLE MIRROR
#dm.coefs are the coefficients for the actuators
#dm.modes are the influence functions (zonal, modal...) (you modify this when you want a modal function)
#dm.modes tells us which and how much dm.coefs should be moved for a particular OPD (when we propagate)
#pitch - distance between actuators


#a funny thing is that only when using zonal modes, the deformable mirror acts like in a simulation
#when using zernike or KL, the actuators are not simulated? (according to Rafael)
#instead, it just displays perfect zernike modes (which is not what happpens with a real DM)





##############################################################################################################


dm = DeformableMirror(telescope      = tel,
                      nSubap         = n_subaperture,
                      mechCoupling   = mechanical_coupling)

#control radius calculation for the dm
control_radius = ((n_subaperture + 1) * src.wavelength) /(2 * tel.D) * rad2arcsec
corr_zone_1 = Circle([0,0], control_radius, fc='none', ec='w', ls=':')
corr_zone_2 = Circle([0,0], control_radius, fc='none', ec='w', ls=':')
print(f"dm pitch {dm.pitch} vs {r_0_src} r_0 value")



##############################################################################################################

#PYRAMID
#pwfs.cam (a child detector class which shows the pyramid sides)

##############################################################################################################


pwfs = Pyramid(nSubap           = n_subaperture,   #resolution of the pwfs cam (i.e. pupil diameter)
               telescope        = tel,
               modulation       = modulation_ratio,
               lightRatio       = light_ratio,
               postProcessing   = post_process,
               n_pix_separation = 10,
               n_pix_edge       = 5
               )



##############################################################################################################

#SHWFS

##############################################################################################################
"""
SHWFS = ShackHartmann(nSubap       = n_subaperture,
                      telescope    = tel,
                      lightRatio   = light_ratio,
                      is_geometric = False)


plt.figure()
plt.subplot(1,2,1)
plt.imshow(SHWFS.valid_subapertures)
plt.title("lenslets of SHWFS")

plt.subplot(1,2,2)
plt.imshow(SHWFS.cam.frame)
plt.title("camera of SHWFS")

pwfs = SHWFS"""



##############################################################################################################

#CALIBRATION
#calib_zonal.D - interaction matrix (takes in modes and outputs pwfs signal)
#calib_zonal.M - pseudo inverse (takes in pwfs signal and outputs modes (zonal, zernike, KL...))
#calib_zonal.nTrunc = 10 (truncate the 10 last singular values)

##############################################################################################################
'''
M2C_KL = compute_KL_basis(tel = tel, atm = atm, dm = dm)
M2C_KL = M2C_KL[:, :300]
'''


##############################################################################################################

#ZERNIKE
#Zernike.modesFullRes - has all of the modes of Zernike that you can later plot
#can use displayMap to plot ALL of the modes
#Z.modes takes in modes and outputs the phase
#dm.modes takes in controle and outputs phase
#1st and 2nd modes is tip tilt (no piston)
#if you use too many zernike coefficients and the DM does not have the resolution to reconstruct
#them, then you will have an unstable and incorrect system

##############################################################################################################
Z = Zernike(tel, Z_coefs)
Z.computeZernike(tel)
M2C_Z = np.linalg.pinv(np.squeeze(dm.modes[tel.pupilLogical, :])) @ Z.modes


#just need to input an identity matrix - the first zonal function moves the first actuator (that is the logic)

M2C_zonal = np.identity(dm.nValidAct)

M2C__ = M2C_Z
#interaction matrix that takes in DM coefs and outputs phase/slopes
calib_zonal = InteractionMatrix(ngs             = src,
                                atm             = atm,
                                tel             = tel,
                                dm              = dm,
                                wfs             = pwfs,
                                M2C             = M2C__,
                                stroke          = 1e-9,
                                nMeasurements   = 1,     #they build up the interaction matrix using nMeasurements columns at a time
                                noise           = "off") #wfs measurement noise is turned off when doing the interaction matrix

input_modes = np.random.randn(M2C__.shape[1]) * 1e-9



dm.coefs = M2C__ @ input_modes
src*tel*dm*pwfs
tel.print_optical_path() #sanity check


plt.figure()
plt.plot(input_modes, label = "input")
plt.plot(calib_zonal.M @ pwfs.signal, label = "reconstructed")
plt.title("Reconstructed vs input dm modes")
plt.ylabel("DM commands")
plt.legend()
plt.show()
##############################################################################################################

#FITTING ERROR
#you cannot calculate the fitting error by propagating through the whole system
#because the wfs for example will introduce phase to slope mapping errors just due to non infinite resolution
#so you instead must reproject the phase of the atmosphere onto the DM using whatever modes you have


##############################################################################################################
modes_inv = np.linalg.pinv(np.squeeze(Z.modes))
Z_coefficient_matrix_atmosphere_test = modes_inv @ atm.OPD[np.where(tel.pupil == 1)]
dm.coefs = M2C__ @ Z_coefficient_matrix_atmosphere_test

fitting_error = atm.OPD - dm.OPD

simulational_fitting_error = np.std(fitting_error[np.where(tel.pupil > 0)]) * 1e9

##############################################################################################################

#CAMERA
#later

##############################################################################################################

#some simulation of my SCAO
nLoop = 100
frame_delay = 2

tel.resetOPD()
dm.coefs = 0
tel+atm


#dm commands
src*tel*dm*pwfs
tel.print_optical_path()


#delay = frame_delay #I don't think you need to add a +1 here because there is already in built delay in the loop
SR                  = np.zeros(nLoop)
SR_running          = np.zeros(nLoop)
total               = np.zeros(nLoop)
residual            = np.zeros(nLoop)
pwfssignal          = np.arange(0, pwfs.nSignal) * 0#[np.zeros(pwfs.nSignal) for i in range(delay)]
atm_OPD_list        = []

gain = 0.4

#takes in pwfs signal and outputs dm controls (coefficients)
reconstructor = M2C__ @ calib_zonal.M

r_0_range              = [0.05, 0.1, 0.15, 0.2, 0.3]
wind_speed_range_1     = [3, 5, 10, 20, 40] #first layer
wind_direction_range_1 = [0, 30, 60, 120, 240]
sampling_time_range    = [1/5000, 1/1000, 1/500, 1/100]
resolution_range       = [20 * 4, 20 * 6, 20 * 8, 20 * 12]
final_residual_phase   = 0
final_atmosphere_OPD   = 0
final_dm_OPD           = 0

for i in range(nLoop):
    #update phase screen
    atm.update()

    atm_OPD_list.append(atm.OPD)
    #phase variance
    total[i] = np.std(tel.OPD[np.where(tel.pupil > 0)]) * 1e9
    #turbulent phase (residual phase left after correction, for corrected phase pls do dm.OPD)
    turbphase = tel.src.phase
    #propagate through AO with the dm commands applied
    src * tel * dm * pwfs
    #propagate to the source with the dm commands applied
    src * tel


    #update the dm commands (i.e. dm.coefs)
    dm.coefs = dm.coefs - gain * np.matmul(reconstructor, pwfssignal)
    pwfssignal = pwfs.signal #2 frame delay
    #metrics
    SR[i] = np.exp(-np.var(tel.src.phase[np.where(tel.pupil == 1)]))
    residual[i] = np.std(tel.OPD[np.where(tel.pupil > 0)]) * 1e9
    print("Loop" + str(i) + "/" + str(nLoop) + "AO residual: " + str(residual[i]) + "nm")
    print(f"strehl {SR[i]}")
    if i == (nLoop - 1):
        final_residual_phase   = tel.src.phase
        final_atmosphere_OPD   = atm.OPD
        final_dm_OPD           = dm.OPD




simulational_temporal_error = np.zeros(len(atm_OPD_list))
for i in range(len(atm_OPD_list)):
    simulational_temporal_error[i] = np.std((atm_OPD_list[0] - atm_OPD_list[i])[np.where(tel.pupil > 0)]) * 1e9


simulational_fitting_error_plot    = np.ones(len(simulational_temporal_error)) * simulational_fitting_error
simulational_temporal_error_plot_2 = np.ones(len(simulational_temporal_error)) * simulational_temporal_error[2]
simulational_temporal_error_plot_1 = np.ones(len(simulational_temporal_error)) * simulational_temporal_error[1]
sum_fitting_temporal               = simulational_temporal_error_plot_2 + simulational_fitting_error_plot

fitting_error_analytical = (0.23) ** (1 / 2) * (dm.pitch/ r_0_src) ** (5 / 6) * src.wavelength / (2 * np.pi) * 1e9
print(f"analytical fitting error {fitting_error_analytical}")
wind_speed = np.array(wind_speed)
fractional_R0 = np.array(fractional_R0)
v_avg = np.sum(wind_speed * fractional_R0)
temporal_error_analytical = 6.88 * ( 2 * v_avg * sampling_time / r_0_src) ** (5 / 6) * src.wavelength / (2 * np.pi) * 1e9
print(f"analytical temporal error {temporal_error_analytical}")


temporal_error_analytical_plot = np.ones(len(simulational_temporal_error)) * temporal_error_analytical
fitting_error_analytical_plot  = np.ones(len(simulational_temporal_error)) * fitting_error_analytical




plt.figure()
plt.subplot(221)
plt.title("simulational temporal error")
plt.plot(simulational_temporal_error, label = "simulational temporal error")
plt.subplot(222)
plt.title("simulational fitting error")
plt.plot(simulational_fitting_error_plot, label = "simulational fitting error")
plt.subplot(223)
plt.title("analytic temporal error")
plt.plot(temporal_error_analytical_plot, label = "analytic temporal error")
plt.subplot(224)
plt.title("analytic fitting error")
plt.plot(fitting_error_analytical_plot, label = "analytic fitting error")
plt.tight_layout()


SR_running = np.convolve(SR, np.ones(30) / 30, mode = "same")


time = np.arange(0, nLoop * sampling_time, sampling_time)
plt.figure()
plt.plot(time, residual, label = "residual")
plt.plot(time, simulational_fitting_error_plot, label = "simulational fitting error")
plt.plot(time, simulational_temporal_error_plot_2, label = "simulational temporal error 2 frame delay")
plt.plot(time, simulational_temporal_error_plot_1, label = "simulational temporal error 1 frame delay")
plt.plot(time, sum_fitting_temporal, label = "sum of fitting + temporal error 2")
plt.title("residual and other error components (nm)")
plt.xlabel("time s")
plt.yscale("log")
plt.legend()

plt.figure()
plt.plot(time, SR, label = "strehl")
plt.plot(time, SR_running, label = "running_strehl")
plt.title("Strehl ratio")
plt.xlabel("time s")
plt.legend()

plt.figure()
tel.computePSF(zeroPaddingFactor)

AO_PSF = tel.PSF[N: -N, N: -N]
AO_PSF = AO_PSF/np.sum(AO_PSF)
plt.imshow(AO_PSF, norm = SymLogNorm(1e-7), extent = [-fov/2, fov/2, -fov/2, fov/2])
plt.title("AO corrected PSF")
plt.gca().add_artist(corr_zone_1)
plt.xlabel("arcsec")
plt.ylabel("arcsec")
plt.colorbar()




plt.figure()
diff = np.abs(AO_PSF - zoomed_PSF) #to avoid zero values
diff = np.clip(diff, 1e-10, None)
PSF1D_x_axis = np.arange(- fov / 2, fov / 2, fov / zoomed_PSF.shape[0])

plt.plot(PSF1D_x_axis, diff[diff.shape[0] // 2, :], label = "residual")
plt.plot(PSF1D_x_axis, zoomed_PSF[zoomed_PSF.shape[0] // 2, :], label = "diffraction_limited")
plt.plot(PSF1D_x_axis, AO_PSF[AO_PSF.shape[0] // 2, :], label = "AO PSF")
plt.title("PSF 1D")
plt.xlabel("arcsec")
plt.ylabel("arcsec")
plt.legend()


OTF_AO = fftshift(fft2(fftshift(tel.PSF / np.sum(tel.PSF))))
x_axis___, OTF_AO_averaged = circular_average(np.abs(OTF_AO).shape, np.abs(OTF_AO))


OTF_AO_averaged_shwfs = np.load("OTF_magnitude_SHWFS.npy")

plt.figure()
plt.plot(x_axis, OTF_dl_averaged, label = "diffraction limited")
plt.plot(x_axis___, OTF_AO_averaged, label = "AO corrected pwfs")
#plt.plot(x_axis___, OTF_AO_averaged_shwfs, label = "AO corrected shwfs")
plt.title("OTF magnitude diffraction limited")
plt.xlabel("Frequency domain")
plt.ylabel("MTF")
plt.xscale("log")
plt.yscale("log")
plt.legend()




#np.save("OTF_magnitude_SHWFS.npy", OTF_AO_averaged)


corrected_phase = final_residual_phase
PSD_corrected = np.abs(fftshift(fft2(corrected_phase))) ** 2 / ((tel.D / 2) ** 2 * np.pi)
x_axis_PSD_residual, PSD_corrected_averaged = circular_average(np.abs(PSD_corrected).shape, np.abs(PSD_corrected))

atmosphere_phase = 2*np.pi * final_atmosphere_OPD / src.wavelength
PSD_atmosphere = np.abs(fftshift(fft2(atmosphere_phase))) ** 2 / ((tel.D / 2) ** 2 * np.pi)
x_axis_PSD_atmosphere_residual, atmosphere_residual_averaged = circular_average(np.abs(PSD_atmosphere).shape, np.abs(PSD_atmosphere))

dm_phase = 2*np.pi * final_dm_OPD / src.wavelength

Z_coefficient_matrix = Z.modes.T @ final_residual_phase[np.where(tel.pupil > 0)] / (np.diag(Z.modes.T @ Z.modes))
Z_coefficient_matrix_atmosphere = Z.modes.T @ atmosphere_phase[np.where(tel.pupil > 0)] / (np.diag(Z.modes.T @ Z.modes))
zernike_names = []
for i in range(len(Z_coefficient_matrix)):
    zernike_names.append(f"z{i+1}")





fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Find global min/max for consistent color scaling
vmin = min(corrected_phase.min(), atmosphere_phase.min(), dm_phase.min())
vmax = max(corrected_phase.max(), atmosphere_phase.max(), dm_phase.max())

# Plot with shared color scale
im1 = axes[0].imshow(corrected_phase, vmin=vmin, vmax=vmax)
axes[0].set_title("DM_Corrected phase")

im2 = axes[1].imshow(atmosphere_phase, vmin=vmin, vmax=vmax)
axes[1].set_title("Atmosphere phase")

im3 = axes[2].imshow(dm_phase, vmin=vmin, vmax=vmax)
axes[2].set_title("DM phase")

# Add a single shared colorbar on the right side
cbar = fig.colorbar(im3, ax=axes.ravel().tolist(), location='right', fraction=0.02, pad=0.04)
cbar.set_label('Phase value')


plt.figure(figsize = (16,10))
plt.bar(zernike_names, Z_coefficient_matrix, color = "red", label = "Zernike coeffs for residual phase")
plt.title("Zernike coefficients for corrected phase")
plt.bar(zernike_names,Z_coefficient_matrix_atmosphere, color = "black", alpha = 0.4, label = "Zernike coeffs for atmospheric phase")
plt.title("Zernike coefficients for atmosphere phase")
plt.tight_layout()


#PSD_residual_shwfs = np.load("PSD_residual_SHWFS.npy")


#for the x_axis I am just using the length of the PSD so it is not related to any real frequencies
#might want to mitigate that
plt.figure()
plt.plot(x_axis_PSD_residual, PSD_corrected_averaged, label = "PSD_residual_pwfs")
#plt.plot(x_axis_PSD_residual, PSD_residual_shwfs, label = "PSD_residual_shwfs")
plt.plot(x_axis_PSD_atmosphere_residual, atmosphere_residual_averaged, label = "atmosphere_PSD")
plt.title("PSD residual vs atmosphere comparison")
plt.xlabel("Frequency domain")
plt.ylabel("PSD")
plt.xscale("log")
plt.yscale("log")
plt.legend()

#np.save("PSD_residual_SHWFS.npy", PSD_corrected_averaged)























plt.show()

'''
time_1000 = np.arange(0, 200 * 1/1000, 1/1000)
time__ = np.arange(0, 200, 1)

SR_r_0_005 = np.load("SR_r_0_005.npy")
SR_r_0_01 = np.load("SR_r_0_01.npy")
SR_r_0_015 = np.load("SR_r_0_015.npy")
SR_r_0_02 = np.load("SR_r_0_02.npy")
SR_r_0_03 = np.load("SR_r_0_03.npy")


SR_resolution_4 = np.load("SR_resolution_4.npy")
SR_resolution_6 = np.load("SR_resolution_6.npy")
SR_resolution_8 = np.load("SR_resolution_8.npy")
SR_resolution_12 = np.load("SR_resolution_12.npy")


SR_sampling_100 = np.load("SR_sampling_100.npy")
SR_sampling_500 = np.load("SR_sampling_500.npy")
SR_sampling_1000 = np.load("SR_sampling_1000.npy")
SR_sampling_5000 = np.load("SR_sampling_5000.npy")
SR_sampling_10000 = np.load("SR_sampling_10000.npy")


SR_wind_3 = np.load("SR_wind_3.npy")
SR_wind_5 = np.load("SR_wind_5.npy")
SR_wind_10 = np.load("SR_wind_10.npy")
SR_wind_20 = np.load("SR_wind_20.npy")
SR_wind_40 = np.load("SR_wind_40.npy")


SR_wind_dir_0 = np.load("SR_wind_dir_0.npy")
SR_wind_dir_30 = np.load("SR_wind_dir_30.npy")
SR_wind_dir_60 = np.load("SR_wind_dir_60.npy")
SR_wind_dir_120 = np.load("SR_wind_dir_120.npy")
SR_wind_dir_240 = np.load("SR_wind_dir_240.npy")


plt.figure()
plt.plot(time_1000, SR_r_0_005, label = "r_0 = 0.05")
plt.plot(time_1000, SR_r_0_01, label = "r_0 = 0.1")
plt.plot(time_1000, SR_r_0_015, label = "r_0 = 0.15")
plt.plot(time_1000, SR_r_0_02, label = "r_0 = 0.2")
plt.plot(time_1000, SR_r_0_03, label = "r_0 = 0.3")
plt.legend()
plt.xlabel("time s")
plt.ylabel("Strehl ratio")
plt.title("SR comparison for different r_0 values")
plt.savefig("SR_r_0.png", dpi = 200)


plt.figure()
plt.plot(time_1000, SR_resolution_4, label = "res = 80")
plt.plot(time_1000, SR_resolution_6, label = "res = 120")
plt.plot(time_1000, SR_resolution_8, label = "res = 160")
plt.plot(time_1000, SR_resolution_12, label = "res = 240")
plt.legend()
plt.xlabel("time s")
plt.ylabel("Strehl ratio")
plt.title("SR comparison for different resolution values")
plt.savefig("SR_resolution.png", dpi = 200)


plt.figure()
plt.plot(time__, SR_sampling_100, label = "f = 100 Hz")
plt.plot(time__, SR_sampling_500, label = "f = 500 Hz")
plt.plot(time__, SR_sampling_1000, label = "f = 1000 Hz")
plt.plot(time__, SR_sampling_5000, label = "f = 5000 Hz")
plt.plot(time__, SR_sampling_10000, label = "f = 10000 Hz")
plt.legend()
plt.xlabel("time s")
plt.ylabel("Strehl ratio")
plt.title("SR comparison for different loop frequency")
plt.savefig("SR_frequency.png", dpi = 200)


plt.figure()
plt.plot(time_1000, SR_wind_3, label = "speed = 3")
plt.plot(time_1000, SR_wind_5, label = "speed = 5")
plt.plot(time_1000, SR_wind_10, label = "speed = 10")
plt.plot(time_1000, SR_wind_20, label = "speed = 20")
plt.plot(time_1000, SR_wind_40, label = "speed = 40")
plt.legend()
plt.xlabel("time s")
plt.ylabel("Strehl ratio")
plt.title("SR comparison for different wind speeds")
plt.savefig("SR_wind_speed.png", dpi = 200)



plt.figure()
plt.plot(time_1000, SR_wind_dir_0, label = "angle = 0 deg")
plt.plot(time_1000, SR_wind_dir_30, label = "angle = 30 deg")
plt.plot(time_1000, SR_wind_dir_60, label = "angle = 60 deg")
plt.plot(time_1000, SR_wind_dir_120, label = "angle = 120 deg")
plt.plot(time_1000, SR_wind_dir_240, label = "angle = 240 deg")
plt.legend()
plt.xlabel("time s")
plt.ylabel("Strehl ratio")
plt.title("SR comparison for different wind directions")
plt.savefig("SR_wind_direction.png", dpi = 200)'''
'''
time_1000 = np.arange(0, 200 * 1/1000, 1/1000)
SR_SHWFS_subap_10 = np.load("SR_SHWFS_subap_10.npy")
SR_SHWFS_subap_20 = np.load("SR_SHWFS_subap_20.npy")
SR_SHWFS_subap_30 = np.load("SR_SHWFS_subap_30.npy")
SR_SHWFS_subap_40 = np.load("SR_SHWFS_subap_40.npy")
SR_SHWFS_subap_50 = np.load("SR_SHWFS_subap_50.npy")

plt.figure()
plt.plot(time_1000, SR_SHWFS_subap_10, label = "n_subaperture = 10 px")
plt.plot(time_1000, SR_SHWFS_subap_20, label = "n_subaperture = 20 px")
plt.plot(time_1000, SR_SHWFS_subap_30, label = "n_subaperture = 30 px")
plt.plot(time_1000, SR_SHWFS_subap_40, label = "n_subaperture = 40 px")
plt.plot(time_1000, SR_SHWFS_subap_50, label = "n_subaperture = 50 px")
plt.legend()
plt.xlabel("time s")
plt.ylabel("Strehl ratio")
plt.title("SR comparison for different number of subapertures (SHWFS)")
plt.savefig("SR_subaperture.png", dpi = 200)



plt.show()'''