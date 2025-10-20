"""

Build a simple AO system simulation
Test the different parameters
Optimize for resolution and potentially other parameters

NGS/SRC -> atmosphere -> telescope -> dm -> wfs -> camera
split somewhere for src/ngs

figure out what is cam and cam.binned

How do I change the dm actuator number? n_subapertures

What is the Z.modesFullRes vs Z.modes?


WHAT DOES THIS REPRESENT/??? print(np.linalg.pinv(np.squeeze(dm.modes[tel.pupilLogical, :])).shape)


NEXT TASK: prepare slides, do ALL tutorials, make some graphs how SR (and other parameters) change when you change:
resolution +
n_subaperture +
actuator number +
wind speed +
r_0 +
sampling time +

Compare PSFs before correction and after correction (so basically SR calculation)
calculate cutoff frequency (formula in PhDs potentially)

Play around with light ratio parameter on the pwfs
"""

import matplotlib.pyplot as plt
import numpy as np


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
sampling_time = 1/1000 #1/1000 og
diameter = 8
central_obstruction = 0.1
r_0 = 0.15 #0.15

#always check whether there are enough pixels for r_0
pixel_size = diameter / resolution

if 3 * pixel_size >= r_0:
    print("PIXELS ARE SMALL ENOUGH")
else:
    print("YOUR PIXELS ARE TOO BIG!!!")


L_0 = 20
wind_speed = [5, 50] #og 5, 50
fractional_R0 = [0.6, 0.4]
wind_direction = [30, 60] #in deg 30, 60 og
alt = [0, 10000]


mechanical_coupling = 0.4
#in terms of lambda/D
modulation_ratio = 3 #full cycle needs to happen in one frame of the pyramid camera to successfully average it out
light_ratio = 0.1 #flux criterion for subaperture pixel consideration (below the threshlod the dm does not react?)
post_process = "slopesMaps"
zeroPaddingFactor = 6



##############################################################################################################

#SOURCE
#usefull function
#src.print_properties()

##############################################################################################################
src = Source(optBand   = "I",
             magnitude = 2)

print(src.type)

##############################################################################################################

#TELESCOPE
#tel.print_optical_path()
#resoolution - describes the resolution of the simulation

"""
_ tel.OPD       - optical path difference
_ tel.src.phase - 2D map of the phase difference (tel.OPD scaled to src wavelength)  

zeroPaddingFactor refers to the ratio between original and padded signal in FFT
i.e. we just add zeros to the original signal
this helps in doing the numerical FFT - smoother and less choppy?
prevents aliasing
"""



##############################################################################################################


tel =  Telescope(resolution           = resolution,
                 diameter             = 8,
                 samplingTime         = sampling_time,
                 centralObstruction   = central_obstruction
                 )
src * tel

tel.print_optical_path()

tel.computePSF(zeroPaddingFactor)
#we just rescale to the resolution size to zoom into the PSF
size_pixel_arcsec = 206265 * (tel.src.wavelength/tel.D) / zeroPaddingFactor
N                 = 200
zoomed_PSF        = tel.PSF[N:-N, N: -N]
zoomed_PSF        = zoomed_PSF/zoomed_PSF.max()
fov               = zoomed_PSF.shape[0] * size_pixel_arcsec




plt.figure()
plt.imshow(np.log10(zoomed_PSF), extent = [ -fov/2, fov/2, -fov/2, fov/2])
plt.title("Source PSF")
plt.xlabel("arcsec")
plt.ylabel("arcsec")
plt.colorbar()

##############################################################################################################

#ATMOSPHERE
#how does the atmosphere look for different object? (surely it is different turbulence or do you make multiple atmosphere object?)
#always initialize atm.initializeAtmosphere(telescope = tel)

##############################################################################################################

atm = Atmosphere(telescope     = tel,
                 r0            = r_0, #defined at 550nm
                 L0            = L_0,
                 windSpeed     = wind_speed,
                 windDirection = wind_direction,
                 fractionalR0  = fractional_R0,
                 altitude      = alt)


atm.initializeAtmosphere(telescope = tel)
atm.display_atm_layers() #just a print out (NOT just a print out, just need to plt.show())



#combining atm + telescope (always compare diffraction limited case with the aosystem corrected case)
#tel+atm


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


plt.figure()
plt.title("dm OPD")
plt.imshow(dm.OPD)


dm.coefs = np.random.rand(dm.nValidAct)
plt.figure()
plt.title("dm OPD with random actuator position")
plt.imshow(dm.OPD)

##############################################################################################################

#PYRAMID
#pwfs.cam (a child detector class which shows the pyramid sides)

##############################################################################################################
print("detector !")

pwfs = Pyramid(nSubap           = n_subaperture,   #resolution of the pwfs cam (i.e. pupil diameter)
               telescope        = tel,
               modulation       = modulation_ratio,
               lightRatio       = light_ratio,
               postProcessing   = post_process,
               n_pix_separation = 10,
               n_pix_edge       = 5
               )
print("detector !!")
plt.figure()
plt.imshow(pwfs.cam.frame)
plt.title("PWFS camera")

plt.figure()
plt.imshow(pwfs.m)
plt.title("PWFS mask")

pwfs*pwfs.focal_plane_camera
plt.figure()
plt.imshow(pwfs.focal_plane_camera.frame)
plt.title("PWFS focal plane (modulation)")



##############################################################################################################

#SHWFS

##############################################################################################################
"""
SHWFS = ShackHartmann(nSubap       = n_subaperture,
                      telescope    = tel,
                      lightRatio   = light_ratio,
                      is_geometric = False,
                      shannon_sampling = True,
                      threshold_cog= 0.1)


plt.figure()
plt.subplot(1,2,1)
plt.imshow(SHWFS.valid_subapertures)
plt.title("lenslets of SHWFS")

plt.subplot(1,2,2)
plt.imshow(SHWFS.cam.frame)
plt.title("camera of SHWFS")




##############################################################################################################

#CALIBRATION
#calib_zonal.D - interaction matrix (takes in modes and outputs pwfs signal)
#calib_zonal.M - pseudo inverse (takes in pwfs signal and outputs modes (zonal, zernike, KL...))
#calib_zonal.nTrunc = 10 (truncate the 10 last singular values)

##############################################################################################################

M2C_KL = compute_KL_basis(tel = tel, atm = atm, dm = dm)
M2C_KL = M2C_KL[:, :300]
print(M2C_KL.shape)
"""
from OOPAO.Zernike import Zernike

##############################################################################################################

#ZERNIKE
#Zernike.modesFullRes - has all of the modes of Zernike that you can later plot
#can use displayMap to plot ALL of the modes
#Z.modes takes in modes and outputs the OPD

##############################################################################################################


#if you use too many zerniked coefficients and the DM does not have the resolution to reconstruct
#them, then you will have an unstable and incorrect system
Z = Zernike(tel, 400)
Z.computeZernike(tel)
#figure out what the following line actually does (converts the default dm.modes into Z.modes)
M2C_Z = np.linalg.pinv(np.squeeze(dm.modes[tel.pupilLogical, :])) @ Z.modes


print("inverse of dm.modes shape", np.linalg.pinv(np.squeeze(dm.modes[tel.pupilLogical, :])).shape)
print("Z modes matrix shape", Z.modes.shape)
print("Z modes but 2D shape", Z.modesFullRes.shape)
error


#test for zonal coefficients (does not work for now, I am not sure if the M2C_zonal matrix is correct)
#just need to input an identity matrix - the first zonal function moves the first actuator (that is the logic)

M2C_zonal = np.identity(dm.nValidAct)

M2C__ = M2C_zonal
#interaction matrix that takes in DM modes
calib_zonal = InteractionMatrix(ngs             = src,
                                atm             = atm,
                                tel             = tel,
                                dm              = dm,
                                wfs             = pwfs,
                                M2C             = M2C__,  #I assume these are zonal modes since I didn't apply anything
                                stroke          = 1e-9,
                                nMeasurements   = 1,    #they build up the interaction matrix using nMeasurements columns at a time
                                noise           = "off") #noise of the wfs measurements


#phase reconstruction using the interaction/reconstruction matrix
#input_modes - Zernike coefficients?
#calib_zonal.M @ pwfs.signal - reconstructed Zernike coefficients

input_modes = np.random.randn(M2C__.shape[1]) * 1e-9


dm.coefs = M2C__ @ input_modes #need this line for propagation
src*tel*dm*pwfs
tel.print_optical_path() #sanity check


plt.figure()
plt.plot(input_modes, label = "input")
plt.plot(calib_zonal.M @ pwfs.signal, label = "reconstructed")
plt.ylabel("DM commands")
plt.legend()



##############################################################################################################

#CAMERA
#later

##############################################################################################################

#some simulation of my SCAO


tel.resetOPD()
dm.coefs = 0
tel+atm


#dm commands
src*tel*dm*pwfs
tel.print_optical_path()

nLoop = 200

SR        = np.zeros(nLoop)
total     = np.zeros(nLoop)
residual  = np.zeros(nLoop)
pwfssignal = np.arange(0, pwfs.nSignal) * 0

gain = 0.4

#takes in pwfs signal and outputs dm controls (coefficients)
reconstructor = M2C__ @ calib_zonal.M

r_0_range = [0.05, 0.1, 0.15, 0.2, 0.3]
wind_speed_range_1 = [3, 5, 10, 20, 40] #first layer
wind_direction_range_1 = [0, 30, 60, 120, 240]
sampling_time_range = [1/5000, 1/1000, 1/500, 1/100]
resolution_range = [20 * 4, 20 * 6, 20 * 8, 20 * 12]

for i in range(nLoop):
    #update phase screen
    atm.update()
    #phase variance
    total[i] = np.std(tel.OPD[np.where(tel.pupil > 0)]) * 1e9
    #turbulent phase
    turbphase = tel.src.phase
    #propagate through AO
    src * tel * dm * pwfs
    #propagate to the source (dm commands are applied now)
    src * tel
    #update the dm commands (i.e. dm.coefs)
    #notice here that you can either use pwfs.signal (no delay) OR pwfssignal (some delay (how do we quantify it?))
    dm.coefs = dm.coefs - gain * np.matmul(reconstructor, pwfssignal)
    #store the slopes after computing the commands
    pwfssignal = pwfs.signal
    #metrics
    SR[i] = np.exp(-np.var(tel.src.phase[np.where(tel.pupil == 1)]))
    residual [i] = np.std(tel.OPD[np.where(tel.pupil > 0)]) * 1e9
    print("Loop" + str(i) + "/" + str(nLoop) + "AO residual: " + str(residual[i]) + "nm")
    print(f"strehl {SR[i]}")



time = np.arange(0, nLoop * sampling_time, sampling_time)
plt.figure()
plt.plot(time, total, label = "total")
plt.plot(time, residual, label = "residual")
plt.title("total and residual (nm)")
plt.xlabel("time s")
plt.legend()

plt.figure()
plt.plot(time, SR, label = "strehl")
plt.title("Strehl ratio")
plt.xlabel("time s")

plt.figure()
tel.computePSF(zeroPaddingFactor)

AO_PSF = tel.PSF[N: -N, N: -N]
AO_PSF = AO_PSF/AO_PSF.max()
plt.imshow(np.log10(AO_PSF), extent = [-fov/2, fov/2, -fov/2, fov/2])
plt.title("AO corrected PSF")
plt.xlabel("arcsec")
plt.ylabel("arcsec")
plt.colorbar()


np.save("SR_PWFS_subap_40x8.npy", SR)
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