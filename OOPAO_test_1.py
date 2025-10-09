import numpy as np
import matplotlib.pyplot as plt

from OOPAO.Atmosphere import Atmosphere
from OOPAO.Zernike import Zernike
from OOPAO.Telescope import Telescope
from OOPAO.Source import Source
from OOPAO.Zernike import Zernike
from OOPAO.DeformableMirror import DeformableMirror
from OOPAO.Pyramid import Pyramid
from OOPAO.calibration.InteractionMatrix import InteractionMatrix
from OOPAO.calibration.compute_KL_modal_basis import compute_KL_basis

#remember that guide stars and sources are not necessarily the same thing
ngs = Source(magnitude = 5,
             optBand = "I")

src = Source(magnitude = 10,
             optBand = "K")

sensing_wavelength = src.wavelength
n_subaperture = 20
diameter = 8
resolution = n_subaperture * 8
pixel_size = diameter/resolution
obs_ratio = 0.1
sampling_time = 1/1000
mechanical_coupling = 0.45


tel = Telescope(diameter = diameter,
                resolution = resolution,
                centralObstruction = obs_ratio,
                samplingTime = sampling_time,
                fov = 4)

ngs*tel # this command only gives the telescope ngs, for more targets pls use asterism
tel.src.print_properties()

src*tel
tel.src.print_properties()

plt.figure()
plt.imshow(tel.pupil)

dm_fried = DeformableMirror(telescope = tel,
                            nSubap = n_subaperture,
                            mechCoupling = mechanical_coupling)
#everything in meters
#you need to couple the telescope with a source object but is it src or ngs?
ngs*tel
atm = Atmosphere(telescope = tel,
                 r0 = 0.15,
                 L0 = 25,
                 fractionalR0=[0.7, 0.3], #these are the C_n^2 coeffs
                 altitude = [0, 10000],
                 windDirection = [0, 20],
                 windSpeed = [5, 10])

pwfs = Pyramid(telescope = tel,
               nSubap = n_subaperture,
               modulation = 3, #radius in terms of lambda/D
               lightRatio = 0.1,
               postProcessing = "slopesMaps",
               n_pix_separation = 10,
               n_pix_edge = 5)

M2C_KL = compute_KL_basis(tel = tel, atm = atm, dm = dm_fried)



#Don't forget to measure the interaction matrix
tel.display_optical_path = False

#this calibrates the response between wfs and dm?
calib_modal = InteractionMatrix(ngs = src,
                                atm = atm,
                                tel = tel,
                                dm = dm_fried,
                                wfs = pwfs,
                                M2C = M2C_KL[:, :300], #mode 2 control matrix (matrix that forces the dm to move in KL modes?)
                                stroke = 1e-9, #stroke for the calibration
                                nMeasurements = 1, #number of measurements in parallel
                                noise = "off") #calibration noise



#need to initialize everything (telescope, atmosphere, dm)
tel.resetOPD() # initialize telescope
atm.initializeAtmosphere(telescope = tel) # atmosphere initialization
dm_fried.coefs = 0 #dm initialization

tel+atm

tel.computePSF(4) #4 is the zeroPaddingfactor (idk)

#calibration data to close the loop

calib_CL = calib_modal
M2C_CL = M2C_KL[:, :300]

#initialize DM commands
ngs*tel*dm_fried*pwfs

nLoop = 200

SR = np.zeros(nLoop)
total = np.zeros(nLoop)
residual = np.zeros(nLoop)
wfsSignal = np.arange(0, pwfs.nSignal)*0

#Loop params
gainCL = 0.4
display = True

#calib_CL.M is the pseudo inverse (takes in slopes (you define it in pwfs postprocessing)
#and outputs modes?)
#M2C_CL takes in modes and outputs the DM controls
reconstructor_matrix = M2C_CL@calib_CL.M


for i in range(nLoop):
    if i == 3:
        plt.subplot(1, 3, 1), plt.imshow(pwfs.m), plt.title("PWFS mask")  # 2D phase mask
        plt.subplot(1, 3, 2), plt.imshow(pwfs.cam.frame), plt.title("PWFS camera framce")  # PWFS camera frame
        plt.subplot(1, 3, 3), plt.imshow(pwfs.focal_plane_camera.frame), plt.title("PWFS focal plane")  # 2D phase mask
        plt.show()
    #update phase screens => overwrite tel.OPD and consequently tel.src.phase
    atm.update()
    #save phase variance
    total[i] = np.std(tel.OPD[np.where(tel.pupil>0)])*1e9 #this is also in nm
    #save turbulent phase
    turbPhase = tel.src.phase
    #propagate to the WFS with the CL commands applied (this is the update step)
    ngs*tel*dm_fried*pwfs
    #propagate to the source with the CL commands applied
    src*tel
    dm_fried.coefs = dm_fried.coefs - gainCL * np.matmul(reconstructor_matrix, wfsSignal)
    #store the slopes after computing the commands => 2 frames delay
    wfsSignal = pwfs.signal
    #store data (where the pupil mask is ==1?
    SR[i] = np.exp(-np.var(tel.src.phase[np.where(tel.pupil==1)]))
    residual[i] = np.std(tel.OPD[np.where(tel.pupil>0)])*1e9 #1e9 is conversion to nm
    print("Loop"+str(i) + '/' + str("nLoop") + "AO residual:" + str(residual[i]) + "nm")

tel.print_optical_path()

plt.figure()
plt.subplot(2, 2, 3)
plt.plot(total)

plt.subplot(2, 2, 4)
plt.plot(residual)

plt.subplot(2, 2, 1)
plt.imshow(pwfs.m)

plt.title("PWFS mask")#2D phase mask
plt.subplot(2, 2, 2)

plt.imshow(pwfs.cam.frame)
plt.title("PWFS camera framce")#PWFS camera frame

plt.show()




