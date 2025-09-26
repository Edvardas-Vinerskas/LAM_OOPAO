# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 13:40:49 2023

@author: cheritier
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from OOPAO.calibration.InteractionMatrix import InteractionMatrix
from OOPAO.tools.displayTools import cl_plot, displayMap

# %%
plt.ion()
# number of subaperture for the WFS
n_subaperture = 20


# %%-----------------------     TELESCOPE   ----------------------------------
from OOPAO.Telescope import Telescope

# create the Telescope object
tel = Telescope(resolution           = 6*n_subaperture,                          # resolution of the telescope in [pix]
                diameter             = 8,                                        # diameter in [m]        
                samplingTime         = 1/1000,                                   # Sampling time in [s] of the AO loop
                centralObstruction   = 0.1,                                      # Central obstruction in [%] of a diameter 
                display_optical_path = False,                                    # Flag to display optical path
                fov                  = 4 )                                     # field of view in [arcsec]. If set to 0 (default) this speeds up the computation of the phase screens but is uncompatible with off-axis targets

# # Apply spiders to the telescope pupil
# thickness_spider    = 0.05                                                       # thickness of the spiders in m
# angle               = [45, 135, 225, 315]                                        # angle in degrees for each spider
# offset_Y            = [-0.2, -0.2, 0.2, 0.2]                                     # shift offsets for each spider
# offset_X            = None

# tel.apply_spiders(angle, thickness_spider, offset_X=offset_X, offset_Y=offset_Y)

# # display current pupil
# plt.figure()
# plt.imshow(tel.pupil)

#%% -----------------------     NGS   ----------------------------------
from OOPAO.Source import Source

# create the Natural Guide Star object
ngs = Source(optBand     = 'I',           # Optical band (see photometry.py)
             magnitude   = 8,             # Source Magnitude
             coordinates = [0,0])         # Source coordinated [arcsec,deg]

# combine the NGS to the telescope using '*'
ngs*tel

# create the Scientific Target object located at 10 arcsec from the  ngs
src = Source(optBand     = 'K',           # Optical band (see photometry.py)
             magnitude   = 8,              # Source Magnitude
             coordinates = [2,0])        # Source coordinated [arcsec,deg]

# combine the SRC to the telescope using '*'
src*tel

# check that the ngs and tel.src objects are the same
tel.src.print_properties()
#%% Computing PSF using the telescope method

# you can apply the focal plane shift of the PSF for off-axis source 
tel.apply_off_axis_tip_tilt = True

tel.computePSF(zeroPaddingFactor = 6)
plt.figure()
plt.imshow(np.log10(np.abs(tel.PSF)),
           extent = [tel.xPSF_arcsec[0],tel.xPSF_arcsec[1],tel.xPSF_arcsec[0],tel.xPSF_arcsec[1]])
plt.clim([-1,4])
plt.xlabel('[Arcsec]')
plt.ylabel('[Arcsec]')
plt.colorbar()
plt.title('src PSF with off-axis tip tilt enabled')

# you can remove the focal plane shift of the PSF 
tel.apply_off_axis_tip_tilt = False


# compute PSF 
tel.computePSF(zeroPaddingFactor = 6)
plt.figure()
plt.imshow(np.log10(np.abs(tel.PSF)),
           extent = [tel.xPSF_arcsec[0],tel.xPSF_arcsec[1],tel.xPSF_arcsec[0],tel.xPSF_arcsec[1]])
plt.clim([-1,4])
plt.xlabel('[Arcsec]')
plt.ylabel('[Arcsec]')
plt.colorbar()
plt.title('src PSF with off-axis tip tilt disabled')


# same with NGS (on-axis) and different wavelength
ngs*tel
tel.computePSF(zeroPaddingFactor = 6)
plt.figure()
plt.imshow(np.log10(np.abs(tel.PSF)),
           extent = [tel.xPSF_arcsec[0],tel.xPSF_arcsec[1],tel.xPSF_arcsec[0],tel.xPSF_arcsec[1]])
plt.clim([-1,4])
plt.xlabel('[Arcsec]')
plt.ylabel('[Arcsec]')
plt.colorbar()
plt.title('ngs PSF')


#%% -----------------------     ATMOSPHERE   ----------------------------------
from OOPAO.Atmosphere import Atmosphere
           
# create the Atmosphere object
atm = Atmosphere(telescope     = tel,                               # Telescope                              
                 r0            = 0.15,                              # Fried Parameter [m]
                 L0            = 25,                                # Outer Scale [m]
                 fractionalR0  = [0.45 ,0.1  ,0.1  ,0.25  ,0.1   ], # Cn2 Profile
                 windSpeed     = [10   ,12   ,11   ,15    ,20    ], # Wind Speed in [m]
                 windDirection = [0    ,72   ,144  ,216   ,288   ], # Wind Direction in [degrees]
                 altitude      = [0    ,1000 ,5000 ,10000 ,12000 ]) # Altitude Layers in [m]


# initialize atmosphere with current Telescope
atm.initializeAtmosphere(tel)

# The phase screen can be updated using atm.update method (Temporal sampling given by tel.samplingTime)
atm.update()

# display the atm.OPD = resulting OPD 
plt.figure()
plt.imshow(atm.OPD*1e9)
plt.title('OPD Turbulence [nm]')
plt.colorbar()

# display the atmosphere layers for the sources specified in list_src: 
atm.display_atm_layers(list_src=[ngs,src])

# the sources coordinates can be updated on the fly: 
src.coordinates = [0,0]
atm.display_atm_layers(list_src=[ngs,src])


#%% -----------------------     Scientific Detector   ----------------------------------
from OOPAO.Detector import Detector

# define a detector with its properties (see Detector class for further documentation)
cam = Detector(integrationTime = tel.samplingTime,      # integration time of the detector
               photonNoise     = True,                  # enable photon noise
               readoutNoise    = 0,                     # readout of the detector in [e-/pixel]
               QE              = 1,                   # quantum efficiency
               psf_sampling    = 2,                     # sampling for the PSF computation 2 = Shannon sampling
               binning         = 1)                     # Binning factor of the PSF

cam_binned = Detector( integrationTime = tel.samplingTime,      # integration time of the detector
                       photonNoise     = True,                  # enable photon noise
                       readoutNoise    = 2,                     # readout of the detector in [e-/pixel]
                       QE              = 0.8,                   # quantum efficiency
                       psf_sampling    = 2,                     # sampling for the PSF computation 2 = Shannon sampling
                       binning         = 4)                     # Binning factor of the PSF


# computation of a PSF on the detector using the '*' operator
src*tel*cam*cam_binned

plt.figure()
plt.imshow(cam.frame,extent=[-cam.fov_arcsec/2,cam.fov_arcsec/2,-cam.fov_arcsec/2,cam.fov_arcsec/2])
plt.xlabel('Angular separation [arcsec]')
plt.ylabel('Angular separation [arcsec]')
plt.title('Pixel size: '+str(np.round(cam.pixel_size_arcsec,3))+'"')

plt.figure()
plt.imshow(cam_binned.frame,extent=[-cam_binned.fov_arcsec/2,cam_binned.fov_arcsec/2,-cam_binned.fov_arcsec/2,cam_binned.fov_arcsec/2])
plt.xlabel('Angular separation [arcsec]')
plt.ylabel('Angular separation [arcsec]')
plt.title('Pixel size: '+str(np.round(cam_binned.pixel_size_arcsec,3))+'"')

#%%         PROPAGATE THE LIGHT THROUGH THE ATMOSPHERE
# The Telescope and Atmosphere can be combined using the '+' operator (Propagation through the atmosphere): 
tel+atm # This operations makes that the tel.OPD is automatically over-written by the value of atm.OPD when atm.OPD is updated. 

# It is possible to print the optical path: 
tel.print_optical_path()

# computation of a PSF on the detector using the '*' operator
atm*ngs*tel*cam*cam_binned

plt.figure()
plt.imshow(cam.frame,extent=[-cam.fov_arcsec/2,cam.fov_arcsec/2,-cam.fov_arcsec/2,cam.fov_arcsec/2])
plt.xlabel('Angular separation [arcsec]')
plt.ylabel('Angular separation [arcsec]')
plt.title('Pixel size: '+str(np.round(cam.pixel_size_arcsec,3))+'"')

plt.figure()
plt.imshow(cam_binned.frame,extent=[-cam_binned.fov_arcsec/2,cam_binned.fov_arcsec/2,-cam_binned.fov_arcsec/2,cam_binned.fov_arcsec/2])
plt.xlabel('Angular separation [arcsec]')
plt.ylabel('Angular separation [arcsec]')
plt.title('Pixel size: '+str(np.round(cam_binned.pixel_size_arcsec,3))+'"')

# The Telescope and Atmosphere can be separated using the '-' operator (Free space propagation) 
tel-atm
tel.print_optical_path()

# computation of a PSF on the detector using the '*' operator
ngs*tel*cam*cam_binned

plt.figure()
plt.imshow(cam.frame,extent=[-cam.fov_arcsec/2,cam.fov_arcsec/2,-cam.fov_arcsec/2,cam.fov_arcsec/2])
plt.xlabel('Angular separation [arcsec]')
plt.ylabel('Angular separation [arcsec]')
plt.title('Pixel size: '+str(np.round(cam.pixel_size_arcsec,3))+'"')

plt.figure()
plt.imshow(cam_binned.frame,extent=[-cam_binned.fov_arcsec/2,cam_binned.fov_arcsec/2,-cam_binned.fov_arcsec/2,cam_binned.fov_arcsec/2])
plt.xlabel('Angular separation [arcsec]')
plt.ylabel('Angular separation [arcsec]')
plt.title('Pixel size: '+str(np.round(cam_binned.pixel_size_arcsec,3))+'"')

#%% -----------------------     DEFORMABLE MIRROR   ----------------------------------
from OOPAO.DeformableMirror import DeformableMirror
from OOPAO.MisRegistration import MisRegistration

# mis-registrations object (rotation, shifts..)
misReg = MisRegistration()
misReg.shiftX = 0           # in [m]
misReg.shiftY = 0           # in [m]
misReg.rotationAngle = 0    # in [deg]


# specifying a given number of actuators along the diameter: 
nAct = n_subaperture+1
    
dm = DeformableMirror(telescope  = tel,                        # Telescope
                    nSubap       = nAct-1,                     # number of subaperture of the system considered (by default the DM has n_subaperture + 1 actuators to be in a Fried Geometry)
                    mechCoupling = 0.35,                       # Mechanical Coupling for the influence functions
                    misReg       = misReg,                     # Mis-registration associated 
                    coordinates  = None,                       # coordinates in [m]. Should be input as an array of size [n_actuators, 2] 
                    pitch        = tel.D/nAct)                 # inter actuator distance. Only used to compute the influence function coupling. The default is based on the n_subaperture value. 
    

# plot the dm actuators coordinates with respect to the pupil

dm.display_dm()
plt.plot(dm.coordinates[:,0],dm.coordinates[:,1],'rx')
plt.xlabel('[m]')
plt.ylabel('[m]')
plt.title('DM Actuator Coordinates')


#%% -----------------------     Pyramid WFS   ----------------------------------
from OOPAO.Pyramid import Pyramid

# make sure that the ngs is propagated to the wfs
ngs*tel

wfs = Pyramid(nSubap            = n_subaperture,                # number of subaperture = number of pixel accros the pupil diameter
              telescope         = tel,                          # telescope object
              lightRatio        = 0.5,                          # flux threshold to select valid sub-subaperture
              modulation        = 3,                            # Tip tilt modulation radius
              binning           = 1,                            # binning factor (applied only on the )
              n_pix_separation  = 4,                            # number of pixel separating the different pupils
              n_pix_edge        = 8,                            # number of pixel on the edges of the pupils
              postProcessing    = 'slopesMaps_incidence_flux')  # slopesMap_incidence_flux, fullFrame_incidence_flux (see documentation)


#  display the Pyramid pupils
plt.figure()
plt.imshow(wfs.cam.frame)

#  display the Pyramid signals
plt.figure()
plt.imshow(wfs.signal_2D)
plt.colorbar()

plt.figure()
plt.imshow(wfs.referenceSignal_2D)
plt.colorbar()


#  propagate to the focal plane camera to see the modulation path
wfs*wfs.focal_plane_camera
plt.figure()
plt.imshow(wfs.focal_plane_camera.frame)


#%% Useful Pyhramid methods an properties

# shift the Pyramid pupils on the detector
wfs.apply_shift_wfs(sx = [6,6,6,6], sy= [4,4,4,4])
plt.figure()
plt.imshow(wfs.cam.frame)
# re-initialize (sx=0 and sy=0)
wfs.apply_shift_wfs()
plt.figure()
plt.imshow(wfs.cam.frame)


# grab a Pyramid quadrant
plt.figure()
plt.imshow(wfs.grabFullQuadrant(n=1))  # full quadrant of the detector

plt.figure()
plt.imshow(wfs.grabQuadrant(n=1))  # quadrant used to compute the slopes-maps




#%% variants of Pyramid 

# user defined number of modulation point    
wfs_ = Pyramid(nSubap            = n_subaperture,                # number of subaperture = number of pixel accros the pupil diameter
              telescope         = tel,                          # telescope object
              lightRatio        = 0.5,                          # flux threshold to select valid sub-subaperture
              modulation        = 3,                            # Tip tilt modulation radius
              binning           = 1,                            # binning factor (applied only on the )
              n_pix_separation  = 4,                            # number of pixel separating the different pupils
              n_pix_edge        = 2,                            # number of pixel on the edges of the pupils
              postProcessing    = 'slopesMaps_incidence_flux',  # slopesMap_incidence_flux, fullFrame_incidence_flux (see documentation)
              nTheta_user_defined = 14)


wfs_*wfs_.focal_plane_camera
plt.figure()
plt.imshow(wfs_.focal_plane_camera.frame)
# user defined modulation (square)
x,y =np.meshgrid( np.linspace(-5,5,11), np.linspace(-5,5,11))

x = x.flatten()
y= y.flatten()

user_modulation = []

for i in range(len(x)):
    user_modulation.append([x[i],y[i]])
    
    
wfs__ = Pyramid(nSubap            = n_subaperture,                # number of subaperture = number of pixel accros the pupil diameter
              telescope         = tel,                          # telescope object
              lightRatio        = 0.5,                          # flux threshold to select valid sub-subaperture
              modulation        = 3,                            # Tip tilt modulation radius
              binning           = 1,                            # binning factor (applied only on the )
              n_pix_separation  = 4,                            # number of pixel separating the different pupils
              n_pix_edge        = 2,                            # number of pixel on the edges of the pupils
              postProcessing    = 'slopesMaps_incidence_flux',  # slopesMap_incidence_flux, fullFrame_incidence_flux (see documentation)
              user_modulation_path = user_modulation)


wfs__*wfs__.focal_plane_camera
plt.figure()
plt.imshow(wfs__.focal_plane_camera.frame)



# user-defined valid pixels

valid_map = wfs.validI4Q
wfs___ = Pyramid(nSubap            = n_subaperture,                # number of subaperture = number of pixel accros the pupil diameter
              telescope         = tel,                          # telescope object
              lightRatio        = 0.5,                          # flux threshold to select valid sub-subaperture
              modulation        = 3,                            # Tip tilt modulation radius
              binning           = 1,                            # binning factor (applied only on the )
              n_pix_separation  = 4,                            # number of pixel separating the different pupils
              n_pix_edge        = 2,                            # number of pixel on the edges of the pupils
              postProcessing    = 'slopesMaps_incidence_flux',  # slopesMap_incidence_flux, fullFrame_incidence_flux (see documentation)
              userValidSignal = valid_map)


plt.figure()
plt.imshow(wfs___.validSignal)

#%% -----------------------     Modal Basis - Zernike  ----------------------------------
# from OOPAO.Zernike import Zernike

# #% ZERNIKE Polynomials
# # create Zernike Object
# Z = Zernike(tel,20)
# # compute polynomials for given telescope
# Z.computeZernike(tel)

# # mode to command matrix to project Zernike Polynomials on DM
# M2C_zernike = np.linalg.pinv(np.squeeze(dm.modes[tel.pupilLogical,:]))@Z.modes

# # show the first 10 zernikes applied on the DM
# dm.coefs = M2C_zernike[:,:10]
# tel*dm
# displayMap(tel.OPD)

#%% -----------------------     Modal Basis - KL Basis  ----------------------------------


from OOPAO.calibration.compute_KL_modal_basis import compute_KL_basis
# use the default definition of the KL modes with forced Tip and Tilt. For more complex KL modes, consider the use of the compute_KL_basis function. 
M2C_KL = compute_KL_basis(tel,
                          atm,
                          dm,
                          lim = 0) # inversion stability criterion

# apply the 10 first KL modes
dm.coefs = M2C_KL[:,:10]
# propagate through the DM
ngs*tel*dm
# show the first 10 KL modes applied on the DM
displayMap(tel.OPD)
#%% -----------------------     Calibration: Interaction Matrix  ----------------------------------

# amplitude of the modes in m
stroke=1e-9
# zonal Interaction Matrix
M2C_zonal = np.eye(dm.nValidAct)

# modal Interaction Matrix for 300 modes
M2C_modal = M2C_KL[:,:300]

# swap to geometric WFS for the calibration
ngs*tel*wfs # make sure that the proper source is propagated to the WFS
wfs.is_geometric = True
# zonal interaction matrix
calib_modal = InteractionMatrix(ngs            = ngs,
                                atm            = atm,
                                tel            = tel,
                                dm             = dm,
                                wfs            = wfs,   
                                M2C            = M2C_modal, # M2C matrix used 
                                stroke         = stroke,    # stroke for the push/pull in M2C units
                                nMeasurements  = 8,        # number of simultaneous measurements
                                noise          = 'off',     # disable wfs.cam noise 
                                display        = True,      # display the time using tqdm
                                single_pass    = True)      # only push to compute the interaction matrix instead of push-pull


plt.figure()
plt.plot(np.std(calib_modal.D,axis=0))
plt.xlabel('Mode Number')
plt.ylabel('WFS slopes STD')

# switch back to diffractive WFS
wfs.is_geometric = False

#%% Define instrument and WFS path detectors
from OOPAO.Detector import Detector
# instrument path
src_cam = Detector(tel.resolution*2)
src_cam.psf_sampling = 4  # sampling of the PSF
src_cam.integrationTime = tel.samplingTime # exposure time for the PSF

# put the scientific target off-axis to simulate anisoplanetism (set to  [0,0] to remove anisoplanetism)
src.coordinates = [2,0]

# WFS path
ngs_cam = Detector(tel.resolution*2)
ngs_cam.psf_sampling = 4
ngs_cam.integrationTime = tel.samplingTime

# initialize the Strehl Ratio computation
tel.resetOPD()

ngs*tel*ngs_cam
ngs_psf_ref = ngs_cam.frame.copy()

src*tel*src_cam

src_psf_ref = src_cam.frame.copy()

#%%  Closed loop simulation
from OOPAO.tools.tools import strehlMeter

plt.close('all')

# These are the calibration data used to close the loop
calib_CL = calib_modal
M2C_CL = M2C_modal
reconstructor = M2C_CL@calib_CL.M 

# initialize Telescope DM commands
tel.resetOPD()
dm.coefs=0
ngs*tel*dm*wfs


# You can update the the atmosphere parameter on the fly
atm.r0 = 0.15
atm.windSpeed = list(np.random.randint(5,20,atm.nLayer))
atm.windDirection = list(np.random.randint(0,360,atm.nLayer))

# To make sure to always replay the same turbulence, generate a new phase screen for the atmosphere and combine it with the Telescope
atm.generateNewPhaseScreen(seed=10)

# combine telescope with atmosphere
tel+atm

# propagate both sources
atm*ngs*tel*ngs_cam
atm*src*tel*src_cam

# loop parameters
nLoop = 500  # number of iterations
gainCL = 0.4  # integrator gain
wfs.cam.photonNoise = False  # enable photon noise on the WFS camera
display = True  # enable the display
frame_delay = 2  # number of frame delay

# variables used to to save closed-loop data data
SR_ngs = np.zeros(nLoop)
SR_src = np.zeros(nLoop)

wfe_atmosphere = np.zeros(nLoop)
wfe_residual_SRC = np.zeros(nLoop)
wfe_residual_NGS = np.zeros(nLoop)
wfsSignal = np.arange(0, wfs.nSignal)*0  # buffer to simulate the loop delay

# configure the display pannel
plot_obj = cl_plot(list_fig=[atm.OPD,  # list of data for the different subplots
                             tel.OPD,
                             tel.OPD,
                             wfs.cam.frame,
                             wfs.focal_plane_camera.frame,
                             [[0, 0], [0, 0], [0, 0]],
                             np.log10(ngs_cam.frame),
                             np.log10(src_cam.frame)],
                   type_fig=['imshow',  # type of figure for the different subplots
                             'imshow',
                             'imshow',
                             'imshow',
                             'imshow',
                             'plot',
                             'imshow',
                             'imshow'],
                   list_title=['Turbulence [nm]',  # list of title for the different subplots
                               'NGS residual [m]',
                               'SRC residual [m]',
                               'WFS Detector',
                               'WFS Foxal Plane Camera',
                               None,
                               None,
                               None],
                   list_legend=[None,  # list of legend labels for the subplots
                                None,
                                None,
                                None,
                                None,
                                ['SRC@'+str(src.coordinates[0])+'"', 'NGS@'+str(ngs.coordinates[0])+'"'],
                                None,
                                None],
                   list_label=[None,  # list of axis labels for the subplots
                               None,
                               None,
                               None,
                               None,
                               ['Time', 'WFE [nm]'],
                               ['NGS PSF@' + str(ngs.coordinates[0]) + '" -- FOV: ' + str(np.round(ngs_cam.fov_arcsec, 2)) + '"', ''],
                               ['SRC PSF@' + str(src.coordinates[0]) + '" -- FOV: ' + str(np.round(src_cam.fov_arcsec, 2)) + '"', '']],
                   n_subplot=[4, 2],
                   list_display_axis=[None,  # list of the subplot for which axis are displayed
                                      None,
                                      None,
                                      None,
                                      None,
                                      True,
                                      None,
                                      None],
                   list_ratio=[[0.95, 0.95, 0.1],
                               [1, 1, 1, 1]],
                   s=20)  # size of the scatter markers


for i in range(nLoop):
    a = time.time()
    # update phase screens => overwrite tel.OPD and consequently tel.src.phase
    atm.update()
    # save the wave-front error of the incoming turbulence within the pupil
    wfe_atmosphere[i] = np.std(tel.OPD[np.where(tel.pupil > 0)])*1e9
    # propagate light from the ngs through the atmosphere, telescope, DM to the WFS and ngs camera
    atm*ngs*tel*dm*wfs*ngs_cam
    # propagate to the focal plane camera
    wfs*wfs.focal_plane_camera
    # save residuals corresponding to the ngs
    wfe_residual_NGS[i] = np.std(tel.OPD[np.where(tel.pupil > 0)])*1e9
    # save Strehl ratio from the PSF image
    SR_ngs[i] = strehlMeter(PSF=ngs_cam.frame, tel=tel, PSF_ref=ngs_psf_ref, display=False)
    # save the OPD seen by the ngs
    OPD_NGS = tel.mean_removed_OPD.copy()
    if display:
        NGS_PSF = np.log10(np.abs(ngs_cam.frame))

    # propagate light from the src through the atmosphere, telescope, DM to the src camera
    atm*src*tel*dm*src_cam
    # save residuals corresponding to the SRC
    wfe_residual_SRC[i] = np.std(tel.OPD[np.where(tel.pupil > 0)])*1e9
    # save the OPD seen by the src
    OPD_SRC = tel.mean_removed_OPD.copy()
    # save Strehl ratio from the PSF image
    SR_src[i] = strehlMeter(PSF=src_cam.frame, tel=tel, PSF_ref=src_psf_ref, display=False)

    # store the slopes after propagating to the WFS <=> 1 frames delay
    if frame_delay == 1:
        wfsSignal = wfs.signal

    # apply the commands on the DM
    dm.coefs = dm.coefs - gainCL*np.matmul(reconstructor, wfsSignal)

    # store the slopes after computing the commands <=> 2 frames delay
    if frame_delay == 2:
        wfsSignal = wfs.signal
    # print('Elapsed time: ' + str(time.time()-a) + ' s')

    # update displays if required
    if display and i > 1:
        SRC_PSF = np.log10(np.abs(src_cam.frame))
        # update range for PSF images
        plot_obj.list_lim = [None,
                             None,
                             None,
                             None,
                             None,
                             None,
                             [NGS_PSF.max()-6, NGS_PSF.max()],
                             [SRC_PSF.max()-6, SRC_PSF.max()]]
        # update title
        plot_obj.list_title = ['Turbulence '+str(np.round(wfe_atmosphere[i]))+'[nm]',
                               'NGS residual '+str(np.round(wfe_residual_NGS[i]))+'[nm]',
                               'SRC residual '+str(np.round(wfe_residual_SRC[i]))+'[nm]',
                               'WFS Detector',
                               'WFS Focal Plance Camera',
                               None,
                               None,
                               None]

        cl_plot(list_fig=[1e9*atm.OPD,
                          1e9*OPD_NGS,
                          1e9*OPD_SRC,
                          wfs.cam.frame,
                          np.log10(wfs.focal_plane_camera.frame),
                          [np.arange(i+1), wfe_residual_SRC[:i+1], wfe_residual_NGS[:i+1]],
                          NGS_PSF,
                          SRC_PSF],
                plt_obj=plot_obj)
        plt.pause(0.01)
        if plot_obj.keep_going is False:
            break
    print('-----------------------------------')
    print('Loop'+str(i) + '/' + str(nLoop))
    print('NGS: Strehl ratio [%] : ', np.round(SR_ngs[i],1), ' WFE [nm] : ', np.round(wfe_residual_NGS[i],2))
    print('SRC: Strehl ratio [%] : ', np.round(SR_src[i],1), ' WFE [nm] : ', np.round(wfe_residual_SRC[i],2))

    
    
#%% Closed Loop data analysis

plt.figure()
plt.plot(np.arange(nLoop)*tel.samplingTime, wfe_atmosphere, label='Turbulence')
plt.plot(np.arange(nLoop)*tel.samplingTime, wfe_residual_NGS, label='NGS')
plt.plot(np.arange(nLoop)*tel.samplingTime, wfe_residual_SRC, label='SRC')
plt.legend()
plt.xlabel('Time [s]')
plt.ylabel('WFE [nm]')

plt.figure()
plt.plot(np.arange(nLoop)*tel.samplingTime, SR_ngs, label='NGS@' + str(np.round(1e9*ngs.wavelength,0)) + ' nm')
plt.plot(np.arange(nLoop)*tel.samplingTime, SR_src, label='SRC@' + str(np.round(1e9*src.wavelength,0)) + ' nm')
plt.legend()
plt.xlabel('Time [s]')
plt.ylabel('SR [%]')
