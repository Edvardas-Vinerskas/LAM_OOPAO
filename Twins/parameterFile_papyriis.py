# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 16:12:28 2024

@author: cheritier
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 07 10:40:42 2023

@author: cheritie - astriffl
"""

from OOPAO.tools.tools import createFolder
import numpy as np
def initializeParameterFile():
    # initialize the dictionaries
    param = dict()
    

    
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ATMOSPHERE PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    param['r0'                   ] = 0.07                                     # value of r0 in the visibile in [m]
    param['L0'                   ] = 30                                             # value of L0 in the visibile in [m]
    param['fractionnalR0'        ] = np.array([0.45,0.1,0.1,0.25,0.1]).astype(np.float64)
    param['V0'        ] = 1.8                       # Cn2 profile
    param['windSpeed'            ] = np.array([5,4,8,10,2]        ).astype(np.float64)* param['V0'        ]/6.15                   # wind speed of the different layers in [m.s-1]
    param['windDirection'        ] = np.array([0,72,144,216,288]    ).astype(np.float64)                          # wind direction of the different layers in [degrees]
    param['altitude'             ] = np.array([0, 1000,5000,10000,12000 ]        ).astype(np.float64)             # altitude of the different layers in [m]
               
    # =============================================================================
    #     PAPYRUS
    # =============================================================================
    
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% M1 PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    param['nSubaperture'         ] = 76                                                                           # number of PWFS subaperture along the telescope diameter
    param['nExtraSubaperture'    ] = 4                                                                            # extra subaperture on the edges
    param['diameter'             ] = 1.52 *(param['nSubaperture'] + param['nExtraSubaperture'])/param['nSubaperture'] # diameter in [m]
    param['ratio'                ] = 80//(param['nSubaperture'] + param['nExtraSubaperture'])                     # ratio factor for binned case
    param['nPixelPerSubap'       ] = 1                                                                            # sampling of the PWFS subapertures in pix
    param['resolution'           ] = (param['nSubaperture'] + param['nExtraSubaperture'])*param['nPixelPerSubap'] # resolution of the telescope driven by the PWFS
    param['sizeSubaperture'      ] = param['diameter']/(param['nSubaperture'] + param['nExtraSubaperture'])       # size of a sub-aperture projected in the M1 space
    param['samplingTime'         ] = 1/200                                                                        # loop sampling time in [s]
    param['m1_reflectivity'      ] = 0.01                                                                         # reflectivity of the pupil
    
    
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% NGS PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    param['magnitude'            ] = -0.04                                          # magnitude of the guide star
    param['opticalBand'          ] = 'K'                                            # optical band of the guide star
    param['opticalBandCalib'     ] = 'R'                                            # optical band of calibration laser
    
    
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% DM PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    param['nActuator'            ] = 17                                             # number of actuators 
    param['mechanicalCoupling'   ] = 0.36
    param['isM4'                 ] = False                                          # tag for the deformable mirror class
    param['dm_coordinates'       ] = None                                           # tag for the deformable mirror class
    
    # mis-registrations                                                             
    # latest value 16062025
    param['rotationAngle'        ] = -89.536                                           # rotation angle of the DM in [degrees]    
    param['shiftX'               ] = -0.004                                              # shift X of the DM in pixel size units ( tel.D/tel.resolution ) 
    param['shiftY'               ] = 0.005                                              # shift Y of the DM in pixel size units ( tel.D/tel.resolution )
    param['anamorphosisAngle'    ] = 0                                              # anamorphosis angle of the DM in [degrees]
    param['tangentialScaling'    ] = -0.025                                          # tangential scaling in percentage of diameter
    param['radialScaling'        ] = -0.031                                           # radial scaling in percentage of diameter

    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PWFS PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    param['modulation'            ] = 5                                             # modulation radius in ratio of wavelength over telescope diameter
    param['n_pix_separation'      ] = 0                                            # separation ratio between the PWFS pupils
    param['psfCentering'          ] = False                                         # centering of the FFT and of the PWFS mask on the 4 central pixels
    param['lightThreshold'        ] = 0.3                                           # light threshold to select the valid pixels
    param['postProcessing'        ] = 'fullFrame'                                   # post-processing of the PWFS signals 'slopesMaps' ou 'fullFrame'
    param['pwfs_pupils_shift_x'   ] = np.array([ 9.45287721, -7.76605308, -7.9417502 ,  7.22082509]).astype(np.float64) 
    param['pwfs_pupils_shift_y'   ] = np.array([-8.61289867, -9.49168905,  8.5352799 ,  9.4927028 ]).astype(np.float64) 

    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% OUTPUT DATA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # name of the system
    param['name'] = 'PAPYRUS_' +  param['opticalBand'] +'_band_'+ str(param['nSubaperture'])+'x'+ str(param['nSubaperture'])  
    
    # location of the calibration data
    param['pathInput'            ] = 'data_calibration/' 
    
    # location of the output data
    param['pathOutput'            ] = 'data_cl/'
    

    print('Reading/Writting calibration data from ' + param['pathInput'])
    print('Writting output data in ' + param['pathOutput'])

    createFolder(param['pathOutput'])
    
    # =============================================================================
    #     OZIRIIS
    # =============================================================================
    
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% M1 PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    param['resolution_2nd'           ] = 90 # resolution of the telescope on calibration
    param['resolution_2nd_sky'           ] = 80 # resolution of the telescope on calibration
    param['pad_sky'           ] = ((6,4),(3,7)) # resolution of the telescope on calibration
    param['sizeSubaperture_2nd'      ] = param['diameter']/param['resolution']      # size of a sub-aperture projected in the M1 space
    param['samplingTime_2nd'         ] = 1/400                                                                        # loop sampling time in [s]
    param['m1_reflectivity_2nd'      ] = 0.01                                                                         # reflectivity of the pupil
    
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% NGS PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    param['magnitude_2nd'            ] = -0.04                                          # magnitude of the guide star
    param['opticalBand_2nd'          ] = 'H'                                            # optical band of the guide star
    param['opticalBandCalib_2nd'     ] = 'H'                                            # optical band of calibration laser
    
    
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% DM PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    param['nActuator_2nd'            ] = 10                                             # number of actuators 
    param['mechanicalCoupling_2nd'   ] = 0.35
    param['isM4_2nd'                 ] = False                                          # tag for the deformable mirror class
    param['dm_coordinates_2nd'       ] = None                                           # tag for the deformable mirror class
    
    # mis-registrations                                                             
    # latest value 16062025
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ZWFS PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    param['depth'            ] = np.array([0.33, -0.74    ]            ).astype(np.float64)                             # modulation radius in ratio of wavelength over telescope diameter
                                               
    param['d_z_sky'          ] = 1.96
    param['d_z_calib'          ] = 2.14
    param['mask_px_size'          ] = 50 #pix size in fourier
    
    return param

