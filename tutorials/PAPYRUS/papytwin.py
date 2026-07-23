# -*- coding: utf-8 -*-
"""
Created on Tue Mar 07 10:40:42 2023

Accurate version of PAPYRUS AO System used for reproducing the real system in details.
- 17/06/2025: Update after change of the WFS camera.
- 23/06/2025: Update to prepare the integration with DAO

@author: cheritie - astriffl
"""

import time
import matplotlib.pyplot as plt
import numpy as np
from OOPAO.calibration.CalibrationVault import CalibrationVault
from OOPAO.calibration.InteractionMatrix import InteractionMatrix
from OOPAO.tools.displayTools import cl_plot, displayMap
from Papyrus import Papyrus
from pymatreader import read_mat

#%% Compute the OOPAO Objects

Papytwin = Papyrus()
tel     = Papytwin.tel
ngs     = Papytwin.ngs
dm      = Papytwin.dm
wfs     = Papytwin.wfs
atm     = Papytwin.atm
slow_tt = Papytwin.slow_tt

# set back to calibration pupil
Papytwin.set_pupil(calibration=True)
ngs*tel*wfs

#%% PAPYRUS input data from the bench
from pymatreader import read_mat


M2C = -np.load("PAPYRIIS_2stage_CNN_RL/M2C_1rst.npy")
im = read_mat('PAPYRIIS_2stage_CNN_RL/intMat_klOOPAO_synthetic_bin=1_F=500_rMod=5_20250604_0307.mat')['matrix_inf']
dm.coefs =  M2C

ngs**tel*dm

#%%  -----------------------     Close loop  ----------------------------------
tel.resetOPD()

end_mode    = 195 
M2C_CL = M2C[:,:end_mode]
# These are the calibration data used to close the loop
# use of experimental calibration
# if full int-mat is available only
calib_CL    = CalibrationVault(im[:,:end_mode])

#%%

from OOPAO.Atmosphere import Atmosphere

atm = Atmosphere(telescope      = tel, 
                 r0             = 0.05,
                 L0             = 25, 
                 windSpeed      = [4], 
                 fractionalR0   = [1], 
                 windDirection  = [0],
                 altitude       = [0])

atm.initializeAtmosphere(tel)
from OOPAO.Detector import Detector
from OOPAO.Source import Source


#create scientific source
src =   Source('IR1310', 0)
src*tel


# initialize Telescope DM commands
tel.resetOPD()
dm.coefs=0
ngs*tel*dm*wfs
# Update the r0 parameter, generate a new phase screen for the atmosphere and combine it with the Telescope
atm.generateNewPhaseScreen(seed = 10)
tel+atm

tel.computePSF(4)
    

# combine telescope with atmosphere
tel+atm

Papytwin.set_pupil(calibration=False,
                   sky_offset=[2,2])
# initialize DM commands
atm*ngs*tel
atm*src*tel

nLoop = 500
# allocate memory to save data
SR_NGS                      = np.zeros(nLoop)
SR_SRC                      = np.zeros(nLoop)
total                       = np.zeros(nLoop)
residual_SRC                = np.zeros(nLoop)
residual_NGS                = np.zeros(nLoop)
dm_commands = np.zeros((nLoop, dm.nValidAct))
wfsSignal               = np.arange(0,wfs.nSignal)*0


# loop parameters
gainCL                  = 0.5
frame_delay             = 2
reconstructor = M2C_CL@calib_CL.M


wfs.cam = Papytwin.OCAM
print(wfs.cam.QE)
print(wfs.cam.darkCurrent)
print(wfs.cam.readoutNoise)
print(wfs.cam.photonNoise)
print(wfs.cam.gain)


for i in range(nLoop):
    a=time.time()
    
    atm.update()
    total[i]=np.std(tel.OPD[np.where(tel.pupil>0)])*1e9
    # propagate light from the NGS through the atmosphere, telescope, DM to the WFS and NGS camera with the CL commands applied
    atm*ngs*tel*dm*slow_tt*wfs
    # save residuals corresponding to the NGS
    residual_NGS[i] = np.std(tel.OPD[np.where(tel.pupil>0)])*1e9
    strehl_ngs = np.exp(-np.var(tel.src.phase[np.where(tel.pupil == 1)]))


    # propagate light from the SRC through the atmosphere, telescope, DM to the Instrument camera
    atm*src*tel*dm*slow_tt
    dm_commands[i,:] = dm.coefs.copy()
    # save residuals corresponding to the NGS
    residual_SRC[i] = np.std(tel.OPD[np.where(tel.pupil>0)])*1e9
    strehl_src = np.exp(-np.var(tel.src.phase[np.where(tel.pupil == 1)]))

    
    # apply the commands on the DM
    dm.coefs=dm.coefs-gainCL*np.matmul(reconstructor,wfsSignal)
    
    # store the slopes after computing the commands => 2 frames delay
    if frame_delay ==2:        
        wfsSignal=wfs.signal
    

    print('Loop'+str(i)+'/'+str(nLoop)+' NGS: '+str(residual_NGS[i])+' -- SRC:' +str(residual_SRC[i])+ '\n' 'strehl: '+str(strehl_ngs) +'\n' +'strehl: '+str(strehl_src))