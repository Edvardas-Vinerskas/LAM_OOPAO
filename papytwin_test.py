import time
import matplotlib.pyplot as plt
import numpy as np
from OOPAO.calibration.CalibrationVault import CalibrationVault
from OOPAO.calibration.InteractionMatrix import InteractionMatrix
from OOPAO.tools.displayTools import cl_plot, displayMap
from tutorials.PAPYRUS.Papyrus import Papyrus
from OOPAO import Atmosphere


Papytwin = Papyrus()
#the telescope, dm, pwfs is obviously preprogrammed
tel       = Papytwin.tel
ngs       = Papytwin.ngs
dm        = Papytwin.dm
wfs       = Papytwin.wfs
atm       = Papytwin.atm
slow_tt   = Papytwin.slow_tt #slow tip/tilt (the dm can slightly change its position?)
param     = Papytwin.param


#Function to switch to on-sky pupil

Papytwin.set_pupil(calibration = False,
                   sky_offset = [2, 2])


#atm.display_atm_layers()

#propagates the ngs through the tele (no atm) and wfs
ngs*tel*wfs

plt.figure()
plt.imshow(tel.pupil)
plt.figure()
plt.imshow(wfs.cam.frame) #.cam alone does not work for imshow plotting
plt.show()


Papytwin.set_pupil(calibration = True)
#after each change you need to repropagate the light
ngs*tel*wfs

#input data from the bench (this is a matlab file reader)
from pymatreader import read_mat

#M2C from the bench I presume
M2C = read_mat("M2C_KL_OOPAO_synthetic_IF.mat")["M2C_KL"] #no file lmao

#from the bench
valid_pixel = read_mat('useful_pixels_20250604_0305.mat')['usefulPix']

#from the bench interaction matrix
im = read_mat('intMat_klOOPAO_synthetic_bin=1_F=500_rMod=5_20250604_0307.mat')['matrix_inf']

# index of the KL modes included in the int-mat
ind = [1, 5, 10, 20, 30, 50, 80, 100, 150]

#experimental interaction matrix
int_mat_extract= im[:,ind]

valid_pixel, int_mat_binned = Papytwin.bin_bench_data(valid_pixel = valid_pixel, full_int_mat = im, ratio = param['ratio'])

#%% PAPYRUS/PAPYTWIN Pyramid Pupils Comparison



#interaction matrix variance calculation
var_im = np.var(im,axis=1).reshape(240,240)
var_im/=var_im.max()
var_im = var_im>0.005

# in case there is a mis-match set the key-word "correct" to True
correct = False
Papytwin.check_pwfs_pupils(valid_pixel_map = var_im, correct=correct)


#%% PAPYRUS/PAPYTWIN Interaction Matrix Comparison

#simulation of the interaction matrix (corresponds to calib.D) for comparison purposes
M2C_CL      = M2C[:,ind]
wfs.modulation = 5
stroke = 0.0001
calib = InteractionMatrix(  ngs            = ngs,
                            atm            = atm,
                            tel            = tel,
                            dm             = dm,
                            wfs            = wfs,
                            M2C            = M2C_CL,
                            stroke         = stroke,
                            phaseOffset    = 0,
                            nMeasurements  = 1,
                            noise          = 'off',
                            print_time=False,
                            display=True)


a = displayMap(int_mat_extract, norma = True,axis=1,returnOutput=True)
plt.title("Experimental Interaction Matrix")
b = displayMap(calib.D[:,:], norma = True,axis=1,returnOutput=True)
plt.title("Synthetic Interaction Matrix")


a[np.isinf(a)] = 0
b[np.isinf(b)] = 0


from OOPAO.tools.displayTools import interactive_show
#now we can plot interactive plots of the experimental and simulated interaction matrices

interactive_show(a, b) # use right and left click to switch between PAPYRUS and PAPYTWIN


###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
#some skipping here


#%%  -----------------------     Close loop  ------------------------------------------
tel.resetOPD()

end_mode    = 195
M2C_CL = M2C[:,:end_mode]
# These are the calibration data used to close the loop
# use of experimental calibration
# if full int-mat is available only
calib_CL    = CalibrationVault(im[:,:end_mode]) #this calculates the pseudo-inverse for the experimental matrix

#%%

from OOPAO.Atmosphere import Atmosphere

atm = Atmosphere(telescope      = tel,
                 r0             = 0.06,
                 L0             = 25,
                 windSpeed      = [0.01],
                 fractionalR0   = [1],
                 windDirection  = [0],
                 altitude       = [0])

atm.initializeAtmosphere(tel)


#ok, in any case I should be replacing the interaction matrix















