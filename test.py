# %%
%load_ext autoreload
%autoreload 2

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import sys
repo = '/home/evinerskas/LAM_OOPAO'
if repo not in sys.path:
    sys.path.insert(0, repo)




# %%
from PAPYRIIS_2stage_CNN_RL.OOPAO_PAPYRIIS_env import OOPAO_environment_PAPYRIIS


PAPYRIIS_env = OOPAO_environment_PAPYRIIS()

# %%


#TODO NEXT UP IS ARCTAN RECONSTRUCTION


#TODO summarise all of the results so far in your slides
#TODO redo strehl calculation whatever good that would do (120 nm of NCPA)
#TODO rerun the gain scan using a different wavelength
#TODO modulation radius is effectively different? (modulation scan 3-6)
#TODO create a separate control theory project with all of theoretical control functions starting with leaky integrator
        #this could also be used for wind fitting and stuff like that
#TODO finish PO4AO gui
#TODO run with 10% rejection for the pyramid signal
#TODO there was a DM scaling issue for the 2nd stage somewhere?
    #ask claude to go through the code and identify how the influence function peak is determined in simulations
    #it should be the sign value
    #on one hand the DM will converge to its shape, on the other the frame to frame correction would be suboptimal due to incorrect scaling
    #on the third hand, the CNN uses a 1e-5 sign in example.ipynb so...


#TODO YOU NEED TO DO A SINGLE RUN WITH THE NEW ATM AND PROJECTIONS TO SANITY CHECK
#TODO reconstruct the phase using arctangent reconstruction from Mathieu's code!!!

#so let's think this through:
    #CNN outputs mode amplitudes from the observed signal -> these should be true
    #the modes are converted to control commands via M2C (I load it from Mathieu and Francisco, you can compare these)
    #they are converted to the 1st stage using the influence functions
    #so I just need to plot the KL variance in 2nd stage basis

#some more thinking:
    #the influence functions map a unit of control to a set OPD size
    #so a deformable mirror unit command on 2nd stage that corresponded to 16e-6
    #would then be converted to approx 4 unit commands in the 1st stage (say 4e-6)
    #if I then used an incorrect projection with 10-e6, then I would necessarily get
    #2.5 unit commands on the 1st stage resulting in a smaller residual
        #a quick check would be to change the 2nd dm influence function size
        #and compare it against another size to see if the 1st stage residuals change

#The solution has been found:
    #the way the 2nd stage deformable mirror was constructed was incorrect
    #it definitely had the wrong sign parameter
    #this did not affect the atmosphere since for that we had the alpao unit 
    #so this is it, this is where you die


#TODO I will just use sign = 1.6e-5 and it will be good enough
# %%

from functions import KL_variance_calculator

# %%
from functions import *

loaddir = 'bench_sky_04_15/arcturus_v7'
RL_telemetry_1st = np.load(f'{loaddir}/2026-04-16T01_19_12_telemetry_data_RLiter50.npy', allow_pickle = True)
CL1st_OL_2nd     = np.load(f'{loaddir}/2026-04-16T01_27_23_telemetry_2nd_data_CL1OL2_noCRED2.npy', allow_pickle = True)

#TODO fun fact about M2C matrices:
    # Mathieu uses the 'm2c_sky matrix
    # Francisco uses either bench or custom
    # and there are differences between them


# on sky 
# %%

#TODO the IF_dm2 influence functions are slightly different from the dm_2nd.modes
    #but for plotting on sky data, there is almost no difference between using IF_dm2 and synthetic dm with sign = 1.6e-5

#TODO I also need to use these new influence functions in the simulations?
    #can't really do that since the CNN deformable mirror was set up for 1e-5
    #you can still try and see what the CNN does

#TODO don't forget to also run the 2nd stage with a pyramid!!!
    #make environment copy

#TODO do one run with M2C_KL and one with M2C from Mathieu for the CNN
#TODO now I need to get the actual residual error from the vZWFS images and that will be that
