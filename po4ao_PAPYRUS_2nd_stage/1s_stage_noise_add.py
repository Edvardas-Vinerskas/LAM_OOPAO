import dao
import sys
# torch.set_deterministic(True)
import numpy as np
import time
import datetime
import os
from collections import deque
import matplotlib.pyplot as plt
import scipy.io
from astropy.io import fits
#import torch.multiprocessing as mp
import json

#flt is the filtration through the KL matrix
m2v_data_1st_stage = dao.shm('/tmp/papyrus_M2C.im.shm').get_data().astype(np.float32)
nmodes = 200 #placeholder for now
KL_projection = m2v_data_1st_stage[:,:nmodes] @ np.linalg.pinv(m2v_data_1st_stage[:,:nmodes])
KL_projection = np.asarray(KL_projection).astype(np.float32)


iters = int(sys.argv[1]) #arg[0] is the name of the script, the rest are arguments
sigma = float(sys.argv[2])


frame_data_1st_stage_shm = dao.shm('/tmp/papyrus_res_wf.im.shm')
dm_shm_1st = dao.shm('/tmp/dmCmd03.im.shm')
# channel 0 is flat
# channel 1 is closed loop
# channel 2 for atm simulation
# use 3 for noise

dm_commands_1st = np.zeros((241, 1), dtype = np.float32)
dm_shm_1st.set_data(dm_commands_1st)



#TODO dm_commands_1st should have shape (241, 1)
#for the wfs measurement projected onto dm space (frame_data_1st_stage), do I need to zero out the first pixel?
for i in range(iters):
    #you need a line with check = True for it to follow frames
    frame_data_1st_stage = frame_data_1st_stage_shm.get_data(check = True).astype(np.float32)

    action_vec = np.matmul(KL_projection, sigma * np.sign(np.random.randn(241).astype(np.float32))).reshape(241, 1)
    dm_commands_1st = action_vec
    dm_shm_1st.set_data(dm_commands_1st)


#don't forget to zero out
dm_commands_1st = np.zeros((241, 1), dtype = np.float32)
dm_shm_1st.set_data(dm_commands_1st)
































