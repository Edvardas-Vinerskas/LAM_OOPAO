import dao
import numpy as np
import time


#zwfs measurement in dm space (already reconstruced with cnn)
frame_data_2nd_stage = dao.shm('/tmp/oziriis_res_wf.im.shm') #'/tmp/oziriis_res_wf.im.shm'
#'/tmp/cred2.im.shm'
#no frame counter in pixel 00
dm_shm_2nd_stage = dao.shm('/tmp/dm2ndStageCmd02.im.shm')


dm_commands = np.zeros((97,1), dtype=np.float32)
dm_shm_2nd_stage.set_data(dm_commands)
prev_commands = np.zeros((97,1), dtype=np.float32)



obs = frame_data_2nd_stage.get_data(check = True).astype(np.float32)
for i in range(10000): 
    a = time.perf_counter()
    obs = frame_data_2nd_stage.get_data(check = True, semNb=1)#.astype(np.float32) 
    action = 0.03 * obs
    dm_commands = dm_commands * 0.99 - action
    #prev_commands = dm_commands#.clip(-0.1, 0.1)
    dm_shm_2nd_stage.set_data(dm_commands)
    #time.sleep(0.00)

    b = time.perf_counter()
    print('frequency', 1/(b-a), end = '\r')


dm_shm_2nd_stage.set_data(dm_shm_2nd_stage.get_data() * 0)


###############################################################################
"""zwfs_measurement = dao.shm("/tmp/oziriis_res_wf.im.shm")
dmShm_cl = dao.shm('/tmp/dm2ndStageCmd01.im.shm')

dm_commands = np.zeros((97,1), dtype=np.float32)
dmShm_cl.set_data(dm_commands)

for ii in range(2000):
    signal = zwfs_measurement.get_data(check=True)
    dm_commands = dm_commands*0.99 - signal * 0.05
    dmShm_cl.set_data(dm_commands)

dm_commands = np.zeros((97,1), dtype=np.float32)
dmShm_cl.set_data(dm_commands)"""



