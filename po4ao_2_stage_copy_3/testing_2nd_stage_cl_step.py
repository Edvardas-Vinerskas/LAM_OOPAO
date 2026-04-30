import dao
import numpy as np
import time
import subprocess
import os


#zwfs measurement in dm space (already reconstruced with cnn)
frame_data_2nd_stage = dao.shm('/tmp/oziriis_res_wf.im.shm')
dm_shm_2nd_stage = dao.shm('/tmp/dm2ndStageCmd02.im.shm')

dm_shm_2nd_stage.set_data(dm_shm_2nd_stage.get_data() * 0)
dm_commands = np.zeros((97,1), dtype=np.float32)
prev_commands =  dm_shm_2nd_stage.get_data() * 0


obs = frame_data_2nd_stage.get_data().astype(np.float32)


def step_2nd_stage(action):
    """
    Pipeline specific function that sends new commands to dm, i.e., sets the action, and reads the
    following WFS measurement projected to DM space trought a linear recontructor. The action and
    the WFS measurement have to be 2D images.

    :param action:             2D image of DM control voltages to be applied
    :return dm_image_torch:   2"D image of WFS measurement projected to DM voltages
    """

    # dm[:] = prev_commands - (dm_image_torch * gain)
    temp = (prev_commands * 0.99) - (action)
    prev_commands[:] = temp.clip(-0.1, 0.1) #clipping limits

    # Calculations are done, move results into the correct SHM buffer
    dm_shm_2nd_stage.set_data(temp)

    obs_image_2nd_stage = frame_data_2nd_stage.get_data(check=True, semNb=5)

    return obs_image_2nd_stage


episode_length = 250
iters = 200
subprocesses = 100
use_1st_stage = True
directory_name = 'integrator_1st_150_2nd300_gain04_20260316_atm2_3' 
savedir = f'~logs/{directory_name}'
next_states_2nd_list = []
if not os.path.exists(savedir):
        os.makedirs(savedir)
for j in range(iters):
    start = time.time()
    if j == subprocesses:
            subprocess.Popen(['python', '/home/daouser/dao/daopapyrus-dev/evinerskas/atm_saving.py', str(int((episode_length * iters)/ 5)), savedir]) #TODO change the length depending on atm fps (this should run at 100 Hz for now)
            if use_1st_stage:
                subprocess.Popen(['python', '/home/daouser/dao/daopapyrus-dev/evinerskas/1st_stage_saving.py', str(int(100 * episode_length/2)), savedir]) #this runs at 250 Hz
    for i in range(episode_length):  
        a = time.perf_counter()
        action = 0.1 * obs
        next_obs = step_2nd_stage(action)
        obs = next_obs
        if j >= 50:
            next_states_2nd_list.append(next_obs)

        b = time.perf_counter()
        print(f'frequency {j}', 1/(b-a), end = '\r')

    print(
            f'******************************************** \n Iteration {i} complete ({time.time() - start:.2f}s) \n\t reward: \n********************************************')

next_states_2nd_list = np.asarray(next_states_2nd_list)
np.save(os.path.join(savedir, f"next_states_2nd.npy"), next_states_2nd_list) #2nd stage wfs measurement in DM space (need C2M for modes)
print('data saved')

dm_shm_2nd_stage.set_data(dm_shm_2nd_stage.get_data() * 0)


