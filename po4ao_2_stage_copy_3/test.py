import dao
import numpy as np
import matplotlib.pyplot as plt
#import torch


"""papyrus_M2C = dao.shm("/tmp/papyrus_M2C.im.shm").get_data()
valid_mask_1st_stage = dao.shm("/tmp/dm241Map.im.shm").get_data()
dm_x_2nd_stage, dm_y_2nd_stage = np.where(papyrus_M2C)
dm_coords_2nd_stage = (dm_x_2nd_stage, dm_y_2nd_stage)


#np.save('valid_mask_1st_stage.npy', valid_mask_1st_stage)
np.save('M2C_1st.npy', papyrus_M2C)
print(papyrus_M2C)
print(dm_coords_2nd_stage)
print(valid_mask_1st_stage)"""

'''valid_mask_2nd = np.load('valid_mask_2nd_stage.npy')
mask = np.ones_like(valid_mask_2nd)
mask[3:6, 4:7] = 0

print(valid_mask_2nd * mask)'''

dm_shm_1st_stage = dao.shm('/tmp/dmCmd01.im.shm')
#1st stage dm channel for noise
dm_shm_noise = dao.shm('/tmp/dmCmd03.im.shm')

print(dm_shm_1st_stage.get_data(check = True))