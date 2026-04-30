import dao

dm_shm_1st = dao.shm('/tmp/dmCmd03.im.shm')
dm_shm_1st.set_data(dm_shm_1st.get_data() * 0)


dm_shm_1st_stage = dao.shm('/tmp/dmCmd01.im.shm')
dm_shm_2nd_stage = dao.shm('/tmp/dm2ndStageCmd02.im.shm')

dm_shm_1st_stage.set_data(dm_shm_1st_stage.get_data() * 0)
dm_shm_2nd_stage.set_data(dm_shm_2nd_stage.get_data() * 0)


