config = {
    'RL': {
        'iterations':  60,
        'episode_length': 800,
        'warmup_episodes':  50,
        'max_sigma':  0.4, #should be around 1% of max stroke (when in CL, i.e. for papyrus max was around 0.3)? change up if no worky
        'min_sigma':0,
        'loss_function_penalty':0.1
    },
    'training': {
        'dynamics_grad_steps':  10,
        'policy_grad_steps':  10,
        'dynamics_grad_steps_warmup':  300,
        'policy_grad_steps_warmup':  300,
    },
    'MDP': {
        'n_history':64,
        'planning_horizon': 4,
        'data_shape_1st_stage': 17, # set by the DM
        'data_shape_2nd_stage': 11, # set by the DM
    },
    'replay_buffers': {
        'replay_size': 20,
        'warmup_memory': 50,
        'train_warmup_percent': 0.2,
    },
    'integrator':{
        'gain': 0.3, # only for the warm up
        'leak': 0.95, # also for RL
        'n_modes': 50,
        'integrator':False # use the integrator as policy
    },
    'NN_models':{
        'filters_per_layer':16,
        'training_batch':128,
        'initial_std':0.01,
        'initial_mean':0,
    },
    'save_and_load':{
        'save_models_pretrained': True,
        'load_models_pretrained':False,
        'save_warmup_buffer': True,
        'load_warmup_buffer': False,
    }
}