config = {
    'RL': {
        'iterations':  40,
        'episode_length': 500, #750 or 1500 for 0.5s or 1s respectively ()
        'warmup_episodes':  50, #increase this to average the atmosphere
        'max_sigma':  0, #used 0.3 in simulations but in a real system need to separately determine
        'min_sigma':0,
        'loss_function_penalty':0.1
    },
    'training': {
        'dynamics_grad_steps':  40,
        'policy_grad_steps':  40,
        'dynamics_grad_steps_warmup':  400,
        'policy_grad_steps_warmup':  400,
    },
    'MDP': {
        'n_history':64,
        'planning_horizon': 4,
        'data_shape_1st_stage': 17, # set by the DM
        'data_shape_2nd_stage': 11, # set by the DM
    },
    'replay_buffers': {
        'replay_size': 50,
        'warmup_memory': 50,
        'train_warmup_percent': 0.2,
    },
    'integrator':{
        'gain': 0.1, # only for the warm up
        'leak': 0.99, # also for RL
        'n_modes': 50, #Francisco is using 50 modes for his CNN
        'integrator':False # use the integrator as policy
    },
    'NN_models':{
        'filters_per_layer':32,
        'training_batch':16,
        'initial_std':0.01,
        'initial_mean':0,
    },
    'save_and_load':{
        'save_models_pretrained': False,
        'load_models_pretrained':False,
        'save_warmup_buffer': False,
        'load_warmup_buffer': False,
    }
}