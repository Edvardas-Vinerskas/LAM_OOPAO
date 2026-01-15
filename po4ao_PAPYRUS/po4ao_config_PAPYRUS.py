config = {
    'RL': {
        'iterations':  200,
        'episode_length': 1000,
        'warmup_episodes':  10,
        'max_sigma':  0.1,
        'min_sigma':0,
        'loss_function_penalty':0.1
    },
    'training': {
        'dynamics_grad_steps':  40,
        'policy_grad_steps':  20,
        'dynamics_grad_steps_warmup':  400,
        'policy_grad_steps_warmup':  200,
    },
    'MDP': {
        'n_history':64,
        'planning_horizon': 4,
        'data_shape': 17, # set by the DM
    },
    'replay_buffers': {
        'replay_size': 20,
        'warmup_memory': 20,
        'train_warmup_percent': 0.2,
    },
    'integrator':{
        'gain': 0.4, # only for the warm up
        'leak': 0.98, # also for RL
        'n_modes': 195,
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