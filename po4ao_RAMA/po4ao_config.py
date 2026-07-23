config = {
    'RL': {
        'iterations':  400,
        'episode_length': 800, #in PO4AO on sky, corresponds to 2s for stability
        'warmup_episodes':  50, #increase this to average the atmosphere
        'max_sigma':  0.4, #should be around 1% of max stroke (when in CL, i.e. for papyrus max was around 0.3)? change up if no worky
        'min_sigma': 0,
        'loss_function_penalty': 0.1
    },
    'training': {
        'dynamics_grad_steps':  10,
        'policy_grad_steps':  10,
        'dynamics_grad_steps_warmup': 300,
        'policy_grad_steps_warmup':  300,
    },
    'MDP': {
        'n_history':64,
        'planning_horizon': 4,
        'data_shape': 17, # set by the DM
    },
    'replay_buffers': {
        'replay_size': 20,
        'warmup_memory': 50,
        'train_warmup_percent': 0.2, #definitely helps with stability but should try other things first
    },
    'integrator':{
        'gain': 0.3, # only for the warm up
        'leak': 0.98, # also for RL
        'n_modes': 35, #Francisco is using 50 modes for his CNN
        'integrator':True # use the integrator as policy
    },
    'NN_models':{
        'filters_per_layer':32, #can even try changing this to 8
        'training_batch':128, #if stable can try changing the batch_size too
        'initial_std':0.01,
        'initial_mean':0,
    },
    'save_and_load':{
        'save_models_pretrained': False,
        'save_models_pretrained_onl': False,
        'load_models_pretrained_onl': False,
        'load_models_pretrained':True,
        'save_warmup_buffer': False,
        'load_warmup_buffer': True,
    }
}