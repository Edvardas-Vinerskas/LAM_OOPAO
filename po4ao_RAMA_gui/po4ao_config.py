config = {
    'RL': {
        'episode_length': 1000,     # in PO4AO on sky, corresponds to 2s for stability
        'warmup_episodes':  20,     # increase this to average the atmosphere
        'max_sigma':  0.01,            # exploration noise at the start of the warm-up 
        'min_sigma': 0,             # warm-up episodes with less noise are not stored in the warm-up buffer
        'loss_function_penalty': 0.5,
        'policy_after_warmup': True,   # switch to the policy (training on) when pre-training ends
        'auto_policy_update': True,    # swap in each new trained policy (False: RELOAD POLICY button)
    },
    'training': {
        'dynamics_grad_steps':  10,    # per online training round
        'policy_grad_steps':  10,
        'dynamics_grad_steps_warmup': 300,
        'policy_grad_steps_warmup':  300,
    },
    'MDP': {
        'n_history':64,
        'planning_horizon': 4,
        'data_shape': 11, # set by the DM
    },
    'replay_buffers': {
        'replay_size': 20,          # episodes kept in the on-line buffer
        'warmup_memory': 20,        # episodes kept in the warm-up buffer
        'train_warmup_percent': 0.2, #definitely helps with stability but should try other things first
    },
    'integrator':{
        'gain': 0.2, # integrator mode and warm-up
        'leak': 0.99, # also for RL
        'n_modes': 80,
        'offset': True,             # subtract dm1CmdOffset from the observation
        'command_clamp': 0.5,       # |dm1Cmd05| limit
    },
    'NN_models':{
        'filters_per_layer':16, #can even try changing this to 8
        'training_batch':32, #16 originally
        'initial_std':0.01,
        'initial_mean':0,
    },
    'save_and_load':{
        'run_name': 'test',            # run directory PO4AO/logs/<timestamp>_<run_name>
        'load_dir': '',                # checkpoint directory loaded at start ('' = none)
        'warmup_at_start': False,      # run warm-up + pre-training right after launch (old behaviour)
        'save_after_pretrain': True,   # checkpoint models + buffers when pre-training ends
        'save_at_exit': True,          # checkpoint models + buffers on quit / Ctrl-C
    }
}

