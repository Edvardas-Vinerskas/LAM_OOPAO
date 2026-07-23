import torch
import subprocess
import dao
# torch.set_deterministic(True)
from torch import optim, save
import numpy as np
import time
import datetime
import os
from collections import deque
import matplotlib.pyplot as plt
import scipy.io
from astropy.io import fits
import torch.multiprocessing as mp
import json

from po4ao_config import config
from po4ao_models_upd import ConvDynamicsFast, EnsembleDynamicsFast, ConvPolicyFastFast #EnsembleDynamicsFast_upsampled, ConvPolicyFastFast_upsampled
from po4ao_util import get_n_params, EfficientExperienceReplay, SharedAdam, SharedAdamW


#you will need to upload your own updated models for po4ao
#this is the updated po4ao version in general and the models are also updated
#this code is written for the case where both stages run at the same frequency
#TODO which cores should I use?


scaling_factor_up   = 1e2 #1e2
scaling_factor_down = 1e-2 #1e-2



# read parameters for optimized performance
iters                   = config['RL']['iterations']
episode_length          = config['RL']['episode_length']
initial_sigma           = config['RL']['max_sigma']
min_sigma               = config['RL']['min_sigma']
warmup_episodes         = config['RL']['warmup_episodes']
loss_penalty            = config['RL']['loss_function_penalty']

n_history               = config['MDP']['n_history']
planning_horizon        = config['MDP']['planning_horizon']
data_shape              = config['MDP']['data_shape']


gain                    = config['integrator']['gain']
leak                    = config['integrator']['leak']
nmodes                  = config['integrator']['n_modes']
integrator              = config['integrator']['integrator']

replay_size             = config['replay_buffers']['replay_size']
warmup_memory           = config['replay_buffers']['warmup_memory']
train_warmup_percent    = config['replay_buffers']['train_warmup_percent']

batch_size = config['NN_models']['training_batch']

device0 = 'cuda:0'
device1 = 'cuda:0'

#TODO M2C matrix 
m2v_data = dao.shm('/tmp/oriziis_M2C.im.shm').get_data().astype(np.float32)

#TODO valid actuator mask
valid_mask = dao.shm('/tmp/dm97Map.im.shm').get_data()
dm_x, dm_y = np.where(valid_mask)
dm_coords = (dm_x, dm_y)


#TODO dm channel to write into
#TODO pyramid measurement in dm space
frame_data = dao.shm('/tmp/oziriis_res_wf.im.shm')
dm_shm = dao.shm('/tmp/dm2ndStageCmd02.im.shm')


prev_commands = torch.as_tensor(np.zeros((data_shape, data_shape)), device=device0).float()
obs_image = torch.as_tensor(np.zeros((data_shape, data_shape)), device=device0).float()


@torch.no_grad()
def sample_noise(sigma, flt, xvalid, yvalid):
    action_vec = torch.matmul(flt, sigma * torch.sign(torch.randn((241,)).to(device0)))
    action_im = torch.zeros((data_shape, data_shape)).to(device0).float()
    action_im[xvalid, yvalid] = action_vec
    return action_im



@torch.no_grad()
def step_2nd_stage(action):
    """
    Pipeline specific function that sends new commands to dm, i.e., sets the action, and reads the
    following WFS measurement projected to DM space trought a linear recontructor. The action and
    the WFS measurement have to be 2D images.

    :param action:             2D image of DM control voltages to be applied
    :return dm_image_torch:   2"D image of WFS measurement projected to DM voltages
    """

    #TODO what are the limits on the dm
        #highest/lowest possible value?
    temp = ((prev_commands * config['integrator']['leak']) + (action.squeeze())).clamp(-0.3, 0.3)

    # Calculations are done, move results into the correct SHM buffer
    dm_shm.set_data(temp[dm_coords].cpu().numpy())

    obs_image[dm_coords] = torch.from_numpy(frame_data.get_data(check=True, semNb=5)).flatten().to(device0)

    return -obs_image




def flatten_dm():
    """
    Pipeline specific function to flatted the DM

    :return dm_image_torch:    2D image of the WFS measurement projected to DM voltages with flattened dm
    """

    #flatten 2nd stage
    dm_shm.set_data(dm_shm.get_data()*0)
    prev_commands[:,:] = prev_commands[:,:]*0
    obs_image[dm_coords] = torch.from_numpy(frame_data.get_data(check=True, semNb=5)).flatten().to(device0)
    
    return -obs_image




def train_dynamics(dynamics: EnsembleDynamicsFast, optimizer: SharedAdam, replay: EfficientExperienceReplay, replay_warmup: EfficientExperienceReplay, dyn_iters=5):
    """
    train_dynamics trains the dynamics model. It samples from the replay buffers (warm-up and new data), 
    forms the state variable by adding the past telemetry data, makes predictions and back-propagates 
    the loss value to optimize the dynamics model parameters.

    :param dynamics:      the dynamics model to be optimized
    :param optimizer:     optimizer for the gradient decent
    :param replay:        new data
    :param replay_warmup: warm-up data
    :prama dyn_iter:      number of gradient steps in training
    :return loss:         loss on the last forward pass
    """

    dynamics.train()
    dynamics_loss = []
    for i in range(dyn_iters):
        optimizer.zero_grad()
        
        loss = 0

        for bs_model in dynamics.models:
            if torch.rand(1) > train_warmup_percent:
                sample = replay.sample_contiguous(n_history, episode_length, batch_size).to(device1)
            else:
                sample = replay_warmup.sample_contiguous(n_history, episode_length, batch_size).to(device1)

            states = sample.state()
            actions = sample.action()

            states = states.view(batch_size, n_history + 1, 1, *states.shape[1:])
            states_unfolded = states[:, :-1]
            
            actions_unfolded = actions.view(batch_size, n_history + 1, *actions.shape[1:])
            actions_unfolded = actions_unfolded[:, :-1].unsqueeze(2)
            
            next_states = states[:, -1]

            state = states_unfolded[:,-1].squeeze(2) 
            action = actions_unfolded[:,-1].squeeze(2)  

            history = torch.cat([states_unfolded[:,:-1].squeeze(2), actions_unfolded[:,:-1].squeeze(2)], dim=1)
            input_dynamics = torch.cat([history, state, action], dim=1)
            
            pred = bs_model(input_dynamics) #shape (batch_size, 1, 10, 10)   

            assert pred.shape == next_states.shape
            pred_loss = (next_states * scaling_factor_up - pred * scaling_factor_up).pow(2).mean()
            loss += pred_loss

        loss.backward()

        dynamics_loss.append(loss.item())
        
        torch.nn.utils.clip_grad_norm_(dynamics.parameters(), 0.5)

        optimizer.step()
        

    return loss.item(), dynamics_loss



def train_policy(optimizer: SharedAdam, policy: ConvPolicyFastFast, dynamics: EnsembleDynamicsFast, replay: EfficientExperienceReplay,
                    replay_warmup: EfficientExperienceReplay, pol_iters=5):
    """
    Train_policy trains the policy model, i.e., optimizes the policy parameters.
    First, it samples initial states from the replay buffers and forms the state variable
    by adding past telemetry. Next, the policy outputs (decides) the actions, and the
    dynamics predicts the next state, the mean over the model in the ensemble.
    Then, it iterates the process over the planning horizon, collects the loss (-reward)
    at each time step, and backpropagate the cumulative loss to policy parameters.

    :param policy:        policy model to be optimized
    :param dynamics:      the dynamics model used for predicting the next states
    :param optimizer:     optimizer for the gradient decent
    :param replay:        new data
    :param replay_warmup: warm-up data
    :prama pol_iter:      number of gradient steps in training
    :return loss:         loss on the last forward pass
    """

    dynamics.train()
    policy.train()

    for p in dynamics.parameters():
        p.requires_grad_(False)

    policy_loss = []
    for i in range(pol_iters):
        optimizer.zero_grad()

        if torch.rand(1) > train_warmup_percent:
            sample = replay.sample_contiguous(n_history, episode_length, batch_size).to(device1)
        else:
            sample = replay_warmup.sample_contiguous(n_history, episode_length, batch_size).to(device1)

        b = len(sample)

        states = sample.state()
        actions = sample.action()

        states_unfolded = states.view(batch_size, n_history + 1, 1, *states.shape[1:])
        states_unfolded = states_unfolded[:, :-1]

        actions_unfolded = actions.view(batch_size, n_history + 1, *actions.shape[1:])
        actions_unfolded = actions_unfolded[:, :-1].unsqueeze(2)

        state = states_unfolded[:,-1].squeeze(2)
        action = actions_unfolded[:,-1].squeeze(2)

        # get past telemetry data
        past_obs = states_unfolded[:,:-1].squeeze(2)
        past_act = actions_unfolded[:,:-1].squeeze(2)

        losses = torch.zeros(b, device=device1)

        for t in range(0, planning_horizon):

            history = torch.cat([past_obs, past_act], dim=1) 

            action = policy(torch.cat([state, history], dim=1))
            
            next_state = dynamics(torch.cat([history, state, action], dim=1))

            #averaging over the 5 dynamics models
            next_state_avg = torch.mean(next_state, dim = 1, keepdim = True)

            losses += loss_fn(torch.mean(next_state, dim = 1) * scaling_factor_up, action * scaling_factor_up)

            # roll history
            past_act = torch.cat([past_act[:,1:,:,:], action], dim = 1)
            past_obs = torch.cat([past_obs[:,1:,:,:], state], dim = 1)

            next_state = next_state_avg
            state = next_state
            
        
        loss = losses.mean()
        loss.backward()
        policy_loss.append(loss.item())
        optimizer.step()

    for p in dynamics.parameters():
        p.requires_grad_(True)

    return loss.item(), policy_loss



def training_thread(start_q, dynamics_q, dynamics_optimizer_q, replay_q, replay_warmup_q, policy_optimizer_q, policy_q, finished_q, dyn_iters = 30, pol_iters = 10):
    """
    training_thread start the parallel training procedure, i.e., trains the dynamics and policy NNs in parallel to controller.


    :param start_q:                   boolean for starting the training procedure. True when there is new data available
    :param dynamics_optimizer_q:      queue for the dynamics optimizer
    :param replay_q:                  queue for the data set to be trained on

    """
    os.sched_setaffinity(0, {11})  # 0 means current process
    print(f"Training process pinned to cores: {os.sched_getaffinity(0)}")


    print("Training process started")
    while(1):
        start = start_q.get()
        if start:
            start_time = time.time()
            dynamics = dynamics_q
            dynamics_optimizer = dynamics_optimizer_q#.get()
            replay = replay_q.get()
            replay_warmup = replay_warmup_q.get()
            policy_optimizer = policy_optimizer_q#.get()
            policy = policy_q

            dyn_loss, dynamics_loss = train_dynamics(dynamics, dynamics_optimizer, replay, replay_warmup, dyn_iters=dyn_iters)
            try:
                torch.cuda.synchronize(device=device1)
            except RuntimeError:
                pass

            pol_loss, policy_loss = train_policy(policy_optimizer, policy, dynamics, replay, replay_warmup, pol_iters=pol_iters)
            try:
                torch.cuda.synchronize(device=device1)
            except RuntimeError:
                pass

            finished_q.put(True)
            start = False
            print(f'--------------------------------------------\n training ({time.time() - start_time:.2f}s). \n\t dyn:{1000*dyn_loss:.4f} pol:{1000*pol_loss:.4f} \n--------------------------------------------')


@torch.no_grad()
def run_episode_policy_warmup(past_obs, past_act, obs, policy, episode_length):
    """
    runs an episode on policy

    :param past_obs:
    :param past_act:
    :param obs:
    :param replay:
    :param policy:
    :param sigma:
    :param episode_length:

    :return reward_sum:
    :return past_act:
    :return past_obs:
    """

    policy.eval()

    for t in range(episode_length):
        a = time.perf_counter()

        
        input_policy = torch.cat([obs.unsqueeze(0).unsqueeze(0), past_obs, past_act],dim = 1)
        policy(input_policy)
        b = time.perf_counter()
        print("frequency, ep_policy", 1/(b - a))
        
    return obs




'''@torch.no_grad() #for when you need to run an integrator after policy failure
def run_episode_integrator(past_obs, past_act, obs, replay, episode_length):#, nxt_sts_2nd_lst, crnt_iter, ctrl_jit):
    """
    runs an episode on policy

    :param past_obs:
    :param past_act:
    :param obs:
    :param replay:
    :param policy:
    :param sigma:
    :param episode_length:

    :return reward_sum:
    :return past_act:
    :return past_obs:
    """
    reward_sum = 0


    for t in range(int(episode_length)):
        a = time.perf_counter()

        action = gain*obs.unsqueeze(0).unsqueeze(0)
        next_obs = step_2nd_stage(action)
        

        # roll telemetry data with new data
        past_obs = torch.cat([past_obs[:,1:,:,:], obs.unsqueeze(0).unsqueeze(0)], dim = 1)
        past_act = torch.cat([past_act[:,1:,:,:], action], dim = 1)

        reward_sum += torch.sum((obs.flatten() * scaling_factor_up) ** 2)

        action_to_save = action.squeeze()
        replay.append(obs, action_to_save, next_obs)
        
        obs = next_obs
        b = time.perf_counter()
        print("integrator frequency, ep_policy", 1/(b - a), end = '\r')
        
    return reward_sum, past_obs, past_act, obs
'''



@torch.no_grad()
def run_episode_policy(past_obs, past_act, obs, replay, policy, sigma, episode_length, current_iteration):#, nxt_sts_2nd_lst, crnt_iter, ctrl_jit):
    """
    runs an episode on policy

    :param past_obs:
    :param past_act:
    :param obs:
    :param replay:
    :param policy:
    :param sigma:
    :param episode_length:

    :return reward_sum:
    :return past_act:
    :return past_obs:
    """

    policy.eval()
    reward_sum = 0

    for t in range(episode_length):
        a = time.perf_counter()

        if integrator == True or current_iteration < 20:
            action = gain*obs.unsqueeze(0).unsqueeze(0)
            #time.sleep(0.0004)
        else:
            input_policy = torch.cat([obs.unsqueeze(0).unsqueeze(0), past_obs, past_act],dim = 1)
            action = policy(input_policy) #/10
            
        next_obs = step_2nd_stage(action)

        # roll telemetry data with new data
        past_obs = torch.cat([past_obs[:,1:,:,:], obs.unsqueeze(0).unsqueeze(0)], dim = 1)
        past_act = torch.cat([past_act[:,1:,:,:], action], dim = 1)

        reward_sum += torch.sum((obs.flatten() * scaling_factor_up) ** 2)

        action_to_save = action.squeeze()
        replay.append(obs, action_to_save, next_obs)
        

        obs = next_obs
        b = time.perf_counter()
        if integrator == True or current_iteration < 20:
            print("int frequency, ep_policy", 1/(b - a), end = '\r')
        else:
            print("policy frequency, ep_policy", 1/(b - a), end = '\r')

    return reward_sum, past_obs, past_act, obs





def run_episode_warmup(replay, replay_warmup, sigma, episode_length, filter, xvalid, yvalid):
    """
    runs an episode on integrator with added noise in control signals. Starts always with a flat mirror

    :param replay:
    :param replay_warmup:
    :param sigma:
    :param filter:
    :param xvalid:
    :param yvalid:

    :return reward_sum:
    :return past_act:
    :return past_obs:
    """

    reward_sum = 0

    obs = flatten_dm()
    past_obs = torch.zeros(1, (n_history-1), *obs.shape, device = device0).squeeze(2) # keep telemetry in memory for the next episode
    past_act = torch.zeros(1, (n_history-1), *obs.shape, device = device0).squeeze(2)
    
    for t in range(episode_length):
        a = time.perf_counter()

        #obs == wfs_measurement
        action = gain * obs.unsqueeze(0).unsqueeze(0)
        action = action + sample_noise(sigma, filter, xvalid, yvalid) * scaling_factor_down

        next_obs = step_2nd_stage(action)

        past_obs = torch.cat([past_obs[:,1:,:,:], obs.unsqueeze(0).unsqueeze(0)], dim = 1) #roll telemetry
        past_act = torch.cat([past_act[:,1:,:,:], action], dim = 1)

        reward_sum += torch.sum((obs.flatten() * scaling_factor_up) ** 2)

        action_to_save = action.squeeze()
        replay.append(obs, action_to_save, next_obs)

        if sigma >= min_sigma:
            replay_warmup.append(obs, action_to_save, next_obs)

        obs = next_obs
        

        b = time.perf_counter()
        print("frequency", 1/(b - a), end = '\r')

    return reward_sum, past_obs, past_act, obs



def loss_fn(state,action):
    "the loss function, i.e, negative reward, for policy training."

    return state.pow(2).mean() + loss_penalty*action.pow(2).mean()

def main():
    os.sched_setaffinity(0, {4, 5}) #0 means current process
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    directory_name = f'test'
    savedir = f'~/PO4AO/logs/{directory_name}_{timestamp}'
    loaddir = f'~/PO4AO/logs/saved'   # copy the models and replay buffers you want use here!!


    if not os.path.exists(savedir):
        os.makedirs(savedir)
    if not os.path.exists(loaddir):
        os.makedirs(loaddir)

    with open(os.path.join(savedir, f"config.txt"), 'w') as convert_file:
        convert_file.write(json.dumps(config))


    ctx = mp.get_context('spawn')
    start_q = ctx.Queue()
    replay_q = ctx.Queue()
    replay_warmup_q = ctx.Queue()
    finished_q = ctx.Queue()
    start_q.put(False)

    KL_projection = m2v_data[:,:nmodes] @ np.linalg.pinv(m2v_data[:,:nmodes])
    KL_projection = torch.from_numpy(np.asarray(KL_projection)).float()

    mask = np.zeros((data_shape, data_shape))
    mask[dm_x, dm_y] = 1
    mask = np.array(mask, dtype = bool)

    xvalid0 = torch.from_numpy(dm_x).to(torch.int32).to(device0).squeeze()
    yvalid0 = torch.from_numpy(dm_y).to(torch.int32).to(device0).squeeze()

    xvalid1 = torch.from_numpy(dm_x).to(torch.int32).to(device1).squeeze()
    yvalid1 = torch.from_numpy(dm_y).to(torch.int32).to(device1).squeeze()

    replay = EfficientExperienceReplay((data_shape,data_shape), (data_shape,data_shape), replay_size* episode_length )
    replay_warmup = EfficientExperienceReplay((data_shape,data_shape), (data_shape,data_shape), warmup_memory* episode_length )

    dynamics = EnsembleDynamicsFast(mask, n_history).to(device1).share_memory()
    policy = ConvPolicyFastFast(xvalid1, yvalid1, KL_projection, n_history).to(device1).share_memory()
    policy_copy = ConvPolicyFastFast(xvalid0, yvalid0, KL_projection, n_history).to(device0).share_memory().eval()

    #dynamics_comp = torch.compile(dynamics, mode="reduce-overhead")
    #policy_comp = torch.compile(policy, mode="reduce-overhead")
    policy_copy_comp = torch.compile(policy_copy, mode="reduce-overhead")

    #dynamics_optimizer = [SharedAdam(model.parameters()) for model in dynamics.models]
    #policy_optimizer = SharedAdam((policy.parameters()))
    #dynamics_optimizer = SharedAdam(dynamics.parameters())
    #policy_optimizer = SharedAdam((policy.parameters()))

    #dynamics_optimizer = [SharedAdamW(model.parameters(), lr = 1e-3, weight_decay=1e-3) for model in dynamics.models] #TODO
    #policy_optimizer = SharedAdamW((policy.parameters()), lr = 1e-3, weight_decay=1e-3)
    dynamics_optimizer = SharedAdamW(dynamics.parameters(), weight_decay=1e-3)
    policy_optimizer = SharedAdamW((policy.parameters()), weight_decay=1e-3)

    #dynamics_optimizer1 = [optim.Adam(model.parameters()) for model in dynamics.models]
    #policy_optimizer1 = optim.Adam(policy.parameters())
    #dynamics_optimizer1 = optim.Adam(dynamics.parameters())
    #policy_optimizer1 = optim.Adam(policy.parameters())

    #dynamics_optimizer1 = [optim.AdamW(model.parameters(), weight_decay = 1e-3) for model in dynamics.models] #TODO
    #policy_optimizer1 = optim.AdamW(policy.parameters(), weight_decay = 1e-3)
    dynamics_optimizer1 = optim.AdamW(dynamics.parameters(), weight_decay = 1e-3)
    policy_optimizer1 = optim.AdamW(policy.parameters(), weight_decay = 1e-3)

    sigma = initial_sigma
    rewards = torch.zeros(iters + warmup_episodes)

    training = False
    obs = flatten_dm()

    past_obs = torch.zeros(1, (n_history-1), *obs.shape, device = device0).squeeze(2)
    past_act = torch.zeros(1, (n_history-1), *obs.shape, device = device0).squeeze(2)


    if not config['save_and_load']['load_warmup_buffer']:
        for i in range(warmup_episodes):
            start = time.time()
            reward_sum, past_obs, past_act, obs = run_episode_warmup(replay, replay_warmup, sigma, episode_length, KL_projection.to(device0), xvalid0, yvalid0)

            rewards[i] = reward_sum

            sigma -= (initial_sigma / (warmup_episodes/1))
            sigma = max(0, sigma)

            print(f'******************************************** \n Warm up {i} complete ({time.time() - start:.2f}s) \n\t reward:{reward_sum:.3f} \n********************************************')

    flatten_dm()

    if config['save_and_load']['save_warmup_buffer']:
        torch.save(replay_warmup.states, os.path.join(savedir, f"states_warmup.pt"))
        torch.save(replay_warmup.next_states, os.path.join(savedir, f"next_states_warmup.pt")) #what the dynamics should predict
        torch.save(replay_warmup.actions, os.path.join(savedir, f"actions_warmup.pt"))

        torch.save(replay.states, os.path.join(savedir, f"states.pt"))
        torch.save(replay.next_states, os.path.join(savedir, f"next_states.pt"))
        torch.save(replay.actions, os.path.join(savedir, f"actions.pt"))

        print(f'--------------------------------------------\n warmup buffer saved! \n--------------------------------------------')


    if config['save_and_load']['load_warmup_buffer']:
        replay_warmup.states = torch.load(os.path.join(loaddir, f"states_warmup.pt"))
        replay_warmup.next_states = torch.load(os.path.join(loaddir, f"next_states_warmup.pt"))
        replay_warmup.actions= torch.load(os.path.join(loaddir, f"actions_warmup.pt"))

        replay_warmup.set_len(50* episode_length -1)

        replay.states = torch.load(os.path.join(loaddir, f"states.pt"))
        replay.next_states = torch.load(os.path.join(loaddir, f"next_states.pt"))
        replay.actions= torch.load(os.path.join(loaddir, f"actions.pt"))

        replay.set_len(20* episode_length -1)
        print(f'--------------------------------------------\n warmup buffer loaded! \n--------------------------------------------')
    
    if config['save_and_load']['load_models_pretrained']:
        dynamics.load_state_dict(torch.load( os.path.join(loaddir, f"dynamics_pretrained.pt"),map_location=lambda storage, loc: storage))
        policy.load_state_dict(torch.load( os.path.join(loaddir, f"policy_pretrained.pt"),map_location=lambda storage, loc: storage))
        policy_copy.load_state_dict(torch.load( os.path.join(loaddir, f"policy_pretrained.pt"),map_location=lambda storage, loc: storage))

        print(f'--------------------------------------------\n Pretrained models loaded! \n--------------------------------------------')


    elif replay_warmup.len > episode_length and config['save_and_load']['load_models_pretrained_onl'] == False:
        start_time = time.time()
        dyn_loss, dynamics_loss = train_dynamics(dynamics, dynamics_optimizer1, replay_warmup, replay_warmup, dyn_iters=config['training']['dynamics_grad_steps_warmup'])
        torch.cuda.synchronize(device=device1)

        pol_loss, policy_loss = train_policy(policy_optimizer1, policy, dynamics, replay_warmup, replay_warmup, pol_iters=config['training']['policy_grad_steps_warmup'])
        torch.cuda.synchronize(device=device1)

        policy_copy.load_state_dict(policy.state_dict())


        print(f'--------------------------------------------\n Warmup training ({time.time() - start_time:.2f}s). \n\t dyn:{1000*dyn_loss:.4f} pol:{1000*pol_loss:.4f} \n--------------------------------------------')

        np.save(f"{savedir}/dynamics_loss_warmup.npy", dynamics_loss)
        np.save(f"{savedir}/policy_loss_warmup.npy", policy_loss)


    if config['save_and_load']['load_models_pretrained_onl']:
        dynamics.load_state_dict(torch.load( os.path.join(loaddir, f"dynamics_pretrained_onl.pt"),map_location=lambda storage, loc: storage))
        policy.load_state_dict(torch.load( os.path.join(loaddir, f"policy_pretrained_onl.pt"),map_location=lambda storage, loc: storage))
        policy_copy.load_state_dict(torch.load( os.path.join(loaddir, f"policy_pretrained_onl.pt"),map_location=lambda storage, loc: storage))

        print(f'--------------------------------------------\n Pretrained models loaded! \n--------------------------------------------')



    if config['save_and_load']['save_models_pretrained']:
        torch.save(dynamics.state_dict(), os.path.join(savedir, f"dynamics_pretrained.pt"))
        torch.save(policy.state_dict(), os.path.join(savedir, f"policy_pretrained.pt"))

        print(f'--------------------------------------------\n Pretrained models saved! \n--------------------------------------------')

    #warmup_done_q = ctx.Queue()  #for parallel network training but using compiled versions
    replay_q.put(replay, False)
    replay_warmup_q.put(replay_warmup, False)

    if replay.len > episode_length and replay_warmup.len > episode_length:
        training_process = ctx.Process(target=training_thread, args=(start_q, dynamics, dynamics_optimizer, replay_q, replay_warmup_q, policy_optimizer, policy, finished_q, config['training']['dynamics_grad_steps'], config['training']['policy_grad_steps'],)) # warmup_done_q TODO
        training_process.start()


        """#TODO
        # Block here until compiled models are warmed up
        print("Main process waiting for warmup...")
        warmup_done_q.get()              # blocks until training_thread signals ready
        print("Warmup done — starting main loop.")"""
    else:
        print("Replay buffers empty --- training not started. Run warm up or load buffers")


    for p in policy_copy.parameters():
        p.grad = None

    obs = flatten_dm()
    
    #warmup is needed for compilation
    print("compilation warmup start")
    for i in range(1):
        run_episode_policy_warmup(past_obs, past_act, obs, policy_copy_comp,
                                                        10)
    print("compilation warmup end")


    for i in range(iters):
        
        start = time.time()
        current_iter = i

        reward_sum, past_obs, past_act, obs = run_episode_policy(past_obs, past_act, obs, replay, policy_copy_comp, sigma,
                                                                 episode_length, current_iter)#, next_states_2nd_list, current_iter, ctrl_jitter_list)


        start_2 = time.time()
        reward_sum_v2 = reward_sum.clone()
        #for running the integrator if the policy fails
        '''if reward_sum > 34000 and i > 3:
            flatten_dm()
            past_obs = past_obs * 0
            past_act = past_act * 0
            reward_sum, past_obs, past_act, obs = run_episode_integrator(past_obs, past_act, obs, replay, episode_length/8)'''



        if config['save_and_load']['save_models_pretrained_onl'] and i == 80:
            torch.save(dynamics.state_dict(), os.path.join(savedir, f"dynamics_pretrained_onl.pt"))
            torch.save(policy.state_dict(), os.path.join(savedir, f"policy_pretrained_onl.pt"))
        
        
        
        rewards[i + warmup_episodes] = reward_sum
        if i > 20:
            try:
                training_finished = finished_q.get(False)
            except:
                training_finished = False

            if training_finished:
                training = False
                policy_copy.load_state_dict(policy.state_dict())

            if not training:
                replay_q.put(replay, False)
                replay_warmup_q.put(replay_warmup, False)
                start_q.put(True)
                training = True
        
        print(
            f'******************************************** \n Iteration {i} complete ({start_2 - start:.2f}s) ({time.time() - start_2:.2f}s)\n\t reward:{reward_sum_v2:.3f} \n********************************************')

    #TODO check that you are saving the correct matrices
    np.save(os.path.join(savedir, f"C2M_1st.npy"), np.asarray(dao.shm("/tmp/papyrus_C2M.im.shm").get_data().astype(np.float32)))
    np.save(os.path.join(savedir, f"M2C_2nd.npy"), np.asarray(dao.shm("/tmp/oriziis_M2C.im.shm").get_data().astype(np.float32)))
    torch.save(rewards, os.path.join(savedir, f"rewards.pt"))
    torch.save(replay.states, os.path.join(savedir, f"states.pt"))
    torch.save(replay.actions, os.path.join(savedir, f"actions.pt")) #output of the policy
    torch.save(replay.next_states, os.path.join(savedir, f"next_states_2nd.pt")) #2nd stage wfs measurement in DM space (need C2M for modes)

    torch.save(dynamics.state_dict(), os.path.join(savedir, f"dynamics_final.pt"))
    torch.save(policy.state_dict(), os.path.join(savedir, f"policy_final.pt"))

    print("data saved!")
    
    flatten_dm()
    print('dms flattened')

if __name__ == '__main__':
    main()
