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


scaling_factor_up = 1e2 #1e2
scaling_factor_down = 1e-2 #1e-2

use_1st_stage = False


# read parameters for optimized performance
iters = config['RL']['iterations']
episode_length = config['RL']['episode_length']
initial_sigma = config['RL']['max_sigma']
min_sigma = config['RL']['min_sigma']
warmup_episodes = config['RL']['warmup_episodes']
loss_penalty = config['RL']['loss_function_penalty']

n_history = config['MDP']['n_history']
planning_horizon = config['MDP']['planning_horizon']
data_shape_1st_stage = config['MDP']['data_shape_1st_stage']
data_shape_2nd_stage = config['MDP']['data_shape_2nd_stage']
#control_delay = config['MDP']['control_delay']

gain = config['integrator']['gain']
leak = config['integrator']['leak']
nmodes = config['integrator']['n_modes']
integrator = config['integrator']['integrator']

replay_size = config['replay_buffers']['replay_size']
warmup_memory = config['replay_buffers']['warmup_memory']
train_warmup_percent = config['replay_buffers']['train_warmup_percent']

batch_size = config['NN_models']['training_batch']

device0 = 'cuda:0'
device1 = 'cuda:0'

#m2v_data_1st_stage = dao.shm('/tmp/papyrus_M2C.im.shm').get_data().astype(np.float32)
#m2v_data_2nd_stage = dao.shm('/tmp/oriziis_M2C.im.shm').get_data().astype(np.float32) #TODO change here to 1st stage
m2v_data_2nd_stage = np.load("M2C_2nd.npy").astype(np.float32) #TODO compilation
#m2v_data_2nd_stage = dao.shm('/tmp/papyrus_M2C.im.shm').get_data().astype(np.float32)
#im_psf_shm = dao.shm('/tmp/cred2im.im.shm') #psf image



valid_mask_2nd_stage = np.load("valid_mask_2nd_stage.npy") #TODO compilation
#valid_mask_2nd_stage = dao.shm('/tmp/dm241Map.im.shm').get_data()
dm_x_2nd_stage, dm_y_2nd_stage = np.where(valid_mask_2nd_stage)
dm_coords_2nd_stage = (dm_x_2nd_stage, dm_y_2nd_stage)

## 1s stage dm closed loop channel#
#dm_shm_1st_stage = dao.shm('/tmp/dmCmd01.im.shm')

#dm channel for noise (1st stage)
#dm_shm_noise = dao.shm('/tmp/dmCmd03.im.shm')

#zwfs measurement in dm space (already reconstruced with cnn)
frame_data_2nd_stage = np.random.randn(97, 1) ##TODO change here to 1st stage
#frame_data_2nd_stage = dao.shm('/tmp/papyrus_res_wf.im.shm') #"/tmp/resWf.im.shm"


#no frame counter in pixel 00
dm_shm_2nd_stage = np.random.randn(97, 1) #'/tmp/dm2ndStageCmd01.im.shm' #TODO change here to 1st stage
#dm_shm_2nd_stage = dao.shm('/tmp/dmCmd03.im.shm') #'/tmp/dmCmd03.im.shm'

## DAO GAIN
# gain=dao.shm('/tmp/lpGain.im.shm')
# leak = dao.shm('/tmp/lpLeak.im.shm')
prev_commands_1st_stage = torch.as_tensor(np.zeros((data_shape_1st_stage, data_shape_1st_stage)), device=device0).float()
#obs_image_1st_stage = torch.as_tensor(np.zeros((data_shape_1st_stage, data_shape_1st_stage)), device=device0).float()


prev_commands_2nd_stage = torch.as_tensor(np.random.randn(data_shape_2nd_stage, data_shape_2nd_stage), device=device0).float()
obs_image_2nd_stage = torch.as_tensor(np.random.randn(data_shape_2nd_stage, data_shape_2nd_stage), device=device0).float()


@torch.no_grad()
def sample_noise(sigma, flt, xvalid, yvalid):
    action_vec = torch.matmul(flt, sigma * torch.sign(torch.randn((97,)).to(device0))) #TODO 97 change the number of controllable actuators
    action_im = torch.zeros((data_shape_2nd_stage, data_shape_2nd_stage)).to(device0).float()
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

    # dm[:] = prev_commands - (dm_image_torch * gain)
    temp = (prev_commands_2nd_stage * config['integrator']['leak']) + (action.squeeze())
    prev_commands_2nd_stage[:,:] = temp.clamp(-0.1, 0.1) #clipping limits

    # Calculations are done, move results into the correct SHM buffer
    dm_shm_2nd_stage = (temp[dm_coords_2nd_stage].cpu().numpy())

    obs_image_2nd_stage[dm_coords_2nd_stage] = torch.from_numpy(np.random.randn(97, 1).astype(np.float32)).flatten().to(device0)

    return -obs_image_2nd_stage #I think this is why po4ao has + in the integrator instead of a -



def flatten_dm():
    """
    Pipeline specific function to flatted the DM

    :return dm_image_torch:    2D image of the WFS measurement projected to DM voltages with flattened dm
    """
    if use_1st_stage:
        #flatten 1st stage
        dm_shm_1st_stage.set_data(dm_shm_1st_stage.get_data()*0)
        prev_commands_1st_stage[:,:] = prev_commands_1st_stage[:,:]*0
        #obs_image_1st_stage[dm_coords_1st_stage] = torch.from_numpy(frame_data_1st_stage.get_data(check=True, semNb=5)).flatten().to(device0)

    #flatten 2nd stage
    dm_shm_2nd_stage = (np.zeros((97, 1)))
    prev_commands_2nd_stage[:,:] = prev_commands_2nd_stage[:,:]*0
    obs_image_2nd_stage[dm_coords_2nd_stage] = torch.from_numpy(np.random.randn(97, 1).astype(np.float32)).flatten().to(device0)
    
    return -obs_image_2nd_stage



'''
def dynamics_train_step(dynamics, optimizer, inputs, next_states):
    """
    Fully compiled training step. No Python control flow inside.
    inputs:      [n_models, batch_size, input_dim]
    next_states: [n_models, batch_size, *state_shape]
    """
    optimizer.zero_grad(True)
    loss = torch.zeros([], device=device1)

    for i, bs_model in enumerate(dynamics.models):
        pred = bs_model(inputs[i])
        pred_loss = (next_states[i] * scaling_factor_up - pred * scaling_factor_up).pow(2).mean()
        loss = loss + pred_loss

    loss.backward()
    torch.nn.utils.clip_grad_norm_(dynamics.parameters(), 0.5)
    optimizer.step()
    return loss
dynamics_train_opt = torch.compile(dynamics_train_step, mode="reduce-overhead")
def train_dynamics(dynamics, optimizer, replay, replay_warmup, dyn_iters=5):
    dynamics.train()
    dynamics_loss = []

    for _ in range(dyn_iters):
        # --- Sampling: Python control flow stays OUTSIDE compiled region ---
        inputs_list = []
        next_states_list = []
        
        for _ in dynamics.models:
            buf = replay if torch.rand(1) > train_warmup_percent else replay_warmup
            sample = buf.sample_contiguous(n_history, episode_length, batch_size)

            states  = sample.state().view(batch_size, n_history + 1, 1, *sample.state().shape[1:])
            actions = sample.action().view(batch_size, n_history + 1, *sample.action().shape[1:])

            states_unfolded  = states[:, :-1]
            actions_unfolded = actions[:, :-1].unsqueeze(2)
            next_states      = states[:, -1]

            state   = states_unfolded[:, -1].squeeze(2)
            action  = actions_unfolded[:, -1].squeeze(2)
            history = torch.cat([states_unfolded[:, :-1].squeeze(2),
                                  actions_unfolded[:, :-1].squeeze(2)], dim=1)

            

            inputs_list.append(torch.cat([history, state, action], dim=1))
            next_states_list.append(next_states)

        # Stack into tensors so the compiled function sees static shapes
        inputs      = torch.stack(inputs_list)       # [n_models, batch_size, input_dim]
        next_states = torch.stack(next_states_list)  # [n_models, batch_size, *state_shape]
        

        # --- Compiled region: pure tensor ops, no Python branching ---
        loss = dynamics_train_opt(dynamics, optimizer, inputs, next_states)
        dynamics_loss.append(loss.item())

    return loss.item(), dynamics_loss
'''

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



'''
def policy_train_step(policy, dynamics, optimizer, state, past_obs, past_act):
    optimizer.zero_grad(True)
    losses = torch.zeros(state.shape[0], device=state.device)

    for t in range(planning_horizon):
        #print(action.shape)
        history        = torch.cat([past_obs, past_act], dim=1)   # [B, C, H, W]
        input_policy   = torch.cat([state, history], dim=1)       # [B, C+..., H, W]
        action         = policy(input_policy)                     # [B, A, H, W]

        input_dynamics = torch.cat([history, state, action], dim=1)
        next_state     = dynamics(input_dynamics)

        losses += loss_fn(
            torch.mean(next_state, dim=1) * scaling_factor_up,
            action * scaling_factor_up
        )

        # Roll by channel count — NOT by index 1
        # past_act: drop oldest A channels, append new action's A channels
        # past_obs: drop oldest 1 channel, append new state's 1 channel
        past_act = torch.cat([past_act[:, action.shape[1]:],  action], dim=1)  # ✅ stays 4D
        past_obs = torch.cat([past_obs[:, state.shape[1]:],   state],  dim=1)  # ✅ stays 4D
        state    = torch.mean(next_state, dim=1, keepdim=True)                 # [B, 1, H, W]

    loss = losses.mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
    optimizer.step()
    return loss
policy_train_opt = torch.compile(policy_train_step, mode="reduce-overhead")
def train_policy(optimizer, policy, dynamics, replay, replay_warmup, pol_iters=5):
    dynamics.train()
    policy.train()

    # Keep requires_grad_ outside compiled region
    for p in dynamics.parameters():
        p.requires_grad_(False)

    policy_loss = []
    for i in range(pol_iters):

        # Sampling stays outside — Python control flow
        buf    = replay if torch.rand(1) > train_warmup_percent else replay_warmup
        sample = buf.sample_contiguous(n_history, episode_length, batch_size)

        states  = sample.state().view(batch_size, n_history + 1, 1, *sample.state().shape[1:])
        actions = sample.action().view(batch_size, n_history + 1, *sample.action().shape[1:])

        state    = states[:, -2].squeeze(2)               # [B, 1, H, W]  — no unsqueeze needed
        past_obs = states[:, :-2].squeeze(2) # [B, (n_history-1)*1, H, W]
        past_act = actions[:, :-2]         # [B, (n_history-1)*A, H, W]  — no unsqueeze

        # Compiled region — pure tensor ops
        loss = policy_train_opt(policy, dynamics, optimizer, state, past_obs, past_act)
        policy_loss.append(loss.item())

    for p in dynamics.parameters():
        p.requires_grad_(True)

    

    return loss.item(), policy_loss
'''


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

            input_policy = torch.cat([state, history], dim=1)
            action = policy(input_policy)
            
            

            
            input_dynamics = torch.cat([history, state, action], dim=1) 
            next_state = dynamics(input_dynamics)

            
            #averaging over the 5 dynamics models
            next_state_avg = torch.mean(next_state, dim = 1, keepdim = True)


            losses += loss_fn(torch.mean(next_state, dim = 1) * scaling_factor_up, action * scaling_factor_up)

            # roll history
            past_act = torch.cat([past_act[:,1:,:,:], action], dim = 1)
            past_obs = torch.cat([past_obs[:,1:,:,:], state], dim = 1)

            next_state = next_state_avg
            state = next_state
            
        
        loss = losses.mean()
        #print(loss.item())
        loss.backward()

        policy_loss.append(loss.item())

        optimizer.step()

    for p in dynamics.parameters():
        p.requires_grad_(True)

    return loss.item(), policy_loss


def training_thread(start_q, ready_q, dynamics_q, dynamics_optimizer_q, replay_q, replay_warmup_q, policy_optimizer_q, policy_q, finished_q, dyn_iters = 30, pol_iters = 10):
    """
    training_thread start the parallel training procedure, i.e., trains the dynamics and policy NNs in parallel to controller.


    :param start_q:                   boolean for starting the training procedure. True when there is new data available
    :param dynamics_optimizer_q:      queue for the dynamics optimizer
    :param replay_q:                  queue for the data set to be trained on

    """

    dummy_inputs   = torch.zeros(5, batch_size, 128, 11, 11, device=device1)
    dummy_states   = torch.zeros(5, batch_size, 1, 11, 11, device=device1)
    dummy_state    = torch.zeros(batch_size, 1, 11, 11, device=device1)
    dummy_past_obs = torch.zeros(batch_size, 63, 11, 11, device=device1)
    dummy_past_act = torch.zeros(batch_size, 63, 11, 11, device=device1)
    
    '''print("Warming up compiled dynamics training step...")
    for _ in range(20):  # a few passes to fully initialize CUDA graphs
        dynamics_train_opt(dynamics_q, dynamics_optimizer_q, dummy_inputs, dummy_states)
    torch.cuda.synchronize()
    print("Warmup complete")


    print("Warming up compiled policy training step...")
    for _ in range(20):  # a few passes to fully initialize CUDA graphs
        policy_train_step(policy_q, dynamics_q, policy_optimizer_q, dummy_state, dummy_past_obs, dummy_past_act)
    torch.cuda.synchronize()
    print("Warmup complete")'''
    ready_q.put(True)

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
            torch.cuda.synchronize(device=device1)

            pol_loss, policy_loss = train_policy(policy_optimizer, policy, dynamics, replay, replay_warmup, pol_iters=pol_iters)
            torch.cuda.synchronize(device=device1)

            finished_q.put(True)
            start = False
            print(f'--------------------------------------------\n training ({time.time() - start_time:.2f}s). \n\t dyn:{1000*dyn_loss:.4f} pol:{1000*pol_loss:.4f} \n--------------------------------------------')


@torch.no_grad()
def run_episode_policy(past_obs, past_act, obs, replay, policy, sigma, episode_length):
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

        
        input_policy = torch.cat([obs.unsqueeze(0).unsqueeze(0), past_obs, past_act],dim = 1)
        #print(input_policy.shape)
        action = policy(input_policy) #/ 10
            

        next_obs = step_2nd_stage(action)

        """if np.max(dm_shm_2nd_stage.get_data()) > 0.075:
            dm_shm_2nd_stage.set_data(dm_shm_2nd_stage.get_data() * 0)
            time.sleep(5e-3)
            past_obs = past_obs * 0
            past_act = past_act * 0"""

        # roll telemetry data with new data
        past_obs = torch.cat([past_obs[:,1:,:,:], obs.unsqueeze(0).unsqueeze(0)], dim = 1)
        past_act = torch.cat([past_act[:,1:,:,:], action], dim = 1)

        reward_sum += torch.sum((obs.flatten() * scaling_factor_up) ** 2)

        action_to_save = action.squeeze()
        replay.append(obs, action_to_save, next_obs)


        obs = next_obs
        b = time.perf_counter()
        #print("frequency, ep_policy", 1/(b - a), end = '\r')
        
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
    
    #I added this to have the most recent vzwfs measurement
    #obs[dm_coords_2nd_stage] =  -torch.from_numpy(frame_data_2nd_stage.get_data(check=True, semNb=5)).flatten().to(device0)
    for t in range(episode_length):
        a = time.perf_counter()

        #here obs would act as Francisco's CNN output projected onto DM space
        #dm_commands = dm_commands*0.9 - wfs_measurement * 0.2
        #obs == wfs_measurement
        action = gain * obs.unsqueeze(0).unsqueeze(0)

        #if t%5 == 0:
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
        #if t%50:
        print("frequency", 1/(b - a), end = '\r')

    return reward_sum, past_obs, past_act, obs



def loss_fn(state,action):
    "the loss function, i.e, negative reward, for policy training."

    return state.pow(2).mean() + loss_penalty*action.pow(2).mean()

def main():
    #TODO change the directories
    subprocesses = 400
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    directory_name = 'test' #test_warmup20_1st150_2nd300_scaling1e2_noise_losspenalty01_reduceaction1_iters500_v15
    savedir = f'~logs/{directory_name}'#{timestamp}'
    loaddir = f'~logs/saved'   # copy the models and replay buffers you want use here!!

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

    KL_projection = m2v_data_2nd_stage[:,:nmodes] @ np.linalg.pinv(m2v_data_2nd_stage[:,:nmodes])
    KL_projection = torch.from_numpy(np.asarray(KL_projection)).float()

    mask = np.zeros((data_shape_2nd_stage, data_shape_2nd_stage))
    mask[dm_x_2nd_stage, dm_y_2nd_stage] = 1
    mask = np.array(mask, dtype = bool)

    xvalid0 = torch.from_numpy(dm_x_2nd_stage).to(torch.int32).to(device0).squeeze()
    yvalid0 = torch.from_numpy(dm_y_2nd_stage).to(torch.int32).to(device0).squeeze()

    xvalid1 = torch.from_numpy(dm_x_2nd_stage).to(torch.int32).to(device1).squeeze()
    yvalid1 = torch.from_numpy(dm_y_2nd_stage).to(torch.int32).to(device1).squeeze()

    replay = EfficientExperienceReplay((data_shape_2nd_stage,data_shape_2nd_stage), (data_shape_2nd_stage,data_shape_2nd_stage), replay_size* episode_length )
    replay_warmup = EfficientExperienceReplay((data_shape_2nd_stage,data_shape_2nd_stage), (data_shape_2nd_stage,data_shape_2nd_stage), warmup_memory* episode_length )


    
    dynamics = EnsembleDynamicsFast(mask, n_history).to(device1).share_memory()
    policy = ConvPolicyFastFast(xvalid1, yvalid1, KL_projection, n_history).to(device1).share_memory()
    policy_copy = ConvPolicyFastFast(xvalid0, yvalid0, KL_projection, n_history).to(device0).share_memory().eval()


    dynamics_comp = torch.compile(dynamics, mode="reduce-overhead")
    policy_comp = torch.compile(policy, mode="reduce-overhead")
    policy_copy_comp = torch.compile(policy_copy, mode="reduce-overhead")

    dynamics_optimizer = SharedAdam(dynamics.parameters())
    policy_optimizer = SharedAdam((policy.parameters()))
    #dynamics_optimizer = SharedAdamW(dynamics.parameters(), weight_decay=1e-3)
    #policy_optimizer = SharedAdamW((policy.parameters()), weight_decay=1e-3)

    dynamics_optimizer1 = optim.Adam(dynamics.parameters())
    policy_optimizer1 = optim.Adam(policy.parameters())
    #dynamics_optimizer1 = optim.AdamW(dynamics.parameters(), weight_decay = 1e-3)
    #policy_optimizer1 = optim.AdamW(policy.parameters(), weight_decay = 1e-3)

    sigma = initial_sigma
    rewards = torch.zeros(iters + warmup_episodes)

    training = False
    obs = flatten_dm()

    past_obs = torch.zeros(1, (n_history-1), *obs.shape, device = device0).squeeze(2)
    past_act = torch.zeros(1, (n_history-1), *obs.shape, device = device0).squeeze(2)

    if use_1st_stage:
        #here you can run the 1st stage in parallel but you can check how delayed it is so the 2nd stage already sees the noise from the 1st stage
        closed_loop_1st_stage = dao.shm('/tmp/lpCmd.im.shm')
        closed_loop_1st_stage.set_data(closed_loop_1st_stage.get_data() * 0 + 1) #starts closing the loop on the first stage
        subprocess.Popen(['taskset', '-c', '4,5', 'python', '/home/daouser/dao/daopapyrus-dev/evinerskas/1s_stage_noise_add.py', str(int(episode_length/2)), str(int(warmup_episodes)), str(sigma)]) #start the noise 
        #for now the Francisco's CNN runs at 500 so we put the same episode_length

        #to wait for the start of the 1st stage subprocess that adds the noise
        dm_shm_noise.get_data(check = True)




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

        replay_warmup.set_len(20* episode_length -1)

        replay.states = torch.load(os.path.join(loaddir, f"states.pt"))
        replay.next_states = torch.load(os.path.join(loaddir, f"next_states.pt"))
        replay.actions= torch.load(os.path.join(loaddir, f"actions.pt"))

        replay.set_len(50* episode_length -1)
        print(f'--------------------------------------------\n warmup buffer loaded! \n--------------------------------------------')

    if config['save_and_load']['load_models_pretrained']:
        dynamics.load_state_dict(torch.load( os.path.join(loaddir, f"dynamics_final.pt"),map_location=lambda storage, loc: storage))
        policy.load_state_dict(torch.load( os.path.join(loaddir, f"policy_final.pt"),map_location=lambda storage, loc: storage))
        policy_copy.load_state_dict(torch.load( os.path.join(loaddir, f"policy_final.pt"),map_location=lambda storage, loc: storage))

        print(f'--------------------------------------------\n Pretrained models loaded! \n--------------------------------------------')


    elif replay_warmup.len > episode_length:
        # pretrain with warmup buffer
        #TODO compilation
        start_time = time.time()
        dyn_loss, dynamics_loss = train_dynamics(dynamics_comp, dynamics_optimizer1, replay_warmup, replay_warmup, dyn_iters=config['training']['dynamics_grad_steps_warmup'])
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True
        ) as prof:
            for _ in range(1):
                train_dynamics(dynamics_comp, dynamics_optimizer1, replay_warmup, replay_warmup, dyn_iters=1)

        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))
        

        torch.cuda.synchronize(device=device1)

        pol_loss, policy_loss = train_policy(policy_optimizer1, policy_comp, dynamics_comp, replay_warmup, replay_warmup, pol_iters=config['training']['policy_grad_steps_warmup'])
        torch.cuda.synchronize(device=device1)

        policy_copy.load_state_dict(policy.state_dict())


        print(f'--------------------------------------------\n Warmup training ({time.time() - start_time:.2f}s). \n\t dyn:{1000*dyn_loss:.4f} pol:{1000*pol_loss:.4f} \n--------------------------------------------')

        np.save(f"{savedir}/dynamics_loss_warmup.npy", dynamics_loss)
        np.save(f"{savedir}/policy_loss_warmup.npy", policy_loss)




    if config['save_and_load']['save_models_pretrained']:
        torch.save(dynamics.state_dict(), os.path.join(savedir, f"dynamics_pretrained.pt"))
        torch.save(policy.state_dict(), os.path.join(savedir, f"policy_pretrained.pt"))

        print(f'--------------------------------------------\n Pretrained models saved! \n--------------------------------------------')



    replay_q.put(replay, False)
    replay_warmup_q.put(replay_warmup, False)
    ready_q = ctx.Queue()
    if replay.len > episode_length and replay_warmup.len > episode_length:
        training_process = ctx.Process(target=training_thread, args=(start_q, ready_q, dynamics_comp, dynamics_optimizer, replay_q, replay_warmup_q, policy_optimizer, policy_comp, finished_q, config['training']['dynamics_grad_steps'], config['training']['policy_grad_steps'],))
        training_process.start()
    else:
        print("Replay buffers empty --- training not started. Run warm up or load buffers")


    for p in policy_copy.parameters():
        p.grad = None

    obs = flatten_dm()

    print("Waiting for training process to warm up...")
    ready_q.get()  # ← blocks here until child finishes warmup
    print("Training process ready, starting episode loop")


    #TODO compilation
    print("compilation warmup start")
    for i in range(10):
        _, _, _, _ = run_episode_policy(past_obs, past_act, obs, replay, policy_copy_comp, sigma,
                                                        episode_length)
    print("compilation warmup end")
    

    for i in range(iters):
        
        start = time.time()

        reward_sum, past_obs, past_act, obs = run_episode_policy(past_obs, past_act, obs, replay, policy_copy_comp, sigma,
                                                                 episode_length)
        

        if i == subprocesses: 
            subprocess.Popen(['python', '/home/daouser/dao/daopapyrus-dev/evinerskas/atm_saving.py', str(int((episode_length * iters)/ 10)), savedir]) 
            if use_1st_stage:
                subprocess.Popen(['python', '/home/daouser/dao/daopapyrus-dev/evinerskas/1st_stage_saving.py', str(12500), savedir])
        
        
        
        rewards[i + warmup_episodes] = reward_sum

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
            f'******************************************** \n Iteration {i} complete ({time.time() - start:.2f}s) \n\t reward:{reward_sum:.3f} \n********************************************')



    #np.save(os.path.join(savedir, f"C2M_1st.npy"), np.asarray(dao.shm("/tmp/papyrus_C2M.im.shm").get_data().astype(np.float32)))
    #np.save(os.path.join(savedir, f"M2C_2nd.npy"), np.asarray(dao.shm("/tmp/oriziis_M2C.im.shm").get_data().astype(np.float32)))
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
