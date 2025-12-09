"""
Code adapted from Jalo Nousiainen
All parameters are Jalo's (for now)

IT IS ALIVE

it is not clear how the replay and replay_warmup differ in practice?
as only replay_warmup is used when training the policy and model
so the batch size was set to 32, I am not sure how OOPAO handled that
    but since you have the buffer data, you can quickly retrain the models and see what is happening

so the reason my stuff is not working as well as expected is because in po4ao they train it more
I am prety sure that for now it is only training on the warmup_buffer
I need to train it on a longer buffer/edit the code again
    I am not sure at which point the regular buffer is being expanded

For next week we do Jalo tests
All of the known information consolidation on Wednesday (AO school, RL dude, RL book, controller lectures)
"""


import torch
from torch import nn
import gym
from OOPAO_environment_ZWFS import OOPAO_environment_ZWFS
#torch.set_deterministic(True)
from torch import optim, save
import numpy as np
#import cupy as cp
import time
import datetime
import os
from collections import deque
import matplotlib.pyplot as plt
import scipy.io
from astropy.io import fits
import torch.multiprocessing as mp
import json
import matplotlib.pyplot as plt

filters_per_layer = 64
n_filt = filters_per_layer


iters                = 51 #what is this
episode_length       = 500
initial_sigma        = 0.015 #what is this
min_sigma            = 0.0 #what is this
warmup_episodes      = 20
loss_penalty         = 0.15 #a coefficient in front of the action when calculating loss

n_history            = 15 # was 32 in Jalo's files and yet that wouldn't make any sense (unless he later used a much longer history)
planning_horizon     = 4
data_shape           = 10 #this is the DM actuators I assume so change it accordingly
control_delay         = 1

gain                 = 0.4 #integrator gain innit
leak                 = 1 #leak parameter for the control law
nmodes               = 250
integrator           = False #use the integrator as policy

replay_size          = 160
warmup_memory        = 20
train_warmup_percent = 0.2

batch_size           = 32 #originally 32 but I don't think it will work with how OOPAO is set up

device0              = 'cpu'
device1              = 'cpu'


#OOPAO environment (all simulation variables are inside)
#maybe later do not declare as a global variable? but if it works it works (until it doesn't)
env = OOPAO_environment_ZWFS()



#NEED TO SUPPLY IT WITH THE PUPIL MASK
#64 filters
#input is the residual + DM.coefs
#activation function LeakyReLU()
#output is a single image of the next observation (residual wavefront!)
#how does batching work?
#so they apply the mask directly in the dynamics model? and yet they don't use this mask
class ConvDynamicsFast(nn.Module):
    def __init__(self, mask, n_history):
        super().__init__()

        self.mask = mask

        self.n_history = n_history

        self.net = nn.Sequential(
            nn.Conv2d(n_history * 2, n_filt, 3, padding=1),
            nn.LeakyReLU(),

            nn.Conv2d(n_filt, n_filt, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(n_filt, 1, 3, padding=1)
        )

        self._hidden = None

    def forward(self, feats):
        #feats shape(batch_size, 2 * n_history, 10, 10)
        #out shape(batch_size, 1, 10, 10)
        out = self.net(feats)


        ret = torch.zeros_like(out)
        #based on the above shapes, the mask is being applied correctly
        ret[:, :, self.mask] = out[:, :, self.mask]

        return out


#an ensemble of dynamics models
class EnsembleDynamicsFast(nn.Module):

    def __init__(self, mask, n_history, n_models=5):
        super().__init__()
        self.n_models = n_models
        self.models = nn.ModuleList([])

        for _ in range(n_models):
            self.models.append(ConvDynamicsFast(mask, n_history))

    def forward(self, feats):
        next_states = []

        for model in self.models:
            next_states.append(model.forward(feats))

        return torch.cat(next_states, dim=1)

#note that the output is clamped and you might need to change the clamping based on your needs for DM.coefs
#takes in wfs measurement reprojected onto DM control and outputs DM control
#on the other hand I think F just reprojects the output phase to a certain number of KL modes that are used for DM control
    #so is it the mode coefficients? It should be just the same phase but restricted to a certain number of KL modes
    #REPLACE F WITH YOUR OWN KL PROJECTION MATRIX
class ConvPolicyFastFast(nn.Module):
    def __init__(self, xvalid, yvalid, F, n_history):
        super().__init__()

        self.xvalid = xvalid
        self.yvalid = yvalid

        self.n_history = n_history

        self.register_buffer('F', F.unsqueeze(0)) #registers F as a non-trainable parameter

        self.net = nn.Sequential(
            nn.Conv2d(n_history * 2 - 1, n_filt, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(n_filt, n_filt, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(n_filt, 1, 3, padding=1),
        )

    def forward(self, feats):
        #feats shape(batch_size, 2 * n_history - 1, 10, 10) #because the current observation is part of the input but not the current action
        #out shape(batch_size, 1, 10, 10)
        out = self.net(feats)
        out = out.clamp(-0.08, 0.08)

        #should you also set the outside values to 0?
        out[:, :, self.xvalid, self.yvalid] = torch.matmul(self.F, out[:, :, self.xvalid, self.yvalid].squeeze(1).unsqueeze(2)).squeeze(-1).unsqueeze(1)

        return out


#class for state, action and next state handling (I guess it is easier this way?)
class ReplaySample():
    def __init__(self, states, actions, next_states):
        self.states = states
        self.next_states = next_states
        self.actions = actions

    def state(self):
        return self.states

    def prev_action(self):
        return self.prev_actions

    def next_state(self):
        return self.next_states

    def action(self):
        return self.actions

    def __len__(self):
        return len(self.states)

    def to(self, device):
        self.states = self.states.to(device)
        self.next_states = self.next_states.to(device)
        self.actions = self.actions.to(device)
        return self



class EfficientExperienceReplay():

    def __init__(self, state_shape, action_shape, max_size=100000, warmup_memory=0):
        self.max_size = max_size

        self.states = torch.empty(max_size, *state_shape).to("cpu")    #.to("cuda:0")  # .share_memory_()
        self.next_states = torch.empty(max_size, *state_shape).to("cpu")   #.to("cuda:0")  # .share_memory_()
        self.actions = torch.empty(max_size, *action_shape).to("cpu")    #.to("cuda:0")  # .share_memory_()

        self.len = 0
        self.index_write = 0
        self.warmup_memory = warmup_memory

    #a function to add batches of new statse, actions and next_states into the replay?
    def add(self, replay):
        cur_len = self.len
        new_len = self.len + len(replay)

        #but ReplaySample doesn't even take in rewards?
        if isinstance(replay, EfficientExperienceReplay):
            replay = ReplaySample(replay.states[:len(replay)], replay.actions[:len(replay)],
                                  replay.next_states[:len(replay)]) #removed replay.rewards[:len(replay)],  as 3rd argument

        self.states[cur_len:new_len] = replay.state()
        self.next_states[cur_len:new_len] = replay.next_state()
        self.actions[cur_len:new_len] = replay.action()

        self.len = new_len

    def __add__(self, replay):
        self.add(replay)
        return self

    #appends a single state, action and next state
    def append(self, obs, action, next_obs):

        if isinstance(obs, np.ndarray):
            raise 'should be torch'

        self.states[self.index_write] = obs
        self.next_states[self.index_write] = next_obs
        self.actions[self.index_write] = action

        self.index_write += 1

        if self.len < self.max_size:
            self.len += 1

        if self.index_write == self.max_size:
            print('Experience Replay Full')
            self.index_write = self.warmup_memory

    #randomly samples horizon length states, actions, next_states for trainign
    def sample_contiguous(self, horizon, max_ts, batch_size=32):
        inds = torch.randint(0, max_ts - (horizon + 1), size=(batch_size,))
        inds += torch.randint(0, len(self) // max_ts, size=(batch_size,)) * max_ts

        indices = torch.cat([torch.arange(ind, ind + horizon + 1) for ind in inds])
        # TODO check correct
        # indices = torch.from_numpy(vrange(inds.numpy(), np.ones_like(inds) * horizon + 1))

        return ReplaySample(self.states[indices], self.actions[indices], self.next_states[indices])

    def next_state(self):
        return self.next_states[:self.len]

    def state(self):
        return self.states[:self.len]

    def action(self):
        return self.actions[:self.len]

    def __len__(self):
        return self.len

    def set_len(self, index):
        self.len = index
        self.index_write = index

    #not used anywhere it seems
    def sample(self, size=512):
        inds = torch.randperm(self.len)[:size]
        return ReplaySample(self.states[inds], self.actions[inds], self.next_states[inds])

    def clear(self):
        self.len = 0








#training the dynamics model
#warm up is for filling up the buffer first
#replay is where our model training data is stored
#what is the states shape?
def train_dynamics(dynamics: EnsembleDynamicsFast, optimizer, replay: EfficientExperienceReplay,
                   replay_warmup: EfficientExperienceReplay, dyn_iters=5):
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
    optimizer = optimizer #optim.Adam(dynamics.parameters(), lr=1e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=0)

    for i in range(dyn_iters):
        optimizer.zero_grad()

        loss = 0

        for bs_model in dynamics.models:
            if torch.rand(1) > train_warmup_percent:
                sample = replay.sample_contiguous(n_history, episode_length, batch_size).to(device1)
            else:
                sample = replay_warmup.sample_contiguous(n_history, episode_length, batch_size).to(device1)

            #shape (batch_size * (n_history + 1), 10, 10)
            states = sample.state()
            actions = sample.action()

            #resizes the states into shape (batch_size, n_history + 1, 1, 10, 10)
            states = states.view(batch_size, n_history + 1, 1, *states.shape[1:])
            #contains all states except the last n_history + 1, so has shape n_history in fact
            #shape (batch_size, n_history, 1, 10, 10)
            states_unfolded = states[:, :-1]

            #shape (batch_size, n_history + 1, 10, 10)
            actions_unfolded = actions.view(batch_size, n_history + 1, *actions.shape[1:])
            #in addition, this also adds another dimension along dim = 2
            # shape (batch_size, n_history, 1, 10, 10)
            actions_unfolded = actions_unfolded[:, :-1].unsqueeze(2)

            #so the next state is in fact the last addition to the states variable
            #shape (batch_size, 1, 1, 10, 10)
            next_states = states[:, -1]

            #shape (batch_size, 1, 1, 10, 10) -> (batch_size, 1, 10, 10)
            #they explicitly separate the CURRENT state and action and then reconcatenate them
            state = states_unfolded[:, -1].squeeze(2)
            action = actions_unfolded[:, -1].squeeze(2)

            #shape (batch_size, n_history - 1, 10, 10) -> (batch_size, 2 * (n_history - 1), 10, 10)
            history = torch.cat([states_unfolded[:, :-1].squeeze(2), actions_unfolded[:, :-1].squeeze(2)], dim=1)

            #shape (batch_size, 1, 10, 10)
            pred = bs_model(torch.cat([history, state, action], dim=1))

            #checks that the prediction shape is the same as next_states shape
            #there might be a mismatch between pred shape and next_states shape
            assert pred.shape == next_states.shape
            pred_loss = (next_states - pred).pow(2).mean()

            loss += pred_loss

        #calculates gradient only after going through all of the dynamics models?
        loss.backward()

        torch.nn.utils.clip_grad_norm_(dynamics.parameters(), 0.5)

        optimizer.step()

    return loss.item()



#are new states appended to the back or to the front of history?
def train_policy(optimizer, policy: ConvPolicyFastFast, dynamics: EnsembleDynamicsFast,
                 replay: EfficientExperienceReplay,
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
    optimizer = optimizer #optim.Adam(policy.parameters(), lr=1e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=0)

    #gradients are not calculated for the dynamics model but it is set to train?
    #gradients flow through dynamics when updating policy (you should draw a sketch or something)
    for p in dynamics.parameters():
        p.requires_grad_(False)


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

        #shape (batch_size, 1, 10, 10)
        state = states_unfolded[:, -1].squeeze(2)
        action = actions_unfolded[:, -1].squeeze(2)

        #get past telemetry data (both current state and current action is removed)
        #shape (batch_size, n_history - 1, 10, 10)
        past_obs = states_unfolded[:, :-1].squeeze(2)
        past_act = actions_unfolded[:, :-1].squeeze(2)

        losses = torch.zeros(b, device=device1)

        for t in range(0, planning_horizon):
            history = torch.cat([past_obs, past_act], dim=1)

            # shape (batch_size, 1, 10, 10) I think?
            action = policy(torch.cat([state, history], dim=1))

            # shape (batch_size, 1, 10, 10) I think? or is it (batch_size, 5, 10, 10)?
            next_state = dynamics(torch.cat([history, state, action], dim=1))

            #sum up of rewards over the horizon time frame
            #if there are 5 next states, why is it only taking the first next state? introduce stochasticity?
            losses += loss_fn(next_state[:, 0], action)

            # roll history (0 index is removed)
            past_act = torch.cat([past_act[:, 1:, :, :], action], dim=1)
            past_obs = torch.cat([past_obs[:, 1:, :, :], state], dim=1)

            next_state = torch.mean(next_state, dim=1, keepdim=True)
            state = next_state

        loss = losses.mean()
        # print(loss.item())
        loss.backward()

        optimizer.step()

    #and now this calculates the gradients? for what purpose?
    for p in dynamics.parameters():
        p.requires_grad_(True)

    return loss.item()




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

    #so this now runs over every state in the episode (500 total)
    residual_error_list = []
    KL_mode_list = []
    strehl_list = []
    total_err_list = []
    env.reset()
    for t in range(episode_length):

        #so the action adds/subtracts from the previous DM commands similar to an integrator?
        if integrator == True:
            action = gain * obs.unsqueeze(0).unsqueeze(0)
            time.sleep(0.0004)
        else:
            #shape(batch_size, 1, 10, 10)
            action = policy(torch.cat([obs.unsqueeze(0).unsqueeze(0), past_obs, past_act], dim=1)) * 1e-7

        #takes in action and outputs the zwfs measurement projected onto DM space
        #dm[:] = (prev_commands * leak) + (action)
        next_obs, _, _, _, INFO = env.step_mod(action.squeeze())
        residual_error_list.append(INFO["residual_err"])
        KL_mode_list.append(INFO["KL_modes"])
        strehl_list.append(INFO["strehl"])
        total_err_list.append(INFO["total_err"])

        strehl_check = INFO["strehl"]
        print(f"Strehl ratio episode policy: {strehl_check}")

        # roll telemetry data with new data
        past_obs = torch.cat([past_obs[:, 1:, :, :], obs.unsqueeze(0).unsqueeze(0)], dim=1)
        past_act = torch.cat([past_act[:, 1:, :, :], action], dim=1)

        #so here the reward is calculated by summing up the CURRENT observations, however, the policy is not being trained here
        #so I guess it doesn't matter as much
        reward_sum += torch.sum((obs.flatten()) ** 2)

        action_to_save = action.squeeze()
        replay.append(obs, action_to_save, next_obs)

        obs = next_obs

    return reward_sum, past_obs, past_act, obs, residual_error_list, KL_mode_list, strehl_list, total_err_list



#20 warm up episodes with 500 states each
#returns reward_sum, past_obs, past_act, obs (is it current or next?) while running for an episode
def run_episode_warmup(replay, replay_warmup, sigma, episode_length): #, filter, xvalid, yvalid):
    """
    runs an episode on integrator with added noise in control signals. Starts always with a flat mirror
    dynamics model is not trained

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

    #if this is just DM flattening then I use my OOPAO environment
    env.DM_zer.coefs = 0
    env.DM_pyr.coefs = 0
    obs = torch.zeros((env.DM_zer.nAct, env.DM_zer.nAct))
    #obs = flatten_dm()


    #zeros with shape(1, 14, 10, 10) (because past only)
    #all single batch, but this shape then probably agrees with what our models want to see
    past_obs = torch.zeros(1, (n_history - 1), *obs.shape, device=device0).squeeze(2)  # keep telemetry in memory for the next episode
    past_act = torch.zeros(1, (n_history - 1), *obs.shape, device=device0).squeeze(2)

    #for an episode length of 500
    #calculates the action from the wfs observation
    #so for this I should just probably write a loop similar to zernike calibration but with random mirror coefficients
    env.reset()
    for t in range(episode_length):

        #shape(1, 1, 10, 10)
        action = gain * obs.unsqueeze(0).unsqueeze(0)
        #action = action + sample_noise(sigma, filter, xvalid, yvalid) #manual injection of noise to the bench

        #so here it just takes the action and outputs the wfs measurements projected onto DM space
        #dm[:] = (prev_commands * leak) + (action)
        #shape(10, 10)?
        next_obs, _, _, _, INFO = env.step_mod(action.squeeze())
        strehl_check = INFO["strehl"]
        print(f"Strehl ratio episode warmup: {strehl_check}")

        #concatenates current observation and action (the zeros get eventually rolled over)
        past_obs = torch.cat([past_obs[:, 1:, :, :], obs.unsqueeze(0).unsqueeze(0)], dim=1)  # roll telemetry
        past_act = torch.cat([past_act[:, 1:, :, :], action], dim=1)

        #again calculates the reward over the whole of 500 episodes from CURRENT observations?
        reward_sum += torch.sum((obs.flatten()) ** 2)

        #added to the replay buffer?
        #not completely clear what is the difference between replay and replay_warmup here
        action_to_save = action.squeeze()
        replay.append(obs, action_to_save, next_obs)

        #added to replay warmup
        if sigma >= min_sigma:
            replay_warmup.append(obs, action_to_save, next_obs)

        obs = next_obs

    return reward_sum, past_obs, past_act, obs


#loss function (why is it calculated in this way?)
#the loss function penalises large action steps which is what we want
def loss_fn(state,action):
    "the loss function, i.e, negative reward, for policy training."

    return state.pow(2).mean() + loss_penalty*action.pow(2).mean()


def main():
    save_buffer = True
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    savedir_model = f'jalo_zwfs_MBRL/models' #/{timestamp}' redo the timestep with directory creation later
    savedir_buffer = f'jalo_zwfs_MBRL/buffer/'
    loaddir = f'jalo_zwfs_MBRL/buffer/'  # copy the models and replay buffers you want use here!!
    loaddir_mod = f'jalo_zwfs_MBRL/models/'

    """if not os.path.exists(savedir):
        os.makedirs(savedir)
    if not os.path.exists(loaddir):
        os.makedirs(loaddir)

    with open(os.path.join(savedir, f"config.txt"), 'w') as convert_file:
        convert_file.write(json.dumps(config))"""



    M2C_zer = env.M2C_zer
    #passes the controle through (FINITE) KL modes and outputs the filtered controle again
    KL_projection = M2C_zer @ np.linalg.pinv(M2C_zer)
    KL_projection = torch.from_numpy(np.asarray(KL_projection)).float()


    mask = np.reshape(env.DM_zer.validAct, (10, 10))
    dm_x, dm_y = np.where(mask)

    #dm_x and dm_y are INDICES where the actuators are valid
    xvalid0 = torch.from_numpy(dm_x).to(torch.int64).to(device0).squeeze()
    yvalid0 = torch.from_numpy(dm_y).to(torch.int64).to(device0).squeeze()

    xvalid1 = torch.from_numpy(dm_x).to(torch.int64).to(device1).squeeze()
    yvalid1 = torch.from_numpy(dm_y).to(torch.int64).to(device1).squeeze()


    #max_size = replay_size (160) * episode_length (500)
    replay = EfficientExperienceReplay((data_shape, data_shape), (data_shape, data_shape), replay_size * episode_length)
    # max_size = warmup_memory (20) * episode_length (500)
    replay_warmup = EfficientExperienceReplay((data_shape, data_shape), (data_shape, data_shape),
                                              warmup_memory * episode_length)


    #-----------------------------------MODEL_initialisation-----------------------------------#

    #the dynamics model takes in the mask innit (which necessarily forces flattening and yet I am pretty sure I need to maintain the 2D output...)
    #the mask is not used in the output of the dynamics model... but is coded correctly as far as I am aware
    dynamics = EnsembleDynamicsFast(mask, n_history).to(device1) #.share_memory()
    policy = ConvPolicyFastFast(xvalid1, yvalid1, KL_projection, n_history).to(device1) #.share_memory()
    policy_copy = ConvPolicyFastFast(xvalid0, yvalid0, KL_projection, n_history).to(device0).eval() #.share_memory().eval()

    #this is for shared memory
    #dynamics_optimizer = SharedAdam(dynamics.parameters())
    #policy_optimizer = SharedAdam((policy.parameters()))

    #optimizer = optim.Adam(policy.parameters(), lr=1e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=0)

    dynamics_optimizer = optim.Adam(dynamics.parameters(), lr=1e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=0)
    policy_optimizer = optim.Adam(policy.parameters(), lr=1e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=0)

    dynamics_optimizer1 = optim.Adam(dynamics.parameters())
    policy_optimizer1 = optim.Adam(policy.parameters())

    sigma = initial_sigma
    #51 + 20
    rewards = torch.zeros(iters + warmup_episodes)

    training = False
    #again, 2D array of the zwfs measurement projected onto the DM
    env.DM_zer.coefs = 0
    env.DM_pyr.coefs = 0
    obs = torch.zeros((env.DM_zer.nAct, env.DM_zer.nAct))
    #obs = flatten_dm()

    #shaped as if not involving current obs or act
    past_obs = torch.zeros(1, (n_history - 1), *obs.shape, device=device0).squeeze(2)
    past_act = torch.zeros(1, (n_history - 1), *obs.shape, device=device0).squeeze(2)

    #i in range(20) 20 * 500
    #run_episode_warmup runs for a single episode
    """for i in range(warmup_episodes): #warmup_episodes
        start = time.time()
        #-----------------------------------sample generation-----------------------------------#
        #also stuffs our warmup buffer if I understand correctly
        reward_sum, past_obs, past_act, obs = run_episode_warmup(replay, replay_warmup, sigma, episode_length)#,KL_projection.to(device0), xvalid0, yvalid0)

        rewards[i] = reward_sum

        sigma -= (initial_sigma / (warmup_episodes / 1))
        sigma = max(0, sigma)
        end = time.time()

        print(
            f'******************************************** \n Warm up {i} complete ({end - start:.2f}s) \n\t reward:{reward_sum:.3f} \n********************************************')


    #saving and loading data
    if save_buffer:
        torch.save(replay_warmup.states, os.path.join(savedir_buffer, f"states_warmup.pt"))
        torch.save(replay_warmup.next_states, os.path.join(savedir_buffer, f"next_states_warmup.pt"))
        torch.save(replay_warmup.actions, os.path.join(savedir_buffer, f"actions_warmup.pt"))

        torch.save(replay.states, os.path.join(savedir_buffer, f"states.pt"))
        torch.save(replay.next_states, os.path.join(savedir_buffer, f"next_states.pt"))
        torch.save(replay.actions, os.path.join(savedir_buffer, f"actions.pt"))

        print(
            f'--------------------------------------------\n warmup buffer saved! \n--------------------------------------------')
    """
    if True:
        replay_warmup.states = torch.load(os.path.join(loaddir, f"states_warmup.pt"))
        replay_warmup.next_states = torch.load(os.path.join(loaddir, f"next_states_warmup.pt"))
        replay_warmup.actions = torch.load(os.path.join(loaddir, f"actions_warmup.pt"))

        replay_warmup.set_len(20 * episode_length - 1)

        replay.states = torch.load(os.path.join(loaddir, f"states.pt"))
        replay.next_states = torch.load(os.path.join(loaddir, f"next_states.pt"))
        replay.actions = torch.load(os.path.join(loaddir, f"actions.pt"))

        replay.set_len(50 * episode_length - 1)
        print(
            f'--------------------------------------------\n warmup buffer loaded! \n--------------------------------------------')

    if True:
        dynamics.load_state_dict(
            torch.load(os.path.join(loaddir_mod, f"dynamics_final.pt"), map_location=lambda storage, loc: storage))
        policy.load_state_dict(
            torch.load(os.path.join(loaddir_mod, f"policy_final.pt"), map_location=lambda storage, loc: storage))
        policy_copy.load_state_dict(
            torch.load(os.path.join(loaddir_mod, f"policy_final.pt"), map_location=lambda storage, loc: storage))

        print(
            f'--------------------------------------------\n Pretrained models loaded! \n--------------------------------------------')

    #was an elif statement before
    print(f"replay warmup length {replay_warmup.len}")
    print(f"episode_length {episode_length}")
    if replay_warmup.len > episode_length:
        # pretrain with warmup buffer
        start_time = time.time()

        # -----------------------------------dynamics train-----------------------------------#
        dyn_loss = train_dynamics(dynamics, dynamics_optimizer1, replay_warmup, replay_warmup,
                                  dyn_iters=300)
        #torch.cuda.synchronize(device="cuda:1")

        # -----------------------------------policy train-----------------------------------#
        pol_loss = train_policy(policy_optimizer1, policy, dynamics, replay_warmup, replay_warmup,
                                pol_iters=300)
        #torch.cuda.synchronize(device="cuda:1")

        policy_copy.load_state_dict(policy.state_dict())

        print(
            f'--------------------------------------------\n Warmup training ({time.time() - start_time:.2f}s). \n\t dyn:{1000 * dyn_loss:.4f} pol:{1000 * pol_loss:.4f} \n--------------------------------------------')


    torch.save(dynamics.state_dict(), os.path.join(savedir_model, f"dynamics_pretrained.pt"))
    torch.save(policy.state_dict(), os.path.join(savedir_model, f"policy_pretrained.pt"))

    print(
            f'--------------------------------------------\n Pretrained models saved! \n--------------------------------------------')

    """replay_q.put(replay, False)
    replay_warmup_q.put(replay_warmup, False)

    if replay.len > episode_length and replay_warmup.len > episode_length:
        training_process = ctx.Process(target=training_thread,
                                       args=(start_q, dynamics, dynamics_optimizer, replay_q, replay_warmup_q,
                                             policy_optimizer, policy, finished_q, 50, 25,))
        training_process.start()
    else:
        print("Replay buffers empty --- training not started. Run warm up or load buffers")"""

    for p in policy_copy.parameters():
        p.grad = None

    env.DM_zer.coefs = 0
    env.DM_pyr.coefs = 0
    obs = torch.zeros((env.DM_zer.nAct, env.DM_zer.nAct))
    #obs = flatten_dm()
    #i in range(51) 51 * 500
    #and then run_episode_policy is run for an episode of 500
    residual_error_list, KL_mode_list, strehl_list, total_err_list = 0, 0, 0, 0
    for i in range(1): #iters

        start = time.time()

        #what is the point of this run episode policy?
        #just the policy evaluation?
        reward_sum, past_obs, past_act, obs, residual_error_list, KL_mode_list, strehl_list, total_err_list = run_episode_policy(past_obs, past_act, obs, replay, policy_copy, sigma,
                                                                 episode_length)

        rewards[i + warmup_episodes] = reward_sum

        #some kind of parallel code training
        """
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
            training = True"""

        print(
            f'******************************************** \n Iteration {i} complete ({time.time() - start:.2f}s) \n\t reward:{reward_sum:.3f} \n********************************************')

    # ---------------------------------------------------Error decomposition---------------------------------------------------#
    timeery = np.arange(0, 500 / 1500, 1/1500)
    plt.figure()
    plt.plot(timeery, residual_error_list, label="residual")
    # plt.plot(time, sim_temp_error_2_frame_delay, label = "simulational temporal error 2 frame delay")
    plt.title("error decomposition (nm)")
    plt.xlabel("time s")
    plt.yscale("log")
    plt.legend()

    # ---------------------------------------------------Strehl---------------------------------------------------#
    sr_mean = np.mean(strehl_list)
    kernel = np.ones(30) / 30

    # pad sr
    pad_left = len(kernel) // 2
    pad_right = len(kernel) - pad_left - 1
    sr_padded = np.pad(strehl_list, (pad_left, pad_right), mode='constant', constant_values=sr_mean)
    sr_running = np.convolve(sr_padded, kernel, mode='valid')

    plt.figure()
    plt.plot(timeery, strehl_list, label="strehl")
    plt.plot(timeery, sr_running, label="running_strehl")
    plt.title("Strehl ratio")
    plt.xlabel("time s")
    plt.ylim(bottom=(sr_mean - 0.3))
    plt.legend()
    plt.show()




    #everythin I need is saved here?
    torch.save(rewards, os.path.join(savedir_model, f"rewards.pt"))
    torch.save(replay.states, os.path.join(savedir_model, f"states.pt"))
    torch.save(replay.actions, os.path.join(savedir_model, f"actions.pt"))

    torch.save(dynamics.state_dict(), os.path.join(savedir_model, f"dynamics_final.pt"))
    torch.save(policy.state_dict(), os.path.join(savedir_model, f"policy_final.pt"))

    print("data saved!")





if __name__ ==  '__main__':
    main()




