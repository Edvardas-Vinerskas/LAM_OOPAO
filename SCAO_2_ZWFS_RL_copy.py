"""
Code adapted from Jalo Nousiainen
All parameters are Jalo's (for now)

For next week we do Jalo tests

* automate the normalisation of inputs and outputs

figure out later what is device0 and device1

figure out how all of the parallel training works, maybe do a short example

you need to formalise the loss calculations because for now this is a bit arbitrary
"""


from scipy import signal
import torch
from po4ao_edw.OOPAO_environment_PWFS import OOPAO_environment_PWFS
from po4ao_edw.OOPAO_environment_ZWFS import OOPAO_environment_ZWFS
from torch import optim
import numpy as np
import time
import os
import torch.multiprocessing as mp
import json
import matplotlib.pyplot as plt


from po4ao_PAPYRUS.po4ao_config_PAPYRUS import config
from po4ao_PAPYRUS.po4ao_models_PAPYRUS_upd import EnsembleDynamicsFast, ConvPolicyFastFast
from po4ao_PAPYRUS.po4ao_util_PAPYRUS import EfficientExperienceReplay, SharedAdam

env = OOPAO_environment_ZWFS()


sample_shape = env.DM_zer.nValidAct
sample_shape_pyr = env.DM_pyr.nValidAct

OOPAO_scaling_up   = 1e7
OOPAO_scaling_down = 1e-7


filters_per_layer = 32
n_filt = filters_per_layer

iters                = 4
episode_length       = 750
initial_sigma        = 0.3
min_sigma            = 0
warmup_episodes      = 20
loss_penalty         = 0.1

train_iter_warmup    = 400 #40
train_iter_parallel  = 40 #40

n_history            = 30 #30
planning_horizon     = 4
data_shape           = env.data_shape
control_delay        = 1

gain                 = 0.4
leak                 = 1
nmodes               = 250
integrator           = False

replay_size          = 20 #20 used in PAPYRUS code
warmup_memory        = warmup_episodes
train_warmup_percent = 0.2

batch_size           = 16 #originally 32

device = "cpu"#torch.device("cuda" if torch.cuda.is_available() else "cpu")

device0              = device
device1              = device


#why the sign function? not necessary, can just do gaussian
@torch.no_grad()
def sample_noise(sigma, flt, xvalid, yvalid):
    action_vec = torch.matmul(flt, sigma * torch.sign(torch.randn((sample_shape,)).to(device0)))
    action_im = torch.zeros((data_shape, data_shape)).to(device0).float()
    action_im[xvalid, yvalid] = action_vec
    return action_im


def sample_noise_pyr(sigma, flt):
    action_vec = np.matmul(flt, sigma * np.sign(np.random.randn(sample_shape_pyr)))
    return action_vec



def train_dynamics(dynamics: EnsembleDynamicsFast, optimizer: SharedAdam, replay: EfficientExperienceReplay,
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

    dynamics_loss = []
    for i in range(dyn_iters):
        optimizer.zero_grad()

        loss = 0

        for bs_model in dynamics.models:
            if torch.rand(1) > train_warmup_percent:
                sample = replay.sample_contiguous(n_history, episode_length, batch_size).to(device1)
            else:
                sample = replay_warmup.sample_contiguous(n_history, episode_length, batch_size).to(device1)


            states = sample.state()                                                         #shape (batch_size * (n_history + 1), 10, 10)
            actions = sample.action()


            states = states.view(batch_size, n_history + 1, 1, *states.shape[1:])           #resizes the states into shape (batch_size, n_history + 1, 1, 10, 10)
            states_unfolded = states[:, :-1]                                                #shape (batch_size, n_history, 1, 10, 10)


            actions_unfolded = actions.view(batch_size, n_history + 1, *actions.shape[1:])  #shape (batch_size, n_history + 1, 10, 10)
            actions_unfolded = actions_unfolded[:, :-1].unsqueeze(2)                        # shape (batch_size, n_history, 1, 10, 10)


            next_states = states[:, -1]                                                     #shape (batch_size, 1, 10, 10)


            state = states_unfolded[:, -1].squeeze(2)                                       #shape (batch_size, 1, 10, 10)
            action = actions_unfolded[:, -1].squeeze(2)


            history = torch.cat([states_unfolded[:, :-1].squeeze(2), actions_unfolded[:, :-1].squeeze(2)], dim=1) #shape (batch_size, 2 * (n_history - 1), 10, 10)
            input_dynamics = torch.cat([history, state, action], dim=1) * OOPAO_scaling_up


            pred = bs_model(input_dynamics)                                                 #shape (batch_size, 1, 10, 10)


            assert pred.shape == next_states.shape
            pred_loss = ((next_states * OOPAO_scaling_up - pred)).pow(2).mean()


            loss += pred_loss


        loss.backward()

        dynamics_loss.append(loss.item())

        torch.nn.utils.clip_grad_norm_(dynamics.parameters(), 0.5)

        optimizer.step()

    return loss.item(), dynamics_loss




def train_policy(optimizer: SharedAdam, policy: ConvPolicyFastFast, dynamics: EnsembleDynamicsFast,
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


        state = states_unfolded[:, -1].squeeze(2)           #shape (batch_size, 1, 10, 10)
        action = actions_unfolded[:, -1].squeeze(2)



        past_obs = states_unfolded[:, :-1].squeeze(2)       #shape (batch_size, n_history - 1, 10, 10)
        past_act = actions_unfolded[:, :-1].squeeze(2)

        losses = torch.zeros(b, device=device1)

        #reward over the horizon timeframe
        for t in range(0, planning_horizon):
            history = torch.cat([past_obs, past_act], dim=1)

            input_policy = torch.cat([state, history], dim=1) * OOPAO_scaling_up
            action = policy(input_policy)                   # shape (batch_size, 1, 10, 10)


            action = action * OOPAO_scaling_down
            input_dynamics = torch.cat([history, state, action], dim=1) * OOPAO_scaling_up
            next_state = dynamics(input_dynamics)           # shape (batch_size, 5, 10, 10)

            next_state = next_state * OOPAO_scaling_down
            #sum up of rewards over the horizon time frame
            #losses += loss_fn(next_state[:, 0] * OOPAO_scaling_up, action * OOPAO_scaling_up)
            losses += loss_fn(torch.mean(next_state, dim=1) * OOPAO_scaling_up, action * OOPAO_scaling_up)

            # roll history (0 index is removed)
            past_act = torch.cat([past_act[:, 1:, :, :], action], dim=1)
            past_obs = torch.cat([past_obs[:, 1:, :, :], state], dim=1)

            next_state = torch.mean(next_state, dim=1, keepdim=True)
            state = next_state

        loss = losses.mean()
        loss.backward()

        policy_loss.append(loss.item())

        optimizer.step()

    #and now this calculates the gradients? for what purpose?
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

            print('training dynamics in training thread')
            dyn_loss, dynamics_loss = train_dynamics(dynamics, dynamics_optimizer, replay, replay_warmup, dyn_iters=train_iter_parallel)
            """try:
                torch.cuda.synchronize(device=device1)
            except RuntimeError:
                pass"""

            print('training policy in training thread')
            pol_loss, policy_loss = train_policy(policy_optimizer, policy, dynamics, replay, replay_warmup, pol_iters=train_iter_parallel)
            """try:
                torch.cuda.synchronize(device=device1)
            except RuntimeError:
                pass"""

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


    INFO_list = []
    for t in range(episode_length):


        if integrator == True:
            action = gain * obs.unsqueeze(0).unsqueeze(0)
            time.sleep(0.0004)

        else:
            #shape(batch_size, 1, 10, 10)
            input_policy = torch.cat([obs.unsqueeze(0).unsqueeze(0).to(device0), past_obs, past_act], dim=1) * OOPAO_scaling_up
            action = policy(input_policy) * OOPAO_scaling_down


        #dm[:] = (prev_commands * leak) + (action)
        next_obs, INFO = env.step(action.squeeze(), pyramid_noise=0)
        INFO_list.append(INFO)

        if t % 50 == 0:
            strehl_check = INFO["strehl"]
            tracker = INFO["tracker"]
            print(f"Strehl ratio episode policy: {strehl_check}; atm tracker: {tracker}")

        # roll telemetry data with new data
        past_obs = torch.cat([past_obs[:, 1:, :, :], obs.unsqueeze(0).unsqueeze(0).to(device0)], dim=1)
        past_act = torch.cat([past_act[:, 1:, :, :], action], dim=1)

        reward_sum += torch.sum((obs.flatten() * OOPAO_scaling_up) ** 2)

        action_to_save = action.squeeze()
        replay.append(obs, action_to_save, next_obs)

        obs = next_obs

    return reward_sum, past_obs, past_act, obs, INFO_list




def run_episode_warmup(replay, replay_warmup, sigma, episode_length, filter, filter_pyr, xvalid, yvalid):
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
    obs = env.flatten_dm()



    past_obs = torch.zeros(1, (n_history - 1), *obs.shape, device=device0).squeeze(2)  # keep telemetry in memory for the next episode
    past_act = torch.zeros(1, (n_history - 1), *obs.shape, device=device0).squeeze(2)


    for t in range(episode_length):



        #shape(1, 1, 10, 10).unsqueeze(0).unsqueeze(0)
        action = gain * obs.unsqueeze(0).unsqueeze(0).to(device0)
        noisy  = sample_noise(sigma, filter, xvalid, yvalid) * OOPAO_scaling_down
        noisy_pyr = sample_noise_pyr(sigma, filter_pyr) * OOPAO_scaling_down
        action = action + noisy.unsqueeze(0).unsqueeze(0) #manual injection of noise to the bench
        #action = noisy.unsqueeze(0).unsqueeze(0)


        #dm[:] = (prev_commands * leak) + (action)
        #shape(10, 10)
        next_obs, INFO = env.step(action.squeeze(), pyramid_noise=noisy_pyr)
        strehl_1 = INFO["strehl_1st"]
        strehl_2 = INFO["strehl"]
        tracker  = INFO["tracker"]
        print(f"warmup strehl 1: {strehl_1}")
        print(f"warmup strehl 2: {strehl_2}; atm tracker: {tracker}")

        past_obs = torch.cat([past_obs[:, 1:, :, :], obs.unsqueeze(0).unsqueeze(0)], dim=1)  # roll telemetry
        past_act = torch.cat([past_act[:, 1:, :, :], action], dim=1)


        reward_sum += torch.sum((obs.flatten() * OOPAO_scaling_up) ** 2)


        action_to_save = action.squeeze()
        replay.append(obs, action_to_save, next_obs)

        #added to replay warmup (this condition is always true btw)
        if sigma >= min_sigma:
            replay_warmup.append(obs, action_to_save, next_obs)

        obs = next_obs

    return reward_sum, past_obs, past_act, obs, strehl_2


def loss_fn(state,action):
    "the loss function, i.e, negative reward, for policy training."

    return state.pow(2).mean() + loss_penalty*action.pow(2).mean() # + loss_penalyty2*(actions_modes[:5]).pow(2).mean()


def main():
    #retraining on single seed with less noise equivalent to single stage system and seeing what we get
    #note that for a single stage the warm up almost always converges
    directory_name = 'test' #vZWFS_1st_2nd_noise_03_phnoise1_read_4_4_QE08_mag2_epis20_eplen1500_noonline
    directory_name_load_buffer = 'vZWFS_1st_2nd_noise_03_phnoise1_read_4_4_QE08_mag2_epis20_warmupperc05'

    save_buffer = False
    load_buffer = True
    load_pretrained_model = False
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    savedir = f'temp_save_dir/{directory_name}/'
    savedir_model = f'temp_save_dir/{directory_name}/models' #/{timestamp}' redo the timestep with directory creation later
    savedir_buffer = f'temp_save_dir/{directory_name}/buffer/'
    loaddir = f'temp_save_dir/{directory_name_load_buffer}/buffer/'  # copy the models and replay buffers you want use here!!
    loaddir_mod = f'temp_save_dir/{directory_name_load_buffer}/models/'

    if not os.path.exists(savedir):
        os.makedirs(savedir)
    if not os.path.exists(savedir_model):
        os.makedirs(savedir_model)
    if not os.path.exists(savedir_buffer):
        os.makedirs(savedir_buffer)
    if not os.path.exists(loaddir):
        os.makedirs(loaddir)
    if not os.path.exists(loaddir_mod):
        os.makedirs(loaddir_mod)

    with open(os.path.join(savedir_buffer, f"config.txt"), 'w') as convert_file:
        convert_file.write(json.dumps(config))

    ctx = mp.get_context('spawn')
    start_q = ctx.Queue()
    replay_q = ctx.Queue()
    replay_warmup_q = ctx.Queue()
    finished_q = ctx.Queue()
    start_q.put(False)

    M2C = env.M2C_
    KL_projection = M2C @ np.linalg.pinv(M2C)
    KL_projection = torch.from_numpy(np.asarray(KL_projection)).float()


    KL_projection_pyr = env.M2C_pyr @ np.linalg.pinv(env.M2C_pyr)


    mask = env.mask
    dm_x, dm_y = np.where(mask)

    #dm_x and dm_y are INDICES where the actuators are valid
    xvalid0 = torch.from_numpy(dm_x).to(torch.int32).to(device0).squeeze()
    yvalid0 = torch.from_numpy(dm_y).to(torch.int32).to(device0).squeeze()

    xvalid1 = torch.from_numpy(dm_x).to(torch.int32).to(device1).squeeze()
    yvalid1 = torch.from_numpy(dm_y).to(torch.int32).to(device1).squeeze()



    replay = EfficientExperienceReplay((data_shape, data_shape), (data_shape, data_shape), replay_size * episode_length)
    replay_warmup = EfficientExperienceReplay((data_shape, data_shape), (data_shape, data_shape),
                                              warmup_memory * episode_length)


    #-----------------------------------MODEL_initialisation-----------------------------------#
    dynamics = EnsembleDynamicsFast(mask, n_history).to(device1).share_memory()
    policy = ConvPolicyFastFast(xvalid1, yvalid1, KL_projection, n_history).to(device1).share_memory()
    policy_copy = ConvPolicyFastFast(xvalid0, yvalid0, KL_projection, n_history).to(device0).eval().share_memory().eval()

    dynamics_optimizer = SharedAdam(dynamics.parameters())
    policy_optimizer = SharedAdam((policy.parameters()))

    dynamics_optimizer1 = optim.Adam(dynamics.parameters())
    policy_optimizer1 = optim.Adam(policy.parameters())

    sigma = initial_sigma
    rewards = torch.zeros(iters + warmup_episodes)

    training = False
    obs, _= env.reset(seed = 5)

    #shaped as if not involving current obs or act
    past_obs = torch.zeros(1, (n_history - 1), *obs.shape, device=device0).squeeze(2)
    past_act = torch.zeros(1, (n_history - 1), *obs.shape, device=device0).squeeze(2)
    strehl_warmup = 0


    if load_buffer == False:
        for i in range(warmup_episodes): #warmup_episodes
            obs, _ = env.reset(seed=np.random.randint(0, 256))
            start = time.time()
            #-----------------------------------sample generation-----------------------------------#
            reward_sum, past_obs, past_act, obs, strehl_warmup = run_episode_warmup(replay, replay_warmup, sigma, episode_length, KL_projection.to(device0), KL_projection_pyr, xvalid0, yvalid0)

            rewards[i] = reward_sum

            #the sigma is the variance innit
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

    if load_buffer:
        replay_warmup.states = torch.load(os.path.join(loaddir, f"states_warmup.pt"))
        replay_warmup.next_states = torch.load(os.path.join(loaddir, f"next_states_warmup.pt"))
        replay_warmup.actions = torch.load(os.path.join(loaddir, f"actions_warmup.pt"))

        replay_warmup.set_len(20 * episode_length - 1)

        replay.states = torch.load(os.path.join(loaddir, f"states.pt"))
        replay.next_states = torch.load(os.path.join(loaddir, f"next_states.pt"))
        replay.actions = torch.load(os.path.join(loaddir, f"actions.pt"))

        replay.set_len(20 * episode_length - 1)
        print(
            f'--------------------------------------------\n warmup buffer loaded! \n--------------------------------------------')

    """if load_pretrained_model:
        dynamics.load_state_dict(
            torch.load(os.path.join(loaddir_mod, f"dynamics_final.pt"), map_location=lambda storage, loc: storage))
        policy.load_state_dict(
            torch.load(os.path.join(loaddir_mod, f"policy_final.pt"), map_location=lambda storage, loc: storage))
        policy_copy.load_state_dict(
            torch.load(os.path.join(loaddir_mod, f"policy_final.pt"), map_location=lambda storage, loc: storage))"""

    if load_pretrained_model:
        dynamics.load_state_dict(
            torch.load(os.path.join(loaddir_mod, f"dynamics_pretrained.pt"), map_location=lambda storage, loc: storage))
        policy.load_state_dict(
            torch.load(os.path.join(loaddir_mod, f"policy_pretrained.pt"), map_location=lambda storage, loc: storage))
        policy_copy.load_state_dict(
            torch.load(os.path.join(loaddir_mod, f"policy_pretrained.pt"), map_location=lambda storage, loc: storage))

        print(
            f'--------------------------------------------\n Pretrained models loaded! \n--------------------------------------------')


    elif replay_warmup.len > episode_length:
        start_time = time.time()

        # -----------------------------------dynamics train-----------------------------------#
        dyn_loss, dynamics_loss = train_dynamics(dynamics, dynamics_optimizer1, replay_warmup, replay_warmup,
                                  dyn_iters=train_iter_warmup)
        #todo check synchronisation
        """try:
            torch.cuda.synchronize(device=device1)
        except RuntimeError:
            pass"""

        # -----------------------------------policy train-----------------------------------#
        pol_loss, policy_loss = train_policy(policy_optimizer1, policy, dynamics, replay_warmup, replay_warmup,
                                pol_iters=train_iter_warmup)
        """try:
            torch.cuda.synchronize(device=device1)
        except RuntimeError:
            pass"""


        policy_copy.load_state_dict(policy.state_dict())

        print(
            f'--------------------------------------------\n Warmup training ({time.time() - start_time:.2f}s). \n\t dyn:{1000 * dyn_loss:.4f} pol:{1000 * pol_loss:.4f} \n--------------------------------------------')

        np.save(f"temp_save_dir/{directory_name}/dynamics_loss", dynamics_loss)
        np.save(f"temp_save_dir/{directory_name}/policy_loss", policy_loss)

        plt.figure()
        plt.subplot(121)
        plt.title("dynamics_loss warmup")
        plt.plot(dynamics_loss)
        plt.grid(True)
        plt.yscale('log')
        plt.subplot(122)
        plt.title("policy_loss warmup")
        plt.grid(True)
        plt.plot(policy_loss)
        plt.yscale('log')




    torch.save(dynamics.state_dict(), os.path.join(savedir_model, f"dynamics_pretrained.pt"))
    torch.save(policy.state_dict(), os.path.join(savedir_model, f"policy_pretrained.pt"))

    print(
            f'--------------------------------------------\n Pretrained models saved! \n--------------------------------------------')

    replay_q.put(replay, False)
    replay_warmup_q.put(replay_warmup, False)


    """if replay.len > episode_length and replay_warmup.len > episode_length:
        training_process = ctx.Process(target=training_thread,
                                       args=(start_q, dynamics, dynamics_optimizer, replay_q, replay_warmup_q,
                                             policy_optimizer, policy, finished_q, 50, 25,))
        training_process.start()
    else:
        print("Replay buffers empty --- training not started. Run warm up or load buffers")"""

    for p in policy_copy.parameters():
        p.grad = None


    obs, _ = env.reset(seed=10)
    obs = env.flatten_dm()
    INFO_list = 0
    INFO_list_final = []
    episode_length_last_pol = 0
    policylosslist = []
    dynamicslosslist = []
    for i in range(iters): #for now set to 1 for faster computation (or set it to 2 for some parallel training?)

        start = time.time()

        reward_sum, past_obs, past_act, obs, INFO_list = run_episode_policy(past_obs, past_act, obs, replay, policy_copy, sigma,
                                                                                episode_length)

        rewards[i + warmup_episodes] = reward_sum
        INFO_list_final.extend(INFO_list)

        print('training dynamics not in training thread')
        dyn_loss, dynamics_loss = train_dynamics(dynamics, dynamics_optimizer1, replay, replay_warmup,
                                                 dyn_iters=train_iter_parallel)

        print('training policy not in training thread')
        pol_loss, policy_loss = train_policy(policy_optimizer1, policy, dynamics, replay, replay_warmup,
                                             pol_iters=train_iter_parallel)

        policy_copy.load_state_dict(policy.state_dict())

        dynamicslosslist.extend(dynamics_loss)
        policylosslist.extend(policy_loss)


        """try:
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

    np.save(f"temp_save_dir/{directory_name}/dynamics_loss_online", np.asarray(dynamicslosslist))
    np.save(f"temp_save_dir/{directory_name}/policy_loss_online", np.asarray(policylosslist))

    #TODO check out full implementation of zernike and how it works
    #TODO automate getting OOPAO updates
    #TODO rewrite OOPAO code (using new tutorial)
    #TODO combine the warmup directly into a working RL loop
    #TODO standardise the outputs from each model (RL and integrator), so that variables are of the same form and plotting is easy
    #TODO fix all of the things that you have plotted (temporal + fitting error)
    #TODO look at OOPAO plotting tools
    #TODO for the run_episode_policy the environment always resets to the same seed, you should only reset it at the start or finish (when gathering data and comparing with integrator)
    #TODO for integrator comparison you should also maybe warmup the RL on the same seed as integrator and then continue with RL policy evaluation with the continuation of said seed
    #also need to save the individual strehls for episode policy
    #for the fair comparison reset the atmosphere for the episode policy
    #INFO_dict = {key: [d[key] for d in INFO_list] for key in INFO_list[0].keys()}
    INFO_dict = {key: [d[key] for d in INFO_list_final] for key in INFO_list_final[0].keys()}


    residual_error = np.asarray(INFO_dict['residual_error'])
    np.save(f"temp_save_dir/{directory_name}/residual_error", residual_error)

    strehl_array_1st = np.asarray(INFO_dict['strehl_1st'])
    np.save(f"temp_save_dir/{directory_name}/strehl_array_1st", strehl_array_1st)

    strehl_array_2nd = np.asarray(INFO_dict['strehl'])
    np.save(f"temp_save_dir/{directory_name}/strehl_array_2nd", strehl_array_2nd)

    modes_1st_stage = np.asarray(INFO_dict['modes_1st_stage'])
    np.save(f"temp_save_dir/{directory_name}/modes_1st_stage", modes_1st_stage)

    modes_2nd_stage = np.asarray(INFO_dict['modes_2nd_stage'])
    np.save(f"temp_save_dir/{directory_name}/modes_2nd_stage", modes_2nd_stage)

    modes_atm = np.asarray(INFO_dict['modes_atm'])
    np.save(f"temp_save_dir/{directory_name}/modes_atm", modes_atm)

    #tel_psf_array = np.asarray(INFO_dict['TEL_PSF'])
    #np.save(f"temp_save_dir/{directory_name}/tel_psf_array", tel_psf_array)

    #residual_OPD_array = np.asarray(INFO_dict['residual_OPD']) #for use in spatial PSD/KL modes var and correlation
    #you can later do the spatial PSD when you save the required atm parameter
    #np.save(f"temp_save_dir/{directory_name}/residual_OPD_array", residual_OPD_array)

    #atm_OPD_array = np.asarray(INFO_dict['atm_OPD']) #not sure where I would use it
    #np.save(f"temp_save_dir/{directory_name}/atm_OPD_array", atm_OPD_array)

    total_err_array = np.asarray(INFO_dict['total_error']) #not useful for now
    np.save(f"temp_save_dir/{directory_name}/total_err_array", total_err_array)


    DM_zer_OPD_array = np.asarray(INFO_dict['DM_zer_OPD'])
    np.save(f"temp_save_dir/{directory_name}/DM_zer_OPD_array", DM_zer_OPD_array)


    #YOU SHOULD ALSO CHECK IF YOU ONLY HAVE AN EPISODE OF DATA IN ALL OF THESE BECAUSE I DON'T REMEMBER
    frequency = env.FREQUENCY
    time_plot = np.arange(0, iters * episode_length / frequency, 1/frequency)
    np.save(f"temp_save_dir/{directory_name}/time_array", time_plot)
    np.save(f"temp_save_dir/{directory_name}/frequency", frequency)



    #everythin I need is saved here?
    torch.save(rewards, os.path.join(savedir_model, f"rewards.pt")) #reward sum for each episode
    torch.save(replay.states, os.path.join(savedir_model, f"states.pt"))
    torch.save(replay.actions, os.path.join(savedir_model, f"actions.pt"))

    torch.save(dynamics.state_dict(), os.path.join(savedir_model, f"dynamics_final.pt"))
    torch.save(policy.state_dict(), os.path.join(savedir_model, f"policy_final.pt"))

    print("data saved!")

    plt.show()



if __name__ ==  '__main__':
    main()




