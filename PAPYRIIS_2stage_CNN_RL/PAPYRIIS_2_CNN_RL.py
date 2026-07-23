"""
Code adapted from Jalo Nousiainen
All parameters are Jalo's (for now)

For next week we do Jalo tests

* automate the normalisation of inputs and outputs
"""


from scipy import signal
import torch

#TODO this is one potential solution to gpu failure problem
#torch.backends.cuda.matmul.allow_tf32 = False
#torch.backends.cudnn.allow_tf32 = False

from PAPYRIIS_2stage_CNN_RL.OOPAO_PAPYRIIS_env import OOPAO_environment_PAPYRIIS
from torch import optim
import numpy as np
import time
import os
import torch.multiprocessing as mp
import json
import matplotlib.pyplot as plt


from PAPYRIIS_2stage_CNN_RL.po4ao_config_PAPYRUS import config
from PAPYRIIS_2stage_CNN_RL.po4ao_models_PAPYRUS_upd import EnsembleDynamicsFast, ConvPolicyFastFast
from PAPYRIIS_2stage_CNN_RL.po4ao_util_PAPYRUS import EfficientExperienceReplay, SharedAdam, SharedAdamW



#TODO calculate the theoretical errors innit (for the noise, i.e. should be a few percent of the max stroke)
sample_shape = 97 #pass it as variable

#TODO you will need to do no scaling or different scaling here
OOPAO_scaling_up   = 1e2
OOPAO_scaling_down = 1e-2


iters                = config['RL']['iterations']
episode_length       = config['RL']['episode_length']
initial_sigma        = config['RL']['max_sigma']
min_sigma            = config['RL']['min_sigma']
warmup_episodes      = config['RL']['warmup_episodes']
loss_penalty         = config['RL']['loss_function_penalty']

n_history            = config['MDP']['n_history']
planning_horizon     = config['MDP']['planning_horizon']
data_shape_2nd_stage = config['MDP']['data_shape_2nd_stage']


gain                 = config['integrator']['gain']
leak                 = config['integrator']['leak']
nmodes               = config['integrator']['n_modes']
integrator           = config['integrator']['integrator']

replay_size          = config['replay_buffers']['replay_size']
warmup_memory        = config['replay_buffers']['warmup_memory']
train_warmup_percent = config['replay_buffers']['train_warmup_percent']

batch_size           = config['NN_models']['training_batch']

device = "cpu" #torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

device0              = device
device1              = device




@torch.no_grad()
def sample_noise(sigma, flt, xvalid, yvalid):
    action_vec = torch.matmul(flt, sigma * torch.sign(torch.randn((sample_shape,)).to(device0)))
    action_im = torch.zeros((data_shape_2nd_stage, data_shape_2nd_stage)).to(device0).float()
    action_im[xvalid, yvalid] = action_vec
    return action_im



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
            input_dynamics = torch.cat([history, state, action], dim=1)


            pred = bs_model(input_dynamics)                                                 #shape (batch_size, 1, 10, 10)


            assert pred.shape == next_states.shape
            pred_loss = ((next_states * OOPAO_scaling_up - pred * OOPAO_scaling_up)).pow(2).mean()
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
            history = torch.cat([past_obs, past_act], dim=1) #shape (batch_size, 2 * (n_history - 1), 10, 10)

            input_policy = torch.cat([state, history], dim=1)
            action = policy(input_policy)                   # shape (batch_size, 1, 10, 10)

            input_dynamics = torch.cat([history, state, action], dim=1)
            next_state = dynamics(input_dynamics)           # shape (batch_size, 5, 10, 10)


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
    :param dynamics_optimizer_q:      queue for start_q, dynamics_q, dynamics_optimizer_q, replay_q, replay_warmup_q dynamics optimizer
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
def run_episode_policy(past_obs, past_act, obs, replay, policy, sigma, episode_length, environment, current_iteration, atm_OPD_1st_residual):
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


    dm_commands = np.zeros((episode_length, 97))
    reconstructed_cmd = np.zeros((episode_length, 97))
    scnd_stage_strehl = np.zeros((episode_length))
    tel_2nd_pupil = 0
    src_opd = np.zeros((episode_length, 90, 90))
    projector_kl_2nd = np.zeros((87, 8100))
    conditional = False
    for t in range(episode_length):

        if integrator == True:
            action = gain * obs.unsqueeze(0).unsqueeze(0)
            time.sleep(0.0004)
            conditional = False
        else:
            #shape(batch_size, 1, 10, 10)
            input_policy = torch.cat([obs.unsqueeze(0).unsqueeze(0).to(device0), past_obs, past_act], dim=1)
            action = policy(input_policy)
            conditional = True



        next_obs, INFO = environment.step(action.squeeze(), atm_OPD_1st_residual) #pass it as variable
        dm_commands[t] =INFO["dm_commands"]
        reconstructed_cmd[t] =INFO["reconstructed_cmd"].detach().cpu().numpy()
        scnd_stage_strehl[t] =INFO["2nd_stage_strehl"]
        src_opd[t] =INFO["src_opd"]
        tel_2nd_pupil = INFO["telescope_pupil"]
        projector_kl_2nd = INFO["projector_kl_2nd"]

        if t % 100 == 0:
            strehl_check = INFO["2nd_stage_strehl"]
            print(f"Strehl ratio episode policy: {strehl_check}")
            print(f'current residual iteration: {environment.CURRENT_STEPS}')
            if conditional:
                print('policy_network')

        # roll telemetry data with new data
        past_obs = torch.cat([past_obs[:, 1:, :, :], obs.unsqueeze(0).unsqueeze(0).to(device0)], dim=1)
        past_act = torch.cat([past_act[:, 1:, :, :], action], dim=1)

        reward_sum += torch.sum((obs.flatten() * OOPAO_scaling_up) ** 2)

        action_to_save = action.squeeze()
        replay.append(obs, action_to_save, next_obs)

        obs = next_obs


    #TODO later you might want to include at least some vzwfs frames for SNR calculations
    INFO_list = {
            "2nd_stage_strehl": scnd_stage_strehl,
            "telescope_pupil": tel_2nd_pupil, 
            "dm_commands": dm_commands,
            "reconstructed_cmd": reconstructed_cmd,
            "src_opd": src_opd,
            "projector_kl_2nd": projector_kl_2nd,
        }

    return reward_sum, past_obs, past_act, obs, INFO_list




def run_episode_warmup(replay, replay_warmup, sigma, episode_length, filter, xvalid, yvalid, environment, atm_OPD_1st_residual):
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
    obs = environment.flatten_dm() #pass it as variable
    past_obs = torch.zeros(1, (n_history - 1), *obs.shape, device=device0).squeeze(2)  # keep telemetry in memory for the next episode
    past_act = torch.zeros(1, (n_history - 1), *obs.shape, device=device0).squeeze(2)


    for t in range(episode_length):
        #shape(1, 1, 10, 10).unsqueeze(0).unsqueeze(0)
        action = gain * obs.unsqueeze(0).unsqueeze(0).to(device0)
        noisy  = sample_noise(sigma, filter, xvalid, yvalid) * OOPAO_scaling_down
        action = action + noisy.unsqueeze(0).unsqueeze(0) #manual injection of noise
        

        next_obs, INFO = environment.step(action.squeeze(), atm_OPD_1st_residual) #pass it as variable
        strehl_check = INFO["2nd_stage_strehl"]

        if t % 100 == 0:
            strehl_check = INFO["2nd_stage_strehl"]
            print(f"Strehl ratio episode warmup: {strehl_check}")
            print(f'current residual iteration: {environment.CURRENT_STEPS}')


        past_obs = torch.cat([past_obs[:, 1:, :, :], obs.unsqueeze(0).unsqueeze(0)], dim=1)  # roll telemetry
        past_act = torch.cat([past_act[:, 1:, :, :], action], dim=1)

        reward_sum += torch.sum((obs.flatten() * OOPAO_scaling_up) ** 2)

        action_to_save = action.squeeze()
        replay.append(obs, action_to_save, next_obs)

        #added to replay warmup (this condition is always true btw)
        if sigma >= min_sigma:
            replay_warmup.append(obs, action_to_save, next_obs)

        obs = next_obs

    return reward_sum, past_obs, past_act, obs


def loss_fn(state,action):
    "the loss function, i.e, negative reward, for policy training."

    return state.pow(2).mean() + loss_penalty*action.pow(2).mean() # + loss_penalyty2*(actions_modes[:5]).pow(2).mean()


def main():
    env = OOPAO_environment_PAPYRIIS()

    directory_name = f'PAPYRIIS_arcturus_noise_quantisation_pwfs_calibration_pupil_EMCCD'#{timestamp}
    savedir = f'PAPYRIIS_2stage_CNN_RL/~2026-06-23/{directory_name}'
    loaddir = f'PAPYRIIS_2stage_CNN_RL/~2026-06-23/saved'   # copy the models and replay buffers you want use here!!

    if not os.path.exists(savedir):
        os.makedirs(savedir)
    if not os.path.exists(loaddir):
        os.makedirs(loaddir)

    #TODO you can run everything from this file!
    #TODO calibrate with CCD and control with EMCCD?
    #TODO for comparison use v10 maybe?
    atm_OPD_1st = np.load(f"PAPYRIIS_2stage_CNN_RL/projected_atm_1st_stage/atm_OPDs_1st_r0_{env.atm_2nd.r0:.3f}_V0_{env.atm_2nd.V0:.3f}_L0_{env.atm_2nd.L0:.3f}_tboil_{env.atm_2nd.t_boiling[0]:.3f}_multi_layer.npz")
    atm_OPD_1st = atm_OPD_1st["atm_OPDs_1st"]
    print(f"PAPYRIIS_2stage_CNN_RL/projected_atm_1st_stage/atm_OPDs_1st_r0_{env.atm_2nd.r0:.3f}_V0_{env.atm_2nd.V0:.3f}_L0_{env.atm_2nd.L0:.3f}_tboil_{env.atm_2nd.t_boiling[0]:.3f}_multi_layer.npz")
    
    
    a = time.perf_counter()
    first_stage_results = env.run_first_stage_loop(90000, atm_OPD_1st)
    np.savez(f"{savedir}/results_1st_stage_r0_{env.atm_2nd.r0:.3f}_V0_{env.atm_2nd.V0:.3f}_L0_{env.atm_2nd.L0:.3f}_tboil_{env.atm_2nd.t_boiling[0]:.3f}_multi_layer.npz", **{
        k: v for k, v in first_stage_results.items() 
        if k != "config"
    },
        # Config fields flattened
        nLoop=first_stage_results["config"].nLoop,
        gainCL=first_stage_results["config"].gainCL,
        leak=first_stage_results["config"].leak,
        frame_delay=first_stage_results["config"].frame_delay,
        photon_noise=first_stage_results["config"].photon_noise,
    )
    

    #----------------------------------------------------1st stage residual entering to 2nd (PO4AO)#----------------------------------------------------

    #PAPYRIIS_2stage_CNN_RL\~2026-06-01\PAPYRIIS_arcturus_nonoise\results_2nd_stage.npz
    # loaddir_test = "PAPYRIIS_2stage_CNN_RL/~2026-06-23/PAPYRIIS_arcturus_noise_centralobs_quantisation_pwfs"
    loaddir_test = savedir
    print(f"{loaddir_test}/results_1st_stage_r0_{env.atm_2nd.r0:.3f}_V0_{env.atm_2nd.V0:.3f}_L0_{env.atm_2nd.L0:.3f}_tboil_{env.atm_2nd.t_boiling[0]:.3f}_multi_layer.npz")

    first_stage_results = np.load(f"{loaddir_test}/results_1st_stage_r0_{env.atm_2nd.r0:.3f}_V0_{env.atm_2nd.V0:.3f}_L0_{env.atm_2nd.L0:.3f}_tboil_{env.atm_2nd.t_boiling[0]:.3f}_multi_layer.npz")
    residuals_opds_1rst = first_stage_results['residuals_opds_1rst']

    #RL start

    with open(os.path.join(savedir, f"config.txt"), 'w') as convert_file:
        convert_file.write(json.dumps(config))

    ctx = mp.get_context('spawn')
    start_q = ctx.Queue()
    replay_q = ctx.Queue()
    replay_warmup_q = ctx.Queue()
    finished_q = ctx.Queue()
    start_q.put(False)

    M2C = env.M2C_2nd
    KL_projection = M2C @ np.linalg.pinv(M2C)
    KL_projection = torch.from_numpy(np.asarray(KL_projection)).float()

    mask = env.mask
    dm_x, dm_y = torch.where(mask)

    #dm_x and dm_y are INDICES where the actuators are valid
    xvalid0 = dm_x.to(torch.int32).to(device0).squeeze()
    yvalid0 = dm_y.to(torch.int32).to(device0).squeeze()

    xvalid1 = dm_x.to(torch.int32).to(device1).squeeze()
    yvalid1 = dm_y.to(torch.int32).to(device1).squeeze()



    replay = EfficientExperienceReplay((data_shape_2nd_stage, data_shape_2nd_stage), (data_shape_2nd_stage, data_shape_2nd_stage), replay_size * episode_length)
    replay_warmup = EfficientExperienceReplay((data_shape_2nd_stage, data_shape_2nd_stage), (data_shape_2nd_stage, data_shape_2nd_stage),
                                              warmup_memory * episode_length)


    #-----------------------------------MODEL_initialisation-----------------------------------#
    dynamics = EnsembleDynamicsFast(mask, n_history).to(device1).share_memory()
    policy = ConvPolicyFastFast(xvalid1, yvalid1, KL_projection, n_history).to(device1).share_memory()
    policy_copy = ConvPolicyFastFast(xvalid0, yvalid0, KL_projection, n_history).to(device0).eval().share_memory().eval()

    #policy_copy_comp = torch.compile(policy_copy, mode="reduce-overhead") problem on windows TODO

    #dynamics_optimizer = SharedAdam(dynamics.parameters())
    #policy_optimizer = SharedAdam((policy.parameters()))
    dynamics_optimizer = optim.AdamW(dynamics.parameters(), weight_decay = 1e-3) #don't need shared since no online training
    policy_optimizer = optim.AdamW(policy.parameters(), weight_decay = 1e-3)

    #dynamics_optimizer1 = optim.Adam(dynamics.parameters())#, weight_decay=1e-6)
    #policy_optimizer1 = optim.Adam(policy.parameters())#, weight_decay=1e-6)
    dynamics_optimizer1 = optim.AdamW(dynamics.parameters(), weight_decay = 1e-3)
    policy_optimizer1 = optim.AdamW(policy.parameters(), weight_decay = 1e-3)

    sigma = initial_sigma
    rewards = torch.zeros(iters + warmup_episodes)

    training = False
    obs, _= env.reset(residuals_opds_1rst, seed = env.seed)
    obs = env.flatten_dm()

    #shaped as if not involving current obs or act
    past_obs = torch.zeros(1, (n_history - 1), *obs.shape, device=device0).squeeze(2)
    past_act = torch.zeros(1, (n_history - 1), *obs.shape, device=device0).squeeze(2)



    if not config['save_and_load']['load_warmup_buffer']:
        for i in range(warmup_episodes): #warmup_episodes
            start = time.time()
            #-----------------------------------sample generation-----------------------------------#
            reward_sum, past_obs, past_act, obs = run_episode_warmup(replay, replay_warmup, sigma, episode_length, KL_projection.to(device0), xvalid0, yvalid0, env, residuals_opds_1rst)

            rewards[i] = reward_sum

            #the sigma is the variance innit
            sigma -= (initial_sigma / (warmup_episodes / 1))
            sigma = max(0, sigma)
            end = time.time()

            print(
                f'******************************************** \n Warm up {i} complete ({end - start:.2f}s) \n\t reward:{reward_sum:.3f} \n********************************************')


    #saving and loading data
    if config['save_and_load']['save_warmup_buffer']:
        torch.save(replay_warmup.states, os.path.join(savedir, f"states_warmup.pt"))
        torch.save(replay_warmup.next_states, os.path.join(savedir, f"next_states_warmup.pt"))
        torch.save(replay_warmup.actions, os.path.join(savedir, f"actions_warmup.pt"))

        torch.save(replay.states, os.path.join(savedir, f"states.pt"))
        torch.save(replay.next_states, os.path.join(savedir, f"next_states.pt"))
        torch.save(replay.actions, os.path.join(savedir, f"actions.pt"))

        print(
            f'--------------------------------------------\n warmup buffer saved! \n--------------------------------------------')

    if config['save_and_load']['load_warmup_buffer']:
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

    if config['save_and_load']['load_models_pretrained']:
        dynamics.load_state_dict(
            torch.load(os.path.join(loaddir, f"dynamics_pretrained.pt"), map_location=lambda storage, loc: storage))
        policy.load_state_dict(
            torch.load(os.path.join(loaddir, f"policy_pretrained.pt"), map_location=lambda storage, loc: storage))
        policy_copy.load_state_dict(
            torch.load(os.path.join(loaddir, f"policy_pretrained.pt"), map_location=lambda storage, loc: storage))

        print(
            f'--------------------------------------------\n Pretrained models loaded! \n--------------------------------------------')


    elif replay_warmup.len > episode_length:
        start_time = time.time()

        # -----------------------------------dynamics train-----------------------------------#
        print("start_training")
        dyn_loss, dynamics_loss = train_dynamics(dynamics, dynamics_optimizer1, replay_warmup, replay_warmup,
                                  dyn_iters=config['training']['dynamics_grad_steps_warmup'])
        #todo check synchronisation
        """try:
            torch.cuda.synchronize(device=device1)
        except RuntimeError:
            pass"""

        # -----------------------------------policy train-----------------------------------#
        pol_loss, policy_loss = train_policy(policy_optimizer1, policy, dynamics, replay_warmup, replay_warmup,
                                pol_iters=config['training']['policy_grad_steps_warmup'])
        """try:
            torch.cuda.synchronize(device=device1)
        except RuntimeError:
            pass"""


        policy_copy.load_state_dict(policy.state_dict())

        print(
            f'--------------------------------------------\n Warmup training ({time.time() - start_time:.2f}s). \n\t dyn:{1000 * dyn_loss:.4f} pol:{1000 * pol_loss:.4f} \n--------------------------------------------')

        np.save(f"{savedir}/dynamics_loss.npy", dynamics_loss)
        np.save(f"{savedir}/policy_loss.npy", policy_loss)

        plt.figure()
        plt.subplot(121)
        plt.title("dynamics_loss warmup")
        plt.plot(dynamics_loss)
        plt.grid(True)
        plt.yscale('log')
        #plt.xscale('log')
        plt.subplot(122)
        plt.title("policy_loss warmup")
        plt.grid(True)
        plt.plot(policy_loss)
        plt.yscale('log')
        #plt.xscale('log')
        #plt.show()


    if config['save_and_load']['save_models_pretrained']:
        torch.save(dynamics.state_dict(), os.path.join(savedir, f"dynamics_pretrained.pt"))
        torch.save(policy.state_dict(), os.path.join(savedir, f"policy_pretrained.pt"))

        print(f'--------------------------------------------\n Pretrained models saved! \n--------------------------------------------')

    replay_q.put(replay, False)
    replay_warmup_q.put(replay_warmup, False)

    #TODO uncomment this to use multiprocessing
    """if replay.len > episode_length and replay_warmup.len > episode_length:
        training_process = ctx.Process(target=training_thread,
                                       args=(start_q, dynamics, dynamics_optimizer, replay_q, replay_warmup_q,
                                             policy_optimizer, policy, finished_q, 50, 25,))
        training_process.start()
    else:
        print("Replay buffers empty --- training not started. Run warm up or load buffers")"""

    for p in policy_copy.parameters():
        p.grad = None



    obs = env.flatten_dm()
    INFO_list = 0
    all_2nd_stage_strehl = []
    telescope_pupil = 0
    all_dm_commands = []
    all_reconstructed_cmd = []
    all_src_opd = []
    projector_kl_2nd = 0
    for i in range(iters):

        start = time.time()
        current_iteration = i

        reward_sum, past_obs, past_act, obs, INFO_list = run_episode_policy(past_obs, past_act, obs, replay, policy_copy, sigma,
                                                                                episode_length, env, current_iteration, residuals_opds_1rst)


        all_2nd_stage_strehl.append(INFO_list["2nd_stage_strehl"])
        telescope_pupil = INFO_list["telescope_pupil"]
        all_dm_commands.append(INFO_list["dm_commands"]) 
        all_reconstructed_cmd.append(INFO_list["reconstructed_cmd"])
        all_src_opd.append(INFO_list["src_opd"])
        projector_kl_2nd = INFO_list["projector_kl_2nd"]


        rewards[i + warmup_episodes] = reward_sum
        

        start_time = time.time()
        #uncomment this to NOT use multiprocessing (sequential training)
        print('training dynamics not in training thread')
        dyn_loss, dynamics_loss = train_dynamics(dynamics, dynamics_optimizer, replay, replay_warmup,
                                                 dyn_iters=config['training']['dynamics_grad_steps'])

        print('training policy not in training thread')
        pol_loss, policy_loss = train_policy(policy_optimizer, policy, dynamics, replay, replay_warmup,
                                             pol_iters=config['training']['policy_grad_steps'])

        policy_copy.load_state_dict(policy.state_dict())

        print(f'--------------------------------------------\n training ({time.time() - start_time:.2f}s). \n\t dyn:{1000*dyn_loss:.4f} pol:{1000*pol_loss:.4f} \n--------------------------------------------')


        #uncomment this to use multiprocessing
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

    all_2nd_stage_strehl    = np.concatenate(all_2nd_stage_strehl, axis=0) 
    all_dm_commands         = np.concatenate(all_dm_commands, axis=0) 
    all_reconstructed_cmd   = np.concatenate(all_reconstructed_cmd, axis=0) 
    all_src_opd             = np.concatenate(all_src_opd, axis=0)  




    #YOU SHOULD ALSO CHECK IF YOU ONLY HAVE AN EPISODE OF DATA IN ALL OF THESE BECAUSE I DON'T REMEMBER
    frequency = 1 / env.tel_2nd.samplingTime 
    time_plot = np.arange(0, iters * episode_length / frequency, 1/frequency)



    np.savez(f"{savedir}/results_2nd_stage.npz",
    # Concatenated across iterations
    all_2nd_stage_strehl    = all_2nd_stage_strehl,
    all_dm_commands         = all_dm_commands,
    all_reconstructed_cmd   = all_reconstructed_cmd,
    residual_opds_2nd       = all_src_opd,
    telescope_pupil         = telescope_pupil,
    projector_kl_2nd        = projector_kl_2nd,
    frequency               = frequency,
    time_array              = time_plot,
    )



    #everythin I need is saved here?
    torch.save(rewards, os.path.join(savedir, f"rewards.pt")) #reward sum for each episode
    torch.save(replay.states, os.path.join(savedir, f"states.pt"))
    torch.save(replay.actions, os.path.join(savedir, f"actions.pt"))

    torch.save(dynamics.state_dict(), os.path.join(savedir, f"dynamics_final.pt"))
    torch.save(policy.state_dict(), os.path.join(savedir, f"policy_final.pt"))

    print("data saved!")
    
    b = time.perf_counter()
    print(b-a)
    plt.show()



if __name__ ==  '__main__':
    main()




