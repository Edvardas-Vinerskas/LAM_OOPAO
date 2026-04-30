import numpy as np
import dao
import torch
from torch import nn
import time
import torch.nn.functional as F_interp
from torch.profiler import profile, ProfilerActivity, record_function

n_filt = 32
scaling_factor_up = 1e2

data_shape_2nd_stage = 11

device0 = "cuda:0"
device1 = "cuda:0"

prev_commands_2nd_stage = torch.as_tensor(np.zeros((11, 11)), device=device0).float()
obs_image_2nd_stage = torch.as_tensor(np.zeros((11, 11)), device=device0).float()

dm_shm_2nd_stage = np.zeros((97, 1), dtype = np.float32)
frame_data_2nd_stage = np.zeros((97, 1), dtype = np.float32)
frame_data_2nd_stage_pinned = torch.zeros(97, device='cpu').pin_memory()


m2v_data_2nd_stage = np.load("M2C_2nd.npy")

valid_mask_2nd_stage = np.load("valid_mask_2nd_stage.npy")
dm_x_2nd_stage, dm_y_2nd_stage = np.where(valid_mask_2nd_stage)
dm_coords_2nd_stage = (dm_x_2nd_stage, dm_y_2nd_stage)


KL_projection = m2v_data_2nd_stage[:,:50] @ np.linalg.pinv(m2v_data_2nd_stage[:,:50])
KL_projection = torch.from_numpy(np.asarray(KL_projection)).float()


mask = np.zeros((data_shape_2nd_stage, data_shape_2nd_stage))
mask[dm_x_2nd_stage, dm_y_2nd_stage] = 1
mask = np.array(mask, dtype = bool)


xvalid0 = torch.from_numpy(dm_x_2nd_stage).to(torch.int32).to(device0).squeeze()
yvalid0 = torch.from_numpy(dm_y_2nd_stage).to(torch.int32).to(device0).squeeze()

xvalid1 = torch.from_numpy(dm_x_2nd_stage).to(torch.int32).to(device1).squeeze()
yvalid1 = torch.from_numpy(dm_y_2nd_stage).to(torch.int32).to(device1).squeeze()

past_obs = torch.zeros(1, (64-1), 11, 11, device = device0).squeeze(2)
past_act = torch.zeros(1, (64-1), 11, 11, device = device0).squeeze(2)


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

    def __init__(self, state_shape, action_shape, max_size=100000, warmup_memory = 0):
        self.max_size = max_size

        self.states = torch.empty(max_size, *state_shape).to("cuda:0")
        self.next_states = torch.empty(max_size, *state_shape).to("cuda:0")
        self.actions = torch.empty(max_size, *action_shape).to("cuda:0")

        self.len   = 0
        self.index_write = 0
        self.warmup_memory = warmup_memory


    def add(self, replay):
        cur_len = self.len
        new_len = self.len + len(replay)


        if isinstance(replay, EfficientExperienceReplay):
            replay = ReplaySample(replay.states[:len(replay)], replay.actions[:len(replay)], replay.rewards[:len(replay)], replay.next_states[:len(replay)])

        self.states[cur_len:new_len] = replay.state()
        self.next_states[cur_len:new_len] = replay.next_state()
        self.actions[cur_len:new_len] = replay.action()

        self.len = new_len

    def __add__(self, replay):
        self.add(replay)
        return self


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


    def sample_contiguous(self, horizon, max_ts, batch_size=32):
        inds = torch.randint(0, max_ts - (horizon + 1), size=(batch_size, ))
        inds += torch.randint(0, len(self) // max_ts, size=(batch_size, )) * max_ts

        indices = torch.cat([torch.arange(ind, ind + horizon + 1) for ind in inds])


        return ReplaySample(self.states[indices], self.actions[indices], self.next_states[indices])

    def next_state(self):
        return self.next_states[:self.len]

    def state(self):
        return self.states[:self.len]

    def action(self):
        return self.actions[:self.len]

    def __len__(self):
        return self.len

    def set_len(self,index):
        self.len = index
        self.index_write = index

    def sample(self, size=512):
        inds = torch.randperm(self.len)[:size]
        return ReplaySample(self.states[inds], self.actions[inds], self.next_states[inds])

    def clear(self):
        self.len = 0

replay = EfficientExperienceReplay((11,11), (11,11), 20* 250)

def flatten_dm():
    """
    Pipeline specific function to flatten the DM

    :return dm_image_torch:    2D image of the WFS measurement projected to DM voltages with flattened dm
    """
    

    dm_shm_2nd_stage = np.zeros((97, 1))
    prev_commands_2nd_stage[:,:] = prev_commands_2nd_stage[:,:]*0
    obs_image_2nd_stage[dm_coords_2nd_stage] = torch.from_numpy(frame_data_2nd_stage).flatten().to(device0)
    
    return -obs_image_2nd_stage


class ConvPolicyFastFast_upsampled(nn.Module):
    def __init__(self, xvalid, yvalid, F, n_history):
        super().__init__()

        self.xvalid = xvalid
        self.yvalid = yvalid

        self.n_history = n_history

        self.register_buffer('F', F.unsqueeze(0))

        self.net = nn.Sequential(
            nn.Conv2d(n_history * 2 - 1, n_filt, 3, padding=1),
            nn.LeakyReLU(),

            nn.Conv2d(n_filt, n_filt, 3, padding=1),
            nn.LeakyReLU(),

            # Downsample back to 11x11
            nn.Conv2d(n_filt, n_filt, 3, stride=2, padding=1),
            nn.LeakyReLU(),

            nn.Conv2d(n_filt, 1, 3, padding=1),
        )

    def forward(self, feats):

        x = F_interp.interpolate(feats, scale_factor=2, mode='bilinear', align_corners=False)


        out = self.net(x * scaling_factor_up) / scaling_factor_up
        out = out.clamp(-0.08, 0.08)

        out[:, :, self.xvalid, self.yvalid] = torch.matmul(self.F, out[:, :, self.xvalid, self.yvalid].squeeze(1).unsqueeze(2)).squeeze(-1).unsqueeze(1)

        return out
    



class ConvPolicyFastFast(nn.Module):
    def __init__(self, xvalid, yvalid, F, n_history):
        super().__init__()

        self.xvalid = xvalid
        self.yvalid = yvalid

        self.n_history = n_history

        self.register_buffer('F', F.unsqueeze(0)) 

        self.net = nn.Sequential(
            nn.Conv2d(n_history * 2 -1, n_filt, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(n_filt, n_filt, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(n_filt, 1, 3, padding=1),

        )

    def forward(self, feats):

        out = self.net(feats * scaling_factor_up) / scaling_factor_up
        out = out.clamp(-0.08, 0.08)

        out[:, :, self.xvalid, self.yvalid] = torch.matmul(self.F, out[:, :, self.xvalid, self.yvalid].squeeze(1).unsqueeze(2)).squeeze(-1).unsqueeze(1)

        return out
    

@torch.no_grad() #@torch.inference_mode()
def step_2nd_stage(action):
    """
    Pipeline specific function that sends new commands to dm, i.e., sets the action, and reads the
    following WFS measurement projected to DM space trought a linear recontructor. The action and
    the WFS measurement have to be 2D images.

    :param action:             2D image of DM control voltages to be applied
    :return dm_image_torch:   2"D image of WFS measurement projected to DM voltages
    """


    temp = (prev_commands_2nd_stage * 0.99) + (action.squeeze())
    prev_commands_2nd_stage[:,:] = temp.clamp(-0.1, 0.1)


    dm_shm_2nd_stage = temp[dm_coords_2nd_stage].cpu().numpy()

    obs_image_2nd_stage[dm_coords_2nd_stage] = torch.from_numpy(frame_data_2nd_stage).flatten().to(device0, non_blocking=True)

    return -obs_image_2nd_stage





#shape (batch_size, 2 * n_history - 1, 10, 10)
policy_input = torch.randn(1, 127, 11, 11, dtype = torch.float32).to(device1)

n_history = 64

policy = ConvPolicyFastFast(xvalid1, yvalid1, KL_projection, n_history).to(device = device1, dtype = torch.float32).share_memory()
policy_upsampled = ConvPolicyFastFast_upsampled(xvalid1, yvalid1, KL_projection, n_history).to(device = device1, dtype = torch.float32).share_memory()


'''class PolicyCompFull(nn.Module):
    def __init__(self, policy, initial_past_obs, initial_past_act):
        super().__init__()
        self.policy = policy

        self.register_buffer('past_obs', initial_past_obs)
        self.register_buffer('past_act', initial_past_act)

    def forward(self, obs):
        input_policy = torch.cat([obs.unsqueeze(0).unsqueeze(0), past_obs, past_act], dim = 1)
        action = self.policy(input_policy)

        """self.past_obs = torch.roll(self.past_obs, shifts = -1, dims = 1)
        self.past_obs[:,-1,:,:] = obs
        self.past_act = torch.roll(self.past_act, shifts = -1, dims = 1)
        self.past_act[:,-1,:,:] = action.squeeze()"""
        self.past_obs = torch.cat([self.past_obs, obs.unsqueeze(0).unsqueeze(0)], dim = 1)
        self.past_act = torch.cat([self.past_act, action], dim = 1)

        return action
'''


'''@torch.no_grad() #@torch.inference_mode()
def run_episode_policy(past_obs, past_act, obs, replay, policy_comp, sigma, episode_length):
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
        

        with record_function("1_Policy_Forward"):
            action = policy_comp(obs)#, past_obs.clone(), past_act.clone())
        with record_function("2_Env_Step"):
            next_obs = step_2nd_stage(action)

        with record_function("data append"):
            reward_sum += torch.sum((obs.flatten() * scaling_factor_up) ** 2)

            action_to_save = action.squeeze()
            replay.append(obs, action_to_save, next_obs)


            obs = next_obs
            
        
    return reward_sum, past_obs, past_act, obs
'''


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



        #start = time.time()
        input_policy = torch.cat([obs.unsqueeze(0).unsqueeze(0), past_obs, past_act],dim = 1)
        action = policy(input_policy) #/ 5
        #if t > 490:
        #    print(f"Elapsed time: {(time.time()-start)*1e6:.2f} us")

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
        print("frequency, ep_policy", 1/(b - a), end = '\r')
        
    return reward_sum, past_obs, past_act, obs


policy = torch.compile(policy, mode = "reduce-overhead")
#policy_comp = PolicyCompFull(policy, past_obs, past_act)
#policy_comp = torch.compile(policy_comp, mode = "reduce-overhead")



policy.eval()
#policy_comp.eval()


start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

obs = flatten_dm()




for i in range(20):
    reward_sum, past_obs, past_act, obs = run_episode_policy(past_obs, past_act, obs, replay, policy, 0.2,
                                                        1)
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    with record_function("model_inference"):
        for i in range(1):
            reward_sum, past_obs, past_act, obs = run_episode_policy(past_obs, past_act, obs, replay, policy, 0.2,
                                                                1)
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))


for _ in range(10000):
    start.record()
    reward_sum, past_obs, past_act, obs = run_episode_policy(past_obs, past_act, obs, replay, policy, 0.2,
                                                                    1)
    end.record()
    torch.cuda.synchronize()

    print(1000/start.elapsed_time(end), end = '\r')
#so you probably should change the mask dtype due to optimisation (default is int64)
#if you don't compile torch.float32 is faster, if you compile torch.float16 is faster
#       this could be just due to some transformation going on under the hood everytime you run if you don't compile since pytorch is written for float32
#       when you compile everything gets hardcoded as much as possible and so it works innit

#you can use the following for graph tracing to see if there are any graph breaks when running
    #import traceback as tb

    #torch._logging.set_logs(graph_code=True)
    #torch._logging.set_logs(graph_breaks=True)

#note that all of the python operations outside pytorch MUST be run on the CPU
#GRAPHS WILL BREAK AT IF STATEMENTS (the value needs to be transported from gpu to cpu and evaluated -> very inefficient)
    #but you can use from functorch.experimental.control_flow import cond
#   tensor.item() -> converts to python number
#   tensor.tolist()
#   tensor.numpy()    i.e. basically everything that makes it a non-tensor object


#for forcing graph break errors
# Reset to clear the torch.compile cache
"""torch._dynamo.reset()

opt_bar_fullgraph = torch.compile(bar, fullgraph=True)
try:
    opt_bar_fullgraph(torch.randn(10), torch.randn(10))
except:
    tb.print_exc()"""

#use compiled_policy = torch.compile(policy) to have a clear view of where you are accessing policy and where the compiled version
#Always use torch.save(model.state_dict(), "path.pt"). (as opposed model_compiled)

#YOU NEED TO WARMUP YOUR COMPILED MODEL BECAUSE FOR THE FIRST FEW ITERATIONS IT OPTIMIZES THE GRAPH