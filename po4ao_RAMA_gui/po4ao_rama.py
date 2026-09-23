"""PO4AO on RAMA as a resident controller (gui/PROPOSAL.md, steps 1-4 in one file).

The process stays alive and is driven through DAO SHMs (po4ao_interface.py) by
po4aoCtrl.py or any script: loop mode open / integrator / policy, training and
exploration on-off, live parameters, warm-up, checkpoint save / load, policy
reload, quit. Models and replay buffers stay on the GPU between mode changes,
so switching controllers or loading a pre-trained policy no longer needs a
restart.

  main process   control loop (one episode at a time) + supervisor between
                 episodes: commands, requests, sigma schedule, policy swap,
                 status publication
  spawned process trainer: while enabled, one round of dynamics + policy training
                 per new episode in the shared replay buffers, then publish the
                 policy weights with a version number

Kept from the monolithic script: integrator + noise warm-up with a decreasing
sigma, pre-training on the warm-up buffer, then the policy with on-line
training. Changed: none of it starts by itself (unless warmup_at_start), the DM
starts flat in OPEN, there is no print in the frame loop.

Mode changes: OPEN and QUIT act immediately (the current episode is dropped);
integrator <-> policy and exploration apply at the next episode boundary so the
replay buffers only ever hold complete, homogeneous episodes.
"""

import os
CONDA_PREFIX = os.environ.get("CONDA_PREFIX", "/home/rama/miniconda3/envs/rtc")
CONDA_LIB = os.path.join(CONDA_PREFIX, "lib")
LIBSTDCXX = os.path.join(CONDA_LIB, "libstdc++.so.6")
os.environ["LD_LIBRARY_PATH"] = CONDA_LIB + ":" + os.environ.get("LD_LIBRARY_PATH", "")
import ctypes
ctypes.CDLL(LIBSTDCXX, mode=ctypes.RTLD_GLOBAL)
import dao

import json
import time

import numpy as np
import torch
from torch import optim
import torch.multiprocessing as mp

from po4ao_config import config
from po4ao_interface import (
    SHM_CMD, SHM_PARAMS, SHM_STATUS, CHECKPOINT_REQUEST_FILE, N_CMD, N_STATUS,
    CMD_LOOP, CMD_TRAIN, CMD_EXPLORE, CMD_SAVE_MODELS, CMD_SAVE_BUFFERS, CMD_RELOAD_POLICY,
    CMD_LOAD_CHECKPOINT, CMD_WARMUP, CMD_QUIT, CMD_SETTINGS,
    LOOP_OPEN, LOOP_INTEGRATOR, LOOP_POLICY, LOOP_NAMES, LIVE_PARAMS,
    ST_EPISODE, ST_STEP, ST_HZ, ST_REWARD, ST_DYN_LOSS, ST_POL_LOSS, ST_TRAINING, ST_VERSION,
    ST_MODE, ST_EXPLORE, ST_SIGMA, ST_STATE, ST_REPLAY_EP, ST_WARMUP_EP, ST_ROUNDS, ST_POLICY_READY,
    STATE_IDLE, STATE_WARMUP, STATE_PRETRAIN, STATE_RUN,
    open_shm)
from po4ao_models_upd import EnsembleDynamicsFast, ConvPolicyFastFast
from po4ao_util import EpisodeReplay

# restart-only parameters (they size tensors / networks or set the schedule)
episode_length = config['RL']['episode_length']
warmup_episodes = config['RL']['warmup_episodes']
min_sigma = config['RL']['min_sigma']
n_history = config['MDP']['n_history']
data_shape = config['MDP']['data_shape']
nmodes = config['integrator']['n_modes']
replay_size = config['replay_buffers']['replay_size']
warmup_memory = config['replay_buffers']['warmup_memory']
batch_size = config['NN_models']['training_batch']

device0 = 'cuda:0'   # control
device1 = 'cuda:0'   # training (synchronise when changing)

# trainer commands / states (multiprocessing Values)
TR_QUIT, TR_IDLE, TR_TRAIN, TR_PRETRAIN = -1, 0, 1, 2


# ============================================================================ live parameters
#describe exactly what this is doing
class LiveParams:
    """Snapshot of po4aoParams, refreshed only when the SHM counter changes."""

    def __init__(self, shm):
        self.shm = shm
        self.counter = None
        self.v = {}
        self.refresh()
    #rereads po4aoParams if the GUI pressed APPLY LIVE
    def refresh(self):
        c = self.shm.get_counter()
        if c == self.counter:
            return False
        vals = np.asarray(self.shm.get_data()).astype(np.float64).ravel()
        self.v = {key: float(vals[i]) for i, (_, key) in enumerate(LIVE_PARAMS)}
        self.counter = c
        return True

    @property
    def max_sigma(self): return self.v["max_sigma"]
    @property
    def penalty(self): return self.v["loss_function_penalty"]
    @property
    def horizon(self): return max(1, int(round(self.v["planning_horizon"])))
    @property
    def warmup_fraction(self): return self.v["train_warmup_percent"]
    @property
    def gain(self): return self.v["gain"]
    @property
    def leak(self): return self.v["leak"]
    @property
    def use_offset(self): return self.v["offset"] > 0.5


def initial_params():
    return np.array([[float(config[s][k])] for s, k in LIVE_PARAMS], dtype=np.float32)


# ============================================================================ bench
class Bench:
    """RAMA SHMs: pyrModesNN in, dm1Cmd05 out. Owns the integrated command."""

    def __init__(self):
        self.m2c = dao.shm('/tmp/dm1M2A.im.shm').get_data().astype(np.float32)
        valid_mask = dao.shm("/tmp/dm1Map.im.shm").get_data().astype(np.float32)
        self.dm_x, self.dm_y = np.where(valid_mask)
        self.n_act = len(self.dm_x)
        self.n_modes_in = self.m2c.shape[1]
        self.frame_shm = dao.shm("/tmp/pyrModesNN.im.shm")
        self.dm_shm = dao.shm("/tmp/dm1Cmd05.im.shm")
        self.offset_shm = dao.shm("/tmp/dm1CmdOffset.im.shm")
        self.clamp = float(config['integrator']['command_clamp'])

        self.xvalid = torch.from_numpy(self.dm_x).to(torch.int32).to(device0)
        self.yvalid = torch.from_numpy(self.dm_y).to(torch.int32).to(device0)
        self.prev_commands = torch.zeros((data_shape, data_shape), device=device0)
        self.obs_image = torch.zeros((data_shape, data_shape), device=device0)

        proj = self.m2c[:, :nmodes] @ np.linalg.pinv(self.m2c[:, :nmodes])
        self.kl_projection = torch.from_numpy(proj).float().to(device0)


    def _read_obs(self, sem, use_offset):
        modes = self.frame_shm.get_data(check=True, semNb=sem).astype(np.float32).squeeze()[:self.n_modes_in]
        vec = self.m2c @ modes
        if use_offset:
            vec = vec - self.offset_shm.get_data().astype(np.float32).squeeze()
        self.obs_image[self.dm_x, self.dm_y] = torch.from_numpy(vec).to(device0)
        return -self.obs_image

    @torch.no_grad()
    def step(self, action, leak, use_offset):
        """Apply a 2D action on the DM, wait for the next WFS frame, return -obs (2D, DM space)."""
        temp = (self.prev_commands * leak + action.squeeze()).clamp(-self.clamp, self.clamp)
        self.prev_commands.copy_(temp)
        self.dm_shm.set_data(temp[self.dm_x, self.dm_y].cpu().numpy())
        return self._read_obs(3, use_offset)


    @torch.no_grad()
    def flatten(self, use_offset, wait_frame=True):
        """Zero dm1Cmd05 and the integrated command; return -obs of the next frame
        (None with wait_frame=False, for shutdown when the CNN may have stopped)."""
        self.dm_shm.set_data(np.zeros_like(self.dm_shm.get_data()))
        self.prev_commands.zero_()
        return self._read_obs(5, use_offset) if wait_frame else None

    @torch.no_grad()
    def sample_noise(self, sigma):
        vec = torch.matmul(self.kl_projection, sigma * torch.sign(torch.randn((self.n_act,), device=device0)))
        im = torch.zeros((data_shape, data_shape), device=device0)
        im[self.xvalid, self.yvalid] = vec
        return im


# ============================================================================ training
#here the unfolding is done in a function instead of inside train_dynamics or train_policy
def unfold(sample, horizon):
    """Split a (batch * (horizon+1)) sample into state, action, history, next_state."""
    states = sample.state().view(batch_size, horizon + 1, 1, *sample.state().shape[1:])
    actions = sample.action().view(batch_size, horizon + 1, *sample.action().shape[1:])[:, :-1].unsqueeze(2)
    next_states = states[:, -1]
    states = states[:, :-1]
    state = states[:, -1].squeeze(2)
    action = actions[:, -1].squeeze(2)
    past_obs = states[:, :-1].squeeze(2)
    past_act = actions[:, :-1].squeeze(2)
    return state, action, past_obs, past_act, next_states



def pick_replay(replay, replay_warmup, warmup_fraction):
    if replay_warmup.n_valid == 0:
        return replay
    if replay.n_valid == 0 or torch.rand(1).item() < warmup_fraction:
        return replay_warmup
    return replay


def train_dynamics(dynamics, optimizer, replay, replay_warmup, warmup_fraction, steps):
    """Gradient steps on the dynamics ensemble; each member gets its own batch."""
    dynamics.train()

    dyn_losses = []
    for _ in range(steps):
        optimizer.zero_grad()
        loss = 0
        for model in dynamics.models:
            sample = pick_replay(replay, replay_warmup, warmup_fraction).sample_contiguous(n_history, batch_size).to(device1)
            state, action, past_obs, past_act, next_states = unfold(sample, n_history)
            pred = model(torch.cat([past_obs, past_act, state, action], dim=1))
            assert pred.shape == next_states.shape
            loss +=(next_states - pred).pow(2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dynamics.parameters(), 0.5)
        optimizer.step()
        dyn_losses.append(loss.item())
    return dyn_losses


def train_policy(policy, dynamics, optimizer, replay, replay_warmup, warmup_fraction, steps, horizon, penalty):
    """Gradient steps on the policy through the (frozen) dynamics over `horizon` roll-out steps."""
    dynamics.train()
    policy.train()
    for p in dynamics.parameters():
        p.requires_grad_(False)
    

    pol_losses = []
    for _ in range(steps):
        optimizer.zero_grad()
        sample = pick_replay(replay, replay_warmup, warmup_fraction).sample_contiguous(n_history, batch_size).to(device1)
        state, _, past_obs, past_act, _ = unfold(sample, n_history)

        total_loss = torch.zeros(batch_size, device=device1)
        for _ in range(horizon):
            history = torch.cat([past_obs, past_act], dim=1)
            action = policy(torch.cat([state, history], dim=1))
            next_state = dynamics(torch.cat([history, state, action], dim=1))   # (b, n_models, n, n)
            next_mean = torch.mean(next_state, dim=1, keepdim=True)              # (b, 1, n, n)
            total_loss+= next_mean.pow(2).mean() + penalty * action.pow(2).mean()
            past_act = torch.cat([past_act[:, 1:], action], dim=1)
            past_obs = torch.cat([past_obs[:, 1:], state], dim=1)
            state = next_mean
        loss = total_loss.mean()
        loss.backward()
        optimizer.step()
        pol_losses.append(loss.item())
    for p in dynamics.parameters():
        p.requires_grad_(True)
    return pol_losses


#this class contains a bunch of multiprocessing values
class TrainerLink:
    """Shared scalars between the control process and the trainer."""

    def __init__(self, ctx):
        self.cmd = ctx.Value('i', TR_IDLE)        # sets the state of controller (idle / train / pretrain / quit) (corresponds to the button press?)
        self.state = ctx.Value('i', TR_IDLE)      # show what it is actually doing now (you press button and there might be a delay between entering the selected state)
        self.policy_version = ctx.Value('i', 0)   # +1 at each publish(); the controller compares it with the version it has swapped in
        self.episodes = ctx.Value('i', 0)         # +1 by the controller at each committed episode; the trainer does one round per new episode
        self.rounds = ctx.Value('i', 0)           # counts how many times dynamics/policy have been updated and saves loss values every 50 updates
        self.pretrain_done = ctx.Value('i', 0)    # +1 when a pre-training pass ends; the controller polls it to know when to switch to the policy
        self.dyn_loss = ctx.Value('d', 0.0)       
        self.pol_loss = ctx.Value('d', 0.0)       
        self.lock = ctx.Lock()                    #multiprocessing function


def trainer_process(dynamics, policy, policy_pub, link, replay, replay_warmup, run_dir):
    """Process 2. While link.cmd == TR_TRAIN: one training round per newly committed
    episode (link.episodes), as in the monolithic script; on TR_PRETRAIN: one
    pre-training pass on the warm-up buffer.

    dynamics / policy / policy_pub / replay buffers are shared CUDA tensors.
    """
    print(f"[trainer] started, cores {os.sched_getaffinity(0)}")

    params = LiveParams(open_shm(SHM_PARAMS))
    dyn_opt = optim.AdamW(dynamics.parameters(), weight_decay=1e-3)
    pol_opt = optim.AdamW(policy.parameters(), weight_decay=1e-3)
    dyn_steps = config['training']['dynamics_grad_steps']
    pol_steps = config['training']['policy_grad_steps']
    dyn_steps_warm = config['training']['dynamics_grad_steps_warmup']
    pol_steps_warm = config['training']['policy_grad_steps_warmup']
    log_dyn, log_pol = [], []
    trained_episodes = link.episodes.value      # last episode count a round was done for


    def publish():
        with link.lock: #the lock here ensures that when reading the policy values, another process does not write into them at the same time, thus mixing "old" with "new"
            pub = policy_pub.state_dict()
            for k, v in policy.state_dict().items():
                pub[k].copy_(v)
            link.policy_version.value += 1

    #breaks the loop if cmd == TR_QUIT
    #goes on to check if it is idle and refreshes params?
    #if pretrain, then trains the dynamics and policy using replay_warmup
    #records the policy state_dcit into policy_pub
    #sets cmd value to TR_IDLE after finishing and sets the pretrained_value += 1
    #else it engages in online training, setting the state value to TR_TRAIN, updating policy_pub
    #if you stop this while loop, then it sets the state to TR_IDLE and saves the loss functions
    try:
        while True:
            cmd = link.cmd.value
            if cmd == TR_QUIT:
                break
            if cmd == TR_IDLE or (replay.n_valid == 0 and replay_warmup.n_valid == 0):
                link.state.value = TR_IDLE
                time.sleep(0.05)
                continue

            params.refresh()
            if cmd == TR_PRETRAIN:
                link.state.value = TR_PRETRAIN
                t0 = time.time()
                # as before: warm-up data only, fresh optimisers, many steps
                dyn_opt = optim.AdamW(dynamics.parameters(), weight_decay=1e-3)
                pol_opt = optim.AdamW(policy.parameters(), weight_decay=1e-3)
                d = train_dynamics(dynamics, dyn_opt, replay_warmup, replay_warmup, 1.0, dyn_steps_warm)
                torch.cuda.synchronize(device=device1)
                p = train_policy(policy, dynamics, pol_opt, replay_warmup, replay_warmup, 1.0, pol_steps_warm,
                                 params.horizon, params.penalty)
                torch.cuda.synchronize(device=device1)
                publish()
                np.save(os.path.join(run_dir, "dynamics_loss_warmup.npy"), d)
                np.save(os.path.join(run_dir, "policy_loss_warmup.npy"), p)
                link.dyn_loss.value, link.pol_loss.value = d[-1], p[-1]
                print(f"[trainer] pre-training done ({time.time() - t0:.1f}s) dyn:{1000 * d[-1]:.4f} pol:{1000 * p[-1]:.4f}")
                with link.cmd.get_lock():
                    if link.cmd.value == TR_PRETRAIN:
                        link.cmd.value = TR_IDLE
                link.pretrain_done.value += 1
            else:  # TR_TRAIN: one round per new episode, otherwise wait for data
                link.state.value = TR_TRAIN
                n_ep = link.episodes.value
                if n_ep == trained_episodes:
                    time.sleep(0.005)
                    continue
                trained_episodes = n_ep
                d = train_dynamics(dynamics, dyn_opt, replay, replay_warmup, params.warmup_fraction, dyn_steps)
                p = train_policy(policy, dynamics, pol_opt, replay, replay_warmup, params.warmup_fraction, pol_steps,
                                 params.horizon, params.penalty)
                torch.cuda.synchronize(device=device1)
                publish()
                link.dyn_loss.value, link.pol_loss.value = d[-1], p[-1]
                link.rounds.value += 1
                log_dyn += d
                log_pol += p

                if link.rounds.value % 50 == 0:
                    np.save(os.path.join(run_dir, "dynamics_loss.npy"), log_dyn)
                    np.save(os.path.join(run_dir, "policy_loss.npy"), log_pol)
    except KeyboardInterrupt:
        pass
    link.state.value = TR_IDLE
    np.save(os.path.join(run_dir, "dynamics_loss.npy"), log_dyn)
    np.save(os.path.join(run_dir, "policy_loss.npy"), log_pol)
    print("[trainer] exit")


# ============================================================================ controller
#the bench class has all the read and write functions to the AO bench
class Controller:
    def __init__(self):
        self.ctx = mp.get_context('spawn')
        self.run_dir = os.path.join("PO4AO_gui", "logs",
                                    time.strftime("%Y%m%d-%H%M%S") + "_" + config['save_and_load']['run_name'])
        os.makedirs(self.run_dir, exist_ok=True)
        with open(os.path.join(self.run_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=1)

        self.bench = Bench()

        # control SHMs: must already exist on the bench (see po4ao_interface.py);
        # zeroed here so stale requests from a previous run are ignored
        self.cmd_shm = open_shm(SHM_CMD)
        self.params_shm = open_shm(SHM_PARAMS)
        self.status_shm = open_shm(SHM_STATUS)
        for path, shm in ((SHM_CMD, self.cmd_shm), (SHM_PARAMS, self.params_shm), (SHM_STATUS, self.status_shm)):
            if shm is None:
                raise SystemExit(f"control SHM {path} not found; create it before starting (see po4ao_interface.py)")
        self.cmd_shm.set_data(np.zeros((N_CMD, 1), dtype=np.int32))
        self.params_shm.set_data(initial_params())     # LIVE_PARAMS from the config, until APPLY LIVE
        self.status_shm.set_data(np.zeros((N_STATUS, 1), dtype=np.float32))
        #what is the difference between LIVE_PARAMS and LiveParams?
        self.params = LiveParams(self.params_shm)
        self.cmd_counter = self.cmd_shm.get_counter()
        self.last_cmd = np.zeros(N_CMD, dtype=np.int64)

        self.requests = set()

        # models and buffers, shared with the trainer (CUDA IPC)
        mask = np.zeros((data_shape, data_shape), dtype=bool)
        mask[self.bench.dm_x, self.bench.dm_y] = True

        self.dynamics = EnsembleDynamicsFast(mask, n_history).to(device1).share_memory()
        self.policy = ConvPolicyFastFast(self.bench.xvalid, self.bench.yvalid, self.bench.kl_projection, n_history).to(device1).share_memory()
        self.policy_pub = ConvPolicyFastFast(self.bench.xvalid, self.bench.yvalid, self.bench.kl_projection, n_history).to(device1).share_memory()
        self.policy_copy = ConvPolicyFastFast(self.bench.xvalid, self.bench.yvalid, self.bench.kl_projection, n_history).to(device0).eval()

        for p in self.policy_copy.parameters():
            p.requires_grad_(False)
        shape = (data_shape, data_shape)
        #TODO double check what kind of device you want here!!!
        self.replay = EpisodeReplay(shape, shape, replay_size, episode_length, self.ctx, device0)
        self.replay_warmup = EpisodeReplay(shape, shape, warmup_memory, episode_length, self.ctx, device0)

        #here is where all of the parallel processes are started
        self.link = TrainerLink(self.ctx) #link has all of the shared values
        self.trainer = self.ctx.Process(target=trainer_process, name="po4ao-trainer",
                                        args=(self.dynamics, self.policy, self.policy_pub, self.link,
                                              self.replay, self.replay_warmup, self.run_dir))
        #so here the parallel process is started (in a new python interpreter)
        self.trainer.start()

        # supervisor state
        #initialises the supervisor variable or something...
        self.mode = LOOP_OPEN 
        self.pending_mode = LOOP_OPEN
        self.explore = False 
        self.explore_req = False
        self.train_req = False
        self.state = STATE_IDLE
        self.warmup_remaining = 0
        self.sigma = 0.0   
        self.policy_ready = False
        self.swapped_version = 0
        self.episode = 0
        self.step_i = 0
        self.hz = 0.0
        self.reward = 0.0
        self.quit = False
        self.last_status_t = 0.0

        self.obs = self.bench.flatten(self.params.use_offset)
        self.reset_history()
        self.warm_cuda()

    # ------------------------------------------------------------------ helpers
    def reset_history(self):
        self.past_obs = torch.zeros(1, n_history - 1, data_shape, data_shape, device=device0)
        self.past_act = torch.zeros(1, n_history - 1, data_shape, data_shape, device=device0)

    #TODO check that all of the devices are correct (check how and when this warms up the policy)
    @torch.no_grad()
    def warm_cuda(self):
        x = torch.zeros(1, 2 * n_history - 1, data_shape, data_shape, device=device0)
        for _ in range(10):
            self.policy_copy(x)
        torch.cuda.synchronize(device0)


    def log(self, msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    def go_open(self):
        if self.mode != LOOP_OPEN:
            self.log("loop OPEN, DM flat")
        self.mode = self.pending_mode = LOOP_OPEN
        self.explore = False
        self.obs = self.bench.flatten(self.params.use_offset)
        self.reset_history()
        if self.state == STATE_WARMUP:
            self.log("warm-up interrupted")
            self.warmup_remaining = 0
            self.state = STATE_IDLE

    #sets the link.cmd value
    def set_trainer(self, cmd):
        with self.link.cmd.get_lock():
            if self.link.cmd.value != TR_QUIT:
                self.link.cmd.value = cmd
    
    
    def pause_trainer(self, timeout=30.0):
        """Stop training and wait until the trainer is idle (models/buffers can be touched)."""
        was = self.link.cmd.value
        self.set_trainer(TR_IDLE)
        t0 = time.time()
        while self.link.state.value != TR_IDLE and time.time() - t0 < timeout:
            time.sleep(0.01)
        if self.link.state.value != TR_IDLE:
            self.log("WARNING: trainer did not pause in time")
        return was


    def swap_policy(self):
        with self.link.lock:
            self.policy_copy.load_state_dict(self.policy_pub.state_dict())
            self.swapped_version = self.link.policy_version.value
        self.policy_ready = True
        self.log(f"inference policy <- version {self.swapped_version}")

    # ------------------------------------------------------------------ commands

    #asks whether the cmd shared memory has changed and changes the variable accordingly
    def poll_cmd(self):
        """Read po4aoCmd if it changed. Only OPEN and QUIT take effect immediately.

        A slot acts only when its value changed, so an unrelated request (e.g.
        SAVE MODELS) never re-applies a stale mode.
        """
        c = self.cmd_shm.get_counter()
        if c == self.cmd_counter:
            return
        self.cmd_counter = c
        #TODO is this the correct data type?
        cmd = np.asarray(self.cmd_shm.get_data()).astype(np.int64).ravel()
        changed = np.flatnonzero(cmd != self.last_cmd)
        self.last_cmd = cmd
        for i in changed:
            if i == CMD_LOOP:
                self.pending_mode = int(cmd[i])
            elif i == CMD_TRAIN:
                self.train_req = bool(cmd[i])
            elif i == CMD_EXPLORE:
                self.explore_req = bool(cmd[i])
            else:
                self.requests.add(int(i))
        if CMD_QUIT in self.requests:
            self.quit = True
        if self.pending_mode == LOOP_OPEN and self.mode != LOOP_OPEN:
            self.go_open()


    def sync_cmd(self):
        """Write the actual mode / flags back into slots 0-2 after the supervisor changed
        them itself (end of warm-up, fallback, refused request), so the GUI's next
        click is a change relative to the real state."""
        want = (self.mode, int(self.train_req), int(self.explore_req))
        if tuple(int(v) for v in self.last_cmd[:len(CMD_SETTINGS)]) == want:
            return
        cmd = np.asarray(self.cmd_shm.get_data()).astype(np.int64).ravel()  # keep the request counters
        cmd[:len(CMD_SETTINGS)] = want
        self.last_cmd = cmd
        self.cmd_shm.set_data(cmd.astype(np.int32).reshape(N_CMD, 1))
        self.cmd_counter = self.cmd_shm.get_counter()
        self.pending_mode = self.mode

    #seems to be the function that is called at the boundary to see whether something changed (saving, loading, stopping etc.)
    def handle_requests(self):
        """Counters that changed since the last boundary."""
        req, self.requests = self.requests, set()
        if not req or self.quit:
            return
        if CMD_SAVE_MODELS in req or CMD_SAVE_BUFFERS in req:
            self.save_checkpoint(models=CMD_SAVE_MODELS in req, buffers=CMD_SAVE_BUFFERS in req)
        if CMD_LOAD_CHECKPOINT in req:
            try: #TODO check CHECKPOINT_REQUEST_FILE file path, something is fishy
                path = open(CHECKPOINT_REQUEST_FILE).read().strip()
            except OSError:
                path = ""
            self.load_checkpoint(path)
        if CMD_RELOAD_POLICY in req:
            if self.link.policy_version.value > 0:
                self.swap_policy()
            else:
                self.log("no trained policy to reload yet")
        if CMD_WARMUP in req:
            self.start_warmup()

    #handles the three settings slots: mode, training flag, exploration flag
    #applied at the boundary
    def apply_settings(self):
        """Mode / exploration / training requested by the GUI, at an episode boundary."""
        if self.state in (STATE_WARMUP, STATE_PRETRAIN):
            return                                    # the schedule owns the mode until it ends
        if self.pending_mode != self.mode:
            if self.pending_mode == LOOP_POLICY and not self.policy_ready:
                self.log("POLICY refused: no trained policy (run a warm-up or load a checkpoint)")
                self.pending_mode = self.mode
            else:
                if self.mode == LOOP_OPEN:
                    self.obs = self.bench.flatten(self.params.use_offset)
                    self.reset_history()
                self.mode = self.pending_mode
                self.log(f"loop mode -> {LOOP_NAMES[self.mode]}")
        self.explore = self.explore_req and self.mode != LOOP_OPEN
        self.sigma = self.params.max_sigma if self.explore else 0.0
        self.set_trainer(TR_TRAIN if self.train_req else TR_IDLE) 
        self.state = STATE_RUN if self.mode != LOOP_OPEN else STATE_IDLE

    # ------------------------------------------------------------------ warm-up / pre-training
    def start_warmup(self):
        if warmup_episodes < 1:
            self.log("warmup_episodes is 0")
            return
        self.log(f"warm-up: {warmup_episodes} integrator episodes with noise {self.params.max_sigma} -> 0")
        self.set_trainer(TR_IDLE)
        self.replay_warmup.clear()
        self.state = STATE_WARMUP
        self.warmup_remaining = warmup_episodes
        self.mode = LOOP_INTEGRATOR
        self.explore = True
        self.sigma = self.params.max_sigma
        self.obs = self.bench.flatten(self.params.use_offset)
        self.reset_history()

    #is reset_history needed here? supposedly yes because of the if self.warmup_remaining > 0:
    def end_warmup_episode(self):
        self.warmup_remaining -= 1
        # same schedule as before: sigma_i = max_sigma * (1 - i / warmup_episodes)
        self.sigma = max(0.0, self.params.max_sigma * self.warmup_remaining / warmup_episodes)
        if self.warmup_remaining > 0:
            # each warm-up episode starts from a flat DM (as before)
            self.obs = self.bench.flatten(self.params.use_offset)
            self.reset_history()
            return
        self.explore = False
        self.sigma = 0.0

        self.log(f"warm-up done, {self.replay_warmup.n_valid} episodes stored; pre-training "
                 f"({config['training']['dynamics_grad_steps_warmup']} + "
                 f"{config['training']['policy_grad_steps_warmup']} steps), integrator keeps running")
        self.state = STATE_PRETRAIN
        self.pretrain_seen = self.link.pretrain_done.value
        self.set_trainer(TR_PRETRAIN)

    #if state == STATE_PRETRAIN, then saves models and turns on policy if selected
    def check_pretrain(self):
        if self.state != STATE_PRETRAIN or self.link.pretrain_done.value == self.pretrain_seen:
            return
        self.swap_policy()
        if config['save_and_load']['save_after_pretrain']:
            self.save_checkpoint(models=True, buffers=True, tag="pretrained")
        self.state = STATE_RUN
        if config['RL']['policy_after_warmup']:
            self.mode = self.pending_mode = LOOP_POLICY
            self.train_req = True
            self.log("pre-training done -> POLICY, training on")
        else:
            self.log("pre-training done, integrator kept; press CLOSE (policy) when ready")

    # ------------------------------------------------------------------ checkpoints
    def save_checkpoint(self, models=True, buffers=True, tag="ckpt"):
        """Write <run_dir>/<tag>_<time>/{models.pt, replay.pt, replay_warmup.pt, config.json}.

        Read-only, so the trainer is not paused (a pause would stall the loop for a
        training round): the policy is taken from the published copy under its lock,
        the buffers export only committed episodes.
        """
        path = os.path.join(self.run_dir, f"{tag}_{time.strftime('%Y%m%d-%H%M%S')}")
        os.makedirs(path, exist_ok=True)
        if models:
            with self.link.lock:
                policy_sd = {k: v.clone() for k, v in self.policy_pub.state_dict().items()}
                version = self.link.policy_version.value
            torch.save(dict(dynamics=self.dynamics.state_dict(), policy=policy_sd, version=version,
                            n_history=n_history, data_shape=data_shape),
                       os.path.join(path, "models.pt"))
        if buffers:
            torch.save(self.replay.export(), os.path.join(path, "replay.pt"))
            torch.save(self.replay_warmup.export(), os.path.join(path, "replay_warmup.pt"))
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f, indent=1)
        self.log(f"saved {'models ' if models else ''}{'buffers ' if buffers else ''}-> {path}")
        return path



    def load_checkpoint(self, path):
        """Load models and/or buffers from a checkpoint directory. Forces OPEN."""
        if not path or not os.path.isdir(path):
            self.log(f"load: '{path}' is not a directory")
            return
        self.go_open()
        self.state = STATE_IDLE
        self.pause_trainer()
        loaded = []
        try:
            f = os.path.join(path, "models.pt")
            if os.path.exists(f):
                ck = torch.load(f, map_location=device1)

                if ck.get("n_history", n_history) != n_history or ck.get("data_shape", data_shape) != data_shape:
                    raise ValueError(f"checkpoint n_history/data_shape {ck.get('n_history')}/{ck.get('data_shape')} "
                                     f"!= {n_history}/{data_shape}")
                self.dynamics.load_state_dict(ck["dynamics"])
                self.policy.load_state_dict(ck["policy"])
                with self.link.lock:
                    self.policy_pub.load_state_dict(ck["policy"])
                    self.link.policy_version.value += 1
                self.swap_policy()
                loaded.append("models")
            for name, buf in (("replay.pt", self.replay), ("replay_warmup.pt", self.replay_warmup)):
                f = os.path.join(path, name)
                if os.path.exists(f):
                    buf.load(torch.load(f, map_location="cpu"))
                    loaded.append(f"{name[:-3]}({buf.n_valid} ep)")
            torch.cuda.synchronize(device0)
        except Exception as e:
            self.log(f"load FAILED from {path}: {e}")
            return
        self.log(f"loaded {', '.join(loaded) or 'nothing'} from {path}; loop stays OPEN")

    # ------------------------------------------------------------------ status
    #writes data into the status shared memory to update the gui display
    def publish_status(self):
        st = np.zeros((N_STATUS, 1), dtype=np.float32)
        st[ST_EPISODE] = self.episode
        st[ST_STEP] = self.step_i
        st[ST_HZ] = self.hz
        st[ST_REWARD] = self.reward
        st[ST_DYN_LOSS] = self.link.dyn_loss.value
        st[ST_POL_LOSS] = self.link.pol_loss.value
        st[ST_TRAINING] = self.link.state.value == TR_TRAIN
        st[ST_VERSION] = self.swapped_version
        st[ST_MODE] = self.mode
        st[ST_EXPLORE] = self.explore
        st[ST_SIGMA] = self.sigma
        st[ST_STATE] = self.state
        st[ST_REPLAY_EP] = self.replay.n_valid
        st[ST_WARMUP_EP] = self.replay_warmup.n_valid
        st[ST_ROUNDS] = self.link.rounds.value
        st[ST_POLICY_READY] = self.policy_ready
        self.status_shm.set_data(st)
        self.last_status_t = time.time()

    # ------------------------------------------------------------------ control loop
    #merges run_episode_warmup and run_episode_policy
    @torch.no_grad()
    def run_episode(self):
        """One episode in the current mode. Returns False if aborted (OPEN / quit)."""
        bench, params = self.bench, self.params
        mode, explore, sigma = self.mode, self.explore, self.sigma
        store_warmup = self.state == STATE_WARMUP and sigma >= min_sigma
        reward = torch.zeros((), device=device0)
        t0 = time.perf_counter()
        for t in range(episode_length):
            self.poll_cmd()
            if self.quit or self.mode == LOOP_OPEN:
                #resets the buffer's write position if you open the loop
                self.replay.abort()
                self.replay_warmup.abort()
                return False
            params.refresh()
            obs = self.obs
            if mode == LOOP_INTEGRATOR:
                action = params.gain * obs.unsqueeze(0).unsqueeze(0)
            else:
                action = self.policy_copy(torch.cat([obs.unsqueeze(0).unsqueeze(0), self.past_obs, self.past_act], dim=1))
            if explore:
                action = action + bench.sample_noise(sigma)
            next_obs = bench.step(action, params.leak, params.use_offset)


            self.past_obs = torch.cat([self.past_obs[:, 1:], obs.unsqueeze(0).unsqueeze(0)], dim=1)
            self.past_act = torch.cat([self.past_act[:, 1:], action], dim=1)
            reward += torch.sum(obs.flatten() ** 2)
            a = action.squeeze()
            self.replay.append(obs, a, next_obs)
            if store_warmup:
                self.replay_warmup.append(obs, a, next_obs)
            self.obs = next_obs
            self.step_i = t
        self.hz = episode_length / (time.perf_counter() - t0)
        #makes the finished episode visible to the trainer
        self.replay.commit()
        if store_warmup:
            self.replay_warmup.commit()
        self.link.episodes.value += 1          # lets the trainer do one round on the new data
        self.reward = reward.item()
        self.episode += 1
        return True

    def boundary(self):
        """Supervisor work between two episodes (or while idle in OPEN)."""
        self.poll_cmd()
        self.check_pretrain() 
        self.handle_requests()
        if self.quit:
            return
        if (self.state == STATE_RUN and config['RL']['auto_policy_update']
                and self.link.policy_version.value > self.swapped_version):
            self.swap_policy()

        self.apply_settings()
        self.sync_cmd()
        self.publish_status()


    def run(self):
        self.log(f"ready in OPEN, run dir {self.run_dir}. Waiting for commands on {SHM_CMD}")
        if config['save_and_load']['load_dir']:
            self.load_checkpoint(config['save_and_load']['load_dir'])
        if config['save_and_load']['warmup_at_start']:
            self.start_warmup()
        self.publish_status()
        try:
            while not self.quit:
                if self.mode == LOOP_OPEN:
                    self.boundary()
                    time.sleep(0.02)
                    continue
                t0 = time.time()
                ok = self.run_episode()
                if ok:
                    if self.state == STATE_WARMUP:
                        self.log(f"warm-up {warmup_episodes - self.warmup_remaining + 1}/{warmup_episodes} "
                                 f"({time.time() - t0:.2f}s) reward {self.reward:.3f} sigma {self.sigma:.4f}")
                        self.end_warmup_episode()
                    else:
                        self.log(f"episode {self.episode} {LOOP_NAMES[self.mode]}"
                                 f"{' +noise' if self.explore else ''} ({time.time() - t0:.2f}s, {self.hz:.0f} Hz) "
                                 f"reward {self.reward:.3f} | replay {self.replay.n_valid}/{replay_size} "
                                 f"trainer {['idle', 'training', 'pre-training'][self.link.state.value]} "
                                 f"v{self.link.policy_version.value} dyn {1000 * self.link.dyn_loss.value:.4f} "
                                 f"pol {1000 * self.link.pol_loss.value:.4f}")
                self.boundary()

        except KeyboardInterrupt:
            self.log("Ctrl-C")
        self.shutdown()

    #shuts everything down and saves the data
    def shutdown(self):
        self.quit = True
        self.bench.flatten(self.params.use_offset, wait_frame=False)
        self.log("DM flat")
        if config['save_and_load']['save_at_exit'] and (self.link.policy_version.value > 0 or self.replay.n_valid > 0):
            self.save_checkpoint(models=True, buffers=True, tag="exit")
        self.set_trainer(TR_QUIT)
        self.trainer.join(timeout=10)
        if self.trainer.is_alive():
            self.trainer.terminate()
        self.state = STATE_IDLE
        self.mode = LOOP_OPEN
        self.publish_status()
        np.save(os.path.join(self.run_dir, "M2C_1st.npy"), self.bench.m2c)
        self.log("exit")



if __name__ == '__main__':
    Controller().run()
