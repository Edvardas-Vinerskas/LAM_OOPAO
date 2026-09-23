import torch
import numpy as np
from torch import nn

class EfficientExperienceReplay():

    def __init__(self, state_shape, action_shape, max_size=100000, warmup_memory = 0):
        self.max_size = max_size

        self.states = torch.empty(max_size, *state_shape).to("cuda:0")#.share_memory_()
        self.next_states = torch.empty(max_size, *state_shape).to("cuda:0")#.share_memory_()
        self.actions = torch.empty(max_size, *action_shape).to("cuda:0")#.share_memory_()

        self.len   = 0
        self.index_write = 0
        self.warmup_memory = warmup_memory

    # a function to add batches of new states, actions and next_states into the replay?
    def add(self, replay):
        cur_len = self.len
        new_len = self.len + len(replay)

        # but ReplaySample doesn't even take in rewards?
        if isinstance(replay, EfficientExperienceReplay):
            replay = ReplaySample(replay.states[:len(replay)], replay.actions[:len(replay)], replay.rewards[:len(replay)], replay.next_states[:len(replay)])

        self.states[cur_len:new_len] = replay.state()
        self.next_states[cur_len:new_len] = replay.next_state()
        self.actions[cur_len:new_len] = replay.action()

        self.len = new_len

    def __add__(self, replay):
        self.add(replay)
        return self

    # appends a single state, action and next state
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

    # randomly samples horizon length states, actions, next_states for training
    def sample_contiguous(self, horizon, max_ts, batch_size=32):
        #chooses batch_size number of random integers from 0 to max_ts - (horizon + 1) (i.e. somehwere within an episode)
        inds = torch.randint(0, max_ts - (horizon + 1), size=(batch_size, ))
        #chooses batch_size number of random integers from 0 to max no of episodes and multiplies by episode_length
        #effectively giving the start index of each episode
        #then adds the previous inds, so you have start_of_ep_ind + rand_int
        #so you effectively sample from a multitude of episodes
        inds += torch.randint(0, len(self) // max_ts, size=(batch_size, )) * max_ts
        #max_ts = 500 (episode length)

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


class EpisodeReplay():
    """GPU ring buffer of whole episodes, shared with the trainer process.

    The tensors go to the trainer through CUDA IPC (pass the object as a Process
    argument). The bookkeeping lives in multiprocessing Values so the trainer
    always sees the current fill level without re-sending the buffer:
      - frames are appended one by one by the control loop;
      - the episode becomes visible to the trainer only at commit();
      - abort() drops a partial episode (loop opened mid-episode).
    Episodes are fixed length, so windows sampled inside one never cross a mode
    change or an open-loop gap, and the trainer never samples the slot being
    written.
    """

    def __init__(self, state_shape, action_shape, n_episodes, episode_length, ctx, device="cuda:0"):
        self.n_episodes = n_episodes
        self.episode_length = episode_length
        self.max_size = n_episodes * episode_length
        self.states = torch.empty(self.max_size, *state_shape, device=device)
        self.next_states = torch.empty(self.max_size, *state_shape, device=device)
        self.actions = torch.empty(self.max_size, *action_shape, device=device)
        self._n_valid = ctx.Value('i', 0)    # committed episodes, <= n_episodes
        self._write_ep = ctx.Value('i', 0)   # episode the control loop writes into
        self._pos = 0                        # frames written in the slot (writer only)

    # ---------------------------------------------------------------- writer
    def append(self, obs, action, next_obs):
        if self._pos >= self.episode_length:
            raise RuntimeError("episode full: commit() or abort() before appending")
        i = self._write_ep.value * self.episode_length + self._pos
        self.states[i] = obs
        self.next_states[i] = next_obs
        self.actions[i] = action
        self._pos += 1

    def commit(self):
        """Make the episode being written visible to the trainer."""
        if self._pos != self.episode_length:
            raise RuntimeError(f"commit() with {self._pos}/{self.episode_length} frames")
        # order matters for the lock-free reader: new write slot first, then the count
        self._write_ep.value = (self._write_ep.value + 1) % self.n_episodes
        self._n_valid.value = min(self._n_valid.value + 1, self.n_episodes)
        self._pos = 0

    def abort(self):
        self._pos = 0

    @property
    def n_valid(self):
        return self._n_valid.value

    def __len__(self):
        return self.n_valid * self.episode_length

    # ---------------------------------------------------------------- reader
    def sample_contiguous(self, horizon, batch_size=32):
        """batch_size windows of horizon+1 consecutive frames from committed episodes."""
        n_valid = self._n_valid.value      # read before write_ep (see commit)
        write_ep = self._write_ep.value
        device = self.states.device
        eps = torch.randint(0, n_valid, (batch_size,), device=device)
        if write_ep < n_valid:             # buffer full: the write slot holds stale data
            eps = torch.where(eps == write_ep, (eps + 1) % n_valid, eps)
        within = torch.randint(0, self.episode_length - (horizon + 1), (batch_size,), device=device)
        starts = eps * self.episode_length + within
        indices = (starts.unsqueeze(1) + torch.arange(horizon + 1, device=device).unsqueeze(0)).flatten()
        return ReplaySample(self.states[indices], self.actions[indices], self.next_states[indices])

    # ---------------------------------------------------------------- checkpoints
    def export(self):
        """Committed episodes, oldest first, as CPU tensors (for torch.save)."""
        n = self.n_valid
        if n < self.n_episodes:
            order = torch.arange(0, n * self.episode_length)
        else:                              # ring: oldest episode is the write slot's successor
            first = (self._write_ep.value + 1) % self.n_episodes
            eps = (torch.arange(self.n_episodes) + first) % self.n_episodes
            order = (eps.unsqueeze(1) * self.episode_length + torch.arange(self.episode_length)).flatten()
        return dict(states=self.states[order].cpu(), actions=self.actions[order].cpu(),
                    next_states=self.next_states[order].cpu(), n_episodes=n,
                    episode_length=self.episode_length)

    def load(self, data):
        """Fill from export() output. Only while the trainer is idle."""
        if data["episode_length"] != self.episode_length:
            raise ValueError(f"episode_length {data['episode_length']} != {self.episode_length}")
        n = min(int(data["n_episodes"]), self.n_episodes)
        keep = n * self.episode_length
        self._n_valid.value = 0
        self.states[:keep] = data["states"][-keep:].to(self.states.device)
        self.next_states[:keep] = data["next_states"][-keep:].to(self.states.device)
        self.actions[:keep] = data["actions"][-keep:].to(self.states.device)
        self._pos = 0
        self._write_ep.value = n % self.n_episodes
        self._n_valid.value = n

    def clear(self):
        self._n_valid.value = 0
        self._write_ep.value = 0
        self._pos = 0


#class for state, action and next state handling
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

import contextlib
import os


@contextlib.contextmanager
def stdchannel_redirected(stdchannel, dest_filename):
    """
    A context manager to temporarily redirect stdout or stderr

    e.g.:


    with stdchannel_redirected(sys.stderr, os.devnull):
        if compiler.has_function('clock_gettime', libraries=['rt']):
            libraries.append('rt')
    """

    try:
        oldstdchannel = os.dup(stdchannel.fileno())
        dest_file = open(dest_filename, 'w')
        os.dup2(dest_file.fileno(), stdchannel.fileno())

        yield
    finally:
        if oldstdchannel is not None:
            os.dup2(oldstdchannel, stdchannel.fileno())
        if dest_file is not None:
            dest_file.close()


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8,
                 weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        # State initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                # share in memory
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()


class SharedAdamW(torch.optim.AdamW):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.01):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

