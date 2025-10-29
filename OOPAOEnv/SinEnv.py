import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
import torch
import os

class MultiSinEnv(gym.Env):
    def __init__(self, n=2, f1=1, a1=0.1, T=5, dt=0.05, d=1, seed=0, freq_multiplier=2/3, amp_multiplier=1.0):
        super(MultiSinEnv, self).__init__()
        
        self.n = n    # Number of independant signals
        self.f1, self.a1 = f1, a1

        self.T = T
        self.dt = dt
        self.seed = seed
        self.t = 0
        self.d = d
        self.epLen = 500

        self.freq_multiplier = freq_multiplier  # Controls how fast frequencies increase
        self.amp_multiplier = amp_multiplier  # Controls amplitude scaling

        # Generate increasing frequencies and amplitudes
        self.frequencies = [self.f1 * (self.freq_multiplier ** i) for i in range(n)]
        self.amplitudes = [self.a1 * (self.amp_multiplier ** i) for i in range(n)]

        # Set random seed
        self.set_seed(self.seed)

        # Env Dynamics
        self.applyActionScale = 1.0


        # Env Spaces
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(n,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(T, n), dtype=np.float32)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.obsHistory = torch.zeros((T, n)).to(self.device)
        self.action_buffer = [torch.zeros((self.n), device=self.device, dtype=torch.float32) for _ in range(self.d)]
        self.reset()

    
    def set_seed(self, seed):
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)



    def reset(self, seed=None, options=None):
        if seed is not None:
            self.set_seed(seed)

        self.t = self.T
        self.obsHistory.zero_()
        self.action_buffer = [torch.zeros((self.n), device=self.device, dtype=torch.float32) for _ in range(self.d)]
        self.randPhase = 2 * torch.pi * torch.rand(self.n)

        # Generate initial history
        t_vals = torch.arange(self.T, device=self.device) * self.dt
        for i in range(self.n):
            self.obsHistory[:, i] = self.amplitudes[i] * torch.sin(2 * torch.pi * self.frequencies[i] * t_vals + self.randPhase[i]).to(self.device)

        info = {}

        return self.obsHistory.cpu().numpy(), info


    def step(self, action):

        # Convert action to tensor and apply delay
        action_tensor = torch.tensor(action, device=self.device, dtype=torch.float32)
        self.action_buffer.append(action_tensor)
        delayed_action = self.action_buffer.pop(0)


        self.obsHistory = torch.roll(self.obsHistory, shifts=1, dims=0)
        
        # Generate new observation values for each sine wave
        for i in range(self.n):
            self.obsHistory[0, i] = self.amplitudes[i] * torch.sin(2 * torch.pi * self.frequencies[i] * self.t * self.dt + self.randPhase[i]).to(self.device)
        
        
        diff = self.applyActionScale * delayed_action - self.obsHistory[0]

        reward = -np.linalg.norm(diff.cpu()) ** 2 / self.n # Normalize by number of signals
        reward = np.clip(reward, -1, 1)

        terminated = False
        truncated = False

        self.t += 1

        info = {}

        done = self.t >= self.epLen
        truncated = done

        if done:
            self.t = self.T

        return self.obsHistory.cpu().numpy(), reward, bool(terminated), bool(truncated), info


class MultiAtmEnv(gym.Env):
    def __init__(self, env, m2opd, n=2, T=5, d=1, seed=0):
        super(MultiAtmEnv, self).__init__()
        
        self.env = env
        self.m2opd = m2opd
        self.opd2m = np.linalg.pinv(m2opd)
        self.xpupil, self.ypupil = np.where(env.tel.pupil == 1)
        self.n = n    # Number of independant signals

        self.T = T
        self.seed = seed
        self.t = 0
        self.d = d
        self.epLen = 500

        # Set random seed
        self.set_seed(self.seed)

        # Env Dynamics
        self.applyActionScale = 1.0
        self.scaleObs = 1e6


        # Env Spaces
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(n,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(T, n), dtype=np.float32)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.obsHistory = torch.zeros((T, n)).to(self.device)
        self.action_buffer = [torch.zeros((self.n), device=self.device, dtype=torch.float32) for _ in range(self.d)]
        self.reset()

    
    def set_seed(self, seed):
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)



    def reset(self, seed=None, options=None):
        if seed is not None:
            self.set_seed(seed)

        self.t = 0
        self.obsHistory.zero_()
        self.action_buffer = [torch.zeros((self.n), device=self.device, dtype=torch.float32) for _ in range(self.d)]

        self.env.atm.generateNewPhaseScreen(np.random.randint(0, 1e8))
        modes = self.opd2m @ self.env.tel.OPD.copy()[self.xpupil, self.ypupil]

        self.obsHistory = torch.roll(self.obsHistory, shifts=1, dims=0)
        for i in range(self.n):
            self.obsHistory[0, i] = modes[i] * self.scaleObs

        info = {}

        return self.obsHistory.cpu().numpy(), info


    def step(self, action):
        # Convert action to tensor and apply delay
        action_tensor = torch.tensor(action, device=self.device, dtype=torch.float32)
        self.action_buffer.append(action_tensor)
        delayed_action = self.action_buffer.pop(0)


        self.obsHistory = torch.roll(self.obsHistory, shifts=1, dims=0)

        self.env.atm.update()

        modes = self.opd2m @ self.env.tel.OPD.copy()[self.xpupil, self.ypupil]
        # Generate new observation values for each sine wave
        for i in range(self.n):
            self.obsHistory[0, i] = modes[i] * self.scaleObs
        
        diff = self.applyActionScale * delayed_action - self.obsHistory[0]

        reward = -np.linalg.norm(diff.cpu()) ** 2 / self.n # Normalize by number of signals
        reward = np.clip(reward, -1, 1)

        terminated = False
        truncated = False

        self.t += 1

        info = {}

        done = self.t >= self.epLen
        truncated = done

        if done:
            self.t = self.T

        return self.obsHistory.cpu().numpy(), reward, bool(terminated), bool(truncated), info

