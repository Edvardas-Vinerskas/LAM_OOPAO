import torch
from torch import nn
import torch.nn.functional as F_interp
import time
# from pytorch_convolutional_rnn.convolutional_rnn.module import Conv2dGRU
# import matplotlib
# matplotlib.use('tkagg')
# import matplotlib.pyplot as plt
from po4ao_config import config

n_filt = config['NN_models']['filters_per_layer'] #32

scaling_factor_up = 1e2 #1e2

class ConvDynamics(nn.Module):
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
            # nn.Tanh()
        )

        self._hidden = None

    def forward(self, states, actions, history=None):
        if states.ndim == 3:
            states = states.view(1, *states.shape)
        if actions.ndim == 3:
            actions = actions.view(1, *actions.shape)

        if history is not None:
            if history.ndim == 3:
                history = history.view(1, 1, *actions.shape)

            feats = torch.cat([history, states, actions], dim=1)
        else:
            feats = torch.cat([states, actions], dim=1)

        out = self.net(feats)

        ret = torch.zeros_like(out)
        ret[:, :, self.mask] = out[:, :, self.mask]

        return out


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
            # nn.Tanh()
        )

        self._hidden = None

    def forward(self, feats):
        # feats shape(batch_size, 2 * n_history, 10, 10)
        # out shape(batch_size, 1, 10, 10)
        out = self.net(feats * scaling_factor_up) / scaling_factor_up

        ret = torch.zeros_like(out, device=feats.device)
        ret[:, :, self.mask] = out[:, :, self.mask]

        return ret


class ConvDynamicsFast_upsampled(nn.Module):
    def __init__(self, mask, n_history):
        super().__init__()

        self.mask = mask
        self.n_history = n_history

        self.net = nn.Sequential(
            # After upsample: (batch, 2*n_history, 22, 22)
            nn.Conv2d(n_history * 2, n_filt, 3, padding=1),
            nn.LeakyReLU(),

            nn.Conv2d(n_filt, n_filt, 3, padding=1),
            nn.LeakyReLU(),

            # Downsample back to 11x11
            nn.Conv2d(n_filt, n_filt, 3, stride=2, padding=1),
            nn.LeakyReLU(),

            nn.Conv2d(n_filt, 1, 3, padding=1)
        )

        self._hidden = None

    def forward(self, feats):
        # feats shape: (batch_size, 2 * n_history, 11, 11)
        # Upsample to 22x22
        x = F_interp.interpolate(feats, scale_factor=2, mode='bilinear', align_corners=False)

        # out shape: (batch_size, 1, 11, 11)
        out = self.net(x * scaling_factor_up) / scaling_factor_up

        ret = torch.zeros_like(out, device=feats.device)
        ret[:, :, self.mask] = out[:, :, self.mask]

        return ret



class ConvPolicyFast(nn.Module):
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
            nn.Conv2d(n_filt, 1, 3, padding=1),
            # nn.Tanh()
        )

    def forward(self, state, history=None, sigma=0.):
        if state.ndim == 3:
            state = state.view(1, *state.shape)
        if state.ndim == 2:
            state = state.view(1, 1, *state.shape)
        if history is not None:
            feats = torch.cat([state, history], dim=1)
        else:
            feats = state

        out = self.net(feats)
        out = out.clamp(-0.08, 0.08)

        out[:, :, self.xvalid, self.yvalid] = torch.matmul(self.F,
                                                           out[:, :, self.xvalid, self.yvalid].squeeze(1).unsqueeze(
                                                               2)).squeeze(-1).unsqueeze(1)

        return out


#clamping changed to -1, 1
#F is the KL reprojection matrix
class ConvPolicyFastFast(nn.Module):
    def __init__(self, xvalid, yvalid, F, n_history):
        super().__init__()

        self.xvalid = xvalid
        self.yvalid = yvalid

        self.n_history = n_history

        self.register_buffer('F', F.unsqueeze(0)) #registers F as a non-trainable parameter

        self.net = nn.Sequential(
            nn.Conv2d(n_history * 2 -1, n_filt, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(n_filt, n_filt, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(n_filt, 1, 3, padding=1),
            #nn.Tanh()
        )

    def forward(self, feats):
        # feats shape(batch_size, 2 * n_history - 1, 10, 10) #because the current observation is part of the input but not the current action
        # out shape(batch_size, 1, 10, 10)
        out = self.net(feats * scaling_factor_up) / scaling_factor_up
        out = out.clamp(-0.03, 0.03)

        #mask = torch.ones_like(out)
        #mask[:, :, 3:6, 4:7] = 0

        # should you also set the outside values to 0?
        out[:, :, self.xvalid, self.yvalid] = torch.matmul(self.F, out[:, :, self.xvalid, self.yvalid].squeeze(1).unsqueeze(2)).squeeze(-1).unsqueeze(1)
        #out = out * mask

        return out


class ConvPolicyFastFast_upsampled(nn.Module):
    def __init__(self, xvalid, yvalid, F, n_history):
        super().__init__()

        self.xvalid = xvalid
        self.yvalid = yvalid

        self.n_history = n_history

        self.register_buffer('F', F.unsqueeze(0))

        self.net = nn.Sequential(
            # After upsample: (batch, 2*n_history - 1, 22, 22)
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
        # feats shape: (batch_size, 2 * n_history - 1, 11, 11)
        # Upsample to 22x22
        x = F_interp.interpolate(feats, scale_factor=2, mode='bilinear', align_corners=False)

        # out shape: (batch_size, 1, 11, 11)
        out = self.net(x * scaling_factor_up) / scaling_factor_up
        out = out.clamp(-0.08, 0.08)

        out[:, :, self.xvalid, self.yvalid] = torch.matmul(self.F, out[:, :, self.xvalid, self.yvalid].squeeze(1).unsqueeze(2)).squeeze(-1).unsqueeze(1)

        return out




#an ensemble of dynamics models
class EnsembleDynamics(nn.Module):

    def __init__(self, mask, n_history, n_models=5):
        super().__init__()
        self.n_models = n_models
        self.models = nn.ModuleList([])

        for _ in range(n_models):
            self.models.append(ConvDynamics(mask, n_history))

    def forward(self, states, actions, history = None):
        next_states = []

        for model in self.models:
            next_states.append(model.forward(states, actions, history))

        return torch.cat(next_states, dim=1)


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
    



class EnsembleDynamicsFast_upsampled(nn.Module):

    def __init__(self, mask, n_history, n_models=5):
        super().__init__()
        self.n_models = n_models
        self.models = nn.ModuleList([])

        for _ in range(n_models):
            self.models.append(ConvDynamicsFast_upsampled(mask, n_history))

    def forward(self, feats):
        next_states = []

        for model in self.models:
            next_states.append(model.forward(feats))

        return torch.cat(next_states, dim=1)
