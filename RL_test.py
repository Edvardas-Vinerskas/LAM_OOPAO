# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter
from cleanrl_utils.buffers import ReplayBuffer
from OOPAO_environment import OOPAO_environment


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "OOPAO_RL_tt"
    """the environment id of the task"""
    total_timesteps: int = 10000
    """total timesteps of the experiments"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = int(5e4)
    """the replay memory buffer size"""
    gamma: float = 0.90
    """the discount factor gamma"""
    tau: float = 0.00385
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 1e3
    """timestep to start learning"""
    policy_lr: float = 0.00001
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 5
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 9
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.02
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    hidden_dim: int = 256


# Change for custom environment
def make_env():
    def thunk():
        env = OOPAO_environment()
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk


# Q network takes in the state and the action (current pwfs measurements and previous dm measurements + the zernike modes output by the policy network)
class SoftQNetwork(nn.Module):
    def __init__(self, env, hidden_dim=256):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = env
        self.n = 664 + 357 # pyramid signal + dm actuators
        self.T = 5  # self.env.get_attr("T")[0]
        #INPUT SHAPE
        self.hidden_dim = hidden_dim

        self.input_dim = self.n * self.T + 2 # it is adding 2 because action is also an input parameter
                                             # action here is the 2 tt modes

        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, 1)
        )

    def forward(self, x, a):
        x = torch.cat([x.view(x.shape[0], -1), a], 1)
        x = x.view(x.shape[0], -1)
        x = self.net(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -10

#takes in the state (current pwfs commands and previous dm commands)
class Actor(nn.Module):
    def __init__(self, env, hidden_dim=256):
        super().__init__()

        self.env = env
        self.n = 664 + 357 #self.env.get_attr("n")[0]
        self.T = 5#self.env.get_attr("T")[0]
        self.hidden_dim = hidden_dim

        self.input_dim = self.n * self.T

        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )


        self.fc_mean = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, np.prod(env.single_action_space.shape))
        )


        self.fc_logstd = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, np.prod(env.single_action_space.shape))
        )

        # Learnable residual scaling factor
        self.residual_scale = nn.Parameter(1e-4 * torch.ones(1))  # Initialized to 1.0


        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.single_action_space.high - env.single_action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.single_action_space.high + env.single_action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):

        batch_size, T, n = x.shape
        assert n == self.n, f"Expected input last dim {self.n}, got {n}"
        assert T == self.T, f"Expected input time dim {self.T}, got {T}"

        # Flatten (T, n) into (T * n)
        x = x.view(batch_size, -1)

        x = self.net(x)
        x = x.view(batch_size, -1)

        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample() # for reparameterization trick (mean + std * N(0,1))


        action = x_t * self.action_scale + self.action_bias

        log_prob = normal.log_prob(x_t)

        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias

        # print(f"base_action: {base_action}, {base_action.shape}")
        # print(f"residual_action: {self.residual_scale * residual_action}, {residual_action.shape}")

        return action, log_prob, mean


if __name__ == "__main__":

    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
    poetry run pip install "stable_baselines3==2.0.0a1"
    """
        )

    num_runs = 1
    seeds = [167640, 813868, 168772, 214449,
             9498, 398085, 753264, 331695,
             950521, 715051]

    envs = gym.vector.SyncVectorEnv([make_env()])

    for i in range(num_runs):

        args = tyro.cli(Args, args=[])
        run_name = f"IM_delay_{args.env_id}__{args.exp_name}__{args.seed}__run_{i}__{int(time.time())}"
        if args.track:
            import wandb

            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=True,
                config=vars(args),
                name=run_name,
                monitor_gym=True,
                save_code=True,
            )
        writer = SummaryWriter(f"./runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )

        # Random seed for multiple runs
        args.seed = seeds[i]  # logged seed values

        # TRY NOT TO MODIFY: seeding
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = args.torch_deterministic

        device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

        # env setup
        assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

        max_action = float(envs.single_action_space.high[0])

        actor = Actor(envs).to(device)
        qf1 = SoftQNetwork(envs).to(device)
        qf2 = SoftQNetwork(envs).to(device)
        qf1_target = SoftQNetwork(envs).to(device)
        qf2_target = SoftQNetwork(envs).to(device)
        qf1_target.load_state_dict(qf1.state_dict())
        qf2_target.load_state_dict(qf2.state_dict())
        q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
        actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

        best_reward = -np.inf
        # Automatic entropy tuning
        if args.autotune:
            target_entropy = - torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
            log_alpha = torch.zeros(1, requires_grad=True, device=device)
            alpha = log_alpha.exp().item()
            a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
        else:
            alpha = args.alpha

        envs.single_observation_space.dtype = np.float32
        rb = ReplayBuffer(
            args.buffer_size,
            envs.single_observation_space,
            envs.single_action_space,
            device,
            handle_timeout_termination=False,
        )
        start_time = time.time()

        # TRY NOT TO MODIFY: start the game
        obs, _ = envs.reset(seed=args.seed)
        for global_step in range(args.total_timesteps):
            # ALGO LOGIC: put action logic here
            if global_step < args.learning_starts:
                if global_step % 100 == 0:
                    print(f"WARMUP: {global_step}/{int(args.learning_starts)}")
                actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
                # Warmup with IM actions
                # actions = np.array([-1 * (obs[i]) for i in range(envs.num_envs)])
            else:
                actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
                actions = actions.detach().cpu().numpy()

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards, terminations, truncations, infos = envs.step(actions)

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            if "final_info" in infos:
                for info in infos["final_info"]:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    with open("./train_returns.txt", "a") as f:  # 'a' mode appends to the file
                        f.write(f"global_step={global_step}, episodic_return={info['episode']['r']} \n")

                    if info['episode']['r'] > best_reward and global_step > args.learning_starts:
                        best_reward = info['episode']['r']
                        torch.save({
                            'epoch': global_step + 1,
                            'model_state_dict': actor.state_dict(),
                            # 'ema_model_state_dict': ema_reconstructor.module.state_dict(),
                            'optimizer_state_dict': actor_optimizer.state_dict(),
                            'reward': best_reward,
                        }, os.path.dirname(__file__) + f"/./models/best_model_delay_run_{i}.pth")
                        with open("./train_returns.txt", "a") as f:  # 'a' mode appends to the file
                            f.write(f"Saving Model \n")

                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                    break

            # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
            real_next_obs = next_obs.copy()
            for idx, trunc in enumerate(truncations):
                if trunc:
                    real_next_obs[idx] = infos["final_observation"][idx]
            rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs

            # ALGO LOGIC: training.
            if global_step > args.learning_starts:
                data = rb.sample(args.batch_size)
                with torch.no_grad():
                    next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                    qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                    qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                    min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                    next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (
                        min_qf_next_target).view(-1)

                qf1_a_values = qf1(data.observations, data.actions).view(-1)
                qf2_a_values = qf2(data.observations, data.actions).view(-1)
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                qf_loss = qf1_loss + qf2_loss

                # optimize the model
                q_optimizer.zero_grad()
                qf_loss.backward()
                nn.utils.clip_grad_norm_(qf1.parameters(), args.max_grad_norm)
                q_optimizer.step()

                if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                    for _ in range(
                            args.policy_frequency
                    ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                        pi, log_pi, _ = actor.get_action(data.observations)
                        qf1_pi = qf1(data.observations, pi)
                        qf2_pi = qf2(data.observations, pi)
                        min_qf_pi = torch.min(qf1_pi, qf2_pi)
                        actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                        actor_optimizer.zero_grad()
                        actor_loss.backward()
                        nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
                        actor_optimizer.step()

                        if args.autotune:
                            with torch.no_grad():
                                _, log_pi, _ = actor.get_action(data.observations)
                            alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                            a_optimizer.zero_grad()
                            alpha_loss.backward()
                            a_optimizer.step()
                            alpha = log_alpha.exp().item()

                # update the target networks
                if global_step % args.target_network_frequency == 0:
                    for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                    for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

                if global_step % 1000 == 0:
                    writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                    writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                    writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                    writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                    writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                    writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                    writer.add_scalar("losses/alpha", alpha, global_step)
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    print(f"Total steps: {global_step}")
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                    if args.autotune:
                        writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

        # envs.close()
        writer.close()
# %%