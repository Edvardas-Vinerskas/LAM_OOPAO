# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
#%%
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import random
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
import optuna
from OOPAOEnv.IM_delayEnv import OOPAO

#%%
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
    env_id: str = "CL_OOPAO-v0"
    """the environment id of the task"""
    total_timesteps: int = 10000#1000000
    """total timesteps of the experiments"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = int(5e4)
    """the replay memory buffer size"""
    gamma: float = 0.
    """the discount factor gamma"""
    tau: float = 0.00385
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 1e3
    """timestep to start learning"""
    policy_lr: float = 0.00001
    """the learning rate of the policy network optimizer"""
    q_lr: float = 0.001
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 3  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.01
    """Entropy regularization coefficient."""
    autotune: bool = False
    """automatic tuning of the entropy coefficient"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    hidden_dim: int = 256


# def make_env(env_id, seed, idx, capture_video, run_name):
#     def thunk():
#         if capture_video and idx == 0:
#             env = gym.make(env_id, render_mode="rgb_array")
#             env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
#         else:
#             env = gym.make(env_id)
#         env = gym.wrappers.RecordEpisodeStatistics(env)
#         env.action_space.seed(seed)
#         return env

#     return thunk


# Change for custom environment
def make_env():
    def thunk():
        env = OOPAO()
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env, hidden_dim=256):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = env
        self.n = 664#self.env.get_attr("n")[0]
        self.T = 5#self.env.get_attr("T")[0]

        self.hidden_dim = hidden_dim

        self.input_dim = self.n * self.T + 2

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


class Actor(nn.Module):
    def __init__(self, env, hidden_dim=256):
        super().__init__()

        self.env = env
        self.n = 664 #self.env.get_attr("n")[0]
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


        # self.fc_mean = nn.Linear(64, np.prod(env.single_action_space.shape))
        # self.fc_logstd = nn.Linear(64, np.prod(env.single_action_space.shape))
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
#%%
def objective(trial):
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
    poetry run pip install "stable_baselines3==2.0.0a1"
    """
        )

    args = tyro.cli(Args, args=[])
    args.gamma = trial.suggest_float("gamma", 0.7, 0.999)
    args.tau = trial.suggest_float("tau", 0.001, 0.1, log=True)
    # args.batch_size = trial.suggest_int("batch_size", 32, 256)
    # args.learning_starts = trial.suggest_int("learning_starts", 1000, 10000)
    args.policy_lr = trial.suggest_float("policy_lr", 1e-5, 1e-3, log=True)
    args.q_lr = trial.suggest_float("q_lr", 1e-5, 1e-3, log=True)
    args.policy_frequency = trial.suggest_int("policy_frequency", 1, 10)
    args.target_network_frequency = trial.suggest_int("target_network_frequency", 1, 10)
    args.alpha = trial.suggest_float("alpha", 0.01, 0.1, log=True)
    # args.hidden_dim = trial.suggest_int("hidden_dim", 64, 256)
    args.exp_name = "optuna_learnIM_0_delay"
    os.makedirs("optuna_runs", exist_ok=True)
    with open("optuna_runs/args.txt", "a") as f:
        f.write(f"Trial {trial.number}\n")
        f.write(f"Gamma: {args.gamma}\n")
        f.write(f"Tau: {args.tau}\n")
        f.write(f"Policy LR: {args.policy_lr}\n")
        f.write(f"Q LR: {args.q_lr}\n")
        f.write(f"Policy Frequency: {args.policy_frequency}\n")
        f.write(f"Target Network Frequency: {args.target_network_frequency}\n")
        f.write(f"Alpha: {args.alpha}\n\n\n")



    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
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
    writer = SummaryWriter(f"optuna_runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv([make_env()])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])

    actor = Actor(envs, hidden_dim=args.hidden_dim).to(device)
    qf1 = SoftQNetwork(envs, hidden_dim=args.hidden_dim).to(device)
    qf2 = SoftQNetwork(envs, hidden_dim=args.hidden_dim).to(device)
    qf1_target = SoftQNetwork(envs, hidden_dim=args.hidden_dim).to(device)
    qf2_target = SoftQNetwork(envs, hidden_dim=args.hidden_dim).to(device)
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
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

                reward_val = info["episode"]["r"]
                trial.report(reward_val, global_step)

                if reward_val > best_reward:
                    best_reward = reward_val

                    with open("optuna_runs/args.txt", "a") as f:
                        f.write(f"Best Reward: {best_reward}\n")

                if trial.should_prune() and global_step > args.learning_starts:
                    raise optuna.exceptions.TrialPruned()
                
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
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

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
    

    return reward_val

if __name__ == "__main__":
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=800, timeout=12*3600)

    pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
# %%
