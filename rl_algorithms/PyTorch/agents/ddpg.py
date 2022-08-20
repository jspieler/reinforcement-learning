import torch
from torch.optim import Adam
import torch.nn.functional as F
import numpy as np

from rl_algorithms.common.env_utils import get_env_params
from rl_algorithms.PyTorch.utils.buffer import Buffer
from rl_algorithms.PyTorch.utils.noise import OUActionNoise
from rl_algorithms.PyTorch.utils.networks import Actor, Critic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DDPG:
    def __init__(
        self,
        env,
        lr_actor=0.001,
        lr_critic=0.002,
        gamma=0.99,
        buffer_capacity=100000,
        tau=0.005,
        hidden_size=(512, 512),
        batch_size=64,
        noise_stddev=0.1,
    ):
        env_params = get_env_params(env)
        self.num_actions, self.num_states, self.min_action, self.max_action = map(
            env_params.get, ("num_actions", "num_states", "lower_bound", "upper_bound")
        )
        self.gamma = gamma
        self.tau = tau
        self.memory = Buffer(
            self.num_actions, self.num_states, buffer_capacity, batch_size
        )
        self.batch_size = batch_size
        self.noise = OUActionNoise(
            mean=np.zeros(1), std_deviation=float(noise_stddev) * np.ones(1)
        )

        self.actor = Actor(self.num_actions, self.num_states, hidden_size).to(device)
        self.critic = Critic(self.num_actions + self.num_states, hidden_size).to(device)
        self.target_actor = Actor(
            self.num_actions, self.num_states, hidden_size, name="target_actor"
        ).to(device)
        self.target_critic = Critic(
            self.num_actions + self.num_states, hidden_size, name="target_critic"
        ).to(device)

        self.actor_optimizer = Adam(self.actor.parameters(), lr_actor)
        self.critic_optimizer = Adam(self.critic.parameters(), lr_critic)

        self.update_network_parameters(tau=1)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        for param, target_param in zip(
            self.actor.parameters(), self.target_actor.parameters()
        ):
            target_param.data.copy_(param.data * tau + target_param.data * (1.0 - tau))

        for param, target_param in zip(
            self.critic.parameters(), self.target_critic.parameters()
        ):
            target_param.data.copy_(param.data * tau + target_param.data * (1.0 - tau))

    def save_models(self):
        torch.save(self.actor.state_dict(), self.actor.checkpoint_file)
        torch.save(self.critic.state_dict(), self.critic.checkpoint_file)
        torch.save(self.target_actor.state_dict(), self.target_actor.checkpoint_file)
        torch.save(self.target_critic.state_dict(), self.target_critic.checkpoint_file)

    def load_models(self):
        self.actor.load_state_dict(torch.load(self.actor.checkpoint_file))
        self.critic.load_state_dict(torch.load(self.critic.checkpoint_file))
        self.target_actor.load_state_dict(torch.load(self.target_actor.checkpoint_file))
        self.target_critic.load_state_dict(
            torch.load(self.target_critic.checkpoint_file)
        )

    def choose_action(self, state, evaluate=False):
        mu = self.actor(state.to(device), self.max_action)
        mu = mu.data

        if not evaluate:
            noise = torch.Tensor(self.noise()).to(device)
            mu += noise

        mu = mu.clamp(self.min_action, self.max_action)

        return mu.cpu().detach().numpy()

    def update(
        self, state_batch, action_batch, reward_batch, next_state_batch, done_batch
    ):
        # Training and updating Actor & Critic networks.
        target_actions = self.target_actor.forward(next_state_batch, self.max_action)
        y = reward_batch + self.gamma * self.target_critic.forward(
            next_state_batch, target_actions
        ) * (1 - done_batch)
        critic_value = self.critic.forward(state_batch, action_batch)
        critic_loss = F.mse_loss(y, critic_value)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actions = self.actor.forward(state_batch, self.max_action)
        critic_value = self.critic.forward(state_batch, actions)
        actor_loss = -torch.mean(critic_value)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def learn(self):
        (
            state_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            done_batch,
        ) = self.memory.sample()

        self.update(
            state_batch, action_batch, reward_batch, next_state_batch, done_batch
        )

        self.update_network_parameters()
