from webbrowser import get
import torch
from torch.optim import Adam
import torch.nn.functional as F
import numpy as np

from rl_algorithms.common.env_utils import get_env_params
from rl_algorithms.PyTorch.utils.buffer import Buffer
from rl_algorithms.PyTorch.utils.networks import SoftActor, Critic


class SAC:
    def __init__(
        self,
        env,
        lr_actor=3e-4,
        lr_critic=3e-4,
        lr_alpha=3e-4,
        gamma=0.99,
        buffer_capacity=100000,
        tau=0.005,
        alpha=0.2,
        hidden_size=(512, 512),
        batch_size=64,
        update_frequency=2,
        target_entropy_tuning=True,
    ):
        env_params = get_env_params(env)
        (
            self.num_actions,
            self.num_states,
            self.min_action,
            self.max_action,
            self.action_space,
        ) = map(
            env_params.get,
            ("num_actions", "num_states", "lower_bound", "upper_bound", "action_space"),
        )
        self.gamma = gamma
        self.tau = tau
        self.memory = Buffer(
            self.num_actions, self.num_states, buffer_capacity, batch_size
        )
        self.batch_size = batch_size
        self.trainstep = 0
        self.update_frequency = update_frequency
        self.target_entropy_tuning = target_entropy_tuning

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = SoftActor(
            self.num_actions, self.num_states, self.action_space, hidden_size
        ).to(device=self.device)
        self.critic = Critic(self.num_actions + self.num_states, hidden_size).to(
            device=self.device
        )
        self.critic2 = Critic(
            self.num_actions + self.num_states, hidden_size, name="critic2"
        ).to(device=self.device)
        self.target_critic = Critic(
            self.num_actions + self.num_states, hidden_size, name="target_critic"
        ).to(device=self.device)
        self.target_critic2 = Critic(
            self.num_actions + self.num_states, hidden_size, name="target_critic2"
        ).to(device=self.device)

        self.actor_optimizer = Adam(self.actor.parameters(), lr_actor)
        self.critic_optimizer = Adam(self.critic.parameters(), lr_critic)
        self.critic2_optimizer = Adam(self.critic2.parameters(), lr_critic)

        self.update_network_parameters(tau=1.0)

        self.alpha = alpha

        if self.target_entropy_tuning is True:
            self.target_entropy = -torch.Tensor([self.num_actions]).to(self.device)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = Adam([self.log_alpha], lr_alpha)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        for param, target_param in zip(
            self.critic.parameters(), self.target_critic.parameters()
        ):
            target_param.data.copy_(param.data * tau + target_param.data * (1.0 - tau))

        for param, target_param in zip(
            self.critic2.parameters(), self.target_critic2.parameters()
        ):
            target_param.data.copy_(param.data * tau + target_param.data * (1.0 - tau))

    def save_models(self):
        torch.save(self.actor.state_dict(), self.actor.checkpoint_file)
        torch.save(self.critic.state_dict(), self.critic.checkpoint_file)
        torch.save(self.critic2.state_dict(), self.critic2.checkpoint_file)
        torch.save(self.target_critic.state_dict(), self.target_critic.checkpoint_file)
        torch.save(
            self.target_critic2.state_dict(), self.target_critic2.checkpoint_file
        )

    def load_models(self):
        self.actor.load_state_dict(torch.load(self.actor.checkpoint_file))
        self.critic.load_state_dict(torch.load(self.critic.checkpoint_file))
        self.critic2.load_state_dict(torch.load(self.critic2.checkpoint_file))
        self.target_critic.load_state_dict(
            torch.load(self.target_critic.checkpoint_file)
        )
        self.target_critic2.load_state_dict(
            torch.load(self.target_critic2.checkpoint_file)
        )

    def choose_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.actor.sample(state)
        else:
            _, _, action = self.actor.sample(state)

        return action.detach().cpu().numpy()[0]

    def update(
        self, state_batch, action_batch, reward_batch, next_state_batch, done_batch
    ):
        target_actions, target_log_pi, _ = self.actor.sample(next_state_batch)
        target_next_state_values = self.target_critic.forward(
            next_state_batch, target_actions
        )
        target_next_state_values2 = self.target_critic2.forward(
            next_state_batch, target_actions
        )
        next_state_target_value = (
            torch.min(target_next_state_values, target_next_state_values2)
            - self.alpha * target_log_pi
        )
        y = reward_batch + self.gamma * next_state_target_value.detach() * (
            1 - done_batch
        )

        critic_value = self.critic.forward(state_batch, action_batch)
        critic_value2 = self.critic2.forward(state_batch, action_batch)
        critic_loss = F.mse_loss(y, critic_value)
        critic_loss2 = F.mse_loss(y, critic_value2)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic_loss2.backward()
        self.critic2_optimizer.step()

        pi, log_pi, _ = self.actor.sample(state_batch)  # rename variables?
        qf1_pi = self.critic(state_batch, pi)
        qf2_pi = self.critic2(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        actor_loss = ((self.alpha * log_pi) - min_qf_pi.detach()).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.target_entropy_tuning:
            alpha_loss = -(
                self.log_alpha * (log_pi + self.target_entropy).detach()
            ).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()

        self.trainstep += 1

        if self.trainstep % self.update_frequency == 0:
            self.update_network_parameters()

    def learn(self):
        if self.memory.buffer_counter < self.batch_size:
            return

        (
            state_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            done_batch,
        ) = self.memory.sample()

        # TODO: check PyTorch implementations on GPU
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        done_batch = torch.FloatTensor(done_batch).to(self.device)

        self.update(
            state_batch, action_batch, reward_batch, next_state_batch, done_batch
        )
