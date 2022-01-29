import torch
from torch.optim import Adam
import torch.nn.functional as F
import numpy as np

from utils.buffer import Buffer
from utils.noise import GaussianActionNoise
from utils.networks import Actor, Critic

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TD3:
    def __init__(self, num_actions, num_states, min_action, max_action, lr_actor=1e-3,
                 lr_critic=2e-3, gamma=0.99, buffer_capacity=100000, tau=0.005,
                 hidden_size=(512,512), batch_size=64, noise_stddev=0.1,
                 warmup_steps=200, update_frequency=2):
        self.gamma = gamma
        self.tau = tau
        self.memory = Buffer(num_actions, num_states, buffer_capacity, batch_size)
        self.batch_size = batch_size
        self.num_actions = num_actions
        self.num_states = num_states
        self.min_action = min_action
        self.max_action = max_action
        self.trainstep = 0
        self.warmup_steps = warmup_steps
        self.actor_update_steps = update_frequency
        self.noise = GaussianActionNoise(mean=np.zeros(num_actions), std_deviation=float(noise_stddev) * np.ones(num_actions))

        self.actor = Actor(num_actions, num_states, hidden_size)
        self.critic = Critic(num_actions+num_states, hidden_size)
        self.critic2 = Critic(num_actions+num_states, hidden_size, name='critic2')
        self.target_actor = Actor(num_actions, num_states, hidden_size, name='target_actor')
        self.target_critic = Critic(num_actions+num_states, hidden_size, name='target_critic')
        self.target_critic2 = Critic(num_actions+num_states, hidden_size, name='target_critic2')

        self.actor_optimizer = Adam(self.actor.parameters(), lr_actor)
        self.critic_optimizer = Adam(self.critic.parameters(), lr_critic)
        self.critic2_optimizer = Adam(self.critic.parameters(), lr_critic)

        self.update_network_parameters(tau=1.0)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
            target_param.data.copy_(param.data * tau + target_param.data * (1.0 - tau))

        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(param.data * tau + target_param.data * (1.0 - tau))

        for param, target_param in zip(self.critic2.parameters(), self.target_critic2.parameters()):
            target_param.data.copy_(param.data * tau + target_param.data * (1.0 - tau))

    def save_models(self):
        torch.save(self.actor.state_dict(), self.actor.checkpoint_file)
        torch.save(self.critic.state_dict(), self.critic.checkpoint_file)
        torch.save(self.critic2.state_dict(), self.critic2.checkpoint_file)
        torch.save(self.target_actor.state_dict(), self.target_actor.checkpoint_file)
        torch.save(self.target_critic.state_dict(), self.target_critic.checkpoint_file)
        torch.save(self.target_critic2.state_dict(), self.target_critic2.checkpoint_file)

    def load_models(self):
        self.actor.load_state_dict(torch.load(self.actor.checkpoint_file))
        self.critic.load_state_dict(torch.load(self.critic.checkpoint_file))
        self.critic2.load_state_dict(torch.load(self.critic2.checkpoint_file))
        self.target_actor.load_state_dict(torch.load(self.target_actor.checkpoint_file))
        self.target_critic.load_state_dict(torch.load(self.target_critic.checkpoint_file))
        self.target_critic2.load_state_dict(torch.load(self.target_critic2.checkpoint_file))

    def choose_action(self, state, evaluate=False):
        if self.trainstep > self.warmup_steps:
            evaluate = True
        
        mu = self.actor(state.to(device), self.max_action)
        mu = mu.data

        if not evaluate:
            noise = torch.Tensor(self.noise()).to(device)
            mu += noise

        mu = mu.clamp(self.min_action, self.max_action)

        return mu.cpu().detach().numpy() 

    def update(self, state_batch, action_batch, reward_batch, next_state_batch, done_batch):
        # Training and updating Actor & Critic networks.
        target_actions = self.target_actor.forward(next_state_batch, self.max_action)
        target_actions += torch.clamp(torch.normal(mean=0.0, std=0.2, size=[*np.shape(target_actions)]), -0.5, 0.5)
        target_actions = torch.clamp(target_actions, self.min_action, self.max_action)

        target_next_state_values = self.target_critic.forward(next_state_batch, target_actions)
        target_next_state_values2 = self.target_critic2.forward(next_state_batch, target_actions)
        next_state_target_value = torch.minimum(target_next_state_values, target_next_state_values2)
        y = reward_batch + self.gamma * next_state_target_value.detach() * (1 - done_batch)

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

        self.trainstep += 1

        if self.trainstep % self.actor_update_steps == 0:

            actions = self.actor.forward(state_batch, self.max_action)
            critic_value = self.critic.forward(state_batch, actions)
            actor_loss = -torch.mean(critic_value)

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

    def learn(self):
        if self.memory.buffer_counter < self.batch_size:
            return

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample()

        self.update(state_batch, action_batch, reward_batch, next_state_batch, done_batch)

        self.update_network_parameters()