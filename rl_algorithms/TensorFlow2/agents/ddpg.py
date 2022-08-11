import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam

from rl_algorithms.TensorFlow2.utils.buffer import Buffer
from rl_algorithms.TensorFlow2.utils.networks import Actor, Critic
from rl_algorithms.TensorFlow2.utils.noise import OUActionNoise


class DDPG:
    def __init__(self, num_actions, num_states, min_action, max_action, lr_actor=0.001,
                 lr_critic=0.002, gamma=0.99, buffer_capacity=100000, tau=0.005,
                 hidden_size=(512,512), batch_size=64, noise_stddev=0.1):
        self.gamma = gamma
        self.tau = tau
        self.memory = Buffer(num_actions, num_states, buffer_capacity, batch_size)
        self.batch_size = batch_size
        self.num_actions = num_actions
        self.num_states = num_states
        self.min_action = min_action
        self.max_action = max_action
        self.noise = OUActionNoise(mean=np.zeros(num_actions), std_deviation=float(noise_stddev) * np.ones(num_actions))

        self.actor = Actor(num_actions, hidden_size)
        self.critic = Critic(hidden_size)
        self.target_actor = Actor(num_actions, hidden_size, name='target_actor')
        self.target_critic = Critic(hidden_size, name='target_critic')

        self.actor_optimizer = Adam(learning_rate=lr_actor)
        self.critic_optimizer = Adam(learning_rate=lr_critic)

        self.actor.compile(self.actor_optimizer)
        self.critic.compile(self.critic_optimizer)
        self.target_actor.compile(self.actor_optimizer)
        self.target_critic.compile(self.critic_optimizer)

        self.update_network_parameters(tau=1)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        weights = []
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.target_actor.set_weights(weights)

        weights = []
        targets = self.target_critic.weights
        for i, weight in enumerate(self.critic.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.target_critic.set_weights(weights)

    def save_models(self):
        self.actor.save_weights(self.actor.checkpoint_file)
        self.target_actor.save_weights(self.target_actor.checkpoint_file)
        self.critic.save_weights(self.critic.checkpoint_file)
        self.target_critic.save_weights(self.target_critic.checkpoint_file)

    def load_models(self):
        self.actor.load_weights(self.actor.checkpoint_file)
        self.target_actor.load_weights(self.target_actor.checkpoint_file)
        self.critic.load_weights(self.critic.checkpoint_file)
        self.target_critic.load_weights(self.target_critic.checkpoint_file)

    def choose_action(self, state, evaluate=False):
        sampled_actions = tf.squeeze(self.actor(state, self.max_action))
        if not evaluate:
            noise = self.noise()
            # add  noise to action
            sampled_actions = sampled_actions.numpy() + noise

        # clip action to bounds
        actions = np.clip(sampled_actions, self.min_action, self.max_action)

        return actions

    @tf.function
    def update(self, state_batch, action_batch, reward_batch, next_state_batch, done_batch):
        # Training and updating Actor & Critic networks.
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_state_batch, self.max_action, training=True)
            y = reward_batch + self.gamma * self.target_critic(next_state_batch, target_actions, training=True) * (1 - done_batch)
            critic_value = self.critic(state_batch, action_batch, training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(critic_grad, self.critic.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = self.actor(state_batch, self.max_action, training=True)
            critic_value = self.critic(state_batch, actions, training=True)
            # Use `-value` as we want to maximize the value given by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor.trainable_variables)
        )

    def learn(self):
        if self.memory.buffer_counter < self.batch_size:
            return

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample()

        self.update(state_batch, action_batch, reward_batch, next_state_batch, done_batch)

        self.update_network_parameters()