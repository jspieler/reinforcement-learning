import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp

from rl_algorithms.TensorFlow2.utils.buffer import Buffer
from rl_algorithms.TensorFlow2.utils.networks import SoftActor, Critic


class SAC:
    def __init__(self, num_actions, num_states, min_action, max_action, action_space=None, 
                 lr_actor=3e-4, lr_critic=3e-4, lr_alpha=3e-4, gamma=0.99, buffer_capacity=100000, 
                 tau=0.005, hidden_size=(512,512), batch_size=64, alpha=0.2, target_entropy_tuning=True, 
                 update_frequency=2):
        self.gamma = gamma
        self.tau = tau
        self.memory = Buffer(num_actions, num_states, buffer_capacity, batch_size)
        self.batch_size = batch_size
        self.num_actions = num_actions
        self.num_states = num_states
        self.min_action = min_action
        self.max_action = max_action
        self.trainstep = 0
        self.update_frequency = update_frequency

        self.alpha = alpha 
        self.target_entropy_tuning = target_entropy_tuning
        self.alpha_optimizer = Adam(learning_rate=lr_alpha)

        if self.target_entropy_tuning:
            self.log_alpha = tf.Variable(0.0)
            self.alpha = tfp.util.DeferredTensor(self.log_alpha, tf.exp)
            self.target_entropy = - tf.Variable(num_actions, dtype=tf.float32)

        self.actor = SoftActor(num_actions, action_space, hidden_size)
        self.critic = Critic(hidden_size)
        self.critic2 = Critic(hidden_size, name='critic2')
        self.target_critic = Critic(hidden_size, name='target_critic')
        self.target_critic2 = Critic(hidden_size, name='target_critic2')

        self.actor_optimizer = Adam(learning_rate=lr_actor)
        self.critic_optimizer = Adam(learning_rate=lr_critic)
        self.critic2_optimizer = Adam(learning_rate=lr_critic)

        self.actor.compile(self.actor_optimizer)
        self.critic.compile(self.critic_optimizer)
        self.critic2.compile(self.critic2_optimizer)

        self.update_network_parameters(tau=1)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        weights = []
        targets = self.target_critic.weights
        for i, weight in enumerate(self.critic.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.target_critic.set_weights(weights)

        weights = []
        targets = self.target_critic2.weights
        for i, weight in enumerate(self.critic2.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.target_critic2.set_weights(weights)

    def save_models(self):
        self.actor.save_weights(self.actor.checkpoint_file)
        self.critic.save_weights(self.critic.checkpoint_file)
        self.critic2.save_weights(self.critic2.checkpoint_file)
        self.target_critic.save_weights(self.target_critic.checkpoint_file)
        self.target_critic2.save_weights(self.target_critic2.checkpoint_file)

    def load_models(self):
        self.actor.load_weights(self.actor.checkpoint_file)
        self.critic.load_weights(self.critic.checkpoint_file)
        self.critic2.load_weights(self.critic2.checkpoint_file)
        self.target_critic.load_weights(self.target_critic.checkpoint_file)
        self.target_critic2.load_weights(self.target_critic2.checkpoint_file)

    def choose_action(self, state, evaluate=False):
        if evaluate is False:
            action, _, _ = self.actor(state)
        else:
            _, _, action = self.actor(state)
        
        return action.numpy()[0]

    # @tf.function # with decoration code is not running bc. networks are not built?
    def update(self, state_batch, action_batch, reward_batch, next_state_batch, done_batch):
        # update critic
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            target_actions, target_log_pi, _ = self.actor(next_state_batch, training=True)
            target_next_state_values = self.target_critic(next_state_batch, target_actions, training=True)
            target_next_state_values2 = self.target_critic2(next_state_batch, target_actions, training=True)
            next_state_target_value = tf.math.minimum(target_next_state_values, target_next_state_values2) - self.alpha * target_log_pi
            target_values = reward_batch + self.gamma * next_state_target_value * (1 - done_batch)

            critic_value = self.critic(state_batch, action_batch, training=True)
            critic_value2 = self.critic2(state_batch, action_batch, training=True)

            next_state_target_value = tf.math.minimum(target_next_state_values, target_next_state_values2)

            target_values = reward_batch + self.gamma * next_state_target_value * (1 - done_batch)
            critic_loss = tf.math.reduce_mean(tf.math.square(target_values - critic_value))
            critic_loss2 = tf.math.reduce_mean(tf.math.square(target_values - critic_value2))

        critic_grad1 = tape1.gradient(critic_loss, self.critic.trainable_variables)
        critic_grad2 = tape2.gradient(critic_loss2, self.critic2.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(critic_grad1, self.critic.trainable_variables)
        )
        self.critic2_optimizer.apply_gradients(
            zip(critic_grad2, self.critic2.trainable_variables)
        )

        # update actor
        with tf.GradientTape() as tape3:
            pi, log_pi, _ = self.actor(state_batch, training=True)
            qf1_pi = self.critic(state_batch, action_batch, training=True)
            qf2_pi = self.critic2(state_batch, action_batch, training=True)

            min_qf_pi = tf.math.minimum(qf1_pi, qf2_pi)
            actor_loss = tf.math.reduce_mean((self.alpha * log_pi) - min_qf_pi)

            actor_grad = tape3.gradient(actor_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(
                zip(actor_grad, self.actor.trainable_variables)
            )
        
        # update alpha  
        if self.target_entropy_tuning:
            with tf.GradientTape() as tape4:
                alpha_loss = - tf.math.reduce_mean(self.log_alpha * tf.stop_gradient(log_pi + self.target_entropy))

            alpha_grad = tape4.gradient(alpha_loss, [self.log_alpha])
            self.alpha_optimizer.apply_gradients(
                zip(alpha_grad, [self.log_alpha])
            )
        
        self.trainstep += 1

        if self.trainstep % self.update_frequency == 0:
            self.update_network_parameters()

    def learn(self):
        if self.memory.buffer_counter < self.batch_size:
            return

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample()

        self.update(state_batch, action_batch, reward_batch, next_state_batch, done_batch)
