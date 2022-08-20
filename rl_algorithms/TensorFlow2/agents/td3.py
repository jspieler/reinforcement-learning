import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from rl_algorithms.common.env_utils import get_env_params
from rl_algorithms.TensorFlow2.utils.buffer import Buffer
from rl_algorithms.TensorFlow2.utils.networks import Actor, Critic
from rl_algorithms.TensorFlow2.utils.noise import GaussianActionNoise


class TD3:
    def __init__(
        self,
        env,
        lr_actor=1e-3,
        lr_critic=2e-3,
        gamma=0.99,
        buffer_capacity=100000,
        tau=0.005,
        hidden_size=(512, 512),
        batch_size=64,
        noise_stddev=0.2,
        warmup_steps=200,
        update_frequency=2,
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
        self.trainstep = 0
        self.warmup_steps = warmup_steps
        self.actor_update_steps = update_frequency
        self.noise = GaussianActionNoise(
            mean=np.zeros(self.num_actions),
            std_deviation=float(noise_stddev) * np.ones(self.num_actions),
        )

        self.actor = Actor(self.num_actions, hidden_size)
        self.critic = Critic(hidden_size)
        self.critic2 = Critic(hidden_size, name="critic2")
        self.target_actor = Actor(self.num_actions, hidden_size, name="target_actor")
        self.target_critic = Critic(hidden_size, name="target_critic")
        self.target_critic2 = Critic(hidden_size, name="target_critic2")

        self.actor_optimizer = Adam(learning_rate=lr_actor)
        self.critic_optimizer = Adam(learning_rate=lr_critic)
        self.critic2_optimizer = Adam(learning_rate=lr_critic)

        self.actor.compile(self.actor_optimizer)
        self.critic.compile(self.critic_optimizer)
        self.critic2.compile(self.critic2_optimizer)

        self.update_network_parameters(tau=1.0)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        weights = []
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight * tau + targets[i] * (1 - tau))
        self.target_actor.set_weights(weights)

        weights = []
        targets = self.target_critic.weights
        for i, weight in enumerate(self.critic.weights):
            weights.append(weight * tau + targets[i] * (1 - tau))
        self.target_critic.set_weights(weights)

        weights = []
        targets = self.target_critic2.weights
        for i, weight in enumerate(self.critic2.weights):
            weights.append(weight * tau + targets[i] * (1 - tau))
        self.target_critic2.set_weights(weights)

    def save_models(self):
        self.actor.save_weights(self.actor.checkpoint_file)
        self.target_actor.save_weights(self.target_actor.checkpoint_file)
        self.critic.save_weights(self.critic.checkpoint_file)
        self.target_critic.save_weights(self.target_critic.checkpoint_file)
        self.critic2.save_weights(self.critic2.checkpoint_file)
        self.target_critic2.save_weights(self.target_critic2.checkpoint_file)

    def load_models(self):
        self.actor.load_weights(self.actor.checkpoint_file)
        self.target_actor.load_weights(self.target_actor.checkpoint_file)
        self.critic.load_weights(self.critic.checkpoint_file)
        self.target_critic.load_weights(self.target_critic.checkpoint_file)
        self.critic2.load_weights(self.critic2.checkpoint_file)
        self.target_critic2.load_weights(self.target_critic2.checkpoint_file)

    def choose_action(self, state, evaluate=False):
        if self.trainstep > self.warmup_steps:
            evaluate = True
        sampled_actions = tf.squeeze(self.actor(state, self.max_action))
        sampled_actions = sampled_actions.numpy()
        if not evaluate:
            noise = self.noise()
            # add noise to action
            sampled_actions += noise

        # make sure action is within bounds
        actions = np.array([1]) * np.clip(
            sampled_actions, self.min_action, self.max_action
        )

        return actions

    # @tf.function
    def update(
        self, state_batch, action_batch, reward_batch, next_state_batch, done_batch
    ):
        # Training and updating Actor & Critic networks.
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            target_actions = self.target_actor(
                next_state_batch, self.max_action, training=True
            )
            target_actions += tf.clip_by_value(
                tf.random.normal(
                    shape=[*np.shape(target_actions)], mean=0.0, stddev=0.2
                ),
                -0.5,
                0.5,
            )
            target_actions = tf.clip_by_value(
                target_actions, self.min_action, self.max_action
            )

            target_next_state_values = self.target_critic(
                next_state_batch, target_actions, training=True
            )
            target_next_state_values2 = self.target_critic2(
                next_state_batch, target_actions, training=True
            )

            critic_value = self.critic(state_batch, action_batch, training=True)
            critic_value2 = self.critic2(state_batch, action_batch, training=True)

            next_state_target_value = tf.math.minimum(
                target_next_state_values, target_next_state_values2
            )

            target_values = reward_batch + self.gamma * next_state_target_value * (
                1 - done_batch
            )
            critic_loss = tf.math.reduce_mean(
                tf.math.square(target_values - critic_value)
            )
            critic_loss2 = tf.math.reduce_mean(
                tf.math.square(target_values - critic_value2)
            )

        critic_grad1 = tape1.gradient(critic_loss, self.critic.trainable_variables)
        critic_grad2 = tape2.gradient(critic_loss2, self.critic2.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(critic_grad1, self.critic.trainable_variables)
        )
        self.critic2_optimizer.apply_gradients(
            zip(critic_grad2, self.critic2.trainable_variables)
        )

        self.trainstep += 1

        if self.trainstep % self.actor_update_steps == 0:

            with tf.GradientTape() as tape3:
                new_policy_actions = self.actor(
                    state_batch, self.max_action, training=True
                )
                critic_value = self.critic(
                    state_batch, new_policy_actions, training=True
                )
                actor_loss = -tf.math.reduce_mean(critic_value)

            actor_grad = tape3.gradient(actor_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(
                zip(actor_grad, self.actor.trainable_variables)
            )

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

        self.update(
            state_batch, action_batch, reward_batch, next_state_batch, done_batch
        )

        self.update_network_parameters()
