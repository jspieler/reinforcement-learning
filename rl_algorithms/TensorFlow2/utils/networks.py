import os
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers


# Initialize weights between -3e-3 and 3-e3
last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)


class Actor(tf.keras.Model):
    """Actor network.

    Args:
        num_actions: The number of actions for output.
        hidden_size: The number of neurons in the hidden layers.
        name: The name of the network.
        chkpt_dir: The directory name where to save the network.
    """

    def __init__(
        self, num_actions, hidden_size=(512, 512), name="actor", chkpt_dir="tmp"
    ):
        super(Actor, self).__init__()
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(
            self.checkpoint_dir, self.model_name + ".h5"
        )

        self.hidden1 = layers.Dense(hidden_size[0], activation="relu")
        self.hidden2 = layers.Dense(hidden_size[1], activation="relu")
        self.mu = layers.Dense(
            num_actions, activation="tanh", kernel_initializer=last_init
        )

    def call(self, state, upper_bound):
        out = self.hidden1(state)
        out = self.hidden2(out)
        mu = self.mu(out)
        # multiply by upper limit
        output = mu * upper_bound
        return output


class Critic(tf.keras.Model):
    """Critic network.

    Args:
        hidden_size: The number of neurons in the hidden layers.
        name: The name of the network.
        chkpt_dir: The directory name where to save the network.
    """

    def __init__(self, hidden_size=(512, 512), name="critic", chkpt_dir="tmp"):
        super(Critic, self).__init__()
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(
            self.checkpoint_dir, self.model_name + ".h5"
        )

        self.hidden1 = layers.Dense(hidden_size[0], activation="relu")
        self.hidden2 = layers.Dense(hidden_size[1], activation="relu")
        self.q = layers.Dense(1, kernel_initializer=last_init)

    def call(self, state, action):
        out = self.hidden1(tf.concat([state, action], axis=1))
        out = self.hidden2(out)
        q = self.q(out)

        return q


class SoftActor(tf.keras.Model):
    """Soft-Actor network.

    Args:
        num_actions: The number of actions for output.
        action_space: The action space of the environment.
        hidden_size: The number of neurons in the hidden layers.
        name: The name of the network.
        chkpt_dir: The directory name where to save the network.
    """

    def __init__(
        self,
        num_actions,
        action_space=None,
        hidden_size=(512, 512),
        name="soft_actor",
        chkpt_dir="tmp",
    ):
        super(SoftActor, self).__init__()
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(
            self.checkpoint_dir, self.model_name + ".h5"
        )

        self.log_std_min = -20
        self.log_std_max = 2
        self.epsilon = 1e-6

        self.hidden1 = layers.Dense(hidden_size[0], activation="relu")
        self.hidden2 = layers.Dense(hidden_size[1], activation="relu")
        self.mean = layers.Dense(num_actions)
        self.log_std = layers.Dense(num_actions)

        # action rescaling
        if action_space is None:
            self.action_scale = tf.constant([1.0])
            self.action_bias = tf.constant([0.0])
        else:
            self.action_scale = tf.constant(
                [(action_space.high - action_space.low) / 2.0], dtype=tf.float32
            )
            self.action_bias = tf.constant(
                [(action_space.high + action_space.low) / 2.0], dtype=tf.float32
            )

    def call(self, state):
        out = self.hidden1(state)
        out = self.hidden2(out)
        mean = self.mean(out)
        log_std = self.log_std(out)
        log_std = tf.clip_by_value(log_std, self.log_std_min, self.log_std_max)

        std = tf.exp(log_std)
        normal = tfp.distributions.Normal(mean, std)
        # reparametrization trick
        x_t = normal.sample()
        y_t = tf.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= tf.math.log(
            self.action_scale * (1 - tf.math.pow(y_t, 2)) + self.epsilon
        )
        log_prob = tf.reduce_sum(log_prob, axis=1, keepdims=True)
        mean = tf.tanh(mean) * self.action_scale + self.action_bias

        return action, log_prob, mean
