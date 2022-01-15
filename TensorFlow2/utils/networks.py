import os
import tensorflow as tf
from tensorflow.keras import layers


# Initialize weights between -3e-3 and 3-e3
last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)


class Actor(tf.keras.Model):
    def __init__(self, num_actions, hidden_size=(512,512), name='actor', chkpt_dir='tmp'):
        super(Actor, self).__init__()
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.model_name + '.h5')

        self.hidden1 = layers.Dense(hidden_size[0], activation="relu")
        self.hidden2 = layers.Dense(hidden_size[1], activation="relu")
        self.mu = layers.Dense(num_actions, activation="tanh", kernel_initializer=last_init)

    def call(self, state, upper_bound):
        out = self.hidden1(state)
        out = self.hidden2(out)
        mu = self.mu(out)
        # multiply by upper limit
        output = mu * upper_bound
        return output


class Critic(tf.keras.Model):
    def __init__(self, hidden_size=(512,512), name='critic', chkpt_dir='tmp'):
        super(Critic, self).__init__()
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.model_name + '.h5')

        self.hidden1 = layers.Dense(hidden_size[0], activation="relu")
        self.hidden2 = layers.Dense(hidden_size[1], activation="relu")
        self.q = layers.Dense(1, kernel_initializer=last_init)

    def call(self, state, action):
        out = self.hidden1(tf.concat([state, action], axis=1))
        out = self.hidden2(out)
        q = self.q(out)

        return q
