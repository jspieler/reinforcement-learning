import numpy as np
import tensorflow as tf


class Buffer:
    def __init__(self, num_actions, num_states, buffer_capacity=100000, batch_size=64):
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size

        self.buffer_counter = 0

        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.done_buffer = np.zeros((self.buffer_capacity, 1))

    def record(self, obs_tuple):
        # Replace old samples if buffer capacity is exceeded
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]
        self.done_buffer[index] = obs_tuple[4]

        self.buffer_counter += 1

    def sample(self):
        # Randomly sample batch
        record_range = min(self.buffer_counter, self.buffer_capacity)
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices], dtype=tf.float32)
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices], dtype=tf.float32)
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices], dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices], dtype=tf.float32)
        done_batch = tf.convert_to_tensor(self.done_buffer[batch_indices], dtype=tf.float32)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch