import numpy as np
import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
        state_batch = torch.tensor(self.state_buffer[batch_indices], dtype=torch.float).to(device)
        action_batch = torch.tensor(self.action_buffer[batch_indices], dtype=torch.float).to(device)
        reward_batch = torch.tensor(self.reward_buffer[batch_indices], dtype=torch.float).to(device)
        next_state_batch = torch.tensor(self.next_state_buffer[batch_indices], dtype=torch.float).to(device)
        done_batch = torch.tensor(self.done_buffer[batch_indices], dtype=torch.float).to(device)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch