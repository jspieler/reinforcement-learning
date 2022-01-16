import os
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Actor(nn.Module):
    def __init__(self, num_actions, num_states, hidden_size=(512,512), name='actor', chkpt_dir='tmp'):
        super(Actor, self).__init__()
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.model_name + '.pt')

        self.hidden1 = nn.Linear(num_states, hidden_size[0])
        self.hidden2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.mu = nn.Linear(hidden_size[1], num_actions)

    def forward(self, state, upper_bound):
        out = self.hidden1(state)
        out = F.relu(out)
        out = self.hidden2(out)
        out = F.relu(out)
        out = self.mu(out)
        out = torch.tanh(out)
        output = out * upper_bound

        return output


class Critic(nn.Module):
    def __init__(self, num_inputs, hidden_size=(512,512), name='critic', chkpt_dir='tmp'):
        super(Critic, self).__init__()
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.model_name + '.pt')

        self.hidden1 = nn.Linear(num_inputs, hidden_size[0])
        self.hidden2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.q = nn.Linear(hidden_size[1], 1)

    def forward(self, state, action):
        inputs = torch.cat((state, action), 1)
        out = self.hidden1(inputs)
        out = F.relu(out)
        out = self.hidden2(out)
        out = F.relu(out)
        q = self.q(out)

        return q