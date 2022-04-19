import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.constant_(m.bias, 0)


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

    
class SoftActor(nn.Module):
    def __init__(self, num_actions, num_states, action_space=None, hidden_size=(512,512), name='soft-actor', chkpt_dir='tmp'):
        super(SoftActor, self).__init__()
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.model_name + '.pt')

        self.log_std_min = -20
        self.log_std_max = 2
        self.epsilon = 1e-6

        self.hidden1 = nn.Linear(num_states, hidden_size[0])
        self.hidden2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.mean = nn.Linear(hidden_size[1], num_actions)
        self.log_std = nn.Linear(hidden_size[1], num_actions)

        self.apply(weights_init)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor((action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor((action_space.high + action_space.low) / 2.)

    def forward(self, state):
        out = self.hidden1(state)
        out = F.relu(out)
        out = self.hidden2(out)
        out = F.relu(out)

        mean = self.mean(out)
        log_std = self.log_std(out)
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)

        return mean, log_std 

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        # reparametrization trick
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias 
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + self.epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias

        return action, log_prob, mean 

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(SoftActor, self).to(device)