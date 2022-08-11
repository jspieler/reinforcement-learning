import os 
import random
import gym
from argparse import ArgumentParser
import numpy as np
import torch

from rl_algorithms.common.plots import plot_avg_reward

from rl_algorithms.PyTorch.agents.ddpg import DDPG 
from rl_algorithms.PyTorch.agents.td3 import TD3
from rl_algorithms.PyTorch.agents.sac import SAC

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seeds(env, seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    env.seed(seed)
    env.action_space.np_random.seed(seed)


def train(agent, env, num_episodes, filename):
    """Runs the training loop for the agent and the environment.
    
    Args:
        agent: An agent object.
        env: An environment object.
        num_episodes: The number of training episodes.
        seed: The random seed.
        filename: The name of the file where the reward plot is saved to.
    """

    # store reward of each episode 
    ep_reward_list = []
    # store average reward for last 'n_smooth' episodes
    avg_reward_list = []
    n_smooth = 40

    for ep in range(num_episodes):

        prev_state = env.reset()
        episodic_reward = 0
        done = False

        while not done:
            # Uncomment to render environment
            # env.render()

            prev_state = torch.tensor(prev_state, dtype=torch.float).to(device)
            action = agent.choose_action(prev_state)

            state, reward, done, info = env.step(action)

            agent.memory.record((prev_state, action, reward, state, done))
            episodic_reward += reward

            agent.learn()

            if done:
                break

            prev_state = state

        ep_reward_list.append(episodic_reward)
        avg_reward = np.mean(ep_reward_list[-n_smooth:])
        print("Episode {} -- average reward: {}".format(ep, avg_reward))
        avg_reward_list.append(avg_reward)

    # Plot average episodic reward against episodes
    plot_avg_reward(avg_reward_list, filename)


if __name__ == '__main__':

    # parse arguments
    parser = ArgumentParser()
    parser.add_argument('--agent', type=str, default='DDPG')
    parser.add_argument('--env', type=str, default='Pendulum-v1')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--ep', type=int, default=150)

    args = parser.parse_args()
    params = vars(args)

    # create the environment
    env = gym.make(params['env'])

    # fix random seeds
    seed = params['seed']
    set_seeds(env, seed)

    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]

    upper_bound = env.action_space.high[0]
    lower_bound = env.action_space.low[0]

    num_episodes = params['ep']

    # create the agent
    if params['agent'] == 'DDPG':
        agent = DDPG(num_actions, num_states, lower_bound, upper_bound)
    elif params['agent'] == 'TD3':
        agent = TD3(num_actions, num_states, lower_bound, upper_bound)
    elif params['agent'] == 'SAC':
        agent = SAC(num_actions, num_states, lower_bound, upper_bound, env.action_space)
    else:
        raise NotImplementedError("Algorithm is not implemented!")

    # train the agent
    train(agent, env, num_episodes, filename=params['agent']+'_'+params['env']+'.png')