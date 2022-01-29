import gym
from argparse import ArgumentParser
import numpy as np
import torch

from utils.plots import plot_avg_reward

from agents.ddpg import DDPG 
from agents.td3 import TD3


device = torch.device('gpu' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--agent', type=str, default='DDPG')
    parser.add_argument('--env', type=str, default='Pendulum-v1')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--ep', type=int, default=150)

    args = parser.parse_args()
    params = vars(args)

    seed = params['seed']
    env = gym.make(params['env'])

    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]

    upper_bound = env.action_space.high[0]
    lower_bound = env.action_space.low[0]

    num_episodes = params['ep']

    if params['agent'] == 'DDPG':
        agent = DDPG(num_actions, num_states, lower_bound, upper_bound)
    elif params['agent'] == 'TD3':
        agent = TD3(num_actions, num_states, lower_bound, upper_bound)
    else:
        raise NotImplementedError("Algorithm is not implemented!")

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
    plot_avg_reward(avg_reward_list, fname=params['agent']+'_'+params['env']+'.png')