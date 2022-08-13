from rl_algorithms.PyTorch.agents.ddpg import DDPG 
from rl_algorithms.PyTorch.agents.td3 import TD3
from rl_algorithms.PyTorch.agents.sac import SAC


class AgentFactory:
    """Agent factory class."""

    def get_agent(algorithm_name, env, params=None):
        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.shape[0]
        action_space = env.action_space

        upper_bound = env.action_space.high[0]
        lower_bound = env.action_space.low[0]

        if not params:
            params = {}

        if algorithm_name == "DDPG":
            agent = DDPG(num_actions, num_states, lower_bound, upper_bound, **params)
        elif algorithm_name == "TD3":
            agent = TD3(num_actions, num_states, lower_bound, upper_bound, **params)
        elif algorithm_name == "SAC":
            agent = SAC(num_actions, num_states, lower_bound, upper_bound, action_space, **params)
        else:
            raise NotImplementedError("Algorithm is not implemented!")
        
        return agent
