from rl_algorithms.PyTorch.agents.ddpg import DDPG
from rl_algorithms.PyTorch.agents.td3 import TD3
from rl_algorithms.PyTorch.agents.sac import SAC


class AgentFactory:
    """Agent factory class."""

    def get_agent(algorithm_name, env, params=None):
        algorithms = {"DDPG": DDPG, "TD3": TD3, "SAC": SAC}

        if not params:
            params = {}

        try:
            agent = algorithms[algorithm_name](env, **params)
        except KeyError:
            raise NotImplementedError("Algorithm is not implemented!")

        return agent
