from typing import Dict

import gym


def get_env_params(env: gym.Env) -> Dict:
    env_params = {
        "num_states": env.observation_space.shape[0],
        "num_actions": env.action_space.shape[0],
        "action_space": env.action_space,
        "upper_bound": env.action_space.high[0],
        "lower_bound": env.action_space.low[0],
    }
    return env_params
