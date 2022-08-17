# Reinforcement Learning
This repository contains minimalistic implementations of several (Deep) Reinforcement Learning algorithms using PyTorch & TensorFlow2. The repository is constantly being updated and new algorithms will be added.

# Algorithms

## Implemented
- [DDPG](#ddpg)
- [TD3](#td3)
- [SAC](#sac)

## Planned
- [MPO](#mpo)
- [Hybrid-MPO](#hybrid-mpo)
- [WD3](https://arxiv.org/pdf/2006.12622.pdf) / [AWD3](https://arxiv.org/pdf/2111.06780.pdf)

# Quickstart
1. Install package via pip:
    ```
    pip install git+https://github.com/jspieler/reinforcement-learning.git
    ```

2. Run algorithms for `OpenAI gym` environments, e.g. DDPG on the `Pendulum-v1` environment for 150 episodes using PyTorch:
    ```
    python rl_algorithms/PyTorch/train_agent.py --agent DDPG --env Pendulum-v1 --seed 1234 --ep 150
    ```
    If you want to use custom parameters for the algorithm instead of the default one, you can add the argument `--config /path/to/config.yaml`. See `config.yaml` for an example.

3. Alternatively, here is a quick example of how to train DDPG on the `Pendulum-v1` environment using PyTorch:
    ```
    import gym 

    from rl_algorithms.PyTorch.agents.ddpg import DDPG
    from rl_algorithms.PyTorch.train_agent import set_seeds, train

    env = gym.make("Pendulum-v1")
    agent = DDPG(num_actions=env.action_space.shape[0], num_states=env.observation_space.shape[0], min_action=env.action_space.low[0], max_action=env.action_space.high[0])
    set_seeds(env, seed=1234)
    train(agent, env, num_episodes=150, filename="ddpg_pendulum_v1_rewards.png")
    ```

# Further information

<a name='ddpg'></a>
## Deep Deterministic Policy Gradient (DDPG)
**Paper:** [Continuous control with deep reinforcement learning](https://arxiv.org/pdf/1509.02971.pdf)<br>
**Method:** Off-Policy / Temporal-Difference / Actor-Critic / Model-Free<br>
**Action space:** Continuous <br>
**Implementation:** [PyTorch](https://github.com/jspieler/reinforcement-learning/blob/main/PyTorch/agents/ddpg.py) / [TensorFlow2](https://github.com/jspieler/reinforcement-learning/blob/main/TensorFlow2/agents/ddpg.py)

**Note:** Implementation is not exactly the same as described in the original paper since specific implementation details are not included (actions are already included in the first layer of the critic network, different weight initialization, no batch normalization, etc.).

<hr>

<a name='td3'></a>
## Twin-Delayed Deep Deterministic Policy Gradient (TD3)
**Paper:** [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/pdf/1802.09477.pdf)<br>
**Method:** Off-Policy / Temporal-Difference / Actor-Critic / Model-Free<br>
**Action space:** Continuous <br>
**Implementation:**  [PyTorch](https://github.com/jspieler/reinforcement-learning/blob/main/PyTorch/agents/td3.py) / [TensorFlow2](https://github.com/jspieler/reinforcement-learning/blob/main/TensorFlow2/agents/td3.py)

<hr>

<a name='sac'></a>
## Soft Actor-Critic (SAC)
**Paper:** [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/pdf/1801.01290.pdf) / [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/pdf/1801.01290.pdf)<br>
**Method:** Off-Policy / Temporal-Difference / Actor-Critic / Model-Free<br>
**Action space:** Continuous <br>
**Implementation:** [PyTorch](https://github.com/jspieler/reinforcement-learning/blob/main/PyTorch/agents/sac.py) / [TensorFlow2](https://github.com/jspieler/reinforcement-learning/blob/main/TensorFlow2/agents/sac.py)

<hr>

<a name='mpo'></a>
## Maximum a Posteriori Policy Optimisation (MPO)
**Paper:** [Maximum a Posteriori Policy Optimisation](https://arxiv.org/pdf/1806.06920.pdf)<br>
**Method:** Off-Policy / Temporal-Difference / Actor-Critic / Model-Free<br>
**Action space:** Continuous & Discrete <br>
**Implementation:** not yet implemented

<hr>

<a name='hybrid-mpo'></a>
## Hybrid Maximum a Posteriori Policy Optimization (Hybrid-MPO)
**Paper:** [Continuous-Discrete Reinforcement Learning for Hybrid Control in Robotics](https://arxiv.org/pdf/2001.00449.pdf)<br>
**Method:** Off-Policy / Temporal-Difference / Actor-Critic / Model-Free<br>
**Action space:** Continuous & Discrete <br>
**Implementation:** not yet implemented

