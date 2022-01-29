# Reinforcement Learning
This repository contains minimalistic implementations of several (Deep) Reinforcement Learning algorithms using PyTorch & TensorFlow2. The repository is constantly being updated and new algorithms will be added.

# Algorithms

## Implemented
- [DDPG](#ddpg)
- [TD3](#td3)
- [SAC](#sac)

## Planned
- [Hybrid-MPO](#hybrid-mpo)
- [WD3](https://arxiv.org/pdf/2006.12622.pdf) / [AWD3](https://arxiv.org/pdf/2111.06780.pdf)

# Quickstart
1. Clone repository using ssh:
    ```
    git clone git@github.com:jspieler/reinforcement-learning.git
    ```
2. Install requirements using `pip`:
    ```
    python -m pip install -r requirements.txt
    ```
3. Run algorithms for `OpenAI gym` environments, e.g. DDPG on `Pendulum-v1` for 150 episodes:
    ```
    python train_agent.py --agent DDPG --env Pendulum-v1 --seed 1234 --ep 150
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
**Paper:** [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/pdf/1801.01290.pdf)<br>
**Method:** Off-Policy / Temporal-Difference / Actor-Critic / Model-Free<br>
**Action space:** Continuous <br>
**Implementation:** not yet implemented

<hr>

<a name='hybrid-mpo'></a>
## Hybrid Maximum a Posteriori Policy Optimization (Hybrid-MPO)
**Paper:** [Continuous-Discrete Reinforcement Learning for Hybrid Control in Robotics](https://arxiv.org/pdf/2001.00449.pdf)<br>
**Method:** Off-Policy / Temporal-Difference / Actor-Critic / Model-Free<br>
**Action space:** Continuous & Discrete <br>
**Implementation:** not yet implemented

