# The Bipedal Skills Benchmark

The bipedal skills benchmark is a suite of reinforcement learning
environments implemented for the MuJoCo physics simulator. It aims to provide a
set of tasks that demand a variety of motor skills beyond locomotion, and is
intended for evaluating skill discovery and hierarchical learning methods. The
majority of tasks exhibit a sparse reward structure.

![Tasks Overview](https://raw.githubusercontent.com/facebookresearch/bipedal-skills/main/img/tasks.png)

This benchmark was introduced in [Hierarchial Skills for Efficient Exploration](https://facebookresearch.github.io/hsd3).

## Usage

In order to run the environments, a working MuJoCo setup (version 2.0 or higher) is required. You
can follow the respective [installation steps of
dm_control](https://github.com/deepmind/dm_control/#requirements-and-installation)
for that.

Afterwards, install the Python package with pip:
```sh
pip install bipedal-skills
```

To install the package from a working copy, do:
```sh
pip install .
```

All tasks are exposed and registered as Gym environments once the `bisk` module
is imported:
```py
import gym
import bisk

env = gym.make('BiskHurdles-v1', robot='Walker')
# Alternatively
env = gym.make('BiskHurdlesWalker-v1')
```

A detailed description of the tasks can be found in the [corresponding
publication](https://arxiv.org/abs/2110.10809).


## Evaluation Protocol

For evaluating agents, we recommend estimating returns on 50 environment
instances with distinct seeds.
This can be acheived in sequence or by using one of Gym's vector wrappers:
```py
# Sequential evaluation
env = gym.make('BiskHurdlesWalker-v1')
env.seed(0)  # determinstic seed
retrns = []
for _ in range(50):
  obs = env.reset()
  retrn = 0
  while True:
    # Retrieve `action` from agent
    obs, reward, done, info = env.step(action)
    retrn += reward
    if done:
      # End of episode
      retrns.append(reward)
      break
print(f'Average return: {sum(retrns)/len(retrns)}')

# Batched evaluation
from gym.vector import SyncVectorEnv
import numpy as np
n = 50
env = SyncVectorEnv([lambda: gym.make('BiskHurdlesWalker-v1')] * n)
env.seed(0)  # determinstic seed
retrns = np.array([0.0] * n)
dones = np.array([False] * n)
obs = env.reset()
while not dones.all():
    # Retrieve `action` from agent
    obs, reward, done, info = env.step(action)
    retrns += reward * np.logical_not(dones)
    dones |= done
print(f'Average return: {retrns.mean()}')
```


## License
The bipedal skills benchmark is MIT licensed, as found in the LICENSE file.

Model definitions have been adapted from:
- [Gym](https://github.com/openai/gym) (HalfCheetah)
- [dm_control](https://github.com/deepmind/dm_control/) (Walker, Humanoid)
