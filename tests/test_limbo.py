# Copyright (c) 2021-present, Facebook, Inc.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import gym
import pytest

import bisk


@pytest.fixture
def env():
    env = gym.make('BiskLimbo-v1', robot='testcube')
    env.seed(0)
    yield env
    env.close()


def test_reward_clear(env):
    upf = (env.p.named.model.body_mass['robot/torso'] * 9.81) / (
        env.p.named.model.actuator_gear['robot/z'][0]
    )
    obs = env.reset()

    # Cross bar
    for _ in range(64):
        next_obs, reward, done, info = env.step([1, 0, 0])
        if next_obs['next_bar'][0] > obs['next_bar'][0]:
            assert reward == 1
            break
        else:
            assert reward == 0
        obs = next_obs

    # Go back
    for _ in range(64):
        obs, reward, done, info = env.step([-1, 0, upf])
        if obs['next_bar'][2] == 0:
            break
    obs, reward, done, info = env.step([-1, 0, upf])
    for _ in range(4):
        obs, reward, done, info = env.step([0, 0, upf])

    # Cross bar again, receive no reward
    for _ in range(64):
        next_obs, reward, done, info = env.step([1, 0, upf])
        if next_obs['next_bar'][0] > obs['next_bar'][0]:
            assert reward == 0
            break
        else:
            assert reward == 0
        obs = next_obs
    obs = next_obs

    # Cross next bar, receive reward
    for _ in range(64):
        next_obs, reward, done, info = env.step([1, 0, upf])
        if next_obs['next_bar'][0] > obs['next_bar'][0]:
            assert reward == 1
            break
        else:
            assert reward == 0
        obs = next_obs


def test_reward_stuck(env):
    upf = (env.p.named.model.body_mass['robot/torso'] * 9.81) / (
        env.p.named.model.actuator_gear['robot/z'][0]
    )
    obs = env.reset()

    # Go up so that we'll be stuck at the first bar
    for _ in range(64):
        obs, reward, done, info = env.step([0, 0, 1])
        # Low threshold, remaining momentum will bring us to the right height
        if obs['observation'][2] - obs['next_bar'][1] >= -0.6:
            break

    # Go forward -- should be stuck at first bar
    for _ in range(64):
        obs, reward, done, info = env.step([1, 0, upf])
        assert reward == 0
