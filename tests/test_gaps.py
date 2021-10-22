# Copyright (c) 2021-present, Facebook, Inc.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import gym
import pytest

import bisk


@pytest.fixture
def env():
    env = gym.make('BiskGaps-v1', robot='testcube')
    env.seed(0)
    yield env
    env.close()


def test_reward_cross(env):
    upf = (env.p.named.model.body_mass['robot/torso'] * 9.81) / (
        env.p.named.model.actuator_gear['robot/z'][0]
    )
    obs = env.reset()
    obs, reward, done, info = env.step([0, 0, 1])

    # Go to platform
    for _ in range(64):
        next_obs, reward, done, info = env.step([1, 0, upf])
        if next_obs['next_gap_platform'][1] > obs['next_gap_platform'][1]:
            assert reward == 1
            break
        else:
            assert reward == 0
        obs = next_obs

    # Go back
    for _ in range(64):
        obs, reward, done, info = env.step([-1, 0, upf])
        if obs['next_gap_platform'][2] == 0:
            break
    obs, reward, done, info = env.step([-1, 0, upf])
    for _ in range(4):
        obs, reward, done, info = env.step([0, 0, upf])

    # Go to platform again, receive no reward
    for _ in range(64):
        next_obs, reward, done, info = env.step([1, 0, upf])
        if next_obs['next_gap_platform'][1] > obs['next_gap_platform'][1]:
            assert reward == 0
            break
        else:
            assert reward == 0
        obs = next_obs
    obs = next_obs

    # Go to next platform, receive reward
    for _ in range(64):
        next_obs, reward, done, info = env.step([1, 0, upf])
        if next_obs['next_gap_platform'][1] > obs['next_gap_platform'][1]:
            assert reward == 1
            break
        else:
            assert reward == 0
        obs = next_obs


def test_touch_gap(env):
    upf = (env.p.named.model.body_mass['robot/torso'] * 9.81) / (
        env.p.named.model.actuator_gear['robot/z'][0]
    )
    obs = env.reset()
    obs, reward, done, info = env.step([0, 0, 1])

    # Go to gap
    for _ in range(64):
        next_obs, reward, done, info = env.step([1, 0, upf])
        if next_obs['next_gap_platform'][0] > obs['next_gap_platform'][0]:
            break
        obs = next_obs

    # Go down into gap
    for _ in range(8):
        obs, reward, done, info = env.step([-1, 0, -0.6])
        if done:
            break
    assert reward == -1
    assert done == True


def test_touch_platform(env):
    upf = (env.p.named.model.body_mass['robot/torso'] * 9.81) / (
        env.p.named.model.actuator_gear['robot/z'][0]
    )
    obs = env.reset()
    obs, reward, done, info = env.step([0, 0, 1])

    # Go to platform
    for _ in range(64):
        next_obs, reward, done, info = env.step([1, 0, upf])
        if next_obs['next_gap_platform'][1] > obs['next_gap_platform'][1]:
            break
        obs = next_obs

    # Go down on platform
    for _ in range(8):
        obs, reward, done, info = env.step([0, 0, -1])
    assert reward == 0
    assert done == False
