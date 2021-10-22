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
    env = gym.make('BiskGoalWall-v1', robot='testcube')
    env.seed(0)
    yield env
    env.close()


@pytest.fixture
def env2d():
    env = gym.make('BiskGoalWall-v1', robot='testcube2d')
    env.seed(0)
    yield env
    env.close()


def test_eoe_if_touched_wall(env):
    upf = (env.p.named.model.body_mass['robot/torso'] * 9.81) / (
        env.p.named.model.actuator_gear['robot/z'][0]
    )
    obs = env.reset()
    done = False
    step = 0
    while not done:
        obs, reward, done, info = env.step([1, 0, upf])
        step += 1
        assert reward == 0
    assert step < 250


def test_reward_goal1(env):
    obs = env.reset()
    env.p.named.data.qvel['ball'][0:3] = [10, -3, 2.5]
    done = False
    ret = 0
    step = 0
    while not done:
        obs, reward, done, info = env.step([0, 0, 0])
        ret += reward
        step += 1
    assert step < 250
    assert ret == 1


def test_reward_goal2(env):
    obs = env.reset()
    env.p.named.data.qvel['ball'][0:3] = [10, 3, 4]
    done = False
    ret = 0
    step = 0
    while not done:
        obs, reward, done, info = env.step([0, 0, 0])
        ret += reward
        step += 1
    assert step < 250
    assert ret == 2


def test_reward_nogoal(env):
    obs = env.reset()
    env.p.named.data.qvel['ball'][0:3] = [10, 0, 2]
    done = False
    ret = 0
    step = 0
    while not done:
        obs, reward, done, info = env.step([0, 0, 0])
        ret += reward
        step += 1
    assert step < 250
    assert ret == 0


def test_no_reward_if_beyond_line(env):
    upf = (env.p.named.model.body_mass['robot/torso'] * 9.81) / (
        env.p.named.model.actuator_gear['robot/z'][0]
    )
    obs = env.reset()

    # Move beyond line without touching the ball
    for _ in range(2):
        obs, reward, done, info = env.step([0, 1, upf])
        assert done == False
    for _ in range(18):
        obs, reward, done, info = env.step([1, 0, upf])
        assert done == False

    env.p.named.data.qvel['ball'][0:3] = [10, 0, 2]
    done = False
    ret = 0
    step = 0
    while not done:
        obs, reward, done, info = env.step([0, 0, 0])
        ret += reward
        step += 1
    assert step < 250
    assert ret == 0

    env.p.named.data.qvel['ball'][0:3] = [10, -3, 2.5]
    done = False
    ret = 0
    step = 0
    while not done:
        obs, reward, done, info = env.step([0, 0, 0])
        ret += reward
        step += 1
    assert step < 250
    assert ret == 0


def test_reward_goal2d(env2d):
    env = env2d
    obs = env.reset()
    env.p.named.data.qvel['ball-x'] = 10
    env.p.named.data.qvel['ball-z'] = 4
    done = False
    ret = 0
    step = 0
    while not done:
        obs, reward, done, info = env.step([0, 0])
        ret += reward
        step += 1
    assert step < 250
    assert ret == 1


def test_reward_nogoal2d_1(env2d):
    env = env2d
    obs = env.reset()
    env.p.named.data.qvel['ball-x'] = 10
    env.p.named.data.qvel['ball-z'] = 0
    done = False
    ret = 0
    step = 0
    while not done:
        obs, reward, done, info = env.step([0, 0])
        ret += reward
        step += 1
    assert step < 250
    assert ret == 0


def test_reward_nogoal2d_2(env2d):
    env = env2d
    obs = env.reset()
    env.p.named.data.qvel['ball-x'] = 10
    env.p.named.data.qvel['ball-z'] = 10
    done = False
    ret = 0
    step = 0
    while not done:
        obs, reward, done, info = env.step([0, 0])
        ret += reward
        step += 1
    assert step < 250
    assert ret == 0
