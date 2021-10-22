# Copyright (c) 2021-present, Facebook, Inc.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import gym
import pytest

import bisk


# Simple test whether we can instantiate the env with a robot and do a single
# step.
def _create_helper(env_name: str, robot: str):
    env = gym.make(env_name, robot='Walker')
    env.reset()
    env.step(env.action_space.sample())
    env.close()

    env = gym.make(env_name.replace('-v', f'{robot}-v'))
    env.reset()
    env.step(env.action_space.sample())
    env.close()


def _envs_helper(robot: str):
    _create_helper('BiskHurdles-v1', robot)
    _create_helper('BiskLimbo-v1', robot)
    _create_helper('BiskHurdlesLimbo-v1', robot)
    _create_helper('BiskGaps-v1', robot)
    _create_helper('BiskStairs-v1', robot)
    _create_helper('BiskGoalWall-v1', robot)
    _create_helper('BiskGoalWall-v1', robot)


def test_halfcheetah_create():
    _envs_helper('HalfCheetah')


def test_walker_create():
    _envs_helper('Walker')


def test_humanoid_create():
    _envs_helper('Humanoid')


def test_humanoidpc_create():
    _envs_helper('HumanoidPC')
