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
    env = gym.make('BiskHurdles-v1', robot='testcube')
    env.seed(0)
    yield env
    env.close()


def test_render(env):
    env.reset()
    img = env.render(width=480, height=480)
    assert img.shape == (480, 480, 3)
