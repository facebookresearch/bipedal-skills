# Copyright (c) 2021-present, Facebook, Inc.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import gym
import pytest

import bisk
from bisk.features.joints import JointsFeaturizer
from bisk.single_robot import BiskSingleRobotEnv


def test_names_walker():
    env = BiskSingleRobotEnv('walker')
    ftzr = JointsFeaturizer(env.p, 'walker', 'robot')
    assert ftzr.observation_space.shape == (60,)
    assert ftzr().shape == ftzr.observation_space.shape
    assert ftzr.feature_names() == [
        'rootz:p',
        'rootx:p',
        'rooty:p',
        'right_hip:p',
        'right_knee:p',
        'right_ankle:p',
        'left_hip:p',
        'left_knee:p',
        'left_ankle:p',
        'rootz:v',
        'rootx:v',
        'rooty:v',
        'right_hip:v',
        'right_knee:v',
        'right_ankle:v',
        'left_hip:v',
        'left_knee:v',
        'left_ankle:v',
        'torso:crx',
        'torso:cry',
        'torso:crz',
        'torso:ctx',
        'torso:cty',
        'torso:ctz',
        'right_thigh:crx',
        'right_thigh:cry',
        'right_thigh:crz',
        'right_thigh:ctx',
        'right_thigh:cty',
        'right_thigh:ctz',
        'right_leg:crx',
        'right_leg:cry',
        'right_leg:crz',
        'right_leg:ctx',
        'right_leg:cty',
        'right_leg:ctz',
        'right_foot:crx',
        'right_foot:cry',
        'right_foot:crz',
        'right_foot:ctx',
        'right_foot:cty',
        'right_foot:ctz',
        'left_thigh:crx',
        'left_thigh:cry',
        'left_thigh:crz',
        'left_thigh:ctx',
        'left_thigh:cty',
        'left_thigh:ctz',
        'left_leg:crx',
        'left_leg:cry',
        'left_leg:crz',
        'left_leg:ctx',
        'left_leg:cty',
        'left_leg:ctz',
        'left_foot:crx',
        'left_foot:cry',
        'left_foot:crz',
        'left_foot:ctx',
        'left_foot:cty',
        'left_foot:ctz',
    ]
    env.close()


def test_exclude_walker():
    env = BiskSingleRobotEnv('walker')
    ftzr = JointsFeaturizer(
        env.p, 'walker', 'robot', exclude='.*/(left|right)_foot'
    )
    assert ftzr.observation_space.shape == (48,)
    assert ftzr().shape == ftzr.observation_space.shape
    assert ftzr.feature_names() == [
        'rootz:p',
        'rootx:p',
        'rooty:p',
        'right_hip:p',
        'right_knee:p',
        'right_ankle:p',
        'left_hip:p',
        'left_knee:p',
        'left_ankle:p',
        'rootz:v',
        'rootx:v',
        'rooty:v',
        'right_hip:v',
        'right_knee:v',
        'right_ankle:v',
        'left_hip:v',
        'left_knee:v',
        'left_ankle:v',
        'torso:crx',
        'torso:cry',
        'torso:crz',
        'torso:ctx',
        'torso:cty',
        'torso:ctz',
        'right_thigh:crx',
        'right_thigh:cry',
        'right_thigh:crz',
        'right_thigh:ctx',
        'right_thigh:cty',
        'right_thigh:ctz',
        'right_leg:crx',
        'right_leg:cry',
        'right_leg:crz',
        'right_leg:ctx',
        'right_leg:cty',
        'right_leg:ctz',
        'left_thigh:crx',
        'left_thigh:cry',
        'left_thigh:crz',
        'left_thigh:ctx',
        'left_thigh:cty',
        'left_thigh:ctz',
        'left_leg:crx',
        'left_leg:cry',
        'left_leg:crz',
        'left_leg:ctx',
        'left_leg:cty',
        'left_leg:ctz',
    ]
    env.close()
