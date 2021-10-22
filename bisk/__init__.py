# Copyright (c) 2021-present, Facebook, Inc.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

__version__ = "1.0"

from gym.envs.registration import register

from bisk.base import BiskEnv
from bisk.single_robot import BiskSingleRobotEnv

for robot in ('', 'HalfCheetah', 'Walker', 'Humanoid', 'HumanoidPC'):
    register(
        id=f'BiskHurdles{robot}-v1',
        entry_point=f'bisk.tasks.hurdles:BiskHurdlesEnv',
        kwargs={
            'robot': robot,
            'features': 'joints',
            'max_height': 0.3,
            'fixed_height': False,
        },
        max_episode_steps=1000,
    )
    register(
        id=f'BiskLimbo{robot}-v1',
        entry_point=f'bisk.tasks.limbo:BiskLimboEnv',
        kwargs={
            'robot': robot,
            'features': 'joints',
            'notouch': False,
            'min_height': 'auto',
            'fixed_height': False,
        },
        max_episode_steps=1000,
    )
    register(
        id=f'BiskHurdlesLimbo{robot}-v1',
        entry_point=f'bisk.tasks.hurdleslimbo:BiskHurdlesLimboEnv',
        kwargs={
            'robot': robot,
            'features': 'joints',
            'notouch': False,
            'min_bar_height': 'auto',
            'max_hurdle_height': 0.3,
            'fixed_height': False,
        },
        max_episode_steps=1000,
    )
    register(
        id=f'BiskGaps{robot}-v1',
        entry_point=f'bisk.tasks.gaps:BiskGapsEnv',
        kwargs={
            'robot': robot,
            'features': 'joints',
            'max_size': 2.5,
            'min_gap': 0.2,
            'max_gap': 0.7,
            'fixed_size': False,
        },
        max_episode_steps=1000,
    )
    register(
        id=f'BiskStairs{robot}-v1',
        entry_point=f'bisk.tasks.stairs:BiskStairsEnv',
        kwargs={
            'robot': robot,
            'features': 'joints-relz',
            'step_height': 0.2,
            'step_length_min': 0.5,
            'step_length_max': 1.0,
        },
        max_episode_steps=1000,
    )
    register(
        id=f'BiskGoalWall{robot}-v1',
        entry_point=f'bisk.tasks.goalwall:BiskGoalWallEnv',
        kwargs={
            'robot': robot,
            'features': 'joints',
            'init_distance': 2.5,
            'touch_ball_reward': 0,
        },
        max_episode_steps=250,
    )
    register(
        id=f'BiskPoleBalance{robot}-v1',
        entry_point=f'bisk.tasks.polebalance:BiskPoleBalanceEnv',
        kwargs={
            'robot': robot,
            'features': 'joints',
            'pole_mass': 0.5,
            'pole_length': 0.5,
            'n_poles': 1,
        },
        max_episode_steps=1000,
    )
