# Copyright (c) 2021-present, Facebook, Inc.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
from bisect import bisect_left

import gym
import numpy as np
from dm_control import mjcf

from bisk.helpers import add_box, add_fwd_corridor
from bisk.single_robot import BiskSingleRobotEnv

log = logging.getLogger(__name__)


class BiskHurdlesEnv(BiskSingleRobotEnv):
    '''
    Jump over hurdles to progress in X-direction.
    '''

    def __init__(
        self,
        robot: str,
        features: str,
        max_height: float,
        fixed_height: bool,
    ):
        super().__init__(robot, features)
        self.max_height = max_height
        self.fixed_height = fixed_height
        self.max_hurdles_cleared = 0

        obs_base = self.featurizer.observation_space
        obs_env = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Dict(
            [
                ('next_hurdle', obs_env),
                ('observation', obs_base),
            ]
        )

    def init_sim(self, root: mjcf.RootElement, frameskip: int = 5):
        W = 8
        add_fwd_corridor(root, W)
        # 200 hurdles should be enough for everybody
        self.n_hurdles = 200
        for i in range(self.n_hurdles):
            b = add_box(
                root, f'hurdle-{i}', size=[0.05, W, 0.1], pos=[2, 0, 0.2]
            )

        super().init_sim(root, frameskip)

    def reset_state(self):
        super().reset_state()
        self.max_hurdles_cleared = 0
        xpos = 1
        intervals = self.np_random.uniform(3, 6, size=(self.n_hurdles,))
        if self.fixed_height:
            heights = np.zeros(self.n_hurdles) + self.max_height
        else:
            heights = self.np_random.uniform(
                0.1, self.max_height, size=(self.n_hurdles,)
            )
        self.hurdle_pos = []
        for i in range(self.n_hurdles):
            xpos += intervals[i]
            self.hurdle_pos.append(xpos)
            self.p.named.model.geom_size[f'hurdle-{i}'][2] = heights[i]
            self.p.named.model.geom_pos[f'hurdle-{i}'][0] = xpos
            self.p.named.model.geom_pos[f'hurdle-{i}'][2] = (
                heights[i] / 2 + 0.01
            )

    def get_observation(self):
        nh = self.next_hurdle_index()
        if nh < len(self.hurdle_pos):
            xpos = self.robot_pos[0]
            nm = self.p.named.model
            next_hurdle_d = nm.geom_pos[f'hurdle-{nh}'][0] - xpos
            next_hurdle_h = nm.geom_size[f'hurdle-{nh}'][2] * 2
        else:
            next_hurdle_d = 10.0
            next_hurdle_h = 0.1
        next_hurdle_cleared = nh < self.max_hurdles_cleared
        return {
            'observation': super().get_observation(),
            'next_hurdle': np.array(
                [next_hurdle_d, next_hurdle_h, not next_hurdle_cleared]
            ),
        }

    def next_hurdle_index(self):
        xpos = self.robot_pos[0]
        return bisect_left(self.hurdle_pos, xpos)

    def step_simulation(self):
        super().step_simulation()
        self.max_hurdles_cleared = max(
            self.max_hurdles_cleared, self.next_hurdle_index()
        )

    def step(self, action):
        mhbefore = self.max_hurdles_cleared
        obs, reward, done, info = super().step(action)

        score = 1 if self.max_hurdles_cleared > mhbefore else 0
        info['score'] = score
        reward = score

        if info.get('fell_over', False):
            done = True
            reward = -1
        return obs, reward, done, info
