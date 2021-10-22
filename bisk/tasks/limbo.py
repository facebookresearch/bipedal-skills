# Copyright (c) 2021-present, Facebook, Inc.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
from bisect import bisect_left
from typing import Dict, List, Union

import gym
import numpy as np
from dm_control import mjcf

from bisk.helpers import add_capsule, add_fwd_corridor
from bisk.single_robot import BiskSingleRobotEnv

log = logging.getLogger(__name__)


class BiskLimboEnv(BiskSingleRobotEnv):
    '''
    A limbo "dance" environment. There are bunch of geoms along the way which
    the robot has to crouch under. Proper limbo posture is not enforced.
    '''

    def __init__(
        self,
        robot: str,
        features: str,
        notouch: bool,
        min_height: Union[float, str],
        fixed_height: bool,
    ):
        super().__init__(robot, features)
        self.notouch = notouch
        self.fixed_height = fixed_height
        self.max_bars_cleared = 0

        if min_height == 'auto':
            if self.robot.startswith('humanoid'):
                self.min_height = 1.0
            elif self.robot.startswith('walker'):
                self.min_height = 1.2
            else:
                self.min_height = 1.0
        else:
            self.min_height = float(min_height)

        self.robot_geoms: List[int] = []
        for g in self.p.named.model.body_geomadr.axes.row.names:
            if g.startswith('robot/'):
                self.robot_geoms.append(self.p.named.model.body_geomadr[g])
        self.bar_geoms: Dict[int, int] = {}
        for i, g in enumerate(self.p.named.model.geom_bodyid.axes.row.names):
            if g.startswith('bar-'):
                self.bar_geoms[i] = int(g.split('-')[1])
        self.bar_geom_ids = set(self.bar_geoms.keys())

        obs_base = self.featurizer.observation_space
        obs_env = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Dict(
            [
                ('next_bar', obs_env),
                ('observation', obs_base),
            ]
        )

    def init_sim(self, root: mjcf.RootElement, frameskip: int = 5):
        W = 8
        add_fwd_corridor(root, W)
        # 200 bars should be enough for everybody
        self.n_bars = 200
        for i in range(self.n_bars):
            b = add_capsule(
                root,
                f'bar-{i}',
                fromto=[2.025, -W, 0.1, 2.025, W, 0.1],
                size=[0.1],
            )

        super().init_sim(root, frameskip)

    def reset_state(self):
        super().reset_state()
        self.max_bars_cleared = 0
        xpos = 1
        intervals = self.np_random.uniform(3, 6, size=(self.n_bars,))
        if self.fixed_height:
            heights = np.zeros(self.n_bars) + self.min_height
        else:
            heights = self.np_random.uniform(
                self.min_height, self.min_height + 0.3, size=(self.n_bars,)
            )
        self.bar_pos = []
        nm = self.p.named.model
        for i in range(self.n_bars):
            xpos += intervals[i]
            self.bar_pos.append(xpos)
            nm.geom_pos[f'bar-{i}'][0] = xpos
            nm.geom_pos[f'bar-{i}'][2] = (
                heights[i] + nm.geom_size[f'bar-{i}'][0]
            )
            nm.geom_rgba[f'bar-{i}'] = [0.8, 0.9, 0.8, 1]
        self.bar_hit = [False] * self.n_bars
        self.new_bars_hit = set()

    def get_observation(self):
        nb = self.next_bar_index()
        if nb < len(self.bar_pos):
            xpos = self.robot_pos[0]
            nm = self.p.named.model
            next_bar_d = nm.geom_pos[f'bar-{nb}'][0] - xpos
            next_bar_h = (
                nm.geom_pos[f'bar-{nb}'][2] + nm.geom_size[f'bar-{nb}'][0]
            )
        else:
            next_bar_d = 1.0
            next_bar_h = 2.0
        next_bar_cleared = nb < self.max_bars_cleared
        return {
            'observation': super().get_observation(),
            'next_bar': np.array(
                [next_bar_d, next_bar_h, not next_bar_cleared]
            ),
        }

    def next_bar_index(self):
        xpos = self.robot_pos[0]
        return bisect_left(self.bar_pos, xpos)

    def on_step_single_frame(self):
        contact = self.p.data.contact
        for i, c in enumerate(contact.geom1):
            if contact.dist[i] > 0:
                continue
            if c not in self.bar_geom_ids:
                continue
            bar = self.bar_geoms[c]
            self.new_bars_hit.add(bar)
        for i, c in enumerate(contact.geom2):
            if contact.dist[i] > 0:
                continue
            if c not in self.bar_geom_ids:
                continue
            bar = self.bar_geoms[c]
            self.new_bars_hit.add(bar)

    def step_simulation(self):
        super().step_simulation()
        self.max_bars_cleared = max(
            self.max_bars_cleared, self.next_bar_index()
        )

    def step(self, action):
        self.new_bars_hit = set()
        mbbefore = self.max_bars_cleared
        obs, reward, done, info = super().step(action)

        score = 1 if self.max_bars_cleared > mbbefore else 0
        touched = False
        for hit in self.new_bars_hit:
            if not self.bar_hit[hit] and self.notouch:
                touched = True
                if self.notouch:
                    marked = [0.8, 0.0, 0.0, 1.0]
                    self.p.named.model.geom_rgba[f'bar-{hit}'] = marked
                    score -= 1
            self.bar_hit[hit] = True
        info['score'] = score
        reward = score

        if not self.allow_fallover and self.fell_over():
            reward = -1
            done = True
        return obs, reward, done, info
