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

from bisk.helpers import add_box, add_capsule, add_fwd_corridor
from bisk.single_robot import BiskSingleRobotEnv

log = logging.getLogger(__name__)


class BiskHurdlesLimboEnv(BiskSingleRobotEnv):
    '''
    Alternating hurdles and limbo bars.
    '''

    def __init__(
        self,
        robot: str,
        features: str,
        notouch: bool,
        min_bar_height: Union[float, str],
        max_hurdle_height: float,
        fixed_height: bool,
    ):
        super().__init__(robot, features)
        self.notouch = notouch
        self.fixed_height = fixed_height
        self.max_obstacles_cleared = 0

        if min_bar_height == 'auto':
            if self.robot.startswith('humanoid'):
                self.min_bar_height = 1.0
            elif self.robot.startswith('walker'):
                self.min_bar_height = 1.2
            else:
                self.min_bar_height = 1.0
        else:
            self.min_bar_height = float(min_bar_height)
        self.max_hurdle_height = max_hurdle_height

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
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Dict(
            [
                ('next_obstacle', obs_env),
                ('observation', obs_base),
            ]
        )

    def init_sim(self, root: mjcf.RootElement, frameskip: int = 5):
        W = 8
        add_fwd_corridor(root, W)
        # 200 obstacles should be enough for everybody
        self.n_obstacles = 200
        for i in range(self.n_obstacles):
            if i % 2 == 0:
                b = add_box(
                    root, f'hurdle-{i}', size=[0.05, W, 0.1], pos=[2, 0, 0.2]
                )
            else:
                b = add_capsule(
                    root,
                    f'bar-{i}',
                    fromto=[2.025, -W, 0.1, 2.025, W, 0.1],
                    size=[0.1],
                )

        super().init_sim(root, frameskip)

    def reset_state(self):
        super().reset_state()
        self.max_obstacles_cleared = 0
        xpos = 1
        intervals = self.np_random.uniform(3, 6, size=(self.n_obstacles,))
        assert self.n_obstacles % 2 == 0
        if self.fixed_height:
            bar_heights = np.zeros(self.n_obstacles // 2) + self.min_bar_height
            hurdle_heights = (
                np.zeros(self.n_obstacles // 2) + self.max_hurdle_height
            )
        else:
            bar_heights = self.np_random.uniform(
                self.min_bar_height,
                self.min_bar_height + 0.3,
                size=(self.n_obstacles // 2,),
            )
            hurdle_heights = self.np_random.uniform(
                0.1, self.max_hurdle_height, size=(self.n_obstacles // 2,)
            )
        self.obstacle_pos = []
        self.obstacle_type = []
        nm = self.p.named.model
        for i in range(self.n_obstacles):
            xpos += intervals[i]
            self.obstacle_pos.append(xpos)
            self.obstacle_type.append(i % 2)
            if i % 2 == 0:
                nm.geom_size[f'hurdle-{i}'][2] = hurdle_heights[i // 2]
                nm.geom_pos[f'hurdle-{i}'][0] = xpos
                nm.geom_pos[f'hurdle-{i}'][2] = (
                    hurdle_heights[i // 2] / 2 + 0.01
                )
            else:
                nm.geom_pos[f'bar-{i}'][0] = xpos
                nm.geom_pos[f'bar-{i}'][2] = (
                    bar_heights[i // 2] + nm.geom_size[f'bar-{i}'][0]
                )
                nm.geom_rgba[f'bar-{i}'] = [0.8, 0.9, 0.8, 1]
        self.bar_hit = [False] * self.n_obstacles
        self.new_bars_hit = set()

    def get_observation(self):
        no = self.next_obstacle_index()
        if no < len(self.obstacle_pos):
            next_obstacle_type = self.obstacle_type[no]
            xpos = self.robot_pos[0]
            nm = self.p.named.model
            if next_obstacle_type == 0:
                next_obstacle_d = nm.geom_pos[f'hurdle-{no}'][0] - xpos
                next_obstacle_h = nm.geom_pos[f'hurdle-{no}'][2] * 2
            else:
                next_obstacle_d = nm.geom_pos[f'bar-{no}'][0] - xpos
                next_obstacle_h = (
                    nm.geom_pos[f'bar-{no}'][2] + nm.geom_size[f'bar-{no}'][0]
                )
        else:
            next_obstacle_d = 10.0
            next_obstacle_h = 0.1
        next_obstacle_cleared = no < self.max_obstacles_cleared
        return {
            'observation': super().get_observation(),
            'next_obstacle': np.array(
                [
                    next_obstacle_type,
                    next_obstacle_d,
                    next_obstacle_h,
                    not next_obstacle_cleared,
                ]
            ),
        }

    def next_obstacle_index(self):
        xpos = self.robot_pos[0]
        return bisect_left(self.obstacle_pos, xpos)

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
        self.max_obstacles_cleared = max(
            self.max_obstacles_cleared, self.next_obstacle_index()
        )

    def step(self, action):
        self.new_bars_hit = set()
        mobefore = self.max_obstacles_cleared
        obs, reward, done, info = super().step(action)

        score = 1 if self.max_obstacles_cleared > mobefore else 0
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

        if info.get('fell_over', False):
            reward = -1
            done = True
        return obs, reward, done, info
