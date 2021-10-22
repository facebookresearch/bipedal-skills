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


class BiskStairsEnv(BiskSingleRobotEnv):
    '''
    Go up and down a flight of fixed-height, varible-length stairs.
    '''

    def __init__(
        self,
        robot: str,
        features: str,
        step_height: float,
        step_length_min: float,
        step_length_max: float,
    ):
        self.step_height = step_height
        self.step_length_min = step_length_min
        self.step_length_max = step_length_max
        super().__init__(robot, features, allow_fallover=False)

        self.max_steps_cleared = 0

        obs_base = self.featurizer.observation_space
        obs_env = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Dict(
            [
                ('next_steps', obs_env),
                ('observation', obs_base),
            ]
        )

    def init_sim(self, root: mjcf.RootElement, frameskip: int = 5):
        W = 8
        add_fwd_corridor(root, W)
        self.n_steps = 10 * 2
        self.start_pos = 3.0
        self.top_width = 3.0
        color1 = [0.8, 0.9, 0.8, 1.0]
        color2 = [0.6, 0.6, 0.6, 1.0]
        length = 0.5

        xpos = self.start_pos + length / 2
        h2 = self.step_height / 2
        for i in range(self.n_steps // 2):
            add_box(
                root,
                f'step-{i}',
                size=[length / 2, W, h2],
                pos=[xpos, 0, h2],
                rgba=color1 if i % 2 == 0 else color2,
            )
            h2 += self.step_height / 2
            xpos += length
        h2 = self.n_steps // 2 * self.step_height / 2
        add_box(
            root,
            'top',
            size=[self.top_width, W, h2],
            pos=[xpos + self.top_width - length / 2, 0, h2],
            rgba=color1,
        )

        xpos += self.top_width * 2
        for i in range(self.n_steps // 2, self.n_steps):
            add_box(
                root,
                f'step-{i}',
                size=[length / 2, W, h2],
                pos=[xpos, 0, h2],
                rgba=color1 if i % 2 == 1 else color2,
            )
            h2 -= self.step_height / 2
            xpos += length

        root.worldbody.add(
            'camera',
            name='stairs_side',
            mode='trackcom',
            pos=[0, -6, 1],
            euler=[80, 0, 0],
        )

        super().init_sim(root, frameskip)

    def reset_state(self):
        super().reset_state()
        self.max_steps_cleared = 0

        self.step_pos: List[float] = []
        lengths = self.np_random.uniform(
            self.step_length_min, self.step_length_max, size=(self.n_steps,)
        )

        xpos = self.start_pos + lengths[0] / 2
        nm = self.p.named.model
        for i in range(self.n_steps // 2):
            nm.geom_size[f'step-{i}'][0] = lengths[i] / 2
            nm.geom_pos[f'step-{i}'][0] = xpos
            self.step_pos.append(xpos)
            xpos += lengths[i] / 2 + lengths[i + 1] / 2
        nm.geom_pos['top'][0] = (
            xpos + self.top_width - lengths[self.n_steps // 2] / 2
        )

        xpos += self.top_width * 2
        for i in range(self.n_steps // 2, self.n_steps):
            nm.geom_size[f'step-{i}'][0] = lengths[i] / 2
            nm.geom_pos[f'step-{i}'][0] = xpos
            self.step_pos.append(xpos)
            if i < self.n_steps - 1:
                xpos += lengths[i] / 2 + lengths[i + 1] / 2

    # Custom fall-over detection because we want to use the featurizer's
    # relative Z position.
    def fell_over(self) -> bool:
        if self.robot.startswith('humanoid'):
            zpos = self.featurizer.relz()
            return bool(zpos < 0.9)
        elif self.robot.startswith('halfcheetah'):
            # Orientation pointing upwards and body almost on the ground
            up = self.p.named.data.xmat['robot/torso', 'zz']
            zpos = self.featurizer.relz()
            if up < -0.8 and zpos < 0.12:
                return True
        elif self.robot.startswith('walker'):
            zpos = self.featurizer.relz()
            r = self.p.named.data.qpos['robot/rooty']
            if zpos < 0.9 or r < -1.4 or r > 1.4:
                return True
        return False

    def get_observation(self):
        ns = self.next_step_index()
        xpos = self.robot_pos[0]
        nm = self.p.named.model
        if ns < len(self.step_pos):
            next_step_d1 = nm.geom_pos[f'step-{ns}'][0] - xpos
            if ns + 1 < len(self.step_pos):
                next_step_d2 = nm.geom_pos[f'step-{(ns+1)}'][0] - xpos
            else:
                next_step_d2 = 10.0
        else:
            next_step_d1 = 10.0
            next_step_d2 = 20.0
        next_step_cleared = ns < self.max_steps_cleared
        return {
            'next_steps': np.array(
                [next_step_d1, next_step_d2, not next_step_cleared]
            ),
            'observation': super().get_observation(),
        }

    def next_step_index(self):
        xpos = self.robot_pos[0]
        return bisect_left(self.step_pos, xpos)

    def step_simulation(self):
        super().step_simulation()
        self.max_steps_cleared = max(
            self.max_steps_cleared, self.next_step_index()
        )

    def step(self, action):
        msbefore = self.max_steps_cleared
        obs, reward, done, info = super().step(action)

        score = 1 if self.max_steps_cleared > msbefore else 0
        info['score'] = score
        reward = score

        if info.get('fell_over', False):
            reward = -1
            done = True
        return obs, reward, done, info
