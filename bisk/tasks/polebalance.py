# Copyright (c) 2021-present, Facebook, Inc.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
from bisect import bisect_left
from typing import List

import gym
import numpy as np
from dm_control import mjcf
from dm_control.mujoco.wrapper.mjbindings import mjlib

from bisk.features import make_featurizer
from bisk.helpers import add_capsule
from bisk.single_robot import BiskSingleRobotEnv

log = logging.getLogger(__name__)


class BiskPoleBalanceEnv(BiskSingleRobotEnv):
    '''
    Classic pole balancing, but with robots. The pole is attached to a suitable
    point (top of the robot's torso or head) with 3 degrees of freedom for
    rotation. If its angle surpasses a threshold, the episode ends.
    If n_poles > 1, multiple poles will be stacked on top of each other, and
    each connection point will be again have 3 degrees of freedom.

    For 2D robots (HalfCheetah, Walker), the pole has just one degree of
    freedom (rotation around Y).
    '''

    def __init__(
        self,
        robot: str,
        features: str,
        pole_mass: float,
        pole_length: float,
        n_poles: int,
    ):
        self.pole_mass = pole_mass
        self.pole_length = pole_length
        self.n_poles = n_poles
        super().__init__(robot, features)

        self.pole_qpos_idx: List[int] = []
        self.pole_qvel_idx: List[int] = []
        if self.robot in {'halfcheetah', 'walker'}:
            for i in range(self.n_poles):
                qppos = self.p.named.model.jnt_qposadr[f'robot/pole-{i}_rot']
                self.pole_qpos_idx.append(qppos)
                qvpos = self.p.named.model.jnt_dofadr[f'robot/pole-{i}_rot']
                self.pole_qvel_idx.append(qvpos)
        else:
            for i in range(self.n_poles):
                for j in range(4):
                    qppos = (
                        j
                        + self.p.named.model.jnt_qposadr[f'robot/pole-{i}_rot']
                    )
                    self.pole_qpos_idx.append(qppos)
                for j in range(3):
                    qvpos = (
                        j + self.p.named.model.jnt_dofadr[f'robot/pole-{i}_rot']
                    )
                    self.pole_qvel_idx.append(qvpos)

        obs_base = self.featurizer.observation_space
        obs_env = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.pole_qpos_idx) + len(self.pole_qvel_idx),),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Dict(
            [
                ('pole', obs_env),
                ('observation', obs_base),
            ]
        )

        self.seed()

    def make_featurizer(self, features: str):
        return make_featurizer(
            features, self.p, self.robot, 'robot', exclude=r'robot/pole'
        )

    def init_sim(self, root: mjcf.RootElement, frameskip: int = 5):
        try:
            from matplotlib import pyplot as plt

            cmap = plt.get_cmap('rainbow')
        except:
            cmap = lambda x: [1, 0, 0, 1]

        size = 0.05
        if self.robot in {'humanoid', 'humanoidpc'}:
            size = 0.02
            head = root.find('body', 'robot/head')
            headg = head.find('geom', 'head')
            zpos = headg.size[0]
            pole = head.add('body', name='pole-0', pos=[0, 0, zpos])
        elif self.robot in {'halfcheetah'}:
            torso = root.find('body', 'robot/torso')
            pole = torso.add('body', name='pole-0', pos=[0, 0, 0])
        elif self.robot in {'walker'}:
            torso = root.find('body', 'robot/torso')
            torsog = torso.find('geom', 'torso')
            pole = torso.add('body', name='pole-0', pos=[0, 0, torsog.size[1]])
        else:
            try:
                torso = root.find('body', 'robot/torso')
                zpos = torso.find('geom', 'torso').size[2]
                pole = torso.add('body', name='pole-0', pos=[0, 0, zpos])
            except:
                raise NotImplementedError(
                    f'Don\'t know how to place poles on a {self.robot} robot'
                )

        if self.robot in {'halfcheetah', 'walker'}:
            # HalfCheetah model is defined in radians
            limit = np.pi if self.robot == 'halfcheetah' else 180
            pole.add(
                'joint',
                name='pole-0_rot',
                type='hinge',
                damping=0.1,
                stiffness=0,
                axis='0 1 0',
                pos=[0, 0, 0],
                range=[-limit, limit],
            )
        else:
            pole.add(
                'joint',
                name='pole-0_rot',
                damping=0.1,
                type='ball',
                pos=[0, 0, 0],
                range=[0, 90],
            )
        pole.add(
            'geom',
            name='pole-0_geom',
            type='capsule',
            fromto=[0, 0, 0, 0, 0, self.pole_length],
            size=[size],
            mass=self.pole_mass,
            rgba=cmap(0),
        )

        for i in range(1, self.n_poles):
            pole = pole.add(
                'body', name=f'pole-{i}', pos=[0, 0, self.pole_length]
            )
            if self.robot in {'halfcheetah', 'walker'}:
                limit = np.pi if self.robot == 'halfcheetah' else 180
                pole.add(
                    'joint',
                    name=f'pole-{i}_rot',
                    type='hinge',
                    damping=0.1,
                    stiffness=0,
                    axis='0 1 0',
                    pos=[0, 0, 0],
                    range=[-limit, limit],
                )
            else:
                pole.add(
                    'joint',
                    name=f'pole-{i}_rot',
                    type='ball',
                    damping=0.1,
                    pos=[0, 0, 0],
                    range=[0, 90],
                )
            pole.add(
                'geom',
                name=f'pole-{i}_geom',
                type='capsule',
                fromto=[0, 0, 0, 0, 0, self.pole_length],
                size=[size],
                mass=self.pole_mass,
                rgba=cmap((i + 1) / self.n_poles),
            )

        super().init_sim(root, frameskip)

    def get_observation(self):
        pole_qpos = self.p.data.qpos[self.pole_qpos_idx]
        pole_qvel = self.p.data.qpos[self.pole_qvel_idx]
        return {
            'observation': super().get_observation(),
            'pole': np.concatenate([pole_qpos, pole_qvel]).astype(np.float32),
        }

    def reset_state(self):
        super().reset_state()

        # Small noise for pole
        noise = 0.01
        qpos = self.init_qpos + self.np_random.uniform(
            low=-noise, high=noise, size=self.p.model.nq
        )
        qvel = self.init_qvel + noise * self.np_random.randn(self.p.model.nv)
        self.p.data.qpos[self.pole_qpos_idx] = qpos[self.pole_qpos_idx]
        self.p.data.qvel[self.pole_qvel_idx] = qvel[self.pole_qvel_idx]

    def step(self, action):
        obs, reward, done, info = super().step(action)
        reward = 1.0

        # Failure is defined as the z range of bottom and top of pole tower
        # falls below 20% of total length.
        xpos = self.p.named.data.xpos
        xquat = self.p.named.data.xquat
        t = np.zeros(3)
        mjlib.mju_rotVecQuat(
            t,
            np.array([0.0, 0.0, -self.pole_length / 2]),
            xquat['robot/pole-0'],
        )
        bottom_z = xpos['robot/pole-0'][2] + t[2]
        mjlib.mju_rotVecQuat(
            t,
            np.array([0.0, 0.0, self.pole_length / 2]),
            xquat[f'robot/pole-{self.n_poles-1}'],
        )
        top_z = xpos[f'robot/pole-{self.n_poles-1}'][2] + t[2]

        zthresh = 0.8 * self.n_poles * self.pole_length
        if top_z - bottom_z < zthresh:
            done = True
        score = 1 if not done else 0
        info['score'] = score
        reward = score

        if info.get('fell_over', False):
            done = True
            reward = -1
        return obs, reward, done, info
