# Copyright (c) 2021-present, Facebook, Inc.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
import re
from typing import List, Set

import gym
import numpy as np
from dm_control import mujoco
from dm_control.mujoco.wrapper.mjbindings import mjlib

from bisk.features.base import Featurizer


class JointsFeaturizer(Featurizer):
    '''
    Featurizes joint observations (qpos, qvel) as well
    as contact forces (clipped to [-1,1]).
    '''

    def __init__(
        self, p: mujoco.Physics, robot: str, prefix: str, exclude: str = None
    ):
        super().__init__(p, robot, prefix, exclude)

        self.qpos_idx: List[int] = []
        self.qvel_idx: List[int] = []
        for jn in self.p.named.model.jnt_type.axes.row.names:
            if not jn.startswith(f'{self.prefix}/'):
                continue
            if exclude is not None and re.match(exclude, jn) is not None:
                continue
            typ = self.p.named.model.jnt_type[jn]
            qpos_adr = self.p.named.model.jnt_qposadr[jn]
            for i in range(self.n_qpos[typ]):
                self.qpos_idx.append(qpos_adr + i)
            qvel_adr = self.p.named.model.jnt_dofadr[jn]
            for i in range(self.n_qvel[typ]):
                self.qvel_idx.append(qvel_adr + i)

        self.cfrc_idx = [
            r
            for r, k in enumerate(self.p.named.data.cfrc_ext.axes.row.names)
            if k.startswith(f'{self.prefix}/')
            and k != f'{self.prefix}/'
            and (exclude is None or re.match(exclude, k) is None)
        ]
        n_obs = len(self.qpos_idx) + len(self.qvel_idx) + len(self.cfrc_idx) * 6
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float32
        )

    def __call__(self) -> np.ndarray:
        qpos = self.p.data.qpos[self.qpos_idx]
        qvel = self.p.data.qvel[self.qvel_idx]
        cfrc_ext = self.p.data.cfrc_ext[self.cfrc_idx]
        return np.concatenate(
            [qpos, qvel, np.clip(cfrc_ext.flat, -1, 1)]
        ).astype(np.float32)

    def feature_names(self) -> List[str]:
        names: List[str] = []
        qp = self.qpos_names()
        names += [qp[i] for i in self.qpos_idx]
        qv = self.qvel_names()
        names += [qv[i] for i in self.qvel_idx]
        cn = self.cfrc_ext_names()
        for i in self.cfrc_idx:
            names += cn[i]
        for i in range(len(names)):
            names[i] = names[i].replace(f'{self.prefix}/', '')
        return names


class JointsRelZFeaturizer(JointsFeaturizer):
    '''
    JointFeaturizer that reports the robots's Z position as relative to the
    closest surface underneath it.
    '''

    def __init__(
        self,
        p: mujoco.Physics,
        robot: str,
        prefix: str = 'robot',
        exclude: str = None,
    ):
        super().__init__(p, robot, prefix, exclude)

        self.robot_geoms: Set[int] = set()
        for i, p in enumerate(self.p.named.model.geom_bodyid.axes.row.names):
            if p.startswith(f'{self.prefix}/'):
                self.robot_geoms.add(i)

        # XXX Hacky lookup of z feature
        try:
            self.zpos_idx = self.feature_names().index(':pz')
        except:
            try:
                self.zpos_idx = self.feature_names().index('slidez:p')
            except:
                self.zpos_idx = self.feature_names().index('rootz:p')

    def relz(self):
        # Find closest non-robot geom from torso downwards
        pos = self.p.named.data.xpos[f'{self.prefix}/torso'].copy()
        dir = np.array([0.0, 0.0, -1.0])
        excl = self.p.named.model.geom_bodyid[f'{self.prefix}/torso']
        id = np.array([0], dtype=np.int32)
        while True:
            d = mjlib.mj_ray(
                self.p.model.ptr, self.p.data.ptr, pos, dir, None, 1, excl, id
            )
            if d < 0.0:  # No geom hit
                break
            pos += dir * d
            if id[0] not in self.robot_geoms:
                break
            excl = self.p.model.geom_bodyid[id[0]]
        return self.p.named.data.xpos[f'{self.prefix}/torso', 'z'] - pos[2]

    def __call__(self) -> np.ndarray:
        obs = super().__call__()
        rz = self.relz()
        obs[self.zpos_idx] = rz
        return obs
