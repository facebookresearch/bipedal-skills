# Copyright (c) 2021-present, Facebook, Inc.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
from typing import Dict, List

import gym
import numpy as np
from dm_control import mujoco
from dm_control.mujoco.wrapper.mjbindings import enums as mjenums
from dm_control.mujoco.wrapper.mjbindings import mjlib

log = logging.getLogger(__name__)


class Featurizer:
    n_qpos: Dict[int, int] = {  # qpos entries per joint type
        mjenums.mjtJoint.mjJNT_FREE: 7,
        mjenums.mjtJoint.mjJNT_BALL: 4,
        mjenums.mjtJoint.mjJNT_SLIDE: 1,
        mjenums.mjtJoint.mjJNT_HINGE: 1,
    }
    n_qvel: Dict[int, int] = {  # qvel entries per joint type
        mjenums.mjtJoint.mjJNT_FREE: 6,
        mjenums.mjtJoint.mjJNT_BALL: 3,
        mjenums.mjtJoint.mjJNT_SLIDE: 1,
        mjenums.mjtJoint.mjJNT_HINGE: 1,
    }

    def __init__(
        self,
        p: mujoco.Physics,
        robot: str,
        prefix: str = 'robot',
        exclude: str = None,
    ):
        self.p = p
        self.prefix = prefix
        self.observation_space: gym.spaces.Box = None

    def __call__(self) -> np.ndarray:
        raise NotImplementedError()

    def set_frame_of_reference(self):
        raise NotImplementedError()

    def feature_names(self) -> List[str]:
        raise NotImplementedError()

    def qpos_names(self) -> List[str]:
        names = ['' for i in range(len(self.p.data.qpos))]
        for jn in self.p.named.model.jnt_type.axes.row.names:
            typ = self.p.named.model.jnt_type[jn]
            adr = self.p.named.model.jnt_qposadr[jn]
            if typ == 0:
                names[adr + 0] = f'{jn}:px'
                names[adr + 1] = f'{jn}:py'
                names[adr + 2] = f'{jn}:pz'
                names[adr + 3] = f'{jn}:ow'
                names[adr + 4] = f'{jn}:ox'
                names[adr + 5] = f'{jn}:oy'
                names[adr + 6] = f'{jn}:oz'
            elif typ == 1:
                names[adr + 0] = f'{jn}:ow'
                names[adr + 1] = f'{jn}:ox'
                names[adr + 2] = f'{jn}:oy'
                names[adr + 3] = f'{jn}:oz'
            elif typ == 2 or typ == 3:
                names[adr] = f'{jn}:p'
            else:
                raise ValueError(f'Unknown joint type {typ}')
        return names

    def qvel_names(self) -> List[str]:
        names = ['' for i in range(len(self.p.data.qvel))]
        for jn in self.p.named.model.jnt_type.axes.row.names:
            typ = self.p.named.model.jnt_type[jn]
            adr = self.p.named.model.jnt_dofadr[jn]
            if typ == 0:
                names[adr + 0] = f'{jn}:lvx'
                names[adr + 1] = f'{jn}:lvy'
                names[adr + 2] = f'{jn}:lvz'
                names[adr + 3] = f'{jn}:avx'
                names[adr + 4] = f'{jn}:avy'
                names[adr + 5] = f'{jn}:avz'
            elif typ == 1:
                names[adr + 0] = f'{jn}:avx'
                names[adr + 1] = f'{jn}:avy'
                names[adr + 2] = f'{jn}:avz'
            elif typ == 2 or typ == 3:
                names[adr] = f'{jn}:v'
            else:
                raise ValueError(f'Unknown joint type {typ}')
        return names

    def cfrc_ext_names(self) -> List[List[str]]:
        names: List[List[str]] = []
        for cn in self.p.named.data.cfrc_ext.axes.row.names:
            names.append(
                [f'{cn}:c{n}' for n in ['rx', 'ry', 'rz', 'tx', 'ty', 'tz']]
            )
        return names

    def sensor_names(self) -> List[str]:
        names = ['' for i in range(len(self.p.data.sensordata))]
        for sn in self.p.named.model.sensor_adr.axes.row.names:
            typ = self.p.named.model.sensor_type[sn]
            adr = self.p.named.model.sensor_adr[sn]
            if typ == mjenums.mjtSensor.mjSENS_GYRO:
                feats = ['avx', 'avy', 'avz']
            elif (
                typ == mjenums.mjtSensor.mjSENS_VELOCIMETER
                or typ == mjenums.mjtSensor.mjSENS_SUBTREELINVEL
            ):
                feats = ['lvx', 'lvy', 'lvz']
            elif typ == mjenums.mjtSensor.mjSENS_ACCELEROMETER:
                feats = ['lax', 'lay', 'laz']
            elif (
                typ == mjenums.mjtSensor.mjSENS_FRAMEPOS
                or typ == mjenums.mjtSensor.mjSENS_SUBTREECOM
            ):
                feats = ['px', 'py', 'pz']
            elif typ == mjenums.mjtSensor.mjSENS_JOINTPOS:
                feats = ['']
            elif typ == mjenums.mjtSensor.mjSENS_JOINTVEL:
                feats = ['']
            elif typ == mjenums.mjtSensor.mjSENS_FORCE:
                feats = ['fx', 'fy', 'fz']
            elif typ == mjenums.mjtSensor.mjSENS_TORQUE:
                feats = ['tx', 'ty', 'tz']
            elif typ == mjenums.mjtSensor.mjSENS_RANGEFINDER:
                feats = ['d']
            elif typ == mjenums.mjtSensor.mjSENS_TOUCH:
                feats = ['f']
            else:
                raise ValueError(f'Unsupported sensor type: {typ}')
            for i, f in enumerate(feats):
                names[adr + i] = f'{sn}:{f}'
        return names
