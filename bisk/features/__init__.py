# Copyright (c) 2021-present, Facebook, Inc.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

from dm_control import mujoco

from bisk.features.base import Featurizer

_registry = {}


def register_featurizer(name, cls):
    global _registry
    _registry[name] = cls


def make_featurizer(
    features: str,
    p: mujoco.Physics,
    robot: str,
    prefix: str = 'robot',
    *args,
    **kwargs,
) -> Featurizer:
    global _registry
    if features == 'joints':
        from bisk.features.joints import JointsFeaturizer

        return JointsFeaturizer(p, robot, prefix, *args, **kwargs)
    elif features == 'joints-relz':
        from bisk.features.joints import JointsRelZFeaturizer

        return JointsRelZFeaturizer(p, robot, prefix, *args, **kwargs)
    elif features in _registry:
        return _registry[features](p, robot, prefix, *args, **kwargs)
    else:
        raise ValueError(f'Unknown feature set {features}')
