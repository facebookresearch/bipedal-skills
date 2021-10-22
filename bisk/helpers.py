# Copyright (c) 2021-present, Facebook, Inc.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import os
from typing import Tuple, Iterable
import logging

import numpy as np
from dm_control import mjcf, mujoco
from dm_control.mujoco.wrapper.mjbindings import mjlib
from gym.utils import seeding

log = logging.getLogger(__name__)

FANCY_SKYBOX = False


def root_with_floor() -> mjcf.RootElement:
    '''
    Constructs a root element with the commonly used checkered floor.
    '''
    root = mjcf.RootElement()
    if FANCY_SKYBOX:
        root.asset.add(
            'texture',
            type='skybox',
            file=f'{asset_path()}/rainbow.png',
            gridsize=[3, 4],
            gridlayout='.U..LFRB.D..',
        )
    else:
        root.asset.add(
            'texture',
            type='skybox',
            builtin='gradient',
            width=800,
            height=800,
            rgb1=[0.3, 0.5, 0.7],
            rgb2=[0, 0, 0],
        )
    root.asset.add(
        'texture',
        name='tex_plane',
        builtin='checker',
        width=100,
        height=100,
        rgb1=[0.2, 0.3, 0.4],
        rgb2=[0.1, 0.15, 0.2],
        type='2d',
    )
    root.asset.add(
        'material',
        name='mat_plane',
        reflectance=0.5,
        shininess=1,
        specular=1,
        texrepeat=[1, 1],
        texuniform=True,
        texture='tex_plane',
    )
    root.worldbody.add(
        'geom',
        name='floor',
        type='plane',
        size=[100, 100, 100],
        rgba=[0.8, 0.9, 0.8, 1.0],
        conaffinity=1,
        condim=3,
        material='mat_plane',
        pos=[0, 0, 0],
    )
    root.worldbody.add(
        'light',
        cutoff=100,
        diffuse=[1, 1, 1],
        dir=[0, 0, -1.3],
        directional=True,
        exponent=1,
        pos=[0, 0, 1.3],
        specular=[0.1, 0.1, 0.1],
    )
    return root


def asset_path() -> str:
    return os.path.join(os.path.dirname(__file__), 'assets')


def add_robot(
    root: mjcf.RootElement, kind: str, name: str, xyoff=None
) -> Tuple[mjcf.Element, np.ndarray]:
    '''
    Add a robot to the root element.
    Returns the attachement frame the original position of the robot's torso.
    If the robot requires a fresh freejoint, it returns its original position
    (so that qpos can be initialized accordingly); otherwise, (0, 0, 0) is
    returned.
    '''
    rm = mjcf.from_path(f'{asset_path()}/{kind.lower()}.xml')
    rm.model = name
    torso = rm.find('body', 'torso')
    if torso is None:
        torso = rm.find('body', 'root')
    pos = torso.pos
    # Use a [0,0,0] torso pos when attaching the frame and rather set
    # the default qpos manually later. dm_control's attachment frame
    # logic (apparently?) resets the frame of reference of the freejoint.
    torso.pos = [0, 0, 0]
    if xyoff:
        pos[0] += xyoff[0]
        pos[1] += xyoff[1]
    root_joint = torso.find('joint', 'root')
    if root_joint and (
        root_joint.tag == 'freejoint' or root_joint.type == 'free'
    ):
        root_joint.remove()
        needs_freejoint = True
    else:
        needs_freejoint = False
    af = root.attach(rm)
    if needs_freejoint:
        af.add('freejoint')
    return af, pos


def add_box(
    root: mjcf.RootElement,
    name: str,
    size: Iterable[float],
    rgba: Iterable[float] = None,
    with_body: bool = False,
    **kwargs,
) -> mjcf.Element:
    if rgba is None:
        rgba = np.array([0.8, 0.9, 0.8, 1])
    body = root.worldbody
    if with_body:
        body = root.worldbody.add('body', name=name)
    box = body.add(
        'geom',
        type='box',
        name=name,
        condim=3,
        size=size,
        rgba=rgba,
        **kwargs,
    )
    return body if with_body else box


def add_capsule(
    root: mjcf.RootElement,
    name: str,
    rgba: Iterable[float] = None,
    with_body: bool = False,
    **kwargs,
) -> mjcf.Element:
    if rgba is None:
        rgba = np.array([0.8, 0.9, 0.8, 1])
    body = root.worldbody
    if with_body:
        body = root.worldbody.add('body', name=name)
    box = body.add(
        'geom',
        type='capsule',
        name=name,
        condim=3,
        rgba=rgba,
        **kwargs,
    )
    return body if with_body else box


def add_fwd_corridor(root: mjcf.RootElement, W=4):
    WH = 2
    wall_alpha = 0.0  # for debugging
    # Change rendering of floor to fit the intended path
    floor = root.find('geom', 'floor')
    floor.size = [100, W, 1]
    floor.pos = [100 - W, 0, 0]
    # Add border walls
    root.worldbody.add(
        'geom',
        type='plane',
        name='wall_left',
        xyaxes=[1, 0, 0, 0, 0, 1],
        size=[100, WH, 1],
        pos=[100 - W, W, WH],
        rgba=[0, 0, 0, wall_alpha],
    )
    root.worldbody.add(
        'geom',
        type='plane',
        name='wall_right',
        xyaxes=[-1, 0, 0, 0, 0, 1],
        size=[100, WH, 1],
        pos=[100 - W, -W, WH],
        rgba=[0, 0, 0, wall_alpha],
    )
    root.worldbody.add(
        'geom',
        type='plane',
        name='wall_back',
        xyaxes=[0, 1, 0, 0, 0, 1],
        size=[W, WH, 1],
        pos=[-4, 0, WH],
        rgba=[0, 0, 0, wall_alpha],
    )


# The ball element follows the element definitions in quadruped.xml from
# dm_control:
# https://github.com/deepmind/dm_control/blob/33cea51/dm_control/suite/quadruped.xml
def add_ball(root: mjcf.RootElement,
             name: str,
             size: float,
             mass: float,
             twod: bool = False,
             **kwargs) -> mjcf.Element:
    root.asset.add('texture',
                   name='ball',
                   builtin='checker',
                   mark='cross',
                   width=151,
                   height=151,
                   rgb1=[0.1, 0.1, 0.1],
                   rgb2=[0.9, 0.9, 0.9],
                   markrgb=[1, 1, 1])
    root.asset.add('material', name='ball', texture='ball')
    ball = root.worldbody.add('body', name=name, pos=[0, 0, 0])
    if twod:
        ball.add('joint',
                 name='ball-x',
                 type='slide',
                 damping=0,
                 axis=[1, 0, 0],
                 pos=[0, 0, 0],
                 range=[-1000, 1000])
        ball.add('joint',
                 name='ball-z',
                 type='slide',
                 damping=0,
                 axis=[0, 0, 1],
                 pos=[0, 0, 0],
                 range=[-1000, 1000])
        ball.add('joint',
                 name='ball-ry',
                 type='hinge',
                 damping=0,
                 axis=[0, 1, 0],
                 pos=[0, 0, 0],
                 range=[-np.pi, np.pi])
    else:
        ball.add('freejoint', name=name)
    ball.add('geom',
             type='sphere',
             name=name,
             size=[size],
             mass=mass,
             condim=6,
             friction=[0.7, 0.005, 0.005],
             solref=[-10000, -30],
             material='ball',
             priority=1)
    return ball
