# Copyright (c) 2021-present, Facebook, Inc.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import logging

import gym
import numpy as np
from dm_control import mjcf, mujoco
from dm_control.mujoco.wrapper.mjbindings import mjlib
from gym.utils import seeding

log = logging.getLogger(__name__)


class BiskEnv(gym.Env):
    metadata = {'render.modes': ['rgb_array']}

    def __init__(self):
        # This is a no-op; run init_sim() with a root element to set up the
        # environment.
        self.p = None

    def init_sim(self, root: mjcf.RootElement, frameskip: int = 5):
        if self.p is not None:
            raise RuntimeError('Simulation already initialized')
        self.p = mjcf.Physics.from_mjcf_model(root)
        self.model = root
        self.frameskip = frameskip
        self.post_init()

    def post_init(self):
        self.init_qpos = self.p.data.qpos.ravel().copy()
        self.init_qvel = self.p.data.qvel.ravel().copy()

        # Expose all actuators
        self.action_space = gym.spaces.Box(
            self.p.model.actuator_ctrlrange[:, 0].astype(np.float32),
            self.p.model.actuator_ctrlrange[:, 1].astype(np.float32),
            dtype=np.float32,
        )
        # Leave observation space undefined in the base environment

    def seed(self, seed=None):
        self.np_random, sd = seeding.np_random(seed)
        if self.action_space is not None:
            self.action_space.seed(seed)
        if self.observation_space is not None:
            self.observation_space.seed(seed)
        return sd

    @property
    def dt(self):
        return self.p.model.opt.timestep * self.frameskip

    def reset_state(self):
        pass

    def get_observation(self):
        raise NotImplementedError()

    def reset(self):
        # Disable contacts during reset to prevent potentially large contact
        # forces that can be applied during initial positioning of bodies in
        # reset_state().
        with self.p.model.disable('contact'):
            self.p.reset()
            self.reset_state()
        self.step_simulation()
        return self.get_observation()

    def render(self, mode='rgb_array', **kwargs):
        width = kwargs.get('width', 480)
        height = kwargs.get('height', 480)
        camera = kwargs.get('camera', 0)
        flags = kwargs.get('flags', {})
        return self.p.render(
            width=width,
            height=height,
            camera_id=camera,
            render_flag_overrides=flags,
        )

    def apply_action(self, action):
        self.p.set_control(action)

    def on_step_single_frame(self):
        pass

    def step_simulation(self):
        for _ in range(self.frameskip):
            self.p.step()
            self.on_step_single_frame()
        # Call mj_rnePostConstraint to populate cfrc_ext (not done automatically
        # in MuJoCo 2.0 unless the model defines the proper sensors)
        mjlib.mj_rnePostConstraint(self.p.model.ptr, self.p.data.ptr)

    def step(self, action):
        self.apply_action(action)
        self.step_simulation()
        return self.get_observation(), 0, False, {}
