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
from bisk.single_robot import BiskSingleRobotWithBallEnv

log = logging.getLogger(__name__)


class BiskGoalWallEnv(BiskSingleRobotWithBallEnv):
    '''
    Goal wall shooting. In the dense-reward setting we allow for falling over
    since the reward is the negative distance to the closest goal.
    '''
    def __init__(self,
                 robot: str,
                 features: str,
                 init_distance: float,
                 touch_ball_reward: float):
        self.init_distance = init_distance
        super().__init__(robot, features)
        self.touch_ball_reward = touch_ball_reward

        if self.touch_ball_reward > 0:
            self.observation_space = gym.spaces.Dict([
                ('ball', self.observation_space.spaces['ball']),
                ('touched_ball',
                 gym.spaces.Box(low=0, high=1, shape=(1, ), dtype=np.float32)),
                ('observation', self.observation_space.spaces['observation']),
            ])

        self.ball_geom = self.p.named.model.body_geomadr['ball']
        self.wall_geom = self.p.named.model.geom_type.axes.row.names.index(
            'wall')

    def init_sim(self, root: mjcf.RootElement, frameskip: int = 5):
        # Add wall
        W = 3
        WH = 1
        WD = 4 + self.init_distance
        root.asset.add('material',
                       name='mat_wall',
                       reflectance=0.5,
                       shininess=1,
                       emission=0.5,
                       specular=1)
        root.worldbody.add('geom',
                           type='plane',
                           name='wall',
                           material='mat_wall',
                           xyaxes=[0, -1, 0, 0, 0, 1],
                           size=[W, WH, 1],
                           pos=[WD, 0, WH],
                           rgba=[0, 0.5, 0.1, 1])

        # Add a visual marker
        root.asset.add('texture',
                       name='tex_dnc',
                       builtin='checker',
                       width=50,
                       height=50,
                       rgb1=[0, 0, 0],
                       rgb2=[1, 0.8, 0],
                       type='2d')
        root.asset.add('material',
                       name='mat_dnc',
                       reflectance=0.5,
                       shininess=1,
                       specular=1,
                       texrepeat=[1, 10],
                       texuniform=False,
                       texture='tex_dnc')
        root.worldbody.add('site',
                           type='box',
                           name='line',
                           size=[
                               0.1,
                               W,
                               0.01,
                           ],
                           pos=[1.5 + self.init_distance, 0, 0.02],
                           material='mat_dnc')
        #rgba=[1, 0, 0, 0.3])

        # Add goals on wall
        if self.is_2d:
            GP = WH
            root.worldbody.add('site',
                               type='ellipsoid',
                               name='goal',
                               material='mat_wall',
                               size=[
                                   0.01,
                                   0.4,
                                   0.4,
                               ],
                               pos=[WD, 0, GP],
                               rgba=[1, 1, 1, 1])
            root.worldbody.add('site',
                               type='ellipsoid',
                               name='goalb',
                               material='mat_wall',
                               size=[
                                   0.005,
                                   0.45,
                                   0.45,
                               ],
                               pos=[WD, 0, GP],
                               rgba=[1, 0, 0, 1])
        else:
            root.worldbody.add('site',
                               type='ellipsoid',
                               name='goal1',
                               material='mat_wall',
                               size=[
                                   0.01,
                                   0.4,
                                   0.4,
                               ],
                               pos=[WD, -1, WH - 0.35],
                               rgba=[1, 1, 1, 1])
            root.worldbody.add('site',
                               type='ellipsoid',
                               name='goal1b',
                               material='mat_wall',
                               size=[
                                   0.005,
                                   0.45,
                                   0.45,
                               ],
                               pos=[WD, -1, WH - 0.35],
                               rgba=[1, 0, 0, 1])
            root.worldbody.add('site',
                               type='ellipsoid',
                               name='goal2',
                               material='mat_wall',
                               size=[
                                   0.01,
                                   0.4,
                                   0.4,
                               ],
                               pos=[WD, 1, WH + 0.35],
                               rgba=[1, 1, 1, 1])
            root.worldbody.add('site',
                               type='ellipsoid',
                               name='goal2b',
                               material='mat_wall',
                               size=[
                                   0.005,
                                   0.45,
                                   0.45,
                               ],
                               pos=[WD, 1, WH + 0.35],
                               rgba=[1, 0, 0, 1])

        # This is the camera we'll use by default
        euler = [80, -5, 0]
        if root.compiler.angle == 'radian':
            euler = [np.deg2rad(e) for e in euler]
        root.worldbody.add('camera',
                           name='sideline',
                           mode='fixed',
                           pos=[WD / 3, -9, 2],
                           euler=euler)

        super().init_sim(root, frameskip)

    def get_observation(self):
        obs = super().get_observation()
        if self.touch_ball_reward > 0:
            obs['touched_ball'] = np.array([float(self.ball_touched)])
        return obs

    def reset_state(self) -> None:
        super().reset_state()

        # Place ball
        ball_size = self.p.named.model.geom_size['ball'][0]
        if self.is_2d:
            self.p.named.data.qpos['ball-x'] += self.init_distance
            self.p.named.data.qpos['ball-z'] += ball_size + 0.1
        else:
            self.p.named.data.qpos['ball'][0] += self.init_distance
            self.p.named.data.qpos['ball'][2] += ball_size + 0.1

        self.ball_yz = None
        self.ball_touched = False

    def on_step_single_frame(self):
        contact = self.p.data.contact
        ball_wall = (np.in1d(contact.geom1, self.wall_geom)
                     & np.in1d(contact.geom2, self.ball_geom))
        touching = contact.dist <= 0
        if np.any(ball_wall & touching):
            if self.is_2d:
                self.ball_yz = [0, self.p.named.data.qpos['ball-z'][0]]
            else:
                self.ball_yz = self.p.named.data.qpos['ball'][1:3].copy()

        if not self.ball_touched:
            for c in contact:
                names = self.p.named.model.name_geomadr.axes.row.names
                if names[c.geom1].startswith('ball') and names[
                        c.geom2].startswith('robot') and c.dist < 0:
                    self.ball_touched = True

    def step(self, action):
        self.ball_yz = None
        btbefore = self.ball_touched
        obs, reward, done, info = super().step(action)

        goal_hit = None
        goal_dists = []
        goal_sizes = []
        if self.ball_yz is not None:
            if self.is_2d:
                goals = ('goal', )
            else:
                goals = ('goal1', 'goal2')
            for g in goals:
                d = np.linalg.norm(self.ball_yz -
                                   self.p.named.data.site_xpos[g][1:3])
                goal_dists.append(d)
                goal_sizes.append(self.p.named.model.site_size[g][2])
                if d <= self.p.named.model.site_size[g][2]:
                    goal_hit = g
                    break

        score = 0
        if goal_hit == 'goal' or goal_hit == 'goal1':
            score = 1
        elif goal_hit == 'goal2':
            score = 2
        info['score'] = score
        reward = score
        if self.touch_ball_reward > 0 and self.ball_touched != btbefore:
            reward += self.touch_ball_reward

        # Zero reward if we're beyond the line
        lpos = self.p.named.data.site_xpos['line', 'x']
        if self.robot_pos[0] > lpos:
            reward = 0

        # Once we've hit the wall we're done
        if self.ball_yz is not None:
            done = True

        if info.get('fell_over', False):
            reward = -1
            done = True
        return obs, reward, done, info
