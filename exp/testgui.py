# Copyright (c) 2021-present, Facebook, Inc.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import time

import gym
from dm_control import _render
from dm_control.viewer import gui, renderer, viewer, views

import bisk

parser = argparse.ArgumentParser()
parser.add_argument('task')
parser.add_argument('robot')
args = parser.parse_args()

env_name = {
    'hurdles': 'BiskHurdles-v1',
    'limbo': 'BiskLimbo-v1',
    'hurdleslimbo': 'BiskHurdlesLimbo-v1',
    'gaps': 'BiskGaps-v1',
    'stairs': 'BiskStairs-v1',
    'goalwall': 'BiskGoalWall-v1',
    'polebalance': 'BiskPoleBalance-v1',
}[args.task.lower()]

env = gym.make(env_name, robot=args.robot)
print(
    f'timestep {env.p.model.opt.timestep}s x frameskip {env.frameskip} = dt {env.dt}s'
)

width = 480
height = 480
title = f'{args.task} - {args.robot}'
render_surface = None
_MAX_FRONTBUFFER_SIZE = 2048
render_surface = _render.Renderer(
    max_width=_MAX_FRONTBUFFER_SIZE, max_height=_MAX_FRONTBUFFER_SIZE
)
ren = renderer.OffScreenRenderer(env.p.model, render_surface)
viewer_layout = views.ViewportLayout()
viewport = renderer.Viewport(width, height)
window = gui.RenderWindow(width, height, title)
vw = viewer.Viewer(viewport, window.mouse, window.keyboard)
ren.components += viewer_layout
vw.initialize(env.p, ren, touchpad=False)

env.seed(0)
step = 0


def tick():
    global step
    global obs
    if step == 0:
        obs = env.reset()
        #env.p.named.data.qvel['ball'][0:3] = [10, 3, 4]

    a = env.action_space.sample()
    a *= 0
    '''
    if step < 1:
        a[2] = 1
    elif step < 100:
        a[0] = 1
    else:
        a[2] = -1
    '''
    obs, r, d, i = env.step(a)
    step += 1
    if step > 200 or d:
        print(r)
        print(f'reset after {step} steps')
        step = 0
    time.sleep(0.05)
    vw.render()


def _tick():
    viewport.set_size(*window.shape)
    tick()
    return ren.pixels


window.event_loop(tick_func=_tick)
window.close()
