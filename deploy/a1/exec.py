"""Script to run policy on sim and robot
"""
from absl import app
from absl import flags
from absl import logging

import numpy as np
import time

import pybullet as pb
import pybullet_data as pd
from pybullet_utils import bullet_client as bc

from motion_imitation.envs import env_builder
from motion_imitation.robots import a1
from motion_imitation.robots import robot_config

import torch
import argparse

from gym import error, spaces

SCALES = {
    'action': 0.25,
    'lin_vel': 2.0,
    'ang_vel': 0.25,
    'dof_pos': 1.0,
    'dof_vel': 0.05
}
CONTROL_TIME_STEP = 0.025
FREQ = 0.1
FLAGS = flags.FLAGS

flags.DEFINE_boolean("rack", False, "put robot on a rack")
flags.DEFINE_boolean("headless", False, "run script without rendering")

swap_idx = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
default_joint_pos = torch.tensor([0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5])

def quat_rotate_inverse(q, v):
    q_w = q[-1]
    q_vec = q[:3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(1, 1, 3), v.view(1, 3, 1)).squeeze(-1) * 2.0
    return a - b + c

def compute_observations(sim_env, commands, prev_a):
    sim_env.robot.GetTrueObservation()
    obs = torch.cat([
        (torch.tensor(sim_env.robot.GetBaseVelocity()) * SCALES['lin_vel']),
        (torch.tensor(sim_env.robot.GetBaseRollPitchYawRate()) * SCALES['ang_vel']),
        quat_rotate_inverse(
            torch.tensor(sim_env.robot.GetBaseOrientation()),
            torch.tensor([0, 0, -1.0])
            )[0],
        commands,
        ((torch.tensor(sim_env.robot.GetMotorAngles()[swap_idx]) - default_joint_pos) * SCALES['dof_pos']),
        (torch.tensor(sim_env.robot.GetMotorVelocities())[swap_idx] * SCALES['dof_vel']),
        prev_a,
    ], dim=0)
    return obs.to(torch.float)

def exec_policy(policy, obs):
    a = policy(obs) # a - FL, FR, RL, RR
    target = (SCALES['action'] * a) + default_joint_pos

    # rewire legs to FR, FL, RR, RL
    return target[swap_idx].detach().numpy(), a

def setup_commands(sim_env, vel_x, vel_y, yaw_rate):
    command_vel_x = sim_env.pybullet_client.addUserDebugParameter(
        paramName='vel_x',
        rangeMin=-1,
        rangeMax=1,
        startValue=0
    )
    command_vel_y = sim_env.pybullet_client.addUserDebugParameter(
        paramName='vel_y',
        rangeMin=-1,
        rangeMax=1,
        startValue=0
    )
    command_yaw_rate = sim_env.pybullet_client.addUserDebugParameter(
        paramName='yaw_rate',
        rangeMin=-1.5,
        rangeMax=1.5,
        startValue=0
    )
    return command_vel_x, command_vel_y, command_yaw_rate

FOLLOW = False

def main(_):
    logging.info("running pybullet")

    # motor controller gains
    # a1.ABDUCTION_P_GAIN = 40
    # a1.ABDUCTION_D_GAIN = 0.5
    # a1.HIP_P_GAIN = 40
    # a1.HIP_D_GAIN = 0.5
    # a1.KNEE_P_GAIN = 40
    # a1.KNEE_D_GAIN = 0.5

    sim_env = env_builder.build_regular_env(
        robot_class=a1.A1, 
        motor_control_mode=robot_config.MotorControlMode.POSITION,
        on_rack=FLAGS.rack,
        enable_rendering=(not FLAGS.headless),
        wrap_trajectory_generator=False)

    sim_env._gym_config.simulation_parameters.num_action_repeat = 4
    sim_env.pybullet_client.addUserDebugParameter( # TODO: why this is required?
        paramName='dummy',
        rangeMin=-0.1,
        rangeMax=0.1,
        startValue=0
    )
    command_vel_x, command_vel_y, command_yaw_rate = setup_commands(sim_env, 0, 0, 0)

    policy = torch.jit.load('policy/a1_cmdtracker.pt')
    policy.eval()

    def handle_key_events():
        global FOLLOW
        pressed_keys = []
        events = sim_env.pybullet_client.getKeyboardEvents()
        key_codes = events.keys()
        if 114 in key_codes: # r - reset
            sim_env.reset()
        if 102 in key_codes: # f - follow toggle
            FOLLOW = not FOLLOW
    
    prev_a = torch.zeros(12)
    for _ in range(10000):
        commands = torch.tensor([
            sim_env.pybullet_client.readUserDebugParameter(command_vel_x) * SCALES['lin_vel'],
            sim_env.pybullet_client.readUserDebugParameter(command_vel_y) * SCALES['lin_vel'],
            sim_env.pybullet_client.readUserDebugParameter(command_yaw_rate) * SCALES['ang_vel'],
        ])
        obs = compute_observations(sim_env, commands, prev_a)
        # print(obs)
        target, a = exec_policy(policy, obs)
        sim_env.step(target)
        prev_a = a.clone()
        handle_key_events()
        if FOLLOW:
            robot_xy = sim_env.robot.GetBasePosition()[:2]
            robot_yaw = sim_env.robot.GetTrueBaseRollPitchYaw()[2] * 180 / np.pi
            sim_env.pybullet_client.resetDebugVisualizerCamera(
                cameraDistance=3,
                cameraYaw=-90,          #robot_yaw-90 -- to track yaw
                cameraPitch=-50,
                cameraTargetPosition=(*robot_xy, 0)
            )
    sim_env.Terminate()


if __name__ == "__main__":
    app.run(main)