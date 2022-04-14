"""Script to target controller execution on simulation

TODO: allow robot testing
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

import argparse

from gym import error, spaces

CONTROL_TIME_STEP = 0.025
FREQ = 0.1
FLAGS = flags.FLAGS

def main(_):
    logging.info("running pybullet")

    a1.ABDUCTION_P_GAIN = 10

    sim_env = env_builder.build_regular_env(
        robot_class=a1.A1, 
        motor_control_mode=robot_config.MotorControlMode.POSITION,
        on_rack=True,
        enable_rendering=True,
        wrap_trajectory_generator=False)

    action_low, action_high = sim_env.action_space.low, sim_env.action_space.high
    dim_action = action_low.shape[0]
    action_selector_ids = []

    robot_motor_angles = sim_env.robot.GetMotorAngles()

    for dim in range(dim_action):
        action_selector_id = sim_env.pybullet_client.addUserDebugParameter(
            paramName=f'dim{dim}',
            rangeMin=action_low[dim],
            rangeMax=action_high[dim],
            startValue=robot_motor_angles[dim]
        )
        action_selector_ids.append(action_selector_id)
    
    for _ in range(10000):
        action = np.zeros(dim_action)
        for dim in range(dim_action):
            action[dim] = sim_env.pybullet_client.readUserDebugParameter(action_selector_ids[dim])
        sim_env.step(action)

    sim_env.Terminate()


if __name__ == "__main__":
    app.run(main)