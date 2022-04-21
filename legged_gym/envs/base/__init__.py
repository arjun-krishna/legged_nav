import legged_gym.envs.base.relcmd as relcmd
import legged_gym.envs.base.nav as nav

from .legged_robot import LeggedRobot
from .legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

# EXPERIMENT
# 1) base design (no normalization, no history) + obs in robot FOP
# 2) (1) + action normalized (bias+scaling per joint)
# 3) (2) + history
# TODO: check (2) + observation normalization (joint pos -- centering wrapping?)
import legged_gym.envs.base.cmd_base as cmd_base # (1)
import legged_gym.envs.base.cmd_base_actnorm as cmd_base_actnorm # (2)

import legged_gym.envs.base.loco_lidar as loco_lidar
import legged_gym.envs.base.obs_aware_loco as obs_aware_loco

base_map = {
    'default': (LeggedRobot, LeggedRobotCfg, LeggedRobotCfgPPO),
    'nav': (nav.LeggedRobot, nav.LeggedRobotCfg, nav.LeggedRobotCfgPPO),
    'relcmd': (relcmd.LeggedRobot, relcmd.LeggedRobotCfg, relcmd.LeggedRobotCfgPPO),
    'cmd_base': (cmd_base.LeggedRobot, cmd_base.LeggedRobotCfg, cmd_base.LeggedRobotCfgPPO),
    'cmd_base_actnorm': (cmd_base_actnorm.LeggedRobot, cmd_base_actnorm.LeggedRobotCfg, cmd_base_actnorm.LeggedRobotCfgPPO),
    'loco_lidar': (loco_lidar.LeggedRobot, loco_lidar.LeggedRobotCfg, loco_lidar.LeggedRobotCfgPPO),
    'obs_aware_loco': (obs_aware_loco.LeggedRobot, obs_aware_loco.LeggedRobotCfg, obs_aware_loco.LeggedRobotCfgPPO)
}
