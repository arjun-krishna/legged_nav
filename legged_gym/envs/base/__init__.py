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


base_map = {
    'default': (LeggedRobot, LeggedRobotCfg, LeggedRobotCfgPPO),
    'nav': (nav.LeggedRobot, nav.LeggedRobotCfg, nav.LeggedRobotCfgPPO),
    'relcmd': (relcmd.LeggedRobot, relcmd.LeggedRobotCfg, relcmd.LeggedRobotCfgPPO),
    'cmd_base': (cmd_base.LeggedRobot, cmd_base.LeggedRobotCfg, cmd_base.LeggedRobotCfgPPO),
}
