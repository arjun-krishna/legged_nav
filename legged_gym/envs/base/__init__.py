import legged_gym.envs.base.relcmd as relcmd
import legged_gym.envs.base.nav as nav

from .legged_robot import LeggedRobot
from .legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

base_map = {
    'default': (LeggedRobot, LeggedRobotCfg, LeggedRobotCfgPPO),
    'nav': (nav.LeggedRobot, nav.LeggedRobotCfg, nav.LeggedRobotCfgPPO),
    'relcmd': (relcmd.LeggedRobot, relcmd.LeggedRobotCfg, relcmd.LeggedRobotCfgPPO),
}
