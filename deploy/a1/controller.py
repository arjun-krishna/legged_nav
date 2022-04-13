from absl import app
from absl import logging
from absl import flags

class Controller:

    FLAGS = flags.FLAGS

    def __init__(self):
        self._init_args()

    def _init_args(self):
        flags.DEFINE_boolean("rack", False, "robot will be put on a rack in sim")
        flags.DEFINE_float("kp", 40.0, "proportional/stiffness for joints position controller")
        flags.DEFINE_float("kd", 0.5, "derivative/damping for joints position controller")
        flags.DEFINE_float("dt", 0.01, "control time step")
        flags.DEFINE_float("ts", 0.001, "time between repeated commands for step calls")
        flags.DEFINE_float("nsteps", 200, "max steps to reach a particular joint position")
        flags.DEFINE_integer("repeat", 1, "times to repeat action/decimation")
    



    