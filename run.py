from typing import Dict
import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

import isaacgym
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.envs import *
from legged_gym.utils import task_registry
import torch
import numpy as np

from util import convert_hydra_cfg, get_args

OmegaConf.register_new_resolver('eq', lambda x, y: x.lower()==y.lower())
OmegaConf.register_new_resolver('contains', lambda x, y: x.lower() in y.lower())
OmegaConf.register_new_resolver('if', lambda pred, a, b: a if pred else b)
OmegaConf.register_new_resolver('arange', lambda start,end,delta: list(np.arange(start, end + 1e-3, delta)))
OmegaConf.register_new_resolver('resolve_default', lambda default, arg: default if arg is None else arg)
@hydra.main(config_name="config", config_path="conf/")
def run(cfg: DictConfig):
    env_cfg, train_cfg = convert_hydra_cfg(cfg)
    task_registry.register(cfg.task.name, LeggedRobot, env_cfg, train_cfg)
    args = get_args(cfg)
    env, env_cfg = task_registry.make_env(name=cfg.task.name, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=cfg.task.name, args=args)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    run()