from typing import Tuple, Dict
from omegaconf import OmegaConf, DictConfig
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from legged_gym.utils.helpers import class_to_dict, update_class_from_dict
import argparse

from isaacgym.gymutil import parse_device_str
from isaacgym import gymapi

def convert_hydra_cfg(cfg: DictConfig) -> Tuple[LeggedRobotCfg, LeggedRobotCfgPPO]:
    env_cfg = LeggedRobotCfg()
    update_class_from_dict(env_cfg, OmegaConf.to_object(cfg.task))
    train_cfg = LeggedRobotCfgPPO()
    update_class_from_dict(train_cfg, OmegaConf.to_object(cfg.train))
    return env_cfg, train_cfg
    
def get_args(cfg: DictConfig):
    # TODO improve validation like in gymutil.parse_arguments
    default_args = {
        'task': 'A1Flat',
        'resume': False,
        'experiment_name': None,
        'run_name': None,
        'load_run': -1,
        'checkpoint': -1,
        'horovod': False,
        'headless': False,
        'rl_device': 'cuda:0',
        'num_envs': None,
        'seed': 1,
        'max_iterations': None,
        'physics_engine': 'physx',
        'sim_device': 'cuda:0',
        'pipeline': 'gpu',
        'graphics_device_id': 0,
        'num_threads': 0,
        'subscenes': 0,
        'slices': None
    }

    args = argparse.Namespace()
    args_d = vars(args)
    for key in default_args:
        if key in cfg:
            args_d[key] = cfg[key]
        else:
            args_d[key] = default_args[key]

    args.sim_device_type, args.compute_device_id = parse_device_str(args.sim_device)
    pipeline = args.pipeline.lower()
    args.use_gpu_pipeline = (pipeline in ('gpu', 'cuda'))

    args.physics_engine = gymapi.SIM_PHYSX if args.physics_engine == 'physx' else gymapi.SIM_FLEX
    args.use_gpu = (args.sim_device_type == 'cuda')

    if args.slices is None:
        args.slices = args.subscenes
    return args
