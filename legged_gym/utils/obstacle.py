from typing import Tuple
import numpy as np
from numpy.random import choice
from scipy import interpolate
import torch
from isaacgym import gymapi

from legged_gym.envs.base.legged_robot_nav_config import LeggedRobotNavCfg

class ObstacleManager:
    """
    Obstacle spawn manager
    """
    def __init__(self, cfg: LeggedRobotNavCfg.obstacle, device: str) -> None:
        self.cfg = cfg
        self.device = device

    def _create_obstacle_asset(self, gym, sim, w_range, h_range, d_range):
        width = np.random.uniform(low=w_range[0], high=w_range[1])
        height = np.random.uniform(low=h_range[0], high=h_range[1])
        depth = np.random.uniform(low=d_range[0], high=d_range[1])

        box_asset = gym.create_box(sim, width, height, depth)
        return box_asset

    def create_obstacles(self, gym, sim, env_handle, i):
        # i - collision group
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(0, 0, 50)
        obstacles = {
            'static': {
                'spawn_range': [1.0, 2.0],
                'handles': [],
            },
            'magic_spawn': {
                'refresh_s': None,
                'spawn_range': [1.0, 2.0],
                'handles': [],
            },
            'dynamic': {
                'refresh_s': None,
                'spawn_range': [1.0, 2.0],
                'v_range': [0.0, 0.0],
                'handles': [],
            },
        }
        for _ in range(self.cfg.static.num):
            asset_handle = self._create_obstacle_asset(
                gym, sim, self.cfg.static.width, self.cfg.static.height, self.cfg.static.depth)
            actor_handle = gym.create_actor(env_handle, asset_handle, start_pose, "static_obstacle", i, 2, 1)
            obstacles['static']['handles'].append(actor_handle)
            obstacles['static']['spawn_range'] = self.cfg.static.spawn_range
        for _ in range(self.cfg.magic_spawn.num):
            asset_handle = self._create_obstacle_asset(
                gym, sim, self.cfg.magic_spawn.width, self.cfg.magic_spawn.height, self.cfg.magic_spawn.depth)
            actor_handle = gym.create_actor(env_handle, asset_handle, start_pose, "magic_spawn_obstacle", i, 2, 1)
            obstacles['magic_spawn']['handles'].append(actor_handle)
            obstacles['magic_spawn']['refresh_s'] = self.cfg.magic_spawn.refresh_s
            obstacles['magic_spawn']['spawn_range'] = self.cfg.magic_spawn.spawn_range            
        for _ in range(self.cfg.dynamic.num):
            asset_handle = self._create_obstacle_asset(
                gym, sim, self.cfg.dynamic.width, self.cfg.dynamic.height, self.cfg.dynamic.depth)
            actor_handle = gym.create_actor(env_handle, asset_handle, start_pose, "dynamic_obstacle", i, 2, 1)
            obstacles['dynamic']['handles'].append(actor_handle)
            obstacles['dynamic']['refresh_s'] = self.cfg.dynamic.refresh_s
            obstacles['dynamic']['spawn_range'] = self.cfg.dynamic.spawn_range
        return obstacles

    def reset_obstacles(obstacle_handles, env_ids):
        # env_id -> [(env_id+1):(env_id:1)+num_obstacles)
        for env_id in env_ids:
            if len(info['static']['handles']): # handle static objects
                pass

    def refresh_obstacles(infos):
        # refresh actor positions for moving obstacles
        pass

    def get_num_obstacles(self):
        return self.cfg.static.num + self.cfg.magic_spawn.num + self.cfg.dynamic.num