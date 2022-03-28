from typing import Tuple
import numpy as np
from numpy.random import choice
from scipy import interpolate
import torch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from legged_gym.envs.base.legged_robot_nav_config import LeggedRobotNavCfg

class ObstacleManager:
    """
    Obstacle spawn manager
    """
    def __init__(self, cfg: LeggedRobotNavCfg.obstacle, device: str) -> None:
        self.cfg = cfg
        self.device = device

        self.asset_shapes = torch.zeros((self.get_num_obstacles(), 3)).to(self.device)

    def _create_obstacle_asset(self, gym, sim, w_range, h_range, d_range):
        width = np.random.uniform(low=w_range[0], high=w_range[1])
        height = np.random.uniform(low=h_range[0], high=h_range[1])
        depth = np.random.uniform(low=d_range[0], high=d_range[1])

        asset_options = gymapi.AssetOptions()
        asset_options.density = 10000 # make it heavy

        box_asset = gym.create_box(sim, depth, width, height, asset_options)
        return box_asset, (depth, width, height)

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
                'curr_t': 0,
                'update_prob': 0.8, # probability of update on hitting refresh_s
            },
            'dynamic': {
                'refresh_s': None,
                'spawn_range': [1.0, 2.0],
                'v_range': [0.0, 0.0],
                'handles': [],
                'curr_t': 0,
                'update_prob': 0.8,
            },
        }
        idx = 0
        for _ in range(self.cfg.static.num):
            asset_handle, shape = self._create_obstacle_asset(
                gym, sim, self.cfg.static.width, self.cfg.static.height, self.cfg.static.depth)
            actor_handle = gym.create_actor(env_handle, asset_handle, start_pose, "static_obstacle", i, 2, 1)
            obstacles['static']['handles'].append(actor_handle)
            obstacles['static']['spawn_range'] = self.cfg.static.spawn_range
            self.asset_shapes[idx] = torch.tensor(shape).to(self.device)
            idx += 1
        for _ in range(self.cfg.magic_spawn.num):
            asset_handle, shape = self._create_obstacle_asset(
                gym, sim, self.cfg.magic_spawn.width, self.cfg.magic_spawn.height, self.cfg.magic_spawn.depth)
            actor_handle = gym.create_actor(env_handle, asset_handle, start_pose, "magic_spawn_obstacle", i, 2, 1)
            obstacles['magic_spawn']['handles'].append(actor_handle)
            obstacles['magic_spawn']['refresh_s'] = self.cfg.magic_spawn.refresh_s
            obstacles['magic_spawn']['spawn_range'] = self.cfg.magic_spawn.spawn_range  
            self.asset_shapes[idx] = torch.tensor(shape).to(self.device)
            idx += 1          
        for _ in range(self.cfg.dynamic.num):
            asset_handle, shape = self._create_obstacle_asset(
                gym, sim, self.cfg.dynamic.width, self.cfg.dynamic.height, self.cfg.dynamic.depth)
            actor_handle = gym.create_actor(env_handle, asset_handle, start_pose, "dynamic_obstacle", i, 2, 1)
            obstacles['dynamic']['handles'].append(actor_handle)
            obstacles['dynamic']['refresh_s'] = self.cfg.dynamic.refresh_s
            obstacles['dynamic']['spawn_range'] = self.cfg.dynamic.spawn_range
            self.asset_shapes[idx] = torch.tensor(shape).to(self.device)
            idx += 1
        return obstacles

    def reset_obstacles(self, obstacle_handles, env_ids):
        # env_id -> [(env_id+1):(env_id:1)+num_obstacles)
        # only static obstacles
        # dynamic, spawnable use refresh_t to update properties
        idxs = []
        reset_pos_vel = torch.zeros((len(env_ids)*self.cfg.static.num, 6), device=self.device) # pos + vel
        curr_id = 0
        for env_id in env_ids:
            offset = env_id * (1 + self.get_num_obstacles())
            spawn_range = obstacle_handles[env_id]['static']['spawn_range']
            reset_pos_vel[curr_id:curr_id+self.cfg.static.num,:2] = self.torch_rand_from_range(spawn_range[0], spawn_range[1], (self.cfg.static.num, 2))
            reset_pos_vel[curr_id:curr_id+self.cfg.static.num, 2] = self.asset_shapes[:self.cfg.static.num, 2]/2
            curr_id += self.cfg.static.num
            idxs.extend(list(range(offset+1,offset+1+self.cfg.static.num)))
        return idxs, reset_pos_vel

    def refresh_obstacles(self, obstacle_handles, env_ids, dt):
        # refresh actor positions for moving/dynamic obstacles
        idxs = []
        reset_pos_vel = torch.zeros((len(env_ids)*(self.cfg.magic_spawn.num+self.cfg.dynamic.num), 6), device=self.device)
        for env_id in env_ids:
            offset = env_id * (1 + self.get_num_obstacles())
            curr_t = obstacle_handles[env_id]['magic_spawn']['curr_t']
            curr_t = obstacle_handles[env_id]['magic_spawn']['curr_t'] = curr_t + dt
            if curr_t >= obstacle_handles[env_id]['magic_spawn']['refresh_s']:
                # TODO refresh logic
                obstacle_handles[env_id]['magic_spawn']['curr_t'] = 0       

    def get_num_obstacles(self):
        return self.cfg.static.num + self.cfg.magic_spawn.num + self.cfg.dynamic.num

    def torch_rand_from_range(self, a, b, shape):
        x = torch_rand_float(a, b, shape, device=self.device)
        y = torch_rand_float(-a, -b, shape, device=self.device)
        z = torch_rand_float(0, 1, shape, device=self.device)
        return torch.where(z < 0.5, x, y)