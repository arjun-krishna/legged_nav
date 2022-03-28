import torch
import numpy as np

class Lidar2D:
    # ref: https://github.com/RansML/simulator_lidar/blob/master/run_simulation.py
    # vectorized
    # NOTE: init this class after obstacle_manager has created obstacles

    def __init__(self, num_envs, obstacle_manager, device, num_reflections: int = 180, fov: float = 180., max_dist: float = 12.0):
        self.num_envs = num_envs
        self.obstacle_manager = obstacle_manager
        self.num_obstacles = obstacle_manager.get_num_obstacles()
        self.device = device
        self.num_reflections = num_reflections
        self.fov = fov * np.pi / 180
        self.resolution = self.fov / self.num_reflections
        self.max_dist = max_dist

        self.obstacle_shapes = self.obstacle_manager.asset_shapes[:,:2].repeat((self.num_envs, 1, 1)) # (num_envs, num_obstacles, 2)


    def get_obstacle_segments(self, obstacle_xy): # ignore yaw (assume it doesn't turn)
        # input: (num_envs, num_obstacles, 2)
        # return: (num_envs, num_segments, 4)
        x, y = obstacle_xy[:,:,0], obstacle_xy[:,:,1]
        dx, dy = self.obstacle_shapes[:,:,0], self.obstacle_shapes[:,:,1]
        dx_c, dx_s = dx, torch.zeros_like(dx) # dx*np.cos(yaw_z), dx*np.sin(yaw_z)
        dy_c, dy_s = dy, torch.zeros_like(dy) # dy*np.cos(yaw_z), dy*np.sin(yaw_z)

        br_x = x + 0.5*(dx_c + dy_s) # (num_envs, num_obstacles)
        br_y = y + 0.5*(dx_s - dy_c)
        bl_x = x - 0.5*(dx_c - dy_s)
        bl_y = y - 0.5*(dx_s + dy_c)
        tl_x = x - 0.5*(dx_c + dy_s)
        tl_y = y - 0.5*(dx_s - dy_c)
        tr_x = x + 0.5*(dx_c - dy_s)
        tr_y = y + 0.5*(dx_s + dy_c)

        seg_bottom = torch.cat([bl_x[...,None], bl_y[...,None], br_x[...,None], br_y[...,None]], dim=-1) # (num_envs, num_obstacles, 4)
        seg_left = torch.cat([bl_x[...,None], bl_y[...,None], tl_x[...,None], tl_y[...,None]], dim=-1)
        seg_top = torch.cat([tl_x[...,None], tl_y[...,None], tr_x[...,None], tr_y[...,None]], dim=-1)
        seg_right = torch.cat([br_x[...,None], br_y[...,None], tr_x[...,None], tr_y[...,None]], dim=-1)

        return torch.cat([seg_bottom, seg_top, seg_left, seg_right], dim=1) # (num_envs, num_segments, 4)


    def yaw_from_quat(self, quat):
        x, y, z, w = quat[:,0], quat[:,1], quat[:,2], quat[:,3]
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        return torch.atan2(siny_cosp, cosy_cosp)


    def compute_intersection_distance(self, a, b):
        # a - (num_envs, num_reflections, num_segments, 4)
        # b - (num_envs, num_reflections, num_segments, 4)
        da = a[...,2:] - a[...,:2]
        db = b[...,2:] - b[...,:2]
        dp = a[...,:2] - b[...,:2]
        dap = torch.zeros_like(dp); dap[...,0] = -da[...,1]; dap[...,1] = da[...,0]
        denom = torch.sum(dap * db, dim=-1, keepdim=True)
        num = torch.sum(dap * dp, dim=-1, keepdim=True)
        eps = 1e-7
        delta = 1e-3
        intersct = b[...,:2] + (num / (denom + eps))*db

        condx_a = torch.logical_and((torch.min(a[...,[0,2]], -1)[0] - delta <= intersct[...,0]), (torch.max(a[...,[0,2]], -1)[0] + delta >= intersct[...,0]))
        condx_b = torch.logical_and((torch.min(b[...,[0,2]], -1)[0] - delta <= intersct[...,0]), (torch.max(b[...,[0,2]], -1)[0] + delta >= intersct[...,0]))
        condy_a = torch.logical_and((torch.min(a[...,[1,3]], -1)[0] - delta <= intersct[...,1]), (torch.max(a[...,[1,3]], -1)[0] + delta >= intersct[...,1]))
        condy_b = torch.logical_and((torch.min(b[...,[1,3]], -1)[0] - delta <= intersct[...,1]), (torch.max(b[...,[1,3]], -1)[0] + delta >= intersct[...,1]))
        return torch.where((condx_a & condx_b & condy_a & condy_b), torch.norm(intersct - a[...,:2], dim=-1), torch.tensor(self.max_dist).to(self.device))


    def lidar_angles(self, root_states):
        robot_xy, robot_quat = root_states[::(1+self.num_obstacles),:2], root_states[::(1+self.num_obstacles),3:7]
        robot_yaw = self.yaw_from_quat(robot_quat) # (num_envs,)

        angles = []
        for i in range(self.num_reflections):
            angles.append(robot_yaw + i*self.resolution)
        angles = torch.vstack(angles).t()  # (num_envs, num_reflections)
        return angles


    def simulate_lidar(self, root_states):
        # root_states - (pos, rot, vel, ang)
        dist = self.max_dist * torch.ones((self.num_envs, self.num_reflections), device=self.device)

        if self.num_obstacles == 0:
            return dist

        # robot_xy - (num_envs, 2)
        robot_xy = root_states[::(1+self.num_obstacles),:2]
        angles = self.lidar_angles(root_states)

        # laser_segments - (num_envs, num_reflections, 4)
        laser_segments = robot_xy.reshape((self.num_envs, 1, 2)).repeat((1, self.num_reflections, 2))
        laser_segments[:,:,2] += torch.cos(angles) * self.max_dist
        laser_segments[:,:,3] += torch.sin(angles) * self.max_dist

        all_idx = torch.arange(self.num_envs * (self.num_obstacles + 1))
        obstacle_idx = torch.nonzero(all_idx % (self.num_obstacles + 1)).flatten()
        # num_segments = num_obstacles * 4
        obstacle_xy = root_states[obstacle_idx,:2].view(self.num_envs, self.num_obstacles, 2)
        obstacle_segments = self.get_obstacle_segments(obstacle_xy) # (num_envs, num_segments, 4)

        # intersection computation => (num_envs, num_reflections, num_segments) [distance]
        scan_grid = torch.meshgrid(torch.arange(self.num_envs), torch.arange(self.num_reflections), torch.arange(self.num_obstacles * 4), indexing='ij')
        # (1, 5, 4)
        a = laser_segments[scan_grid[0], scan_grid[1]]
        b = obstacle_segments[scan_grid[0], scan_grid[2]]

        dist = self.compute_intersection_distance(a, b).min(dim=-1)[0] # (num_envs, num_reflections, num_segments)
        return dist
