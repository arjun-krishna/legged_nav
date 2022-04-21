from legged_gym.envs.base.obs_aware_loco.legged_robot_config import LeggedRobotCfg
import numpy as np
from numpy.random import choice
from scipy import interpolate

from isaacgym import terrain_utils

class ObstacleCourse:
    def __init__(self, cfg: LeggedRobotCfg.obstacle_course, num_robots) -> None:
        self.cfg = cfg
        self.num_robots = num_robots
        
        self.length = cfg.length
        self.width = cfg.width

        self.num_sub_courses = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))

        self.length_pixels = int(cfg.length / cfg.xy_scale)
        self.width_pixels = int(cfg.width / cfg.xy_scale)
        
        self.border = int(cfg.border_size / cfg.xy_scale)
        self.tot_cols = int(cfg.num_cols * self.width_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int8)
        if cfg.curriculum:
            raise NotImplementedError("implement curriculum creation")
            self.curiculum()
        else:
            self.randomized_course()
        
        self.heightsamples = self.height_field_raw
        
        self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(
            self.heightsamples,
            cfg.xy_scale,
            cfg.z_scale,
            0.75            # slope threshold isn't used
        )

    def randomized_course(self):
        for k in range(self.num_sub_courses):
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0, 1)
            difficulty = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0])
            course = self.make_course(choice, difficulty)
            self.add_course_to_map(course, i, j)
    
    def make_course(self, choice, difficulty):
        terrain = terrain_utils.SubTerrain(
            "terrain",
            width=self.width_pixels,
            length=self.length_pixels,
            vertical_scale=self.cfg.z_scale,
            horizontal_scale=self.cfg.xy_scale)
        num_rects = 20 + int(difficulty * 10)
        self.discrete_obstacles_terrain(
            terrain, 
            max_height=2,
            min_size=0.5,
            max_size=1.0,
            num_rects=num_rects,
            platform_size=3.0
        )
        return terrain

    def add_course_to_map(self, terrain, i, j):
        # map coordinate system
        start_x = self.border + i * self.length_pixels
        end_x = self.border + (i + 1) * self.length_pixels
        start_y = self.border + j * self.width_pixels
        end_y = self.border + (j + 1) * self.width_pixels
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

        env_origin_x = (i + 0.5) * self.length
        env_origin_y = (j + 0.5) * self.width
        x1 = int((self.length/2. - 1) / terrain.horizontal_scale)
        x2 = int((self.length/2. + 1) / terrain.horizontal_scale)
        y1 = int((self.width/2. - 1) / terrain.horizontal_scale)
        y2 = int((self.width/2. + 1) / terrain.horizontal_scale)
        env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*terrain.vertical_scale
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]

    def discrete_obstacles_terrain(self, terrain, max_height, min_size, max_size, num_rects, platform_size=1.):
        """
        Generate a terrain with gaps

        Parameters:
            terrain (terrain): the terrain
            max_height (float): maximum height of the obstacles (range=[-max, -max/2, max/2, max]) [meters]
            min_size (float): minimum size of a rectangle obstacle [meters]
            max_size (float): maximum size of a rectangle obstacle [meters]
            num_rects (int): number of randomly generated obstacles
            platform_size (float): size of the flat platform at the center of the terrain [meters]
        Returns:
            terrain (SubTerrain): update terrain
        """
        # switch parameters to discrete units
        max_height = int(max_height / terrain.vertical_scale)
        min_size = int(min_size / terrain.horizontal_scale)
        max_size = int(max_size / terrain.horizontal_scale)
        platform_size = int(platform_size / terrain.horizontal_scale)

        (i, j) = terrain.height_field_raw.shape
        height_range = [-max_height] #, -max_height // 2]
        width_range = range(min_size, max_size, 4)
        length_range = range(min_size, max_size, 4)

        for _ in range(num_rects):
            width = np.random.choice(width_range)
            length = np.random.choice(length_range)
            start_i = np.random.choice(range(0, i-width, 4))
            start_j = np.random.choice(range(0, j-length, 4))
            terrain.height_field_raw[start_i:start_i+width, start_j:start_j+length] = np.random.choice(height_range)

        x1 = (terrain.width - platform_size) // 2
        x2 = (terrain.width + platform_size) // 2
        y1 = (terrain.length - platform_size) // 2
        y2 = (terrain.length + platform_size) // 2
        terrain.height_field_raw[x1:x2, y1:y2] = 0
        return terrain

