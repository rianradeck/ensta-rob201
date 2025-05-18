""" A simple robotics navigation code including SLAM, exploration, planning"""

import cv2
import numpy as np
from occupancy_grid import OccupancyGrid
from place_bot.entities.lidar import Lidar

class TinySlam:
    """Simple occupancy grid SLAM"""

    def __init__(self, occupancy_grid: OccupancyGrid):
        self.grid = occupancy_grid

        # Origin of the odom frame in the map frame
        self.odom_pose_ref = np.array([0, 0, 0])

    def get_obstacles(self, lidar: Lidar, pose):
        indexes = np.where(lidar.get_ray_angles() < lidar.max_range)
        angles = lidar.get_ray_angles()[indexes]
        ranges = lidar.get_sensor_values()[indexes]
        obstacles = self.polar_to_cartesian(
            ranges,
            angles, 
            pose
        )
        obstacles = (
            np.clip(obstacles[0], self.grid.x_min_world, self.grid.x_max_world),
            np.clip(obstacles[1], self.grid.y_min_world, self.grid.y_max_world)
        )
        
        map_coords = self.grid.conv_world_to_map(obstacles[0], obstacles[1])
        map_coords = (
            np.clip(map_coords[0], 0, self.grid.x_max_map - 1),
            np.clip(map_coords[1], 0, self.grid.y_max_map - 1)
        )
        return map_coords
        

    def get_corrected_pose(self, odom_pose, odom_pose_ref=None):
        """
        Compute corrected pose in map frame from raw odom pose + odom frame pose,
        either given as second param or using the ref from the object
        odom : raw odometry position
        odom_pose_ref : optional, origin of the odom frame if given,
                        use self.odom_pose_ref if not given
        """
        if odom_pose_ref is None:
            odom_pose_ref = self.odom_pose_ref
        
        x0, y0, t0 = odom_pose
        x0_ref, y0_ref, t0_ref = odom_pose_ref
        d0 = np.sqrt(x0 ** 2 + y0 ** 2)
        alpha0 = np.arctan2(y0, x0)
        
        x = x0_ref + d0 * np.cos(t0_ref + alpha0)
        y = y0_ref + d0 * np.sin(t0_ref + alpha0)
        t = t0_ref + t0

        return np.array([x, y, t])
    
    def _score(self, lidar, pose):
        obstacles_coords = self.get_obstacles(lidar, pose)
        return np.sum(self.grid.occupancy_map[obstacles_coords[0], obstacles_coords[1]])

    def __score(self, pose, true_pose):
        pos_weight = 1
        theta_weight = 100
        return -(
            pos_weight * np.abs(true_pose[0] - pose[0]) + 
            pos_weight * np.abs(true_pose[1] - pose[1]) + 
            theta_weight * np.abs(true_pose[2] - pose[2])
        )

    def localise(self, lidar, raw_odom_pose):
        """
        Compute the robot position wrt the map, and updates the odometry reference
        lidar : placebot object with lidar data
        odom : [x, y, theta] nparray, raw odometry position
        """

        best_score = -np.inf
        original_score = self._score(lidar, raw_odom_pose)
        original_ref = self.odom_pose_ref.copy()
        xy_var = 0.15
        xy_size = 300
        t_var = 0.01
        t_size = 200
        
        for dt in np.random.normal(0, t_var, t_size):
            pose_ref = np.array([
                self.odom_pose_ref[0],
                self.odom_pose_ref[1],
                self.odom_pose_ref[2] + dt
            ])
            new_corrected_pose = self.get_corrected_pose(raw_odom_pose, pose_ref)
            score = self._score(lidar, new_corrected_pose)
            if score > best_score:
                best_score = score
                self.odom_pose_ref = pose_ref
        # print("Best angle", self.odom_pose_ref[2])
        # print("Corrected pose (angle adjusted)", self.get_corrected_pose(raw_odom_pose, self.odom_pose_ref))
        
        for dx in np.random.normal(0, xy_var, xy_size):
            pose_ref = np.array([
                self.odom_pose_ref[0] + dx,
                self.odom_pose_ref[1],
                self.odom_pose_ref[2]
            ])
            new_corrected_pose = self.get_corrected_pose(raw_odom_pose, pose_ref)
            score = self._score(lidar, new_corrected_pose)
            if score > best_score:
                best_score = score
                self.odom_pose_ref = pose_ref
        for dy in np.random.normal(0, xy_var, xy_size):
            pose_ref = np.array([
                self.odom_pose_ref[0],
                self.odom_pose_ref[1] + dy,
                self.odom_pose_ref[2]
            ])
            new_corrected_pose = self.get_corrected_pose(raw_odom_pose, pose_ref)
            score = self._score(lidar, new_corrected_pose)
            if score > best_score:
                best_score = score
                self.odom_pose_ref = pose_ref
        # print("Best score", best_score)
        # print("Best pose ref", self.odom_pose_ref)
        # new_corrected_pose = self.get_corrected_pose(raw_odom_pose, self.odom_pose_ref)
        # print("Best corrected pose", new_corrected_pose)
        if best_score < original_score:
            self.odom_pose_ref = original_ref

    def polar_to_cartesian(self, ranges, angles, pose):
        return np.array([pose[0] + ranges * np.cos(angles + pose[2]),
                         pose[1] + ranges * np.sin(angles + pose[2])])

    def update_map(self, lidar: Lidar, pose):
        """
        Bayesian map update with new observation
        lidar : placebot object with lidar data
        pose : [x, y, theta] nparray, corrected pose in world coordinates
        """
        obstacles = self.polar_to_cartesian(
            lidar.get_sensor_values(), 
            lidar.get_ray_angles(), 
            pose
        )
        
        thickness = 3
        jitter = 1
        
        for obstacle_x, obstacle_y in zip(obstacles[0], obstacles[1]):
            self.grid.add_value_along_line(pose[0], pose[1], obstacle_x, obstacle_y, -1)
            
            for dx in np.arange(-jitter, jitter, 1):
                for dy in np.arange(-jitter, jitter, 1):
                    u_pose_obs_vec = obstacle_x + dx - pose[0], obstacle_y + dy - pose[1]
                    u_pose_obs_vec = u_pose_obs_vec / np.linalg.norm(u_pose_obs_vec)
                    
                    segment_start_x = obstacle_x + dx - u_pose_obs_vec[0] * thickness
                    segment_start_y = obstacle_y + dy - u_pose_obs_vec[1] * thickness
                    segment_end_x = obstacle_x + dx + u_pose_obs_vec[0] * thickness
                    segment_end_y = obstacle_y + dy + u_pose_obs_vec[1] * thickness
                    
                    self.grid.add_value_along_line(segment_start_x, segment_start_y, segment_end_x, segment_end_y, 1)
                    
        self.grid.add_map_points(obstacles[0], obstacles[1], 2)
        self.grid.occupancy_map = np.clip(self.grid.occupancy_map, -40, 40)
    
    def compute(self):
        """ Useless function, just for the exercise on using the profiler """
        # Remove after TP1

        ranges = np.random.rand(3600)
        ray_angles = np.arange(-np.pi, np.pi, np.pi / 1800)

        # Poor implementation of polar to cartesian conversion
        points = []
        for i in range(3600):
            pt_x = ranges[i] * np.cos(ray_angles[i])
            pt_y = ranges[i] * np.sin(ray_angles[i])
            points.append([pt_x, pt_y])
