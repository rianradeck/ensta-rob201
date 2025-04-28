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

    def _score(self, lidar: Lidar, pose):
        """
        Computes the sum of log probabilities of laser end points in the map
        lidar : placebot object with lidar data
        pose : [x, y, theta] nparray, position of the robot to evaluate, in world coordinates
        """
        indexes = np.where(lidar.get_ray_angles() < lidar.max_range)
        angles = lidar.get_ray_angles()[indexes]
        ranges = lidar.get_sensor_values()[indexes]
        obstacles = self.polar_to_cartesian(
            ranges,
            angles, 
            pose
        )
        world_coords = self.grid.conv_world_to_map(obstacles[0], obstacles[1])
        # print(len(world_coords[0]))
        # print("Five first detected obstacles", world_coords[0][:5], world_coords[1][:5])
        # print("Total detected obstacles", len(world_coords[0]))
        
        # print("Five first map obstacles", np.where(self.grid.occupancy_map > 0)[0][:5], np.where(self.grid.occupancy_map > 0)[1][:5])
        # print("Total map obstacles", len(np.where(self.grid.occupancy_map > 0)[0]))
        # print(self.grid.occupancy_map[np.where(self.grid.occupancy_map > 0)[0], np.where(self.grid.occupancy_map > 0)[1]])

        # world_coords == self.grid.occupancy_map
        try:
            score = np.sum(np.clip(self.grid.occupancy_map[world_coords[0], world_coords[1]], 0, 1))
        except IndexError:
            # print("IndexError: world_coords", np.where(world_coords[0] > self.grid.x_max_map), np.where(world_coords[1] > self.grid.y_max_map))
            score = -np.inf
        # print("Score", score)
        # input()
        return score

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
        d0 = np.sqrt((x0_ref - x0) ** 2 + (y0_ref - y0) ** 2)
        alpha0 = np.arctan2(y0 - y0_ref, x0 - x0_ref)
        
        x = x0_ref + d0 * np.cos(t0_ref + alpha0)
        y = y0_ref + d0 * np.sin(t0_ref + alpha0)
        t = t0_ref + t0

        return np.array([x, y, t])

    def localise(self, lidar, raw_odom_pose):
        """
        Compute the robot position wrt the map, and updates the odometry reference
        lidar : placebot object with lidar data
        odom : [x, y, theta] nparray, raw odometry position
        """
        # TODO for TP4

        best_score = self._score(lidar, raw_odom_pose)
        N = 100
        cartesian_diff = 0.1
        polar_diff = np.pi / 360
        while N > 0:
            random_pose = np.array([raw_odom_pose[0] + np.random.uniform(-1, 1) * cartesian_diff,
                                    raw_odom_pose[1] + np.random.uniform(-1, 1) * cartesian_diff,
                                    raw_odom_pose[2] + np.random.uniform(-1, 1) * polar_diff])
            corrected_pose = self.get_corrected_pose(raw_odom_pose, random_pose)
            # print("Random corrected pose", corrected_pose)
            score = self._score(lidar, corrected_pose)
            if score > best_score:
                best_score = score
                self.odom_pose_ref = corrected_pose
            N -= 1
        print("Best score", best_score)
        return best_score

    def polar_to_cartesian(self, ranges, angles, pose):
        return np.array([pose[0] + ranges * np.cos(angles + pose[2]),
                         pose[1] + ranges * np.sin(angles + pose[2])])


    def update_map(self, lidar: Lidar, pose):
        """
        Bayesian map update with new observation
        lidar : placebot object with lidar data
        pose : [x, y, theta] nparray, corrected pose in world coordinates
        """
        # print(lidar.get_sensor_values(), pose)
        obstacles = self.polar_to_cartesian(
            lidar.get_sensor_values(), 
            lidar.get_ray_angles(), 
            pose
        )
        # print(lidar.get_ray_angles())
        
        # world_coords = self.grid.conv_world_to_map(obstacles[0], obstacles[1])
        thickness = 2
        for obstacle_x, obstacle_y in zip(obstacles[0], obstacles[1]):
            self.grid.add_value_along_line(pose[0], pose[1], obstacle_x, obstacle_y, -1)
            # self.grid.add_value_along_line(
            #     obstacle_x - thickness,
            #     obstacle_y - thickness,
            #     obstacle_x + thickness,
            #     obstacle_y + thickness,
            #     1
            # )
        self.grid.add_map_points(obstacles[0], obstacles[1], 2)
        
        self.grid.occupancy_map = np.clip(self.grid.occupancy_map, -40, 40)
        # TODO for TP3
    
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
