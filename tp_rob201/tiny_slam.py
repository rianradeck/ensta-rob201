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
        obstacles = (
            np.clip(obstacles[0], self.grid.x_min_world, self.grid.x_max_world),
            np.clip(obstacles[1], self.grid.y_min_world, self.grid.y_max_world)
        )
        
        map_coords = self.grid.conv_world_to_map(obstacles[0], obstacles[1])
        map_coords = (
            np.clip(map_coords[0], 0, self.grid.x_max_map - 1),
            np.clip(map_coords[1], 0, self.grid.y_max_map - 1)
        )
        
        # print(map_coords)
        # print(len(map_coords[0]))
        # print("Five first detected obstacles", map_coords[0][:5], map_coords[1][:5])
        # print("Total detected obstacles", len(map_coords[0]))
        
        # print("Five first map obstacles", np.where(self.grid.occupancy_map > 0)[0][:5], np.where(self.grid.occupancy_map > 0)[1][:5])
        # print("Total map obstacles", len(np.where(self.grid.occupancy_map > 0)[0]))
        # print(self.grid.occupancy_map[np.where(self.grid.occupancy_map > 0)[0], np.where(self.grid.occupancy_map > 0)[1]])

        # map_coords == self.grid.occupancy_map
        # try::
        # score = np.sum(np.clip(self.grid.occupancy_map[map_coords[0], map_coords[1]], 0, 1))
        score = np.sum(self.grid.occupancy_map[map_coords[0], map_coords[1]])
        # except IndexError:
        #     # print("IndexError: map_coords", np.where(map_coords[0] > self.grid.x_max_map), np.where(map_coords[1] > self.grid.y_max_map))
        #     score = -np.inf
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
        d0 = np.sqrt(x0 ** 2 + y0 ** 2)
        alpha0 = np.arctan2(y0, x0)
        
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

        best_score = -np.inf
        # for _ in range(10):
        # best_score = max(best_score, self._score(lidar, raw_odom_pose))
        N = 50 * 2
        cartesian_sd_x = 2
        cartesian_sd_y = 2
        polar_sd_theta = 0.3
        fov = .8

        best_angle = raw_odom_pose[2]
        for theta_diff in np.arange(-fov, fov, 200):
            random_rotation_pose = np.array([
                raw_odom_pose[0],
                raw_odom_pose[1],
                raw_odom_pose[2] + theta_diff
            ])
            corrected_pose = self.get_corrected_pose(raw_odom_pose, random_rotation_pose)
            # print("Random corrected pose", corrected_pose)
            score = self._score(lidar, corrected_pose)
            if score > best_score:
                best_score = score
                best_angle = random_rotation_pose[2]

        print("Best angle", best_angle, best_score)
        best_score = -np.inf
        for _ in range(N):
            random_pose = np.array([
                raw_odom_pose[0] + np.random.normal(0, np.sqrt(cartesian_sd_x)),
                raw_odom_pose[1] + np.random.normal(0, np.sqrt(cartesian_sd_y)),
                best_angle
            ])
            corrected_pose = self.get_corrected_pose(raw_odom_pose, random_pose)
            # print("Random corrected pose", corrected_pose)
            score = self._score(lidar, corrected_pose)
            if score > best_score:
                best_score = score
                self.odom_pose_ref = random_pose
        
        print("Best score", best_score, self.odom_pose_ref)
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
        obstacles = self.polar_to_cartesian(
            lidar.get_sensor_values(), 
            lidar.get_ray_angles(), 
            pose
        )
        
        thickness = 2
        jitter = 0.1
        
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

    def plan(self, start, goal, pose, _goal):
        """
        Plan a path from start to goal
        start : [x, y] map coordinates
        goal : [x, y] map coordinates
        """
        
        class pq:
            def __init__(self):
                self.queue = []
            
            def push(self, item):
                self.queue.append(item)
            
            def pop(self):
                return self.queue.pop(self.queue.index(min(self.queue, key=lambda x: x[0])))
            
            def is_empty(self):
                return len(self.queue) == 0
            
            def __len__(self):
                return len(self.queue)
        
        def heuristic(a, b):
            return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
        
        start = (start[0], start[1])
        goal = (goal[0], goal[1])
        
        priority_queue = pq()
        priority_queue.push((0, start))
        visited = []
        for _ in range(self.grid.x_max_map):
            aux = []
            for _ in range(self.grid.y_max_map):
                aux.append(0)
            visited.append(aux)
        
        came_from = {}
        from collections import defaultdict
        dist = defaultdict(lambda: np.inf)
        dist[start] = 0
        explored_x = []
        explored_y = []
        
        print(start, goal)
        while not priority_queue.is_empty():
            u = priority_queue.pop()
            if u[1] == goal:
                print("Goal reached")
                print("Path found")
                break
            if u[1] in visited:
                continue
            # print("Current node", u[1])
            visited[u[1][0]][u[1][1]] = 1
            explored_x.append(self.grid.conv_map_to_world(u[1][0], u[1][1])[0])
            explored_y.append(self.grid.conv_map_to_world(u[1][0], u[1][1])[1])
            # self.grid.display_cv(pose, _goal, np.array((explored_x, explored_y)))
            
            dx = [-1, 0, 1, 0, 1, -1, 1, -1]
            dy = [0, -1, 0, 1, 1, -1, -1, 1]
            obs_threshold = 0
            for i in range(8):
                v = (int(u[1][0] + dx[i]), int(u[1][1] + dy[i]))
                # print("Next node", v)
                # print(self.grid.occupancy_map[v[0]][v[1]])
                if v[0] < 0 or v[0] >= self.grid.x_max_map or v[1] < 0 or v[1] >= self.grid.y_max_map:
                    continue
                if self.grid.occupancy_map[v[0]][v[1]] > obs_threshold: # obstacle
                    continue
                if not visited[v[0]][v[1]]:
                    uv = 1 if (dx[i] == 0 or dy[i] == 0) else np.sqrt(2)
                    if dist[u[1]] + uv < dist[v]:
                        dist[v] = dist[u[1]] + uv
                        priority_queue.push((dist[v] + heuristic(v, goal), v))
                        
                        came_from[v] = u[1]
                
        path = []
        current = goal
        while current != start:
            path.append(self.grid.conv_map_to_world(current[0], current[1]))
            current = came_from[current]
        path.append(self.grid.conv_map_to_world(start[0], start[1]))
        pathx = []
        pathy = []
        for i in range(len(path)):
            pathx.append(path[i][0])
            pathy.append(path[i][1])
        path.reverse()
        self.grid.display_cv(pose, _goal, np.array((pathx, pathy)))
        print(len(path))
        input()
        return path