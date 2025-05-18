"""
Planner class
Implementation of A*
"""

import numpy as np
import heapq

from occupancy_grid import OccupancyGrid


class Planner:
    """Simple occupancy grid Planner"""

    def __init__(self, occupancy_grid: OccupancyGrid):
        self.grid = occupancy_grid


        # Origin of the odom frame in the map frame
        self.odom_pose_ref = np.array([0, 0, 0])

    def plan(self, start, goal):
        """
        Plan a path from start to goal
        start : [x, y] map coordinates
        goal : [x, y] map coordinates
        """
        
        def get_closest_obstacle(pose, K=8):
            # look in K evenly spaced directions around pose and return the distance to the closest obstacle
            least_dist = 1000
            for k in range(K):
                angle = 2 * np.pi * k / K
                dx = np.cos(angle)
                dy = np.sin(angle)
                for i in range(25):
                    x = int(round(pose[0] + dx * i))
                    y = int(round(pose[1] + dy * i))
                    if x < 0 or x >= self.grid.x_max_map or y < 0 or y >= self.grid.y_max_map:
                        break
                    if self.grid.occupancy_map[x][y] == 40:
                        least_dist = min(least_dist, i)
                        break 
            return -least_dist

        def heuristic(a, b):
            return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) + get_closest_obstacle(a) * 100
        
        start = (start[0], start[1])
        goal = (goal[0], goal[1])

        priority_queue = []
        heapq.heappush(priority_queue, (0, start))
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
        while priority_queue:
            u = heapq.heappop(priority_queue)
            if u[1] == goal:
                print("Goal reached")
                print("Path found")
                break
            if u[1] in visited:
                continue
            visited[u[1][0]][u[1][1]] = 1
            explored_x.append(self.grid.conv_map_to_world(u[1][0], u[1][1])[0])
            explored_y.append(self.grid.conv_map_to_world(u[1][0], u[1][1])[1])
            # self.grid.display_cv(pose, _goal, np.array((explored_x, explored_y)))
            
            dx = [-1, 0, 1, 0, 1, -1, 1, -1]
            dy = [0, -1, 0, 1, 1, -1, -1, 1]
            obs_threshold = -20
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
                        heapq.heappush(priority_queue, (dist[v] + heuristic(v, goal), v))
                        
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
        # self.grid.display_cv(pose, _goal, np.array((pathx, pathy)))
        # print(len(path))
        # input()
        return path

    def explore_frontiers(self):
        """ Frontier based exploration """
        goal = np.array([0, 0, 0])  # frontier to reach for exploration
        return goal
