"""
Robot controller definition
Complete controller including SLAM, planning, path following
"""
import numpy as np

from place_bot.entities.robot_abstract import RobotAbstract
from place_bot.entities.odometer import OdometerParams
from place_bot.entities.lidar import LidarParams

from tiny_slam import TinySlam

from control import potential_field_control, reactive_obst_avoid
from occupancy_grid import OccupancyGrid
from planner import Planner
import cv2

# Definition of our robot controller
class MyRobotSlam(RobotAbstract):
    """A robot controller including SLAM, path planning and path following"""

    def __init__(self,
                 lidar_params: LidarParams = LidarParams(),
                 odometer_params: OdometerParams = OdometerParams()):
        # Passing parameter to parent class
        super().__init__(should_display_lidar=False,
                         lidar_params=lidar_params,
                         odometer_params=odometer_params)

        # step counter to deal with init and display
        self.counter = 0

        # Init SLAM object
        # Here we cheat to get an occupancy grid size that's not too large, by using the
        # robot's starting position and the maximum map size that we shouldn't know.
        size_area = (1400, 1000)
        robot_position = (439.0, 195)
        self.original_pose = np.array([robot_position[0], robot_position[1], 0])
        self._original_pose = np.array([robot_position[0], robot_position[1], 0])
        self.occupancy_grid = OccupancyGrid(x_min=-(size_area[0] / 2 + robot_position[0]),
                                            x_max=size_area[0] / 2 - robot_position[0],
                                            y_min=-(size_area[1] / 2 + robot_position[1]),
                                            y_max=size_area[1] / 2 - robot_position[1],
                                            resolution=2)

        self.tiny_slam = TinySlam(self.occupancy_grid)
        self.planner = Planner(self.occupancy_grid)
        self.is_turning = 0
        self.odo_before_turning = None
        self.idle = 0
        self.stopped = 0
        self.goal_count = 0
        self.score_threshold = 11e3
        self.pathx = []
        self.pathy = []
        self.starting_coord = np.array([0, 0])

        self.goals_list = [
            [-200,-400,0],
            [-220,-260,0],
            [-500,-260,0],
            [-500, 0,0],
        ]


        # storage for pose after localization
        self.corrected_pose = np.array([0, 0, 0])

    def control(self):
        """
        Main control function executed at each time step
        """
        return self.control_tp2()

    def control_tp1(self):
        """
        Control function for TP1
        Control funtion with minimal random motion
        """
        self.tiny_slam.compute()

        # Compute new command speed to perform obstacle avoidance
        if not self.is_turning:
            command, turning = reactive_obst_avoid(self.lidar())
            self.is_turning = turning
            self.odo_before_turning = self.odometer_values()
        else:
            turning_factor = 0.2
            command = {"forward": 0,
               "rotation": self.is_turning * turning_factor}

            if np.abs((self.odometer_values()[2] - self.odo_before_turning[2] + np.pi) % (2 * np.pi) - np.pi) > np.pi / 4:
                self.is_turning = 0
                
        
        return command

    def control_tp2(self):
        """
        Control function for TP2
        Main control function with full SLAM, random exploration and path planning
        """
        
        self.counter += 1
        corrected_pose = self.tiny_slam.get_corrected_pose(self.odometer_values())
        true_pose = np.array((self.true_position()[0], self.true_position()[1], self.true_angle())) - self._original_pose
        
        print(self.counter)
        if self.stopped:
            self.goals_list = []
            for i in range(0, len(self.pathx), 30):
                self.goals_list.append((self.pathx[i], self.pathy[i]))
            self.goals_list.append(self.starting_coord)
            self.goal_count = 0
            self.stopped = 0
            self.counter = 0
            self.starting_coord = (corrected_pose[0], corrected_pose[1])
            self.original_pose = np.array((corrected_pose[0], corrected_pose[1], corrected_pose[2]))
            return {"forward": 0.0, "rotation": 0.0}

        goal = self.goals_list[self.goal_count]
        if self.counter < 50:
            self.tiny_slam.update_map(self.lidar(), corrected_pose)
            self.tiny_slam.grid.display_cv(corrected_pose, goal=goal)
            return { "forward": 0.0, "rotation": 0.0 }
        
        self.tiny_slam.localise(self.lidar(), self.odometer_values())
        corrected_pose = self.tiny_slam.get_corrected_pose(self.odometer_values())
        score = self.tiny_slam._score(self.lidar(), corrected_pose)

        if self.counter % 10 == 0:
            self.tiny_slam.grid.display_cv(corrected_pose, goal=goal)
        self.counter += 1
        
        print("corrected pose", corrected_pose)
        print("true pose", true_pose)
        print("score", score)

        if self.idle > 100:
            self.score_threshold *= 0.95
        actual_threshold = max(9e3, self.score_threshold * (1 + self.counter / 7000))
        print("actual threshold", actual_threshold)
        if score > actual_threshold:
            self.tiny_slam.update_map(self.lidar(), corrected_pose)
            self.idle = 0
        if score < actual_threshold:
            self.idle += 1
            return {"forward": 0.0, "rotation": 0.0}
        
        command = potential_field_control(self.lidar(), corrected_pose, goal, self.tiny_slam.grid)
        

        if command == {"forward": 0.0, "rotation": 0.0}:
            print(self.counter, self.idle)
            self.goal_count += 1
        if self.goal_count >= len(self.goals_list):
            self.tiny_slam.grid.display_cv(corrected_pose, self.starting_coord)
            map_pose = self.occupancy_grid.conv_world_to_map(corrected_pose[0], corrected_pose[1])
            original_pose = self.occupancy_grid.conv_world_to_map(
                self.starting_coord[0], self.starting_coord[1]
            )
            path = self.planner.plan(map_pose, original_pose)
            self.pathx = []
            self.pathy = []
            for i in range(len(path)):
                self.pathx.append(path[i][0])
                self.pathy.append(path[i][1])
            for _ in range(100): # only for video purpose
                self.tiny_slam.grid.display_cv(corrected_pose, self.starting_coord, np.array((self.pathx, self.pathy)))
            self.stopped = 1
            self.counter = 0
            print("stopped")
            # command = {"forward": 0.0, "rotation": 0.0}
        # self.stopped = 1
            
        return command

    def control_tp3(self):
        # self.map_pose = self.occupancy_grid.conv_world_to_map(self.odometer_values()[0], self.odometer_values()[1])
        odom_pose_ref = None
        corrected_pose = self.tiny_slam.get_corrected_pose(self.odometer_values(), odom_pose_ref)
        self.tiny_slam.update_map(self.lidar(), corrected_pose)
        
        if self.counter % 100 == 0:
            self.tiny_slam.grid.display_cv(corrected_pose)
        self.counter += 1
        
        return {
            "forward": 0.0,
            "rotation": 0.0
        }
