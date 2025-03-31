""" A set of robotics control functions """

import random
import numpy as np
from place_bot.entities.lidar import Lidar

def reactive_obst_avoid(lidar: Lidar) -> dict:
    """
    Simple obstacle avoidance
    lidar : placebot object with lidar data
    """
    # TODO for TP1

    laser_dist = lidar.get_sensor_values()
    # print(laser_dist.shape)
    speed = 0.0
    rotation_speed = 0.0
    
    turning = 0
    cone = 10
    front_dist = np.min(laser_dist[180-cone:180+cone])
    if laser_dist[180] < 50:
        left_laser = laser_dist[0]
        right_laser = laser_dist[360]
        if left_laser < right_laser:
            turning = 1
        else:
            turning = -1
        speed = 0.0
    else:
        # Forward motion
        speed = 1
        rotation_speed = 0.0
    
    turning_factor = .50
    speed_factor = .50
    command = {"forward": speed * speed_factor,
               "rotation": rotation_speed * turning_factor}

    return command, turning
    


def potential_field_control(lidar, current_pose, goal_pose):
    """
    Control using potential field for goal reaching and obstacle avoidance
    lidar : placebot object with lidar data
    current_pose : [x, y, theta] nparray, current pose in odom or world frame
    goal_pose : [x, y, theta] nparray, target pose in odom or world frame
    Notes: As lidar and odom are local only data, goal and gradient will be defined either in
    robot (x,y) frame (centered on robot, x forward, y on left) or in odom (centered / aligned
    on initial pose, x forward, y on left)
    """
    # TODO for TP2

    command = {"forward": 0,
               "rotation": 0}

    return command
