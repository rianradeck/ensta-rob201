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
    

def tod(angle):
    """
    Convert angle from radian to degree
    """
    return angle * 180 / np.pi

def correct_precision(vec):
    a, b = np.round(vec, 2)
    a = -a if a == -0.0 else a
    b = -b if b == -0.0 else b
    return np.array([a, b])

def opposite_angle(angle):
    op = angle + np.pi
    if op > 2 * np.pi:
        op -= 2 * np.pi
    return op

def angle_difference(alpha, beta):
    angle_diff = np.abs(alpha - beta)
    if angle_diff > np.pi:
        angle_diff -= np.pi
    return angle_diff

def to_right(alpha, beta):
    """if beta is to right of alpha"""
    op_alpha = opposite_angle(alpha)
    if angle_difference(alpha, beta) < angle_difference(op_alpha, beta):
        return 1
    else:
        return -1

def polar_to_cartesian(ranges, angles, pose):
        return np.array([pose[0] + ranges * np.cos(angles + pose[2]),
                         pose[1] + ranges * np.sin(angles + pose[2])])

def potential_field_control(lidar: Lidar, current_pose, goal_pose, grid):
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
    
    
    vec_goal = goal_pose[:2]
    vec_rob = current_pose[:2]
    
    beta = current_pose[2]
    if beta < 0:
        beta += 2 * np.pi
    
    fov = 180
    front_point = 180
    sensor_dists = lidar.get_sensor_values()
    sensor_angles = lidar.get_ray_angles()
    obs_dist = np.min(sensor_dists[front_point - fov//2:front_point + fov//2])
    idx_obs = np.where(sensor_dists == obs_dist)[0][0]
    angle_obs = sensor_angles[idx_obs]
    if angle_obs < 0:
        angle_obs += 2 * np.pi
    # print(idx_obs, obs_dist, tod(angle_obs))
    # angle_obs = angle_difference(beta, angle_obs)
    # vec_obs = np.array([vec_rob[0] + obs_dist * np.cos(angle_obs), vec_rob[1] + obs_dist * np.sin(angle_obs)])
    vec_obs = polar_to_cartesian(obs_dist, angle_obs, current_pose)
    # grid.display_cv(current_pose, goal=vec_obs)
    # print("vec obs", vec_obs)
    # print("vec rob", vec_rob)
    vec_obs_to_rob = vec_obs - vec_rob
    # print("vec obs - rob", vec_obs_to_rob)
    # input()
    obs_weight = 5e2 / obs_dist
    if obs_dist > 100:
        obs_weight = 0
    
    vec_res = vec_goal - vec_rob - obs_weight * vec_obs_to_rob
    dist = np.linalg.norm(vec_res)
    u_vec_res = vec_res / dist
    u_vec_res = correct_precision(u_vec_res)
    alpha = np.arctan2(u_vec_res[1], u_vec_res[0])
    if alpha < 0:
        alpha += 2 * np.pi
    angle_diff = angle_difference(alpha, beta)
    # print("res:", u_vec_res)
    # print("beta:", tod(beta), "alpha:", tod(alpha))
    # print(tod(angle_diff))
    

    eps = 0.01
    rotation_factor = to_right(alpha, beta) * 0.2 / angle_diff
    # if angle_diff > np.pi:
    #     rotation_factor *= -1
    if np.abs(angle_diff) > np.pi / 2:
        return {"forward": 0,
               "rotation": angle_diff * rotation_factor}
    if np.linalg.norm(vec_rob - vec_goal) > 10:
        return {"forward": min(0.5, dist),
               "rotation": angle_diff * rotation_factor}

    print("goal reached")
    command = {"forward": 0,
               "rotation": 0}

    return command
