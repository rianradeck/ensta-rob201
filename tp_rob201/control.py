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
    
    vec_goal = goal_pose[:2]
    vec_rob = current_pose[:2]
    
    # Values from -pi to pi
    sensor_dists = lidar.get_sensor_values()
    sensor_angles = lidar.get_ray_angles()
    fov = 360
    sensor_angles = sensor_angles[180 - fov//2:180 + fov//2]
    sensor_dists = sensor_dists[180 - fov//2:180 + fov//2]

    obstacles_vectors = []
    aux_obstacles_vectors = []
    for dist, angle in zip(sensor_dists, sensor_angles):
        if dist > 100:
            continue
        obs_vec = polar_to_cartesian(dist, angle, current_pose)
        obstacles_vectors.append(obs_vec)
        unitary_vec = obs_vec - vec_rob
        unitary_vec /= np.linalg.norm(unitary_vec)
        aux_obstacles_vectors.append((dist, unitary_vec))

    obstacles_vectors = np.array(obstacles_vectors)

    obstacles_factor = 3e1
    goal_factor = 2e4
    res_vec = np.zeros(2)
    for vec in obstacles_vectors:
        too_close = 1
        if np.linalg.norm(vec - vec_rob) < 50:
            too_close = 10
        if np.linalg.norm(vec - vec_rob) < 20:
            too_close = 100
        res_vec -= too_close * obstacles_factor * (vec - vec_rob) / np.linalg.norm(vec - vec_rob)**2.5

    res_vec += goal_factor * (vec_goal - vec_rob) / np.linalg.norm(vec_goal - vec_rob)**2

    res_vec = correct_precision(res_vec)
    norm_res_vec = np.linalg.norm(res_vec)
    if norm_res_vec > 100: # limit res_vec norm
        res_vec = res_vec / norm_res_vec * 100
    print("res_vec", res_vec)
    to_draw = [(res_vec, (0, 255, 0))]
    top_obstacles_vectors = sorted(aux_obstacles_vectors, key=lambda x: x[0])
    top_k = 3
    for i in range(top_k):
        if i >= len(top_obstacles_vectors):
            break
        dist = top_obstacles_vectors[i][0]
        resized_vec = top_obstacles_vectors[i][1] * dist / 2
        to_draw.append((resized_vec, (255, 0, 255)))

    grid.display_cv(current_pose, goal_pose, vectors=to_draw)

    vec_angle = np.arctan2(res_vec[1], res_vec[0])
    robot_angle = current_pose[2] 

    vec_angle_deg = np.degrees(vec_angle)
    robot_angle_deg = np.degrees(robot_angle)

    angle_diff = vec_angle_deg - robot_angle_deg
    angle_diff = (angle_diff + 180) % 360 - 180

    # Threshold in degrees
    threshold = 20
    max_speed = .5
    norm_res_vec = np.linalg.norm(res_vec)
    dist_to_goal = np.linalg.norm(vec_goal - vec_rob)
    if dist_to_goal < 10:
        return {"forward": 0, "rotation": 0}
    if dist_to_goal < 150:
        threshold = 10
        max_speed = 0.3
    
    if abs(angle_diff) < threshold:
        forward = min(norm_res_vec * 0.03, max_speed)
    else:
        forward = 0.0
    
    # Rotation proportional to angle difference (scaled)
    rotation = np.clip(angle_diff / 90, -1, 1)

    command = {"forward": forward, "rotation": rotation}

    return command
