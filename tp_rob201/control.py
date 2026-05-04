""" A set of robotics control functions """
import random  
from turtle import delay
import numpy as np


def reactive_obst_avoid(lidar):
    """
    Simple obstacle avoidance
    lidar : placebot object with lidar data
    """
    # TODO for TP1

    laser_dist = lidar.get_sensor_values()
    #print(lidar.get_sensor_values().min())

    """
    # First way i implemented obstacle avoidance: if any of the laser distances is less than 10, we stop and rotate randomly
    if (laser_dist.min() < 12):  #Since laser_dist is an array, we check if any of the values is less than 10
        speed = 0
        rotation_speed = random.uniform(0.5, 1.0) * random.choice([-1, 1])
    else:
        speed = 0.3  # Reduzindo a velocidade para uma reação mais suave
        rotation_speed = 0

    command = {"forward": speed,
               "rotation": rotation_speed}
    
    return command

    """
    """
# Second way: See only the front Secteur

    laser_dist = lidar.get_sensor_values()

    # --- Nova Abordagem: Frente do robô está no meio do array ---
    num_rays = len(laser_dist) #mesure the number of rays in the lidar
    mid_point = num_rays // 2 #the middle index of the lidar rays, assuming the front is in the middle of the array (180 DEGRES)
    num_rays_side = 50 # (180 - 50) = 130 e (180 + 50) = 230, so we will look at the rays from index 130 to 230, which correspond to the front sector of the robot
    
    start_index = max(0, mid_point - num_rays_side) #130
    end_index = min(num_rays, mid_point + num_rays_side) #230
    front_sector = laser_dist[start_index:end_index] #Copy the desired elements of the lidar vector to the new vector (front sect)

    min_front_dist = front_sector.min() #Find the minimum distance in the front sector
    print(f"Distância Mínima Frontal: {min_front_dist:.2f}")

    SAFE_DISTANCE = 30

    if min_front_dist < SAFE_DISTANCE:
        speed = 0
        rotation_speed = random.uniform(0.7, 1.0) * random.choice([-1, 1])
    else:
        
        speed = 0.3
        rotation_speed = 0

    command = {"forward": speed,
               "rotation": rotation_speed}
    
    return command
    """

    #Third Way: Extension Possible: Change la direction base on the angle
    #Verify the direction whose objet is the closest, and turn in the opposite direction

    laser_dist = lidar.get_sensor_values()

    num_rays = len(laser_dist) #mesure the number of rays in the lidar
    mid_point = num_rays // 2 #the middle index of the lidar rays, assuming the front is in the middle of the array (180 DEGRES)
    num_rays_side = 50 # (180 - 50) = 130 e (180 + 50) = 230, so we will look at the rays from index 130 to 230, which correspond to the front sector of the robot
    
    start_index = max(0, mid_point - num_rays_side) #130
    end_index = min(num_rays, mid_point + num_rays_side) #230

    left_sector = laser_dist[start_index:mid_point] #Copy the desired elements of the lidar vector to the new vector (left sect)
    right_sector = laser_dist[mid_point:end_index] #Copy the desired elements of the lidar vector to the new vector (right sect)

    SAFE_DISTANCE = 30
    if ((left_sector.min() < right_sector.min()) and (left_sector.min() < SAFE_DISTANCE)): #If the closest object is on the left, turn right
        speed = 0
        rotation_speed = random.uniform(0.7, 1.0) * 1 #Turn right
    elif ((right_sector.min() < left_sector.min()) and (right_sector.min() < SAFE_DISTANCE)): #If the closest object is on the right, turn left
        speed = 0
        rotation_speed = random.uniform(0.7, 1.0) * -1 #Turn left
    else: #If there is no object in front, go forward
        speed = 0.3
        rotation_speed = 0

    command = {"forward": speed,
               "rotation": rotation_speed}

    return command


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
