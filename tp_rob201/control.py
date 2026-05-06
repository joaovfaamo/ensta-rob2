""" A set of robotics control functions """
import random  
from turtle import delay, speed
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
    ...
    """
    
    kgoal = 0.5  # Gain for the attractive potential
    
    qgoal = np.array(goal_pose[:2])  # Eliminate the orientation component, only (x, y)
    qcurrent = np.array(current_pose[:2])  # Current position (x, y)
    distance = np.linalg.norm(qcurrent - qgoal) #Calculate the euclidean distance between the current position and the goal position

    finalstop_goal_angle =  goal_pose[2] 
    finalstop_current_angle = current_pose[2]
    finalstop_angle_diff = finalstop_goal_angle - finalstop_current_angle
    finalstop_angle_diff = (finalstop_angle_diff + np.pi) % (2 * np.pi) - np.pi # Normalize the angle difference to the range [-pi, pi]


    speed = 0
    rotation_speed = 0

    if distance > 2:
        gradient_attractive = (kgoal * (qgoal - qcurrent))/distance
        norme = np.linalg.norm(gradient_attractive)
        direction = gradient_attractive / norme# Gradient of the attractive potential
        
    else:
        if finalstop_angle_diff > 0.1: # If the robot is close to the goal but not well oriented, we rotate in place to correct the orientation
            speed = 0
            rotation_speed = 0.1
        else:
            gradient_attractive = np.array([0.0, 0.0])
            speed = 0
            rotation_speed = 0
        return {"forward": speed, "rotation": rotation_speed}
    
    #VERIFICAR ESSA PARTE 
    
    # Calcula o ângulo do vetor FORÇA no mundo (-pi a pi)
    angle_force = np.arctan2(direction[1], direction[0])  #(Y dps X)
    robot_theta = current_pose[2] # ORIEtacao atuald o robo

    # Calcula A DIFERENÇA entre onde eu quero ir e onde estou olhando (Angulo Objetivo)
    angle_diff = angle_force - robot_theta
    angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi

    speed = 0.4 * kgoal 
    rotation_speed = 0.5 * angle_diff  # Gira proporcionalmente ao ERRO de ângulo
    # SATURAÇÃO MÁXIMA/MÍNIMA para o motor do simulador [-1.0, 1.0]
    speed = float(np.clip(speed, -1.0, 1.0))
    rotation_speed = float(np.clip(rotation_speed, -1.0, 1.0))
    print(f"Alvo(Mundo): {angle_force:.2f}, Robô: {robot_theta:.2f}, Erro(Giro): {angle_diff:.2f}, distance: {distance:.2f}")
    

    #Conjuntos de Parametros Funcionais:
    #kgoal = 1.0, speed = 0.3, rotation_speed = 0.5 
    #kgoal = 0.5, speed = 0.2, rotation_speed = 0.3
    
    command = {"forward": speed,
               "rotation": rotation_speed}

    return command
