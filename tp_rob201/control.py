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
    #Implementacao Gradiente Atrativo
    kgoal = 0.8  # Gain for the attractive potential
    qgoal = np.array(goal_pose[:2])  # Eliminate the orientation component, only (x, y)
    qcurrent = np.array(current_pose[:2])  # Current position (x, y)
    distance = np.linalg.norm(qcurrent - qgoal) #Calculate the euclidean distance between the current position and the goal position

    #Implementacao Gradiente Repulsivo 
    kobstacle = 30  # Reduzido de 100 para 30 para o "empurrão" ser muito mais suave e não derrapar
    safe_distance = 25 # Aumentado de 20 para 25 para a repulsão começar mais longe, mas ser gentil
    #Como a ideia é ter varias forças de repulsão, uma para cada obstaculo, precisamos calcular o gradiente de cada um deles e somar as forças de repulsão
    laser_dist = lidar.get_sensor_values()
    laser_angles = np.linspace(-np.pi, np.pi, len(laser_dist))  # Gera um vetor de angulos correspondente a cada leitura do lidar 
    #Quero somar laser_angles com current_pose[2] para obter os angulos dos obstaculos no mundo
    #Preciso definir os objetos com coordenas globais (mundo)
    # Angulo global = Angulo carro + Angulo Lidar
    # Xglobal = Xcarro + distancia_lidar * cos(angulo_global)
    # Yglobal = Ycarro + distancia_lidar * sin(angulo_global)

    obs_angles_world = laser_angles + current_pose[2]  # Angles of obstacles in the world frame
    obs_x = current_pose[0] + laser_dist * np.cos(obs_angles_world)  # X coordinates of obstacles in the world frame
    obs_y = current_pose[1] + laser_dist * np.sin(obs_angles_world)  # Y coordinates of obstacles in the world frame
    qobs = np.vstack((obs_x, obs_y)).T  # Combine obs_x and obs_y into a single array of shape (num_obstacles, 2)

    #Parte de Correcao do Angulo Final (Para ajudar o angulo na posicao de parada)
    finalstop_goal_angle =  goal_pose[2] 
    finalstop_current_angle = current_pose[2]
    finalstop_angle_diff = finalstop_goal_angle - finalstop_current_angle #Angulo de Parada
    finalstop_angle_diff = (finalstop_angle_diff + np.pi) % (2 * np.pi) - np.pi # Normalize the angle difference to the range [-pi, pi]


    speed = 0
    rotation_speed = 0

    #Se a distancia for maior que 2, o robo calcula os gradientes, se for menor (ele para)
    if distance > 2:
        gradient_attractive = (kgoal * (qgoal - qcurrent))/distance
        norme_attractive = np.linalg.norm(gradient_attractive) #calcula norma do vetor
        direction_atractive = gradient_attractive / norme_attractive# Gradient of the attractive potential
        #Gradiente de repulsão para cada obstáculo
        direction_repulsive = np.array([0.0, 0.0])
        for obs in qobs:
            obs_distance = np.linalg.norm(qcurrent - obs) #Calculate the distance from the robot to the obstacle
            if obs_distance < safe_distance: #If the obstacle is within the safe distance, we calculate the repulsive force
                gradient_repulsive = (kobstacle/ (obs_distance**3)) * (1/obs_distance - 1/safe_distance) * (obs - qcurrent) #Calculate the repulsive force using the formula for the gradient of the repulsive potential
                norme_repulsive = np.linalg.norm(gradient_repulsive)
                direction_repulsive += gradient_repulsive/norme_repulsive #We add the repulsive force to the attractive force to get the final direction of movement
        final_direction = direction_atractive - direction_repulsive  #Como o veto repulsivo aponta do carro para o obstaculo, tivemos que subtrair
    else:
        if abs(finalstop_angle_diff) > 0.1: # If the robot is close to the goal but not well oriented, we rotate in place to correct the orientation
            speed = 0
            rotation_speed = 0.1
            print(f"Final Stop: Alvo(Mundo): {finalstop_goal_angle:.2f}, Robô: {finalstop_current_angle:.2f}, Erro(Giro): {finalstop_angle_diff:.2f}")
        else:
            gradient_attractive = np.array([0.0, 0.0])
            speed = 0
            rotation_speed = 0
        return {"forward": speed, "rotation": rotation_speed}
    
    
    # Calcula o ângulo do vetor FORÇA no mundo (-pi a pi)
    angle_force = np.arctan2(final_direction[1], final_direction[0])  #(Y dps X)
    robot_theta = current_pose[2] # ORIEtacao atuald o robo

    # Calcula A DIFERENÇA entre onde eu quero ir e onde estou olhando (Angulo Objetivo)
    angle_diff = angle_force - robot_theta
    angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi

    # Calcula a FORÇA TOTAL (Norma do vetor final)
    force_magnitude = np.linalg.norm(final_direction)

    speed_base = 0.4 * force_magnitude  # Reduzido de 0.4 para 0.2 para ele andar muito mais devagar por padrão
    rotation_speed = 0.4 * angle_diff  # Reduzido de 0.5 para 0.3 para giros mais contidos
    
    # --- REDUTOR DE VELOCIDADE ANTI-DERRAPAGEM ---
    # Se o robô estiver sentindo alguma repulsão das paredes (mag > 0.0), a gente corta a 
    # velocidade máxima pela metade para que os pneus não patinem ("escorreguem" na odometria)
    if 'direction_repulsive' in locals():
        mag_repulsive = np.linalg.norm(direction_repulsive)
        if mag_repulsive > 0.1:
            speed_base = min(speed_base, 0.3)
            rotation_speed = float(np.clip(rotation_speed, -0.5, 0.5)) # Gira suavemente para esquivar
            
    # Reduz a velocidade se o erro de angulo for muito grande (pra n ir reto na parede enqnto vira)
    if abs(angle_diff) > np.pi / 4: # Se o erro for maior que 45 graus
        speed_base = 0.05 # Anda quase parando para girar com segurança
    
    # SATURAÇÃO MÁXIMA/MÍNIMA para o motor do simulador [-1.0, 1.0]
    speed = float(np.clip(speed_base, 0.0, 0.7)) # Cap de velocidade máxima travado em 0.4 (antes era 1.0) para não patinar
    rotation_speed = float(np.clip(rotation_speed, -0.5, 0.5)) # Cap do limite de giro seguro
    print(f"Alvo(Mundo): {angle_force:.2f}, Robô: {robot_theta:.2f}, Erro(Giro): {angle_diff:.2f}, distance: {distance:.2f}")

    # Para modificar o desempenho do sistema, só devemos modificar os Ks (kgoal e kobstacle)

    command = {"forward": speed,
               "rotation": rotation_speed}
    
    #Parte das Extensoes Foram Feitas (Que é no caso a utilizacao de todas as forcas de repulsao a partir do lidar)

    return command
