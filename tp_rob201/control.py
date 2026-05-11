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
    # First approach: obstacle avoidance: if any of the laser distances is less than 10, we stop and rotate randomly
    if (laser_dist.min() < 12):  #Since laser_dist is an array, we check if any of the values is less than 10
        speed = 0
        rotation_speed = random.uniform(0.5, 1.0) * random.choice([-1, 1])
    else:
        speed = 0.3  # Reducing speed for smoother reaction
        rotation_speed = 0

    command = {"forward": speed,
               "rotation": rotation_speed}
    
    return command

    """
    """
# Second way: See only the front Secteur

    laser_dist = lidar.get_sensor_values()

    # --- Front approach: front of the robot is in the middle of the array ---
    num_rays = len(laser_dist) #mesure the number of rays in the lidar
    mid_point = num_rays // 2 #the middle index of the lidar rays, assuming the front is in the middle of the array (180 DEGRES)
    num_rays_side = 50 # (180 - 50) = 130 e (180 + 50) = 230, so we will look at the rays from index 130 to 230, which correspond to the front sector of the robot
    
    start_index = max(0, mid_point - num_rays_side) #130
    end_index = min(num_rays, mid_point + num_rays_side) #230
    front_sector = laser_dist[start_index:end_index] #Copy the desired elements of the lidar vector to the new vector (front sect)

    min_front_dist = front_sector.min() #Find the minimum distance in the front sector
    print(f"Minimum Front Distance: {min_front_dist:.2f}")

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
    # Attractive Gradient
    kgoal = 0.8  # Gain for the attractive potential
    qgoal = np.array(goal_pose[:2])  # Eliminate the orientation component, only (x, y)
    qcurrent = np.array(current_pose[:2])  # Current position (x, y)
    distance = np.linalg.norm(qcurrent - qgoal) #Calculate the euclidean distance between the current position and the goal position

    # Repulsive Gradient 
    kobstacle = 30  # # Reduced from 100 to 30 for smoother push to avoiding drifting
    safe_distance = 25 # # Increased from 20 to 25 to start repulsion further away but gently
    ## As the idea is to have several repulsive forces, one for each obstacle, we need to calculate the gradient of each one and sum the repulsive forces
    laser_dist = lidar.get_sensor_values()
    laser_angles = np.linspace(-np.pi, np.pi, len(laser_dist))  # # Generate an array of angles corresponding to each lidar reading 
    ## Add laser angles with current_pose[2] to get obstacle angles in the world
    ## Define objects with global coordinates (world)
    # # Global angle = Car angle + Lidar angle
    # # Xglobal = Xcar + lidar_distance * cos(global_angle)
    # # Yglobal = Ycar + lidar_distance * sin(global_angle)

    obs_angles_world = laser_angles + current_pose[2]  # Angles of obstacles in the world frame
    obs_x = current_pose[0] + laser_dist * np.cos(obs_angles_world)  # X coordinates of obstacles in the world frame
    obs_y = current_pose[1] + laser_dist * np.sin(obs_angles_world)  # Y coordinates of obstacles in the world frame
    qobs = np.vstack((obs_x, obs_y)).T  # Combine obs_x and obs_y into a single array of shape (num_obstacles, 2)

    ## Final Angle Correction (To help with angle at the stopping position)
    finalstop_goal_angle =  goal_pose[2] 
    finalstop_current_angle = current_pose[2]
    finalstop_angle_diff = finalstop_goal_angle - finalstop_current_angle ## Stopping Angle
    finalstop_angle_diff = (finalstop_angle_diff + np.pi) % (2 * np.pi) - np.pi # Normalize the angle difference to the range [-pi, pi]


    speed = 0
    rotation_speed = 0

    ## If distance is greater than 2, compute gradients, if less (stop)
    if distance > 2:
        gradient_attractive = (kgoal * (qgoal - qcurrent))/distance
        norme_attractive = np.linalg.norm(gradient_attractive) ## calculate vector norm
        direction_atractive = gradient_attractive / norme_attractive# Gradient of the attractive potential
        #Gradiente de repulsão para cada obstáculo
        direction_repulsive = np.array([0.0, 0.0])
        for obs in qobs:
            obs_distance = np.linalg.norm(qcurrent - obs) #Calculate the distance from the robot to the obstacle
            if obs_distance < safe_distance: #If the obstacle is within the safe distance, we calculate the repulsive force
                gradient_repulsive = (kobstacle/ (obs_distance**3)) * (1/obs_distance - 1/safe_distance) * (obs - qcurrent) #Calculate the repulsive force using the formula for the gradient of the repulsive potential
                norme_repulsive = np.linalg.norm(gradient_repulsive)
                direction_repulsive += gradient_repulsive/norme_repulsive #We add the repulsive force to the attractive force to get the final direction of movement
        final_direction = direction_atractive - direction_repulsive  ## Subtract since the repulsive vector points from the car to the obstacle
    else:
        if abs(finalstop_angle_diff) > 0.1: # If the robot is close to the goal but not well oriented, we rotate in place to correct the orientation
            speed = 0
            rotation_speed = 0.1
            print(f"Final Stop: Target(World): {finalstop_goal_angle:.2f}, Robot: {finalstop_current_angle:.2f}, Error(Turn): {finalstop_angle_diff:.2f}")
        else:
            gradient_attractive = np.array([0.0, 0.0])
            speed = 0
            rotation_speed = 0
        return {"forward": speed, "rotation": rotation_speed}
    
    
    # # Calculate angle of the FORCE vector in the world (-pi to pi)
    angle_force = np.arctan2(final_direction[1], final_direction[0])  #(Y dps X)
    robot_theta = current_pose[2] # # Current robot orientation

    # # Calculate THE DIFFERENCE between target and look direction (Goal Angle)
    angle_diff = angle_force - robot_theta
    angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi

    # # Calculate TOTAL FORCE (Norm of final vector)
    force_magnitude = np.linalg.norm(final_direction)

    speed_base = 0.4 * force_magnitude  # # Reduced from 0.4 to 0.2 to walk slower by default
    rotation_speed = 0.4 * angle_diff  # # Reduced from 0.5 to 0.3 for tighter turns
    
    # --- # ANTI-DRIFT SPEED REDUCER ---
    # # If robot feels repulsive force from walls, cut 
    # # max speed in half so tires don't slip (odometry slip)
    if 'direction_repulsive' in locals():
        mag_repulsive = np.linalg.norm(direction_repulsive)
        if mag_repulsive > 0.1:
            speed_base = min(speed_base, 0.3)
            rotation_speed = float(np.clip(rotation_speed, -0.5, 0.5)) # # Turn smoothly to avoid
            
    # # Reduce speed if angle error is large (avoid driving straight into wall while turning)
    if abs(angle_diff) > np.pi / 4: # # If error is greater than 45 degrees
        speed_base = 0.05 # # Drive almost stopping for safe turn
    
    # # MAXIMUM/MINIMUM SATURATION for simulator motor [-1.0, 1.0]
    speed = float(np.clip(speed_base, 0.0, 0.7)) # # Max speed cap locked at 0.4 (previously 1.0) to avoid slipping
    rotation_speed = float(np.clip(rotation_speed, -0.5, 0.5)) # # Safe turn limit cap
    print(f"Target(World): {angle_force:.2f}, Robot: {robot_theta:.2f}, Error(Turn): {angle_diff:.2f}, distance: {distance:.2f}")

    # # To modify system performance, just modify Ks (kgoal and kobstacle)

    command = {"forward": speed,
               "rotation": rotation_speed}
    
    ## Extensions were made (such as using all repulsive forces from the lidar)

    return command
