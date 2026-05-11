""" A simple robotics navigation code including SLAM, exploration, planning"""

import cv2
import numpy as np
from occupancy_grid import OccupancyGrid


class TinySlam:
    """Simple occupancy grid SLAM"""

    def __init__(self, occupancy_grid: OccupancyGrid):
        self.grid = occupancy_grid

        # Origin of the odom frame in the map frame
        self.odom_pose_ref = np.array([0, 0, 0])

    def _score(self, lidar, pose):
        """
        Computes the sum of log probabilities of laser end points in the map
        lidar : placebot object with lidar data
        pose : [x, y, theta] nparray, position of the robot to evaluate, in world coordinates
        """
        # 1. # Get distances and respective angles
        laser_dist = lidar.get_sensor_values()
        laser_angles = np.linspace(-np.pi, np.pi, len(laser_dist)) 
        
        # 2. # Filter out max distance points (not actual obstacle detections)
        valid_mask = laser_dist < lidar.max_range
        valid_dist = laser_dist[valid_mask] ## Keep only valid distances (less than maximum lidar range)
        valid_angles = laser_angles[valid_mask] ## Keep only angles corresponding to valid distances
        
        if len(valid_dist) == 0:
            return 0
        
        # 3. # Estimate absolute map positions of detections
        obs_angles_world = valid_angles + pose[2]
        obs_x = pose[0] + valid_dist * np.cos(obs_angles_world)
        obs_y = pose[1] + valid_dist * np.sin(obs_angles_world)
        
        # 4. # Convert metric positions (x,y) to grid indices (pixels)
        map_x, map_y = self.grid.conv_world_to_map(obs_x, obs_y)
        
        # 5. # Remove lidar-points falling out of mapping grid bounds
        inside_map_mask = (map_x >= 0) & (map_x < self.grid.x_max_map) & \
                          (map_y >= 0) & (map_y < self.grid.y_max_map)
        
        valid_map_x = map_x[inside_map_mask]
        valid_map_y = map_y[inside_map_mask]
        
        # 6. # Sum the score of marked cells 
        score = np.sum(self.grid.occupancy_map[valid_map_x, valid_map_y])

        return score

    #Get the corrected position of the robot in world coordinates
    #REVISE
    def get_corrected_pose(self, odom_pose, odom_pose_ref=None):
        if odom_pose_ref is None:
            odom_pose_ref = self.odom_pose_ref
            
        # # Extract reference parameters
        ref_x, ref_y, ref_t = odom_pose_ref
        
        # # Rotate odometry using corrected angle from reference
        # # and then apply translation
    
        corr_x = ref_x + odom_pose[0] * np.cos(ref_t) - odom_pose[1] * np.sin(ref_t)
        corr_y = ref_y + odom_pose[0] * np.sin(ref_t) + odom_pose[1] * np.cos(ref_t)
        
        # # Final angle (theta) is just basic sum
        corr_t = ref_t + odom_pose[2]
        
        # # It is important to keep theta mapped [-pi, pi] if code spins forever
        corr_t = (corr_t + np.pi) % (2 * np.pi) - np.pi
        
        corrected_pose = np.array([corr_x, corr_y, corr_t])
        return corrected_pose


    def localise(self, lidar, raw_odom_pose):
        """
        Compute the robot position wrt the map, and updates the odometry reference
        lidar : placebot object with lidar data
        odom : [x, y, theta] nparray, raw odometry position
        """
        # TODO for TP4

        # 1. # Calculate initial score using CURRENT reference position
        best_odom_ref = self.odom_pose_ref.copy()
        corrected_pose = self.get_corrected_pose(raw_odom_pose, best_odom_ref)
        best_score = self._score(lidar, corrected_pose)

        # 2. # Random search to find best reference
        # # N_max was reduced to optimize performance, otherwise simulation runs slow.
        N_max = 200  
        no_improve_count = 0
        
        # # Standard deviations [x, y, theta] for noise matrix
        # # A balanced sigma to allow correcting jumps without purely random search.
        sigma = np.array([0.1, 0.1, 0.05])

        while no_improve_count < N_max:
            # 3. Adicionar ruído na POSIÇÃO DE REFERÊNCIA, não na odometria bruta
            noise = np.random.normal(0, sigma)
            test_odom_ref = best_odom_ref + noise
            
            # Normalizar theta
            test_odom_ref[2] = (test_odom_ref[2] + np.pi) % (2 * np.pi) - np.pi #Mantém o ângulo entre -pi e pi para evitar problemas de rotação contínua

            # 4. Calcular score com a nova referência testada
            test_pose = self.get_corrected_pose(raw_odom_pose, test_odom_ref)
            test_score = self._score(lidar, test_pose)

            # 5. Se for melhor, memorizar o score e atualizar a referência (e resetar o contador)
            if test_score > best_score:
                best_score = test_score
                best_odom_ref = test_odom_ref
                no_improve_count = 0
            else:
                no_improve_count += 1

        # 6. # Save the best obtained robot reference in object attribute
        self.odom_pose_ref = best_odom_ref

        return best_score


    def update_map(self, lidar, pose):
        """
        Bayesian map update with new observation
        lidar : placebot object with lidar data
        pose : [x, y, theta] nparray, corrected pose in world coordinates
        """
        # TODO for TP3
        
        ## Implementation to Upload Cell Probabilities of Map
        ## Gets current robot position
        current_pose = pose
        laser_dist = lidar.get_sensor_values()
        laser_angles = np.linspace(-np.pi, np.pi, len(laser_dist)) 
        obs_angles_world = laser_angles + current_pose[2]  # Angles of obstacles in the world frame
        obs_x = current_pose[0] + laser_dist * np.cos(obs_angles_world)  # X coordinates of obstacles in the world frame
        obs_y = current_pose[1] + laser_dist * np.sin(obs_angles_world)  # Y coordinates of obstacles in the world frame
        
        ## The logic here is to put uncertainty radius around laser impact point (avoids instability)
        free_dist = laser_dist - 0.1  # # Decreased to 10cm - Smaller margin so walls get redder (more concentrated)
        free_dist = np.maximum(free_dist, 0) ## Keep positive values

         # # Safe coordinates to set as free (just before impact)
        free_x = current_pose[0] + free_dist * np.cos(obs_angles_world)
        free_y = current_pose[1] + free_dist * np.sin(obs_angles_world)

        # # Add higher weight for the wall (hit = +2 instead of +1)
        self.grid.add_map_points(obs_x, obs_y, val=2) #Adiciona os pontos de obstaculos no mapa 
        self.grid.add_map_points(np.array([current_pose[0]]), np.array([current_pose[1]]), val=-1) 
        
       # Como o metodo aceita só valores unicos, usa o for para adicionar os pontos 
        for i in range(len(free_x)):
             self.grid.add_value_along_line(current_pose[0], current_pose[1], free_x[i], free_y[i], val=-0.2)

        #Normaliza os valores do grid para evitar que fiquem muito extremos
        #A propria funcao display ja cria uma varia de cores baseada nesse clip (5 vermelho, e -5 azul escuro, 0 cinza)
        self.grid.occupancy_map = np.clip(self.grid.occupancy_map, -5, 5)
        
 