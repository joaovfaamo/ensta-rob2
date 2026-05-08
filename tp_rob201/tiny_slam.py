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
        # TODO for TP4

        score = 0

        return score

    def get_corrected_pose(self, odom_pose, odom_pose_ref=None):
        """
        Compute corrected pose in map frame from raw odom pose + odom frame pose,
        either given as second param or using the ref from the object
        odom : raw odometry position
        odom_pose_ref : optional, origin of the odom frame if given,
                        use self.odom_pose_ref if not given
        """
        # TODO for TP4
        corrected_pose = odom_pose

        return corrected_pose

    def localise(self, lidar, raw_odom_pose):
        """
        Compute the robot position wrt the map, and updates the odometry reference
        lidar : placebot object with lidar data
        odom : [x, y, theta] nparray, raw odometry position
        """
        # TODO for TP4

        best_score = 0

        return best_score

    def update_map(self, lidar, pose):
        """
        Bayesian map update with new observation
        lidar : placebot object with lidar data
        pose : [x, y, theta] nparray, corrected pose in world coordinates
        """
        # TODO for TP3
        
        #Implementacao do Upload das Probs das Celulas do Mapa
        #Pega a posicao atual do robo
        current_pose = pose
        laser_dist = lidar.get_sensor_values()
        laser_angles = np.linspace(-np.pi, np.pi, len(laser_dist)) 
        obs_angles_world = laser_angles + current_pose[2]  # Angles of obstacles in the world frame
        obs_x = current_pose[0] + laser_dist * np.cos(obs_angles_world)  # X coordinates of obstacles in the world frame
        obs_y = current_pose[1] + laser_dist * np.sin(obs_angles_world)  # Y coordinates of obstacles in the world frame
        
        #A logica aqui é deixar um raio de incerteza ao redor do ponto de impacto do laser (evita instabilidade)
        free_dist = laser_dist - 0.2  # Diminuí 20cm do limite do impacto (Ajuste o valor aqui dependendo da estabilidade) 
        free_dist = np.maximum(free_dist, 0) #Mantem os valores em positivos

         # Coordenadas Seguras para marcar como livres (um pouco antes do impacto)
        free_x = current_pose[0] + free_dist * np.cos(obs_angles_world)
        free_y = current_pose[1] + free_dist * np.sin(obs_angles_world)

        #Adiciona 1 as celulas dos obstaculos e -1 as celulas em linha reta entre os dois e a posicao do robo
        self.grid.add_map_points(obs_x, obs_y, val=1) #Adiciona os pontos de obstaculos no mapa (com valor 1, ou seja, mais provavel de ser ocupado) 
        self.grid.add_map_points(np.array([current_pose[0]]), np.array([current_pose[1]]), val=-1) #Adiciona a posicao do robo no mapa (com valor -1, ou seja, mais provavel de ser livre)
        
       #Adciona -1 para os pontos entre o robo e a posicao segura de impacto
       # Como o metodo aceita só valores unicos, usa o for para adicionar os pontos 
        for i in range(len(free_x)):
             self.grid.add_value_along_line(current_pose[0], current_pose[1], free_x[i], free_y[i], val=-0.5)

        self.grid.occupancy_map = np.clip(self.grid.occupancy_map, -5, 5)



    def compute(self):
        """ Useless function, just for the exercise on using the profiler """
       