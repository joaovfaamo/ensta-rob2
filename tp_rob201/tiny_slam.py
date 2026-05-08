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
        if odom_pose_ref is None:
            odom_pose_ref = self.odom_pose_ref
            
        # Extrair parâmetros da referência
        ref_x, ref_y, ref_t = odom_pose_ref
        
        # Rotacionar a odometria "mente" usando o ângulo corrigido da referência
        # e depois aplicar a translação
        corr_x = ref_x + odom_pose[0] * np.cos(ref_t) - odom_pose[1] * np.sin(ref_t)
        corr_y = ref_y + odom_pose[0] * np.sin(ref_t) + odom_pose[1] * np.cos(ref_t)
        
        # O ângulo final (theta) é só a soma básica
        corr_t = ref_t + odom_pose[2]
        
        # É importante manter o theta restrito entre -pi e pi se o código ficar girando eternamente
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
        free_dist = laser_dist - 0.1  # Diminou para 10cm - Margem menor para paredes ficarem mais vermelhas (mais concentradas)
        free_dist = np.maximum(free_dist, 0) #Mantem os valores em positivos

         # Coordenadas Seguras para marcar como livres (um pouco antes do impacto)
        free_x = current_pose[0] + free_dist * np.cos(obs_angles_world)
        free_y = current_pose[1] + free_dist * np.sin(obs_angles_world)

        # Adicionando um peso mais alto para a parede (bateu = +2 em vez de +1)
        self.grid.add_map_points(obs_x, obs_y, val=2) #Adiciona os pontos de obstaculos no mapa 
        self.grid.add_map_points(np.array([current_pose[0]]), np.array([current_pose[1]]), val=-1) 
        
       # Como o metodo aceita só valores unicos, usa o for para adicionar os pontos 
        for i in range(len(free_x)):
             self.grid.add_value_along_line(current_pose[0], current_pose[1], free_x[i], free_y[i], val=-0.2)

        #Normaliza os valores do grid para evitar que fiquem muito extremos
        #A propria funcao display ja cria uma varia de cores baseada nesse clip (5 vermelho, e -5 azul escuro, 0 cinza)
        self.grid.occupancy_map = np.clip(self.grid.occupancy_map, -5, 5)
        
 