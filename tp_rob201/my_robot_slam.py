"""
Robot controller definition
Complete controller including SLAM, planning, path following
"""
import numpy as np

from place_bot.simulation.robot.robot_abstract import RobotAbstract
from place_bot.simulation.robot.odometer import OdometerParams
from place_bot.simulation.ray_sensors.lidar import LidarParams

from tiny_slam import TinySlam

from control import potential_field_control, reactive_obst_avoid
from occupancy_grid import OccupancyGrid
from planner import Planner


# Definition of our robot controller
class MyRobotSlam(RobotAbstract):
    """A robot controller including SLAM, path planning and path following"""

    def __init__(self,
                 lidar_params: LidarParams = LidarParams(),
                 odometer_params: OdometerParams = OdometerParams()):
        # Passing parameter to parent class
        super().__init__(lidar_params=lidar_params,
                         odometer_params=odometer_params)

        # step counter to deal with init and display
        self.counter = 0

        # Init SLAM object
        # Here we cheat to get an occupancy grid size that's not too large, by using the
        # robot's starting position and the maximum map size that we shouldn't know.
        size_area = (1400, 1000)
        robot_position = (439.0, 195)
        self.occupancy_grid = OccupancyGrid(x_min=-(size_area[0] / 2 + robot_position[0]),
                                            x_max=size_area[0] / 2 - robot_position[0],
                                            y_min=-(size_area[1] / 2 + robot_position[1]),
                                            y_max=size_area[1] / 2 - robot_position[1],
                                            resolution=2)

        self.tiny_slam = TinySlam(self.occupancy_grid)
        self.planner = Planner(self.occupancy_grid)

        # storage for pose after localization
        self.corrected_pose = np.array([0, 0, 0])

    def control(self):
        """
        Main control function executed at each time step
        """
        return self.control_so_para_teste()

    def control_tp1(self):
        """
        Control function for TP1
        Control funtion with minimal random motion
        """
        self.tiny_slam.compute()

        # Compute new command speed to perform obstacle avoidance
        command = reactive_obst_avoid(self.lidar())
        return command

    def control_tp2(self):
        """
        Control function for TP2
        Main control function with full SLAM, random exploration and path planning
        """
        pose = self.odometer_values()
        goal = [-300, -300, 3.14/2 ]

        # Compute new command speed to perform obstacle avoidance
        command = potential_field_control(self.lidar(), pose, goal)

        return command

    def control_so_para_teste(self):
        """
        Criei esta funcao para ter um controle que nao faz nada, para poder testar o SLAM e o mapeamento sem o robot se mover, e assim verificar se o mapa esta sendo atualizado corretamente.
        Posso excluir ela depois
        """
        # 1. Mise à jour de la carte
        self.tiny_slam.update_map(self.lidar(), self.odometer_values())

        # 2. Incrementa o contador
        self.counter += 1

        # 3. Affichage (1 vez a cada 10)
        if self.counter % 10 == 0:
            # Salva o mapa original
            original_map = np.copy(self.occupancy_grid.occupancy_map)
            
            # Aplica o seuillage para visualização (4 para parede, -4 para livre)
            clean_map = np.zeros_like(original_map)
            clean_map[original_map > 2] = 4
            clean_map[original_map < -2] = -4
            
            # Coloca o mapa limpo na grid temporariamente e exibe
            self.occupancy_grid.occupancy_map = clean_map
            self.occupancy_grid.display_cv(self.odometer_values())
            
            # Restaura o mapa original de probabilidades
            self.occupancy_grid.occupancy_map = original_map

        command = {"forward": 0,
                   "rotation": 0}
        return command