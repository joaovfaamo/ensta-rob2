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
        raw_odom = self.odometer_values()

        # 1. Tenta melhorar o "self.odom_pose_ref" (descobrir o erro) e recupera o quão confiável essa medição foi (best_score)
        best_score = self.tiny_slam.localise(self.lidar(), raw_odom)

        # 2. Constrói a posição absoluta final para onde o Lidar será colado no mapa de probabilidades
        self.corrected_pose = self.tiny_slam.get_corrected_pose(raw_odom)

        score_threshold = -1# Ou qualquer constante que você otimizou empiricamente assistindo o score
        
        if best_score > score_threshold:
            self.tiny_slam.update_map(self.lidar(), self.corrected_pose)
                

        # 2. Incrementa o contador
        self.counter += 1

        # 3. Affichage (1 vez a cada 10)
        if self.counter % 10 == 0:
            # Exibe o mapa original de probabilidades diretemente, permitindo ver o degrade de log-odds
            # sem forcar hard limits (seuillage visual)
            self.occupancy_grid.display_cv(self.odometer_values())


        return self.control_tp2()

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
        goal = [-100, -400, 3.14/2 ]

        # Compute new command speed to perform obstacle avoidance
        command = potential_field_control(self.lidar(), pose, goal)

        return command

  