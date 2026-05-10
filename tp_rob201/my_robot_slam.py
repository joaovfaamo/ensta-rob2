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

        # storage for path planning
        self.traj = None
        self.target_idx = 0
        self.replanning_counter = 0  # Contador para replanning dinâmico
        self.destino = None  # Armazena o destino para usar no replanning

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
            if self.traj is not None:
                self.occupancy_grid.display_cv(self.corrected_pose, np.array([0, 0, 0]), self.traj)
            else:
                self.occupancy_grid.display_cv(self.corrected_pose)

        return self.control_tp5()

    
    def control_tp1(self):
        """
        Control function for TP1
        Control funtion with minimal random motion
        """
        # Compute new command speed to perform obstacle avoidance
        command = reactive_obst_avoid(self.lidar())
        return command

    def control_tp2(self):
        """
        Control function for TP2
        Main control function with full SLAM, random exploration and path planning
        """
        pose = self.odometer_values()
        goal = [-400, -400, 3.14/2 ]

        # Compute new command speed to perform obstacle avoidance
        command = potential_field_control(self.lidar(), pose, goal)

        return command

    def control_tp5(self):
        """
        Implementation of the final planning routine for TP5
        """
        exploration_iterations = 200

        if self.counter < exploration_iterations:
            # Cartographie/exploration
            return self.control_tp1() # Usa reactive obstacle avoidance para bater perna e mapear
            
        elif self.counter == exploration_iterations:
            # À une itération choisie, calculez le plus court chemin
            print("Calculando caminho de volta para a origem...")
            
            # VOCÊ PODE ALTERAR O DESTINO AQUI: (x, y, theta)
            self.destino = np.array([-200, -200, 0.0])
            self.traj = self.planner.plan(self.corrected_pose, self.destino)
            
            self.target_idx = 0
            return {"forward": 0.0, "rotation": 0.0} # Para para pensar
            
        else:
            # Pour les itérations suivantes, utilisez un contrôleur local
            if self.traj is None:
                return {"forward": 0.0, "rotation": 0.0}
                
            # Arrêtez-vous lorsque le robot est revenu au point de départ
            if self.target_idx >= self.traj.shape[1]:
                print("🏁 Ponto de partida alcançado!")
                return {"forward": 0.0, "rotation": 0.0}
            
            # --- REPLANNING DINÂMICO ---
            # A cada 20 iterações, recalcula a rota do ponto atual até o destino
            # Isso permite adaptar-se dinamicamente se descobrir novas paredes
            self.replanning_counter += 1
            if self.replanning_counter >= 20:
                print(f"[REPLANNING] Recalculando rota da posição {self.corrected_pose[:2]}...")
                new_traj = self.planner.plan(self.corrected_pose, self.destino)
                
                if new_traj is not None:
                    # Conseguiu encontrar um novo caminho, atualiza
                    self.traj = new_traj
                    self.target_idx = 0
                    print(f"[REPLANNING] ✓ Nova rota calculada com {self.traj.shape[1]} waypoints")
                else:
                    # Caminho bloqueado! Continua tentando com o caminho antigo
                    print(f"[REPLANNING] ⚠ Caminho bloqueado! Continuando com trajetória anterior...")
                
                self.replanning_counter = 0
                
            # Pega o nó atual da trajetória para usar como alvo intermediário
            target_x = self.traj[0, self.target_idx]
            target_y = self.traj[1, self.target_idx]
            local_goal = np.array([target_x, target_y, 0.0]) # Usa a mesma formatação [x,y,theta]
            
            # Segue esse nó da trajetória local usando o seu potential field control do TP passado
            command = potential_field_control(self.lidar(), self.corrected_pose, local_goal)
            
            # Checa a distância em relação a esse pequeno nó atual. Se estiver perto (~10 px/cm), mira no próximo nó
            dist = np.sqrt((self.corrected_pose[0] - target_x)**2 + (self.corrected_pose[1] - target_y)**2)
            if dist < 25.0:
                self.target_idx += 1
                
            return command
