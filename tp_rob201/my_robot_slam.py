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
# No __init__ de MyRobotSlam:
        self.returning_to_home = False  # Novo flag para o estado de retorno
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

        # 1. Tenta se localizar e pega o score
        best_score = self.tiny_slam.localise(self.lidar(), raw_odom)
        self.corrected_pose = self.tiny_slam.get_corrected_pose(raw_odom)

        # DESCOBRINDO O SEU SCORE REAL (Descomente a linha abaixo para ver no terminal)
        # print(f"Iter: {self.counter} | Score: {best_score:.2f}")

        score_threshold = 100  # Diminua um pouco para começar

        # 2. O SEGREDO: Atualiza o mapa cegamente nas primeiras 50 iterações (Cold Start)
        # Depois disso, só atualiza se o score for bom o suficiente.
        if self.counter < 50 or best_score > score_threshold:
            self.tiny_slam.update_map(self.lidar(), self.corrected_pose)
                
        # 3. Incrementa o contador
        self.counter += 1

        # ... (resto do seu código de affichage e return) ...
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
        Implementation of the final planning routine for TP5 with Return to Base
        """
        exploration_iterations = 200

        if self.counter < exploration_iterations:
            # Cartographie/exploration
            return self.control_tp1() # Usa reactive obstacle avoidance para bater perna e mapear
            
        elif self.counter == exploration_iterations:
            # À une itération choisie, calculez le plus court chemin
            print("Fase de exploração concluída. Calculando rota para o objetivo...")
            
            # VOCÊ PODE ALTERAR O DESTINO AQUI: (x, y, theta)
            self.destino = np.array([-900, -50, 0.0])
            self.traj = self.planner.plan(self.corrected_pose, self.destino)
            
            self.target_idx = 0
            return {"forward": 0.0, "rotation": 0.0} # Para para pensar
            
        else:
            # Pour les itérations suivantes, utilisez un contrôleur local
            if self.traj is None:
                return {"forward": 0.0, "rotation": 0.0}
                
            # --- MÁQUINA DE ESTADOS: Verifica se chegou ao fim da trajetória atual ---
            if self.target_idx >= self.traj.shape[1]:
                if not self.returning_to_home:
                    # ACABOU DE CHEGAR NO OBJETIVO -> HORA DE VOLTAR
                    print("🏁 Objetivo atingido! Calculando rota de retorno para a origem...")
                    self.returning_to_home = True
                    # Ponto de partida inicial
                    self.destino = np.array([0.0, 0.0, 0.0]) 
                    self.traj = self.planner.plan(self.corrected_pose, self.destino)
                    self.target_idx = 0
                    
                    if self.traj is None:
                        print("Erro: Não foi possível encontrar caminho de volta.")
                        return {"forward": 0.0, "rotation": 0.0}
                else:
                    # JÁ CHEGOU A CASA
                    print("🏠 Missão cumprida: Robô de volta à base!")
                    return {"forward": 0.0, "rotation": 0.0}
            
            # 1. Pega o nó atual da trajetória para usar como alvo intermediário
            target_x = self.traj[0, self.target_idx]
            target_y = self.traj[1, self.target_idx]
            
            # --- REPLANNING INTELIGENTE BASEADO NO MAPA ---
            map_coord = self.occupancy_grid.conv_world_to_map(target_x, target_y)
            i, j = int(map_coord[0]), int(map_coord[1])
            
            is_path_blocked = False
            # Checa se o índice está dentro dos limites do mapa para evitar erros
            if 0 <= i < self.occupancy_grid.x_max_map and 0 <= j < self.occupancy_grid.y_max_map:
                # Se a probabilidade log-odds for maior que 0, é uma parede confirmada
                if self.occupancy_grid.occupancy_map[i, j] > 0.0:
                    is_path_blocked = True

            # Se o caminho à frente bloqueou, paramos para recalcular
            if is_path_blocked:
                print(f"[REPLANNING] Obstáculo no caminho! Recalculando rota da posição {self.corrected_pose[:2]}...")
                new_traj = self.planner.plan(self.corrected_pose, self.destino)
                
                if new_traj is not None:
                    self.traj = new_traj
                    self.target_idx = 0
                    print(f"[REPLANNING] ✓ Nova rota calculada com {self.traj.shape[1]} waypoints")
                    target_x = self.traj[0, self.target_idx]
                    target_y = self.traj[1, self.target_idx]
                else:
                    print(f"[REPLANNING] ⚠ Caminho global bloqueado! Tentando seguir a rota antiga...")

            # 2. Monta o local goal atualizado e envia para o campo potencial
            local_goal = np.array([target_x, target_y, 0.0]) 
            
            # Segue esse nó da trajetória local usando o seu potential field control do TP passado
            command = potential_field_control(self.lidar(), self.corrected_pose, local_goal)
            
            # Checa a distância em relação a esse pequeno nó atual. Se estiver perto (~10 px/cm), mira no próximo nó
            dist = np.sqrt((self.corrected_pose[0] - target_x)**2 + (self.corrected_pose[1] - target_y)**2)
            if dist < 40.0:
                self.target_idx += 1
                
            return command