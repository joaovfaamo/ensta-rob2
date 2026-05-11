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
        self.returning_to_home = False  # # New flag for return state
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
        self.replanning_counter = 0  # # Counter for dynamic replanning
        self.destino = None  # # Store destination for replanning

    def control(self):
        """
        Main control function executed at each time step
        """
        raw_odom = self.odometer_values()

        # 1. # Try to localise and get score
        best_score = self.tiny_slam.localise(self.lidar(), raw_odom)
        self.corrected_pose = self.tiny_slam.get_corrected_pose(raw_odom)

        # # DISCOVERING YOUR REAL SCORE (Uncomment the line below to view in terminal)
        print(f"Iter: {self.counter} | Score: {best_score:.2f}")

        # # Increased threshold to be stricter and avoid mapping fake walls (ghosts)
        score_threshold = 250 

        # 2. # THE SECRET: Only update map if score is good enough or in the first iterations
        if self.counter < 50 or best_score > score_threshold:
            self.tiny_slam.update_map(self.lidar(), self.corrected_pose)
                
        # 3. # Increment counter
        self.counter += 1

        # ... (rest of the display and return code) ...
        # 3. Display (1 time every 10)
        if self.counter % 10 == 0:
            # # Display original probability map directly, allowing log-odds gradient to be seen
            # # without forcing hard limits (visual thresholding)
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
        goal = [-600, -10, 3.14/2 ]
       #  # is an easier destination to find route, but feel free to test other spots!
       # [-400, -400, 3.14/2 ]
     
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
            return self.control_tp1() # # Uses reactive obstacle avoidance to wander and map
            
        elif self.counter == exploration_iterations:
            # À une itération choisie, calculez le plus court chemin
            print("Exploration phase concluded. Calculating route to target...")
            
            # # YOU CAN CHANGE DESTINATION HERE: (x, y, theta)
            self.destino = np.array([-600, -20, 0.0])
            #-300,-50
            # -600, -20
            # -500, -500
            #-150, -200
            #-950, -70, 0.0 # is an easier destination to find route, but feel free to test other spots!
            self.traj = self.planner.plan(self.corrected_pose, self.destino)
            
            self.target_idx = 0
            return {"forward": 0.0, "rotation": 0.0} # # Stop to think
            
        else:
            # Pour les itérations suivantes, utilisez un contrôleur local
            if self.traj is None:
                return {"forward": 0.0, "rotation": 0.0}
                
            # --- # STATE MACHINE: Check if arrived at end of current trajectory ---
            if self.target_idx >= self.traj.shape[1]:
                if not self.returning_to_home:
                    # # JUST REACHED THE TARGET -> TIME TO RETURN
                    print("🏁 Target reached! Recalibrating SLAM intensely before returning...")
                    
                    # # FORCE RE-LOCALISATION: Since robot is stopped at destination, we run localisation 
                    # # of SLAM dozens of times. This makes the algorithm align map perfectly, 
                    # # resetting temporary accumulation before turning to leave mapping!
                    raw_odom = self.odometer_values()
                    for _ in range(15):
                        self.tiny_slam.localise(self.lidar(), raw_odom)
                    self.corrected_pose = self.tiny_slam.get_corrected_pose(raw_odom)

                    self.returning_to_home = True
                    # # Initial starting point
                    self.destino = np.array([0.0, 0.0, 0.0]) 
                    self.traj = self.planner.plan(self.corrected_pose, self.destino)
                    self.target_idx = 0
                    
                    if self.traj is None:
                        print("Error: Could not find return path.")
                        return {"forward": 0.0, "rotation": 0.0}
                else:
                    # # ALREADY ARRIVED HOME
                    print("🏠 Mission accomplished: Robot returned to base!")
                    return {"forward": 0.0, "rotation": 0.0}
            
            # 1. # Get current trajectory node to use as intermediate target
            target_x = self.traj[0, self.target_idx]
            target_y = self.traj[1, self.target_idx]
            
            # --- # SMART EXPLORATION REPLANNING BASED ON MAP ---
            map_coord = self.occupancy_grid.conv_world_to_map(target_x, target_y)
            i, j = int(map_coord[0]), int(map_coord[1])
            
            # # Check if path to next waypoint is blocked in occupancy map
            is_path_blocked = False
            # # Check if index is within map bounds to avoid errors
            if 0 <= i < self.occupancy_grid.x_max_map and 0 <= j < self.occupancy_grid.y_max_map:
                # # If log-odds probability is greater than 0, it is a confirmed wall
                if self.occupancy_grid.occupancy_map[i, j] > 0.0:
                    is_path_blocked = True

            # # If path ahead is blocked, stop to recalculate
            if is_path_blocked:
                print(f"[REPLANNING] Obstacle in path! Recalculating route from position {self.corrected_pose[:2]}...")
                new_traj = self.planner.plan(self.corrected_pose, self.destino)
                
                if new_traj is not None:
                    self.traj = new_traj
                    self.target_idx = 0
                    print(f"[REPLANNING] ✓ New route calculated with {self.traj.shape[1]} waypoints")
                    target_x = self.traj[0, self.target_idx]
                    target_y = self.traj[1, self.target_idx]
                else:
                    print(f"[REPLANNING] ⚠ Caminho global bloqueado! Tentando seguir a rota antiga...")

            # 2. Monta o local goal atualizado e envia para o campo potencial
            local_goal = np.array([target_x, target_y, 0.0]) 
            
            # Segue esse nó da trajetória local usando o potential field control do TP passado
            command = potential_field_control(self.lidar(), self.corrected_pose, local_goal)
            
            # --- VERIFICAÇÃO DE CHEGADA NO WAYPOINT ---
            dist = np.sqrt((self.corrected_pose[0] - target_x)**2 + (self.corrected_pose[1] - target_y)**2)
            
            # Verifica se este é o ÚLTIMO ponto da rota (o destino final)
            if self.target_idx == self.traj.shape[1] - 1:
                # Relaxando a precisão do destino final para evitar que o robô trave
                # se as forças repulsivas não deixarem ele chegar no 'zero' exato.
                if dist <= 15.0: 
                    self.target_idx += 1
            else:
                # Se for apenas um waypoint no meio do caminho, mantém 40.0 para fluidez
                if dist < 40.0:
                    self.target_idx += 1
                
            return command