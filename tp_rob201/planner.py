"""
Planner class
Implementation of A*
"""

import copy
import heapq
import math
from collections import defaultdict
from typing import Tuple


import cv2
import numpy as np
from occupancy_grid import OccupancyGrid




class Planner:
    """Simple occupancy grid Planner"""


    def __init__(self, occupancy_grid: OccupancyGrid):
        self.grid = occupancy_grid
        self.map_walls = None


    def get_neighbors(self, current_cell):
        """ Return list of free (i.e. not obstacle) neighbour cells 
            with the format of current_cell: (i, j) in the map frame
        """
        neighbor_list = []
        # TODO for TP5: iterate through neighbors and add free ones to neighbor_list

        current_cell_i, current_cell_j = current_cell #Desempacotamos as coordenadas da célula atual para facilitar a iteração
        for i in range(current_cell_i - 1, current_cell_i + 2):
            for j in range(current_cell_j - 1, current_cell_j + 2):
                if (i, j) != current_cell:  # Exclude the current cell itself
                    if 0 <= i < self.grid.x_max_map and 0 <= j < self.grid.y_max_map:  # Check bounds
                        # Permitimos andar pelo verde (< 1) novamente para não travar nos "Buraquinhos" deixados pelo laser
                        if self.map_walls[i, j] < 1:  #Verificamos se a célula é livre (menor que 1, ou seja, azul ou verde)
                            neighbor_list.append((i, j))

        return neighbor_list


    def heuristic(self, cell_1: Tuple[int, int], cell_2: Tuple[int, int]):
        """ Return heuristic goal distance """
        #Return the Euclidean distance between the two cells
        h = 0
            # TODO for TP5: compute heuristic distance between cell_1 and cell_2
        h = math.sqrt((cell_1[0] - cell_2[0]) ** 2 + (cell_1[1] - cell_2[1]) ** 2)
        
        return h


    def reconstruct_path(self, came_from, goal):
        """ Extract path after cost computation """
        total_path = [goal]
        cell = goal
        while cell in came_from.keys():
            cell = came_from[cell]
            total_path.insert(0, cell)


        total_path = np.array(total_path)
        traj_world_x, traj_world_y = self.grid.conv_map_to_world(total_path[:, 0], total_path[:, 1])
        return np.vstack((traj_world_x, traj_world_y))


    def plan(self, start, goal):
        """
        Compute a path using A*, recompute plan if start or goal change
        start : [x, y, theta] nparray, start pose in world coordinates (theta unused)
        goal : [x, y, theta] nparray, goal pose in world coordinates (theta unused)
        """

        # Convert start and goal from world coordinates to map coordinates (i, j)
        start: Tuple[int, int] = self.grid.conv_world_to_map(start[0], start[1])
        goal: Tuple[int, int] = self.grid.conv_world_to_map(goal[0], goal[1])


        # creates a copy of occupancy map to modify it and take into account
        # a margin in the walls
        self.map_walls = copy.deepcopy(self.grid.occupancy_map)
        
        # TODO for TP5: dilate walls in self.map_walls to take into account a margin around obstacles
        # Consideramos como parede o que tiver probabilidade log-odds maior que um limiar (ex: > 0)
        walls_mask = (self.map_walls > 0).astype(np.uint8)
        
        # Cria um kernel menor para bloqueio rígido (evita colisão sem tapar corredores de vez)
        # Aumentei levemente para 7x7 para que ele passe as quinas ainda com uma bordinha extra de segurança física
        kernel_obst = np.ones((9, 9), np.uint8)
        dilated_walls = cv2.dilate(walls_mask, kernel_obst, iterations=1)
        
        # Cria uma "Aura / Zona de Desconforto" ainda mais larga para forçar o A* bem pelo meio
        kernel_soft = np.ones((25, 25), np.uint8)
        self.soft_walls = cv2.dilate(walls_mask, kernel_soft, iterations=1)
        
        # Aplica os obstáculos dilatados rígidos ao mapa de paredes
        self.map_walls[dilated_walls > 0] = 5

        # Garante que as áreas do ponto de partida e do objetivo estejam sempre livres (desobstrui a dilatação neles)
        # Isso evita que o robô não consiga gerar uma rota por estar "preso" dentro de uma margem virtual
        for x in range(start[0]-2, start[0]+3):
            for y in range(start[1]-2, start[1]+3):
                if 0 <= x < self.grid.x_max_map and 0 <= y < self.grid.y_max_map:
                    if self.map_walls[x, y] == 5: # Só limpa as margens virtuais, mantém obstáculos reais (>0)
                        self.map_walls[x, y] = 0
                    
        for x in range(goal[0]-2, goal[0]+3):
            for y in range(goal[1]-2, goal[1]+3):
                if 0 <= x < self.grid.x_max_map and 0 <= y < self.grid.y_max_map:
                    if self.map_walls[x, y] == 5:
                        self.map_walls[x, y] = 0

        # cv2.imshow("map_walls", sel.map_walls)

        # min heap to contain values to explore next
        open_set = [(0.0, start)]
        heapq.heapify(open_set)


        # dictionary to trace back route
        came_from = {}

        #Para que
        # cost to get to each cell
        g_score = defaultdict(lambda: math.inf)
        g_score[start] = 0.0


        # best guess of cost for each cell (cost + heuristic)
        f_score = defaultdict(lambda: math.inf)
        f_score[start] = 0.0 + self.heuristic(start, goal)

        # main loop of A*
        while len(open_set) > 0:
            current = heapq.heappop(open_set)
            current_f, current_cell = current
            # lazy deletion: skip stale entries
            if current_f > f_score[current_cell]:
                continue
            if current_cell == goal:
                return self.reconstruct_path(came_from, goal)


            neighbours = self.get_neighbors(current_cell)
            for cell in neighbours:
                
                # --- NOVO SISTEMA DE CUSTOS INTELIGENTE ---
                # A base do custo é a distância natural
                step_cost = self.heuristic(current_cell, cell)
                
                # Penaliza o A* brutalmente se ele andar muito perto da parede (na zona soft_walls)
                # Assim ele é obrigado a escolher as células no MEIO dos corredores!
                if hasattr(self, "soft_walls") and self.soft_walls[cell[0], cell[1]] > 0:
                    step_cost *= 25.0
                
                # Se for uma célula VERDE (desconhecida/não validada pelo Lidar)...
                # Multiplicamos o peso absurdamente! (x50). Assim, ele pode pisar nelas
                # pra consertar pequenos "buraquinhos" de raio laser espalhados no mapa,
                # MAS se tiver que atravessar uma PONTE verde (onde seria a parede escondida),
                # vai custar tão mais caro que o A* julgará melhor rodear por todo corredor AZUL!
                if self.map_walls[cell[0], cell[1]] >= -0.1:
                    step_cost *= 50.0
                    
                tentative_g_score = g_score[current_cell] + step_cost
                
                if tentative_g_score < g_score[cell]:
                    # better path, recording it
                    came_from[cell] = current_cell
                    g_score[cell] = tentative_g_score
                    f_score[cell] = tentative_g_score + self.heuristic(cell, goal)
                    heapq.heappush(open_set, (f_score[cell], cell))


        # goal was never reached
        print('failed getting to objective')
        return None


    def explore_frontiers(self):
        """ Frontier based exploration """
        goal = np.array([0, 0, 0])  # frontier to reach for exploration
        return goal

        