import numpy as np
import heapq
from typing import List, Tuple, Optional
import plot_path


class AStarPathfinder:
    def __init__(self, occupancy_map: np.ndarray):
        self.map = occupancy_map
        self.layers, self.rows, self.cols = occupancy_map.shape

        # 10 directional movement
        self.directions = [
            # sideways movement
            (0, -1, -1), (0, -1, 0), (0, -1, 1),
            (0, 0, -1),               (0, 0, 1),
            (0, 1, -1),  (0, 1, 0),   (0, 1, 1),

            # up down movement
            (1, 0, 0), (-1, 0, 0),
        ]

        # movement costs
        self.straight_cost = 1.0
        self.diagonal_cost = np.sqrt(2)
        self.up_cost = 2.0
        self.flight_straight_cost = 1.5 * self.straight_cost
        self.flight_diagonal_cost = 1.5 * self.diagonal_cost

    def heuristic(self, pos1: Tuple[int, int, int], pos2: Tuple[int, int, int]) -> float:
        # Euclidian Distance
        return np.linalg.norm(np.array(pos1) - np.array(pos2))

    def get_movement_cost(self, from_pos: Tuple[int, int, int], to_pos: Tuple[int, int, int]) -> float:
        dz = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        dx = to_pos[2] - from_pos[2]

        if dz == 1:
            cost = self.up_cost
        elif dx + dy == 2:
            if to_pos[0] == 1:
                cost = self.flight_diagonal_cost
            else:
                cost = self.diagonal_cost
        else:
            if to_pos[0] == 1:
                cost = self.flight_straight_cost
            else:
                cost = self.straight_cost

        return cost

    def get_neighbors(self, pos: Tuple[int, int, int]) -> List[Tuple[Tuple[int, int, int], float]]:
        # Go through each neighbor and get cost
        neighbors = []
        layer, row, col = pos

        for i, (dl, dc, dr) in enumerate(self.directions):
            new_pos = (layer + dl, row + dr, col + dc)

            # Check bounds
            if self.is_valid_position(new_pos):
                cost = self.get_movement_cost(pos, new_pos)
                neighbors.append((new_pos, cost))

        return neighbors

    def is_valid_position(self, pos: Tuple[int, int, int]) -> bool:
        layer, row, col = pos

        return (0 <= layer < self.layers and
                0 <= row < self.rows and
                0 <= col < self.cols and
                self.map[layer, row, col] == 0)

    def find_path(self, start: Tuple[int, int, int], goal: Tuple[int, int, int]) -> \
            Optional[List[Tuple[int, int, int]]]:

        # Validate start and goal positions
        if not self.is_valid_position(start) or not self.is_valid_position(goal):
            return None

        open_set = [(0, 0, start)]
        came_from = {}

        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}

        visited = set()

        while open_set:
            current_f, current_g, current = heapq.heappop(open_set)

            if current in visited:
                continue

            visited.add(current)

            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            for neighbor, move_cost in self.get_neighbors(current):
                if neighbor in visited:
                    continue

                tentative_g = g_score[current] + move_cost

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + \
                        self.heuristic(neighbor, goal)

                    heapq.heappush(
                        open_set, (f_score[neighbor], tentative_g, neighbor))

        return None  # No path found


num = 11
occ = np.load(f'benchmark_maps/20x20/map{num}.npy')

top = np.zeros(occ.shape)
grid = np.stack([occ, top], axis=0)

# Initialize pathfinder
pathfinder = AStarPathfinder(grid)

# Set start and goal (z,y,x)
start = (0, 0, 0)
goal = (0, 19, 19)

# Find path using A*
path = pathfinder.find_path(start, goal)

if path:
    print(f"Path found with {len(path)} points")
    print("Path points:", path)

    # Visualize results
    plot_path.visualize_path(path, grid, start, goal,
                             "A* Pathfinding Results")
else:
    print("No path found!")
