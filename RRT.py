import numpy as np
import random
import math
from typing import List, Tuple, Optional
import plot_path


class RRTNode:
    def __init__(self, position: Tuple[int, int, int], parent=None, cost: float = 0.0):
        self.position = position
        self.parent = parent
        self.cost = cost
        self.children = []

    def add_child(self, child_node):
        self.children.append(child_node)


class RRT3D:
    def __init__(self, grid_map: np.ndarray, start: Tuple[int, int, int], goal: Tuple[int, int, int]):
        self.grid_map = grid_map
        self.map = grid_map
        self.start = start
        self.goal = goal
        self.dimensions = grid_map.shape

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

        # RRT parameters
        self.max_iterations = 5000
        self.step_size = 1
        self.goal_threshold = 1.1
        self.goal_bias = 0.1  # chance to sample goal directly

        # Tree storage
        self.nodes = []
        self.root = None

    def get_movement_cost(self, from_pos: Tuple[int, int, int], to_pos: Tuple[int, int, int]) -> float:
        dz, dy, dx = (to_pos[i] - from_pos[i] for i in range(3))

        # Check if movement involves flying (z > 0 or dz != 0)
        is_flying = from_pos[2] > 0 or to_pos[2] > 0 or dz != 0

        # Vertical movement cost
        if dz != 0:
            return self.up_cost

        # Horizontal movement costs
        if dx != 0 and dy != 0:  # Diagonal movement
            return self.flight_diagonal_cost if is_flying else self.diagonal_cost
        else:  # Straight movement
            return self.flight_straight_cost if is_flying else self.straight_cost

    def is_valid_position(self, pos: Tuple[int, int, int]) -> bool:
        z, y, x = pos

        # Check bounds
        if not (0 <= z < self.dimensions[0] and
                0 <= y < self.dimensions[1] and
                0 <= x < self.dimensions[2]):
            return False

        # Check for obstacles
        return self.grid_map[z, y, x] == 0

    def get_neighbors(self, pos: Tuple[int, int, int]) ->  \
            List[Tuple[int, int, int]]:
        neighbors = []
        z, y, x = pos

        for dl, dc, dr in self.directions:
            new_pos = (z + dl, y + dc, x + dr)
            if self.is_valid_position(new_pos):
                neighbors.append(new_pos)

        return neighbors

    def distance(self, pos1: Tuple[int, int, int], pos2: Tuple[int, int, int]) -> float:
        return math.sqrt(sum((pos1[i] - pos2[i]) ** 2 for i in range(3)))

    def sample_random_position(self) -> Tuple[int, int, int]:
        # chance to sample goal
        if random.random() < self.goal_bias:
            return self.goal

        # Sample random position
        max_attempts = 100
        for _ in range(max_attempts):
            z = random.randint(0, self.dimensions[0] - 1)
            y = random.randint(0, self.dimensions[1] - 1)
            x = random.randint(0, self.dimensions[2] - 1)
            pos = (z, y, x)

            if self.is_valid_position(pos):
                return pos

        # Fallback to goal if no valid random position found
        return self.goal

    def find_nearest_node(self, target_pos: Tuple[int, int, int]) -> RRTNode:
        min_distance = float('inf')
        nearest_node = None

        for node in self.nodes:
            dist = self.distance(node.position, target_pos)
            if dist < min_distance:
                min_distance = dist
                nearest_node = node

        return nearest_node

    def steer(self, from_pos: Tuple[int, int, int], to_pos: Tuple[int, int, int]) -> \
            Tuple[int, int, int]:
        dist = self.distance(from_pos, to_pos)

        if dist <= self.step_size:
            return to_pos

        # Calculate unit vector
        dz = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        dx = to_pos[2] - from_pos[2]

        # Normalize and scale by step size
        scale = self.step_size / dist
        new_z = from_pos[0] + int(dz * scale)
        new_y = from_pos[1] + int(dy * scale)
        new_x = from_pos[2] + int(dx * scale)

        new_pos = (new_z, new_y, new_x)

        # Ensure the new position is valid
        if self.is_valid_position(new_pos):
            return new_pos
        else:
            return from_pos

    def is_collision_free(self, from_pos: Tuple[int, int, int], to_pos: Tuple[int, int, int]) -> Tuple[bool, Optional[RRTNode]]:
        # Simple check - both positions should be valid
        return self.is_valid_position(from_pos) and self.is_valid_position(to_pos)

    def build_tree(self) -> bool:
        # Initialize tree with start node
        self.root = RRTNode(self.start, None, 0.0)
        self.nodes.append(self.root)

        for iteration in range(self.max_iterations):
            # Sample random position
            random_pos = self.sample_random_position()

            # Find nearest node
            nearest_node = self.find_nearest_node(random_pos)

            # Steer towards random position
            new_pos = self.steer(nearest_node.position, random_pos)

            # Check if movement is valid
            if (new_pos != nearest_node.position and
                    self.is_collision_free(nearest_node.position, new_pos)):

                # Calculate cost
                movement_cost = self.get_movement_cost(
                    nearest_node.position, new_pos)
                new_cost = nearest_node.cost + movement_cost

                # Create new node and add to tree
                new_node = RRTNode(new_pos, nearest_node, new_cost)
                self.nodes.append(new_node)
                nearest_node.add_child(new_node)

                # Check if we reached the goal
                if self.distance(new_pos, self.goal) <= self.goal_threshold:
                    movement_cost = self.get_movement_cost(
                        new_pos, self.goal)
                    goal_node = RRTNode(self.goal, new_node,
                                        new_cost + movement_cost)
                    new_node.add_child(goal_node)
                    return True, goal_node

        print(f"Failed to reach goal in {self.max_iterations} iterations")
        return False, None

    def find_path(self) -> Optional[List[Tuple[int, int, int]]]:
        success, goal_node = self.build_tree()

        if not success:
            return None

        # Reconstruct path
        if goal_node is None:
            return None

        path = []
        current = goal_node

        while current is not None:
            path.append(current.position)
            current = current.parent

        path.reverse()
        return path


num = 11
occ = np.load(f'benchmark_maps/20x20/map{num}.npy')

top = np.zeros(occ.shape)
grid = np.stack([occ, top], axis=0)

# Set start and goal (z,y,x)
start = (0, 0, 0)
goal = (0, 19, 19)

# Create RRT solver
rrt = RRT3D(grid, start, goal)

# Find path
path = rrt.find_path()

if path:
    print(f"Path found with {len(path)} points")
    print("Path points:", path)

    # Calculate total cost
    total_cost = 0.0
    for i in range(1, len(path)):
        cost = rrt.get_movement_cost(path[i-1], path[i])
        total_cost += cost

    # Visualize results
    plot_path.visualize_path(path, grid, start, goal,
                             "RRT Pathfinding Results")

else:
    print("\nNo path found!")
