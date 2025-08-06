import numpy as np
import random
from typing import List, Tuple, Optional
from collections import defaultdict
import plot_path


class Graph3D:
    def __init__(self, occupancy_map: np.ndarray):
        self.map = occupancy_map
        self.nodes = []
        self.node_dict = {}
        self.edges = defaultdict(list)
        self.edge_weights = {}
        self.pheromones = defaultdict(float)
        self.start_idx = None
        self.goal_idx = None

    def add_node(self, position: Tuple[int, int, int]) -> int:
        if position in self.node_dict:
            return self.node_dict[position]

        node_idx = len(self.nodes)
        self.nodes.append(position)
        self.node_dict[position] = node_idx
        return node_idx

    def add_edge(self, pos1: Tuple[int, int, int], pos2: Tuple[int, int, int], weight: float):
        idx1 = self.add_node(pos1)
        idx2 = self.add_node(pos2)

        self.edges[idx1].append(idx2)
        self.edges[idx2].append(idx1)

        edge_key = tuple(sorted([idx1, idx2]))
        self.edge_weights[edge_key] = weight

        # Initialize pheromones
        self.pheromones[edge_key] = 0.1

    def set_start_goal(self, start: Tuple[int, int, int], goal: Tuple[int, int, int]):
        self.start_idx = self.node_dict.get(start)
        self.goal_idx = self.node_dict.get(goal)

    def get_neighbors(self, node_idx: int) -> List[int]:
        return self.edges[node_idx]

    def get_edge_key(self, node1_idx: int, node2_idx: int) -> Tuple[int, int]:
        return tuple(sorted([node1_idx, node2_idx]))

    def get_edge_weight(self, node1_idx: int, node2_idx: int) -> float:
        edge_key = self.get_edge_key(node1_idx, node2_idx)
        return self.edge_weights.get(edge_key, float('inf'))

    def get_pheromone(self, node1_idx: int, node2_idx: int) -> float:
        edge_key = self.get_edge_key(node1_idx, node2_idx)
        return self.pheromones[edge_key]

    def update_pheromone(self, node1_idx: int, node2_idx: int, value: float):
        edge_key = self.get_edge_key(node1_idx, node2_idx)
        self.pheromones[edge_key] = value

    def distance(self, node1_idx: int, node2_idx: int) -> float:
        pos1 = self.nodes[node1_idx]
        pos2 = self.nodes[node2_idx]
        return np.linalg.norm(np.array(pos1) - np.array(pos2))


def create_3d_graph_from_array(start: Tuple[int, int, int], goal: Tuple[int, int, int], node_array: np.ndarray) -> Graph3D:

    # Initialize graph
    graph = Graph3D(node_array)
    # Add nodes
    node_positions = np.where(node_array == 0)
    node_coords = list(
        zip(node_positions[0], node_positions[1], node_positions[2]))
    for coord in node_coords:
        graph.add_node(coord)
    node_set = set(node_coords)
    # Add edges
    for node in node_coords:
        neighbors = get_neighbors_3d(node, node_array.shape)
        for neighbor in neighbors:
            if neighbor in node_set:
                if node < neighbor:
                    weight = calculate_edge_weight(node, neighbor)
                    graph.add_edge(node, neighbor, weight)

    graph.set_start_goal(start, goal)

    return graph


def get_neighbors_3d(pos, shape):
    z, y, x = pos
    neighbors = []

    # 10 directional movement
    directions = [
        # sideways movement
        (0, -1, -1), (0, -1, 0), (0, -1, 1),
        (0, 0, -1),               (0, 0, 1),
        (0, 1, -1),  (0, 1, 0),   (0, 1, 1),

        # up down movement
        (1, 0, 0), (-1, 0, 0),
    ]

    for dz, dy, dx in directions:
        nz, ny, nx = z + dz, y + dy, x + dx
        if 0 <= nz < shape[0] and 0 <= ny < shape[1] and 0 <= nx < shape[2]:
            neighbors.append((nz, ny, nx))

    return neighbors


def calculate_edge_weight(pos1, pos2):
    distance = np.linalg.norm(np.array(pos1) - np.array(pos2))

    # Penalize vertical movement (z-axis) and upper level movement
    if pos1[0] != pos2[0]:  # Different z levels
        distance *= 2
    elif pos1[0] == 1:  # Upper level (z=1)
        distance *= 1.5

    return distance


class Ant:
    def __init__(self, start_idx: int):
        self.current_idx = start_idx
        self.path = [start_idx]
        self.visited = {start_idx}
        self.total_distance = 0.0
        self.stuck = False

    def reset(self, start_idx: int):
        self.current_idx = start_idx
        self.path = [start_idx]
        self.visited = {start_idx}
        self.total_distance = 0.0
        self.stuck = False

    def can_move_to(self, node_idx: int) -> bool:
        return node_idx not in self.visited


class ACO:
    def __init__(self, graph: Graph3D, n_ants: int = 20,
                 max_iterations: int = 100, alpha: float = 1.0,
                 beta: float = 2.0, rho: float = 0.1,
                 q: float = 0.8, xi: float = 0.1):
        self.graph = graph
        self.n_ants = n_ants
        self.max_iterations = max_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q
        self.xi = xi

        # Pheromone bounds
        self.tau_min = 0.01
        self.tau_max = 1.0

        self.best_path = None
        self.best_distance = float('inf')
        self.convergence_data = []

    def triangle_inequality_heuristic(self, current_idx: int, next_idx: int) \
            -> float:
        if self.graph.goal_idx is None:
            return 1.0

        # Distance from current to next
        d1 = self.graph.distance(current_idx, next_idx)
        # Distance from next to goal
        d2 = self.graph.distance(next_idx, self.graph.goal_idx)
        # Distance from current to goal
        d3 = self.graph.distance(current_idx, self.graph.goal_idx)

        if d1 + d2 >= d3:
            return 1.0 / (d2 + 0.1)
        else:
            return 1.0 / (d2 + d3 + 0.1)

    def pseudo_random_state_transition(self, ant: Ant) -> Optional[int]:
        neighbors = self.graph.get_neighbors(ant.current_idx)
        valid_neighbors = [idx for idx in neighbors if ant.can_move_to(idx)]

        if not valid_neighbors:
            return None

        # Calculate selection probabilities
        probabilities = []
        for neighbor_idx in valid_neighbors:
            tau = self.graph.get_pheromone(ant.current_idx, neighbor_idx)
            eta = self.triangle_inequality_heuristic(
                ant.current_idx, neighbor_idx)
            prob = (tau ** self.alpha) * (eta ** self.beta)
            probabilities.append(prob)

        # Normalize probabilities
        total_prob = sum(probabilities)
        if total_prob == 0:
            return random.choice(valid_neighbors)

        probabilities = [p / total_prob for p in probabilities]

        # Pseudo-random selection
        if random.random() < self.q:
            # Exploitation
            best_idx = np.argmax(probabilities)
            return valid_neighbors[best_idx]
        else:
            # Exploration
            selected_idx = np.random.choice(
                len(valid_neighbors), p=probabilities)
            return valid_neighbors[selected_idx]

    def backtrack_ant(self, ant: Ant) -> bool:
        if len(ant.path) <= 1:
            return False

        # Remove current position from visited
        ant.visited.remove(ant.current_idx)

        # Backtrack
        ant.path.pop()
        if ant.path:
            ant.current_idx = ant.path[-1]

            # Check for valid moves
            neighbors = self.graph.get_neighbors(ant.current_idx)
            valid_neighbors = [
                idx for idx in neighbors if ant.can_move_to(idx)]

            if valid_neighbors:
                return True
            else:
                return self.backtrack_ant(ant)

        return False

    def move_ant(self, ant: Ant) -> bool:
        next_idx = self.pseudo_random_state_transition(ant)

        if next_idx is None:
            # Try backtracking
            if self.backtrack_ant(ant):
                next_idx = self.pseudo_random_state_transition(ant)
                if next_idx is None:
                    ant.stuck = True
                    return False
            else:
                ant.stuck = True
                return False

        # Move ant
        distance = self.graph.get_edge_weight(ant.current_idx, next_idx)
        ant.total_distance += distance
        ant.current_idx = next_idx
        ant.path.append(next_idx)
        ant.visited.add(next_idx)

        return True

    def pheromone_update(self, best_ant: Ant):
        if not best_ant or not best_ant.path:
            return

        # Evaporation
        for edge_key in self.graph.pheromones:
            current = self.graph.pheromones[edge_key]
            self.graph.pheromones[edge_key] = max(
                self.tau_min, (1 - self.rho) * current)

        # New Pheromone
        delta_tau = 1.0 / best_ant.total_distance
        for i in range(len(best_ant.path) - 1):
            node1_idx = best_ant.path[i]
            node2_idx = best_ant.path[i + 1]

            current_pheromone = self.graph.get_pheromone(node1_idx, node2_idx)
            new_pheromone = min(self.tau_max, current_pheromone + delta_tau)
            self.graph.update_pheromone(node1_idx, node2_idx, new_pheromone)

    def run_single_ant(self, ant: Ant) -> bool:
        max_steps = len(self.graph.nodes) * 2
        steps = 0

        while (ant.current_idx != self.graph.goal_idx and
               not ant.stuck and steps < max_steps):
            if not self.move_ant(ant):
                break
            steps += 1

        return ant.current_idx == self.graph.goal_idx

    def optimize(self) -> Tuple[List[int], float]:
        for iteration in range(self.max_iterations):
            # Initialize ants
            ants = [Ant(self.graph.start_idx) for _ in range(self.n_ants)]

            # Run all ants
            successful_ants = []
            for ant in ants:
                if self.run_single_ant(ant):
                    successful_ants.append(ant)

            # Find best ant
            if successful_ants:
                iteration_best = min(
                    successful_ants, key=lambda a: a.total_distance)

                if iteration_best.total_distance < self.best_distance:
                    self.best_distance = iteration_best.total_distance
                    self.best_path = iteration_best.path.copy()

                # Pheromone update
                self.pheromone_update(iteration_best)

            # Record convergence
            current_best = self.best_distance if self.best_path else float(
                'inf')
            self.convergence_data.append(current_best)

            # Early convergence check
            if len(self.convergence_data) > 10:
                recent_improvements = [abs(self.convergence_data[i] - self.convergence_data[i-1])
                                       for i in range(-10, -1)]
                if all(imp < 0.01 for imp in recent_improvements):
                    print(f"Converged at iteration {iteration}")
                    break

        return self.best_path, self.best_distance


# Set start and goal (z,y,x)
start = (0, 0, 0)
goal = (0, 19, 19)

# Load the map
num = 11
occ = np.load(f'benchmark_maps/20x20/map{num}.npy')
top = np.zeros(occ.shape)
grid = np.stack([occ, top], axis=0)

graph = create_3d_graph_from_array(start, goal, grid)

# Initialize ACO algorithm
aco = ACO(
    graph=graph,
    n_ants=5,
    max_iterations=100,
    alpha=1.0,
    beta=2.0,
    rho=0.1,
    q=0.8,
    xi=0.1
)

# Run optimization
best_path, best_distance = aco.optimize()

# Display results
if best_path:
    print(f"Path found with {len(best_path)} points")

    print(f"Path Points: {[tuple(map(int, graph.nodes[i]))
          for i in best_path]}")

    # Visualize results
    plot_path.visualize_path([graph.nodes[i] for i in best_path],
                             graph.map, start, goal,
                             "ACO Pathfinding Results")

else:
    print("No path found!")
