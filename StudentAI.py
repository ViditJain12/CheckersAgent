import random
from BoardClasses import Move, Board
from copy import deepcopy
from math import sqrt, log
from operator import attrgetter

# Constants
PLAYER_MAPPING = {1: 2, 2: 1}  # Maps each player to their opponent


def select_random_action(game_state, player) -> Move:
    """
    Selects a random move from the available options for the given player.
    """
    possible_actions = game_state.get_all_possible_moves(player)
    return random.choice(random.choice(possible_actions))


class StudentAI:
    def __init__(self, columns, rows, pieces):
        self.columns = columns
        self.rows = rows
        self.pieces = pieces
        self.game_state = Board(columns, rows, pieces)
        self.game_state.initialize_game()
        self.current_player = 2  # Default player
        self.mcts_tree = MonteCarloTree(TreeNode(self.game_state, self.current_player, None, None))
        self.simulation_limit = 1500  # Number of simulations per move

    def get_move(self, opponent_action) -> Move:
        """
        Determines the best move using MCTS and updates the game state.
        Core logic flow:
        1. Handle opponent's move or initialize first move
        2. Check for forced moves
        3. Perform MCTS search if needed
        4. Execute and return chosen move
        """
        # Handle initial move or opponent's move
        if not opponent_action:
            return self._handle_first_move()
        
        self._process_opponent_move(opponent_action)
        
        # Check for forced move situation
        if (forced := self._get_forced_move()):
            return forced
        
        # Perform MCTS exploration and return best move
        return self._perform_mcts_search()

    # Helper methods to maintain the same logic in modular form
    def _handle_first_move(self) -> Move:
        """Handles the initial move of the game"""
        self.current_player = 1
        self.mcts_tree.root = TreeNode(self.game_state, self.current_player, None, None)
        initial_move = self.game_state.get_all_possible_moves(self.current_player)[0][1]
        self.execute_action(initial_move, self.current_player)
        return initial_move

    def _process_opponent_move(self, move):
        """Processes the opponent's move in the game"""
        self.execute_action(move, PLAYER_MAPPING[self.current_player])

    def _get_forced_move(self) -> Move | None:
        """Returns forced move if only one exists, otherwise None"""
        moves = self.game_state.get_all_possible_moves(self.current_player)
        if len(moves) == 1 and len(moves[0]) == 1:
            forced_move = moves[0][0]
            self.execute_action(forced_move, self.current_player)
            return forced_move
        return None

    def _perform_mcts_search(self) -> Move:
        """Executes MCTS and returns the best found move"""
        best_move = self.mcts_tree.explore(self.simulation_limit)
        self.execute_action(best_move, self.current_player)
        return best_move

    def execute_action(self, action, player):
        """
        Updates the game state and adjusts the MCTS tree accordingly.
        """
        self.game_state.make_move(action, player)
        
        if action in self.mcts_tree.root.children and self.mcts_tree.root.children[action]:
            self.mcts_tree.root = self.mcts_tree.root.children[action]
            self.mcts_tree.root.parent = None
        else:
            self.mcts_tree.root = TreeNode(self.game_state, PLAYER_MAPPING[player], None, None)

class MonteCarloTree:
    def __init__(self, root):
        self.root = root

    def simulate(self, node) -> int:
        """Simulates a game from the given node and returns the result."""
        simulation_state, current_player = self._prepare_simulation(node)
        final_outcome = self._run_simulation_cycles(simulation_state, current_player)
        return self._determine_result(node, final_outcome)

    def _prepare_simulation(self, node):
        """Creates a copy of the game state and initializes player"""
        return deepcopy(node.game_state), node.current_player

    def _run_simulation_cycles(self, state, player):
        """Executes simulation moves until game conclusion"""
        outcome = self._check_initial_win(state, player)
        while not outcome:
            outcome = self._execute_simulation_step(state, player)
            player = PLAYER_MAPPING[player]
        return outcome

    def _check_initial_win(self, state, player):
        """Checks if opponent has already won before any moves"""
        return state.is_win(PLAYER_MAPPING[player])

    def _execute_simulation_step(self, state, player):
        """Performs one complete simulation step (move + win check)"""
        action = select_random_action(state, player)
        state.make_move(action, player)
        return state.is_win(player)

    def _determine_result(self, node, outcome):
        """Translates game outcome to MCTS result values"""
        if outcome == PLAYER_MAPPING[node.current_player]:
            return 1  # Win for parent node
        if outcome == node.current_player:
            return -1  # Loss for parent node
        return 0  # Draw

    def explore(self, simulation_limit) -> Move:
        """
        Runs MCTS for a set number of simulations and returns the best move.
        """
        for _ in range(simulation_limit):
            node = self.select_node(self.root)
            outcome = self.simulate(node)
            node.update_tree(outcome)
        return self.select_best_action()

    def select_node(self, node) -> 'TreeNode':
        """
        Selects a node to explore based on UCB values or expands a new node.
        """
        if not node.children:
            return node
        if None in node.children.values():
            for action, child in node.children.items():
                if child is None:
                    node.children[action] = TreeNode(node.game_state, PLAYER_MAPPING[node.current_player], action, node)
                    return node.children[action]
        return self.select_node(max(node.children.values(), key=attrgetter('ucb_value')))

    def select_best_action(self) -> Move:
        """
        Chooses the most visited action.
        """
        return max(self.root.children.items(), key=lambda x: x[1].visit_count)[0]

class TreeNode:
    def __init__(self, game_state, current_player, action, parent):
        self.game_state = deepcopy(game_state)
        self.current_player = current_player
        self.parent = parent
        self.visit_count = 1
        self.wins_for_parent = 0
        self.ucb_value = 0

        if action:
            self.game_state.make_move(action, PLAYER_MAPPING[self.current_player])

        self.children = {}
        if self.game_state.is_win(PLAYER_MAPPING[self.current_player]) == 0:
            for action_group in self.game_state.get_all_possible_moves(self.current_player):
                for action in action_group:
                    self.children[action] = None

    def update_tree(self, result):
        """
        Updates node statistics and propagates results upwards.
        """
        self.visit_count += 1
        if self.parent:
            self.parent.update_tree(-result)
            self.wins_for_parent += max(result, 0)
            self.ucb_value = (self.wins_for_parent / self.visit_count + sqrt(2) * sqrt(log(self.parent.visit_count) / self.visit_count))
