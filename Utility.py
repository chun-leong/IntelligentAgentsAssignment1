"""
Helper methods.
"""

import matplotlib.pyplot as plt
import random
import os
import copy
from pprint import pprint

from Constants import RESULTS_DIR_PATH, G, B, W, E
from Maze import Maze, MazeActions


def generate_maze(side_length: int, green_dist: float = 1/6, brown_dist: float = 1/6, wall_dist: float = 1/6):
    """ 
    Generates a square maze with random placement of objects.

    Parameters
    ----------
    - side_length : int - Length of each edge of the maze.
    - green_dist: float - Probability distribution of green squares in the maze. Value should be between 0-1. (Default = 1/6).
    - brown_dist: float - Probability distribution of brown squares in the maze. Value should be between 0-1. (Default = 1/6).
    - wall_dist: float - Probability distribution of walls in the maze. Value should be between 0-1. (Default = 1/6).

    Remarks
    -------
    Sum of distributions should be no more than 1. Default values for distribution will be used otherwise.
    """
    if (green_dist + brown_dist + wall_dist > 1):
        green_dist = 1/6
        brown_dist = 1/6
        wall_dist = 1/6

    random.seed()
    maze = []

    for row in range(side_length):
        maze.append([])
        
        for col in range(side_length):
            object = random.random()

            # Object allocation chance: |-Wall (1/6)-|-Brown (1/6)-|-Green(1/6)-|-----------Empty (1/2)-----------|
            if object < green_dist:
                maze[row].append(G)
            elif object < green_dist + brown_dist:
                maze[row].append(B)
            elif object < green_dist + brown_dist + wall_dist:
                maze[row].append(W)
            else:
                maze[row].append(E)

    return maze


def plot_utility_iteration_graph(iteration_utilities, save_file_name=None):
    """
    Utility function to plot the changes of each state's utility over iterations of updating.
    
    Parameters
    ----------
    - iteration_utilities: {state: [record_of_utilities]} - The record of utilities for each state.
    - save_file_name = None: string - Name of file to save plot as.
    """
    plt.figure(figsize=(16, 8))

    for state_position in iteration_utilities:
        plt.plot(iteration_utilities[state_position])

    plt.legend(iteration_utilities, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.title('Utility of each state against iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Utility estimate')

    if save_file_name is not None:
        plt.savefig(RESULTS_DIR_PATH + save_file_name)

    plt.show()


def log_mdp_result(maze, result, save_file_name=None):
    """
    Logs the results of MDP into a text file.

    Parameters
    ----------
    - maze: Maze - Maze MDP.
    - result: - Result of solving the `maze` MDP. Should contain the final utilities 
    of each state, the optimal policy, the record of utilities per state, and the 
    number of iterations of utility updating.
    - save_file_name: string - The name of file to save logs to. (Default is None - no file will be saved)
    """
    lines = []

    line = 'Number of iterations: ' + str(result['num_iterations'])
    print(line)
    lines.append(line)

    utilities, optimal_policy = result['converged_utilities'], result['optimal_policy']

    optimal_policy_display = copy.deepcopy(maze.environment)
    action_symbol_map = {
        MazeActions.UP: '∧',
        MazeActions.DOWN: 'v',
        MazeActions.LEFT: '<',
        MazeActions.RIGHT: '>',
    }

    line = '---Utility for each state in the maze---'
    print(line)
    lines.append(line)

    for state_position in maze.states:
        if state_position[1] == 0:
            print()

        line = str(state_position) + ': {:.3f}'.format(utilities[state_position])
        print(line)
        lines.append(line)

        action = optimal_policy[state_position]
        action_symbol = action_symbol_map[action] if action else ' '

        optimal_policy_display[state_position[0]][state_position[1]] = action_symbol

    for row in range(len(optimal_policy_display)):
        for col in range(len(optimal_policy_display[0])):
            if not optimal_policy_display[row][col]:
                optimal_policy_display[row][col] = '-'

    line = '---Optimal policy grid---'
    print(line)
    lines.append(line)

    pprint(optimal_policy_display)
    print()

    if not os.path.exists(RESULTS_DIR_PATH):
        os.makedirs(RESULTS_DIR_PATH)

    if save_file_name is not None:
        with open(RESULTS_DIR_PATH + save_file_name, 'w+', encoding="utf-8") as file:
            for line in lines:
                file.write(line + '\n')
            pprint(optimal_policy_display, file)
            