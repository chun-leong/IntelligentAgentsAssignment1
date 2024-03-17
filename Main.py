"""
Main driver code.
"""

from pprint import pprint

from Algorithms import value_iteration, policy_iteration
from Constants import *
from Maze import Maze
from Utility import plot_utility_iteration_graph, generate_maze, log_mdp_result


def solve_assignment1():
    """
    Runs all the code necessary to answer the questions in Assignment 1.
    """
    solve_reference_utilities()
    solve_q1_value_iteration()
    solve_q1_policy_iteration()
    solve_q2()


def solve_reference_utilities():
    # Reference utilities
    maze = Maze(
        environment=PART1_MAZE,
        start_state=START_STATE,
        discount_factor=REFERENCE_DISCOUNT_FACTOR
    )
    result = value_iteration(maze, max_error=REFERENCE_MAX_ERROR)

    log_mdp_result(maze, result, 'approximate_reference_utilities_result.txt')

    plot_utility_iteration_graph(
        result['iteration_utilities'],
        save_file_name='approximate_reference_utilities.png'
    )


def solve_q1_value_iteration():
    # Q1 - Value iteration
    maze = Maze(
        environment=PART1_MAZE,
        start_state=START_STATE,
        discount_factor=DISCOUNT_FACTOR
    )
    result = value_iteration(maze, max_error=MAX_ERROR)

    log_mdp_result(maze, result, 'value_iteration_result.txt')

    plot_utility_iteration_graph(
        result['iteration_utilities'],
        save_file_name='value_iteration_utilities.png'
    )

def solve_q1_policy_iteration():
    # Q1 - Policy iteration
    maze = Maze(
        environment=PART1_MAZE,
        start_state=START_STATE,
        discount_factor=DISCOUNT_FACTOR
    )
    result = policy_iteration(maze, num_policy_evaluation=100)
    
    log_mdp_result(maze, result, 'policy_iteration_result.txt')

    plot_utility_iteration_graph(
        result['iteration_utilities'],
        save_file_name='policy_iteration_utilities.png'
    )


def solve_q2():
    # Q2 - Custom maze environment
    bonus_grid_length = 100
    bonus_grid = generate_maze(bonus_grid_length)

    print('bonus grid, length =', bonus_grid_length)
    with open(RESULTS_DIR_PATH + 'bonus_maze.txt', 'w') as file:
        pprint(bonus_grid, file)

    # Q2 - Value iteration
    bonus_maze = Maze(
        environment=bonus_grid,
        start_state=START_STATE,
        discount_factor=DISCOUNT_FACTOR
    )

    result = value_iteration(bonus_maze, max_error=MAX_ERROR)

    log_mdp_result(
        bonus_maze, 
        result, 
        'bonus_value_iteration_result.txt'
    )

    plot_utility_iteration_graph(
        result['iteration_utilities'],
        'bonus_value_iteration_utilities.png'
    )

    # Q2 - Policy iteration
    result = policy_iteration(bonus_maze, NUM_POLICY_EVALUATION)

    log_mdp_result(
        bonus_maze, 
        result, 
        'bonus_policy_iteration_result.txt'
    )

    plot_utility_iteration_graph(
        result['iteration_utilities'],
        'bonus_policy_iteration_utilities.png'
    )


if __name__ == '__main__':
    solve_assignment1()