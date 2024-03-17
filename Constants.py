"""
Constants Declaration.
"""

# Maze objects and their respective rewards.
G = +1
""" Green squares with +1 reward. """

B = -1
""" Brown squares with -1 reward. """

W = None
""" Walls. Unaccessible states in the maze. """

E = -0.04
""" Empty squares with -0.04 reward. """

PART1_MAZE = [[G, W, G, E, E, G],
              [E, B, E, G, W, B],
              [E, E, B, E, G, E],
              [E, E, E, B, E, G],
              [E, W, W, W, B, E],
              [E, E, E, E, E, E]]
""" Maze given in assignment Part 1. """

START_STATE = (3,2)
""" Start position in format `(row,col)`. """

DISCOUNT_FACTOR = 0.99
""" Discount factor given in assignment. """

RESULTS_DIR_PATH = 'results/'

# Approximation of reference utilities
REFERENCE_DISCOUNT_FACTOR = 0.95
""" Approximation of reference discount factor. """

REFERENCE_MAX_ERROR = 1.5
""" Approximation of reference maximum error in value iteration. """

MAX_ERROR = 20
""" Maximum error in value iteration. """

NUM_POLICY_EVALUATION = 100
""" Number of times policy evaluation is carried out before every policy update in policy iteration. """