from MarkovDecisionProcess import MarkovDecisionProcess
from enum import Enum
from operator import sub

class MazeActions(Enum):
    """ Enums of the possible actions that agents can perform at each stage in the maze. """
    
    UP = (-1,0)
    DOWN = (+1,0)
    LEFT = (0,-1)
    RIGHT = (0,+1)
    
class Maze(MarkovDecisionProcess):
    """
    Represents a 2D maze problem as a Markov Decision Process.

    Attributes
    ----------
    - states - The set of possible states that agents in this maze can achieve.
    - actions - The set of actions that agents can perform at each state in the maze.
    - discount_factor - The factor which future rewards are discounted at.
    - environment - Representation of the environment as a 2D array of rewards obtainable at each state.
    - start_state - The initial state of the agent in the environment.

    Methods
    -------
    - transition_model(current_state, action_taken: MazeActions, next_state) - Returns the probability of achieving `next_state`, after performing `action_taken`, when in state `current_state`.
    - reward_function(state) - Returns the reward achieved by agents in `state`.
    - get_next_state_probabilities(state, action: MazeActions) - Returns the possible states as a result of performing `action` in `state`.
    - is_wall(self, state) - Utility method to check if `state` is blocked off to an agent. I.e. out of bounds of the maze or a wall.
    - build_action_next_state_map(self, state) - Builder method to return all possible states obtainable and their probabilities at a given `state`.
    """
    
    def __init__(self, environment, start_state, discount_factor):
        """
        Parameters
        ----------
        - environment - Representation of the environment as a 2D array of states and the rewards obtainable in each one.
        - start_state - The initial state of the agent in the environment.
        - discount_factor - The factor which future rewards are discounted at.
        """
        
        self.environment = environment
        self.start_state = start_state
        
        states = {(row,col): {} for row in range(len(environment)) for col in range(len(environment[0])) if not self.is_wall((row, col))}
        for state in states:
            states[state] = self.build_action_next_state_map(state)
        
        actions = [action for action in MazeActions]

        super().__init__(states, actions, discount_factor)

        
    def transition_model(self, current_state, intended_action: MazeActions, actual_action: MazeActions):
        """
        Returns the probability of achieving `next_state`, after performing `action_taken`, when in state `current_state`.

        Parameters
        ----------
        - current_state - Coordinates in the 2D maze formatted as `(row,col)`.
        - action_taken: MazeActions - The intended action taken by the agent.
        - next_state - The resultant state, of which the probability of obtaining is returned.

        Returns
        -------
        float - Probability of the input scenario happening.
        """
        
        next_state_probability_map = self.states[current_state][intended_action]           
        next_state_probability = next_state_probability_map.get(actual_action)
        return next_state_probability["probability"] if next_state_probability else 0
    

    def reward_function(self, state):
        """
        Returns the reward achieved by agents in `state`.

        Parameters
        ----------
        - state - Coordinates in the 2D maze formatted as `(row,col)`.

        Returns
        -------
        float - Reward achieved by an agent in this state.
        """
        
        return self.environment[state[0]][state[1]]


    def is_wall(self, state):
        """
        Utility method to check if `state` is blocked off to an agent. I.e. out of bounds of the maze or a wall.

        Parameters
        ----------
        - state - Coordinates in the 2D maze formatted as `(row,col)`.

        Returns
        -------
        boolean - `True` if this state is obstructed from the agent; `False` otherwise.
        """
        
        return state[0] < 0 or state[0] >= len(self.environment) or \
            state[1] < 0 or state[1] >= len(self.environment[0]) or \
            self.environment[state[0]][state[1]] is None
    

    def build_action_next_state_map(self, state):
        """
        Builder method to return all possible states obtainable and their probabilities at a given `state`.

        Parameters
        ----------
        - state - Coordinates in the 2D maze formatted as `(row,col)`.

        Returns
        -------
        dict - `key`s represent the possible actions taken; `value`s represent the possible states obtainable from performing that action and their probabilities.
        """
        
        return {
            action: self.get_next_state_probabilities(state, action)
            for action in MazeActions
        }


    def get_next_state_probabilities(self, state, action: MazeActions):
        """
        Returns the possible states obtainable and their probabilities by performing `action` at a given `state`.

        Parameters
        ----------
        - state - Coordinates in the 2D maze formatted as `(row,col)`.
        - action - Intended action to take.

        Returns
        -------
        dict - `key`s represent the possible states; `value`s represent the probabilities of that state.
        """
        
        up_state = (state[0] - 1, state[1])
        up_state = state if self.is_wall(up_state) else up_state

        down_state = (state[0] + 1, state[1])
        down_state = state if self.is_wall(down_state) else down_state

        left_state = (state[0], state[1] - 1)
        left_state = state if self.is_wall(left_state) else left_state

        right_state = (state[0], state[1] + 1)
        right_state = state if self.is_wall(right_state) else right_state

        if action is MazeActions.UP:
            next_states = { 
                MazeActions.UP: {
                    "next_state": up_state,
                    "probability": 0.8
                    },
                MazeActions.LEFT: {
                    "next_state": left_state,
                    "probability": 0.1
                    },
                MazeActions.RIGHT: {
                    "next_state": right_state,
                    "probability": 0.1
                    }
            }

        elif action is MazeActions.DOWN:
            next_states = {
                MazeActions.DOWN: {
                    "next_state": down_state,
                    "probability": 0.8
                    },
                MazeActions.LEFT: {
                    "next_state": left_state,
                    "probability": 0.1
                    },
                MazeActions.RIGHT: {
                    "next_state": right_state,
                    "probability": 0.1
                    }
            }

        elif action is MazeActions.LEFT:
            next_states = {
                MazeActions.LEFT: {
                    "next_state": left_state,
                    "probability": 0.8
                    },
                MazeActions.UP: {
                    "next_state": up_state,
                    "probability": 0.1
                    },
                MazeActions.DOWN: {
                    "next_state": down_state,
                    "probability": 0.1
                    }
            }

        elif action is MazeActions.RIGHT:
            next_states = {
                MazeActions.RIGHT: {
                    "next_state": right_state,
                    "probability": 0.8
                    },
                MazeActions.UP: {
                    "next_state": up_state,
                    "probability": 0.1
                    },
                MazeActions.DOWN: {
                    "next_state": down_state,
                    "probability": 0.1
                    }
            }

        return next_states