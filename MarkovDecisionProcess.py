class MarkovDecisionProcess:
    """ 
    The base representation of a Markov Decision Process. 
    
    A sequential decision problem for a fully observable, stochastic environment with a Markovian transition model and additive rewards.

    Attributes
    ----------
    - states - The set of possible states that agents in this environment can achieve.
    - actions - The set of actions that agents in this environment can perform at each state.
    - discount_factor - The factor which future rewards are discounted at.

    Methods
    -------
    - transition_model() - Returns the probability of achieving a certain state s', after performing a certain action a, when in state s.
    - reward_function() - Returns the reward achieved by agents in a certain state.
    """
    
    def __init__(self, states, actions, discount_factor):
        """
        Parameters
        ----------
        - states - The set of possible states that agents in this environment can achieve.
        - actions - The set of actions that agents in this environment can perform at each state.
        - discount_factor - The factor which future rewards are discounted at.
        """
        
        self.states = states
        self.actions = actions
        self.discount_factor = discount_factor
        

    def transition_model(self):
        """ Returns the probability of achieving a certain state s', after performing a certain action a, when in state s. """
        
        pass
    
    
    def reward_function(self):
        """ Returns the reward achieved by agents in a certain state. """
        
        pass