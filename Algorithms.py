"""
Methods used in value iteration and policy iteration.
"""

from MarkovDecisionProcess import MarkovDecisionProcess
from Maze import MazeActions

def value_iteration(
    mdp: MarkovDecisionProcess,
    max_error=1,
):
    """
    An iterative approach of updating the utility of each state from the utilities of its neighbors using the Bellman equation until equilibrium is reached.

    Parameters
    ----------
    - mdp: MarkovDecisionProcess - An MDP with states S, actions A(s), transition model P (s' | s, a), rewards R(s), discount γ.
    - max_error - The maximum error, ε, allowed in the utility of any state.

    Returns
    -------
    - converged_utilities: {state: utility} - The utilities of each state at equilibrium.
    - optimal_policy: {state: optimal_action} - The action to take at each state to maximise expected utility.
    - iteration_utilities: {state: [record_of_utilities]} - A record of every state's utilities at each iteration.
    """
    
    # Local variables U and U' respectively
    current_utilities, new_utilities = {}, {}

    # Used to record of each state's utilities at each iteration
    all_iteration_utilities = {}

    # Used to store the action that maximises expected utility at each state 
    optimal_policy = {}

    for state in mdp.states:
        # Initialise U' (which is used to initialise U later) as zero
        new_utilities[state] = 0

        # Initialise storage
        all_iteration_utilities[state] = []
        optimal_policy[state] = None

    while True:
        # U ← U'
        current_utilities.update(new_utilities)

        for state in mdp.states:
            all_iteration_utilities[state].append(current_utilities[state])

        # δ
        max_utility_difference = 0

        for state in mdp.states:
            new_utilities[state], optimal_policy[state] = bellman_equation(
                mdp,
                state,
                current_utilities, 
            )

            max_utility_difference = max(max_utility_difference, abs(new_utilities[state] - current_utilities[state]))

        # until δ < ε(1−γ)/γ
        if (max_utility_difference < max_error * (1 - mdp.discount_factor) / mdp.discount_factor): 
            break

    return {
        'converged_utilities': current_utilities,
        'optimal_policy': optimal_policy,
        'num_iterations': len(all_iteration_utilities.get(mdp.start_state)),
        'iteration_utilities': all_iteration_utilities,
    }


def policy_iteration(
    mdp: MarkovDecisionProcess, 
    num_policy_evaluation: int=1
    ):
    """
    An iterative approach of evaluating and improving some initial policy until the utilities no longer change after each improvement iteration.

    Parameters
    ----------
    - mdp: MarkovDecisionProcess - An MDP with states S, actions A(s), transition model P (s' | s, a), rewards R(s), discount γ.
    - num_policy_evaluation: int - The number of iterations of policy evaluation to perform at once to produce the next utility estimate.
    When this value is more than one, the algorithm is known as the modified policy iteration.

    Returns
    -------
    - converged_utilities: {state: utility} - The utilities of each state at equilibrium.
    - optimal_policy: {state: optimal_action} - The action to take at each state to maximise expected utility.
    - iteration_utilities: {state: [record_of_utilities]} - A record of every state's utilities at each iteration.
    """
    
    # Local variables U and π respectively
    current_utilities, policy = {}, {}

    # Used to record of each state's utilities at each iteration
    all_iteration_utilities = {}

    for state_position in mdp.states:
        # Initialise U as zero
        current_utilities[state_position] = 0
        # Initialise π arbitrarily
        policy[state_position] = MazeActions.UP

        # Initialise storage and record the zeroth iteration's utilities
        all_iteration_utilities[state_position] = [0]

    hasChanged = True
    num_iterations = 0

    while hasChanged:
        # Policy evaluation
        current_utilities, new_iteration_utilities = policy_evaluation(
            mdp, 
            policy, 
            current_utilities, 
            num_policy_evaluation, 
        )

        # Policy improvement
        policy, hasChanged = policy_improvement(mdp, policy, current_utilities)

        # Useful if multiple policy evaluations are performed before each policy improvement
        num_iterations += num_policy_evaluation

        for state_position in mdp.states:
            all_iteration_utilities[state_position].extend(new_iteration_utilities[state_position])

    return {
        'converged_utilities': current_utilities,
        'optimal_policy': policy,
        'num_iterations': len(all_iteration_utilities.get(mdp.start_state)),
        'iteration_utilities': all_iteration_utilities,
    }


def policy_evaluation(
    mdp: MarkovDecisionProcess,
    policy: dict, 
    utilities: dict, 
    num_policy_evaluation: int,
):
    """
    Updates utilities by using a simplified version of the Bellman equation because of fixing the action at each state by the policy in policy iteration.

    Parameters
    ----------
    - mdp: MarkovDecisionProcess - An MDP with states S, actions A(s), transition model P (s' | s, a), rewards R(s), discount γ.
    - policy: : {state: policy_action} - The action to take at each state to maximise expected utility in the current iteration.
    - utilities: {state: utility} - The utilities of each state at the current iteration.
    - num_policy_evaluation: int - The number of iterations of policy evaluation to perform at once to produce the next utility estimate.

    Returns
    -------
    - current_utilities {state: utility} - The utility estimates after performing policy evaluation for the number of times specified by `num_policy_evaluation`.
    - new_iteration_utilities {state: [record_of_utilities]} - The record of every state's utilities at each iteration from input to the current utility values. 
    )
    """
    
    current_utilities, updated_utilities = {}, {}
    new_iteration_utilities = {}

    for state in mdp.states:
        current_utilities[state] = utilities[state]
        new_iteration_utilities[state] = []

    for i in range(num_policy_evaluation):
        for state in mdp.states:
            current_state_reward = mdp.reward_function(state)

            expected_utility = get_expected_utility(
                mdp,
                state,
                policy[state],
                current_utilities
            )

            # Simplified Bellman Equation where the action is fixed (max operator is removed)
            updated_utilities[state] = current_state_reward + mdp.discount_factor * expected_utility

        for state in mdp.states:
            current_utilities[state] = updated_utilities[state]
            new_iteration_utilities[state].append(current_utilities[state])
            
    return (current_utilities, new_iteration_utilities)


def policy_improvement(
    mdp,
    policy,
    utilities, 
):
    """
    Calculates a new policy, using based on utilities obtained from policy evaluation.

    Parameters
    ----------
    - mdp: MarkovDecisionProcess - An MDP with states S, actions A(s), transition model P (s' | s, a), rewards R(s), discount γ.
    - policy: {state: policy_action} - The action to take at each state to maximise expected utility according to the old policy.
    - utilities: {state: utility} - The utilities of each state at the current iteration.

    Returns
    -------
    - improved_policy: {state: policy_action} - The action to take at each state to maximise expected utility using the new utility values.
    - hasChanged: bool - Indicates if utility has improved after improving the policy.
    )
    """
    
    improved_policy = {}
    hasChanged = False

    for state in mdp.states:
        max_expected_utility = float('-inf')
        best_action = None

        action_next_state_map = mdp.states[state]

        for action in action_next_state_map:
            expected_utility = get_expected_utility(
                mdp,
                state,
                action,
                utilities
            )

            if expected_utility > max_expected_utility:
                max_expected_utility = expected_utility
                best_action = action

        current_expected_utility = get_expected_utility(
            mdp,
            state,
            policy[state],
            utilities
        )

        if max_expected_utility > current_expected_utility:
            improved_policy[state] = best_action
            hasChanged = True
        else:
            improved_policy[state] = policy[state]

    return (improved_policy, hasChanged)


def bellman_equation(
    mdp: MarkovDecisionProcess,
    current_state: tuple, 
    current_utilities: dict, 
):
    """
    Defines the utility of a state as the immediate reward for that state plus the 
    expected discounted utility of the next state, assuming that the agent chooses 
    the optimal action.

    Parameters
    ----------
    - mdp: MarkovDecisionProcess - An MDP with states S, actions A(s), transition model P (s' | s, a), rewards R(s), discount γ.
    - current_state: tuple - Coordinates in the 2D maze formatted as `(row,col)`.
    - current_utilities: {state: utility} - The utilities of each state at the current iteration.

    Returns
    -------
    - updated_utility: float - The updated maximum expected utility achievable at the `current_state`.
    - best_action: MazeActions - The action to take at the `current_state` to maximise expected utility.
    )
    """
    
    max_expected_utility = float('-inf')
    best_action = None

    action_next_state_map = mdp.states[current_state]

    current_state_reward = mdp.reward_function(current_state)

    for action in action_next_state_map:
        expected_utility = get_expected_utility(
            mdp,
            current_state,
            action,
            current_utilities
        )

        if expected_utility > max_expected_utility:
            max_expected_utility = expected_utility
            best_action = action

    updated_utility = current_state_reward + mdp.discount_factor * max_expected_utility
    
    return (updated_utility, best_action)


def get_expected_utility(
    mdp: MarkovDecisionProcess,
    current_state: tuple, 
    intended_action: MazeActions,
    utilities: dict, 
) -> float:
    """
    The sum of the expected utility of all possible states after taking an action at a specific state.

    params:
    - mdp: MarkovDecisionProcess - An MDP with states S, actions A(s), transition model P (s' | s, a), rewards R(s), discount γ.
    - current_state: tuple - Coordinates in the 2D maze formatted as `(row,col)`.
    - action: MazeAction - Intended action to take.
    - utilities: {state: utility} - The utilities of each state at the current iteration.

    Returns
    -------
    float - Expected utility value of the input scenario
    """
    
    expected_utility = 0

    stochastic_next_states = mdp.states[current_state][intended_action]

    for stochastic_next_state in stochastic_next_states:
        next_state = stochastic_next_states[stochastic_next_state]["next_state"]

        next_state_utility = utilities[next_state]

        expected_utility += mdp.transition_model(current_state, intended_action, stochastic_next_state) * next_state_utility

    return expected_utility