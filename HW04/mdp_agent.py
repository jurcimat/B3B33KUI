#!/usr/bin/env python3

import kuimaze
import random
import os
import time
import sys
import copy

MAP = 'maps/easy/easy1.bmp'
MAP = os.path.join(os.path.dirname(os.path.abspath(__file__)), MAP)
PROBS = [0.4, 0.3, 0.3, 0]
GRAD = (0, 0)
SKIP = False
SAVE_EPS = False
VERBOSITY = 0


GRID_WORLD4 = [[[255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 0, 0]],
               [[255, 255, 255], [0, 0, 0], [255, 255, 255], [255, 255, 255]],
               [[0, 0, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255]],
               [[255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255]]]

GRID_WORLD3 = [[[255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 0, 0]],
               [[255, 255, 255], [0, 0, 0], [255, 255, 255], [255, 0, 0]],
               [[0, 0, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255]]]

REWARD_NORMAL_STATE = -0.04
REWARD_GOAL_STATE = 1
REWARD_DANGEROUS_STATE = -1

GRID_WORLD3_REWARDS = [[REWARD_NORMAL_STATE, REWARD_NORMAL_STATE, REWARD_NORMAL_STATE, REWARD_GOAL_STATE],
                       [REWARD_NORMAL_STATE, 0, REWARD_NORMAL_STATE, REWARD_DANGEROUS_STATE],
                       [REWARD_NORMAL_STATE, REWARD_NORMAL_STATE, REWARD_NORMAL_STATE, REWARD_NORMAL_STATE]]


def wait_n_or_s():
    # This function was given in assignment
    def wait_key():
        '''
        returns key pressed ... works only in terminal! NOT in IDE!
        '''
        result = None
        if os.name == 'nt':
            import msvcrt
            result = msvcrt.getch()
        else:
            import termios
            fd = sys.stdin.fileno()

            oldterm = termios.tcgetattr(fd)
            newattr = termios.tcgetattr(fd)
            newattr[3] = newattr[3] & ~termios.ICANON & ~termios.ECHO
            termios.tcsetattr(fd, termios.TCSANOW, newattr)
            try:
                result = sys.stdin.read(1)
            except IOError:
                pass
            finally:
                termios.tcsetattr(fd, termios.TCSAFLUSH, oldterm)
        return result
    '''
    press n - next, s - skip to end ... write into terminal
    '''
    global SKIP
    x = SKIP
    while not x:
        key = wait_key()
        x = key == 'n'
        if key == 's':
            SKIP = True
            break


def get_visualisation_values(dictvalues):
    # This function was given in assignment
    if dictvalues is None:
        return None
    ret = []
    for key, value in dictvalues.items():
        # ret.append({'x': key[0], 'y': key[1], 'value': [value, value, value, value]})
        ret.append({'x': key[0], 'y': key[1], 'value': value})
    return ret


# the init functions are provided for your convenience, modify, use ...
def init_policy(problem):
    # This function was given in assignment, not changes were applied
    policy = dict()
    for state in problem.get_all_states():
        if problem.is_goal_state(state):
            policy[state.x, state.y] = None
            continue
        actions = [action for action in problem.get_actions(state)]
        policy[state.x, state.y] = random.choice(actions)
    return policy


def init_utils(problem):
    '''
    Initialize all state utilities to zero except the goal states
    :param problem: problem - object, for us it will be kuimaze.Maze object
    :return: dictionary of utilities, indexed by state coordinates
    '''
    utils = dict()
    x_dims = problem.observation_space.spaces[0].n
    y_dims = problem.observation_space.spaces[1].n

    for x in range(x_dims):
        for y in range(y_dims):
            utils[(x,y)] = 0

    for state in problem.get_all_states():
        utils[(state.x, state.y)] = state.reward # problem.get_state_reward(state)
    return utils


def get_not_terminal_states(problem):
    # Function returns list of not terminal states at maze
    list_of_states = []
    for state in problem.get_all_states():
        if not problem.is_goal_state(state):
            list_of_states.append(state)
    return list_of_states


def get_value_of_action(list_of_action_with_probabilities, utils):
    # Function returns sum of utilities of neighbouring states with corresponding probability of that state
    value_with_probabilities = 0
    for T in list_of_action_with_probabilities:
        value_with_probabilities += utils[T[0].x, T[0].y] * T[1]
    return value_with_probabilities


def choose_best_action(problem, state, utils):
    # Returns action with highest score and the score of that action
    score_of_actions = {}
    for action in problem.get_actions(state):
        actions_with_probabilities = problem.get_next_states_and_probs(state, action)
        score_of_actions[action] = get_value_of_action(actions_with_probabilities, utils)
    max_action = max(score_of_actions, key=score_of_actions.get)
    return max_action, score_of_actions[max_action]


def find_policy_via_value_iteration(problem, discount_factor, epsilon):
    # Returns policy based on value iteration algorithm
    utils = init_utils(problem)
    new_utils = copy.deepcopy(utils)
    policy = init_policy(problem)
    delta = float("inf")
    while delta > (epsilon*(1 - discount_factor)/discount_factor):
        delta = 0
        for state in get_not_terminal_states(problem):                          # Calculate new policy
            max_action, max_action_score = choose_best_action(problem, state, utils)
            new_utils[state.x, state.y] = state.reward + discount_factor*max_action_score
            policy[state.x, state.y] = max_action
            delta = max(abs(new_utils[state.x, state.y] - utils[state.x, state.y]), delta)
        temp = copy.deepcopy(utils)             # exchange utilities
        utils = new_utils
        new_utils = temp
    return policy


def find_policy_via_policy_iteration(problem, discount_factor, epsilon=0.01):
    # Returns policy based on policy iteration algorithm
    utils = init_utils(problem)
    new_utils = copy.deepcopy(utils)
    policy = init_policy(problem)
    without_change = False
    while not without_change:
        without_change = True
        delta = float("inf")
        # Update utilities
        while delta > epsilon:
            delta = 0
            for state in get_not_terminal_states(problem):
                # state is evaluated by current policy
                actions_with_probabilities = problem.get_next_states_and_probs(state, policy[state.x, state.y])
                value_of_action = get_value_of_action(actions_with_probabilities, utils)
                new_utils[state.x, state.y] = state.reward + discount_factor * value_of_action
                delta = max(abs(new_utils[state.x, state.y] - utils[state.x, state.y]), delta)
            temp = copy.deepcopy(utils)
            utils = new_utils
            new_utils = temp
        # This part evaluates policy
        for state in get_not_terminal_states(problem):
            max_action = choose_best_action(problem, state, utils)[0]
            if max_action != policy[state.x, state.y]:
                policy[state.x, state.y] = max_action
                without_change = False
    return policy

if __name__ == "__main__":
    # Initialize the maze environment
    env = kuimaze.MDPMaze(map_image=GRID_WORLD3, probs=PROBS, grad=GRAD, node_rewards=GRID_WORLD3_REWARDS)
    # env = kuimaze.MDPMaze(map_image=GRID_WORLD3, probs=PROBS, grad=GRAD, node_rewards=None)
    # env = kuimaze.MDPMaze(map_image=MAP, probs=PROBS, grad=GRAD, node_rewards=None)
    env.reset()

    print('====================')
    print('works only in terminal! NOT in IDE!')
    print('press n - next')
    print('press s - skip to end')
    print('====================')

    print(env.get_all_states())
    #policy = find_policy_via_policy_iteration(env,0.9)
    policy = find_policy_via_value_iteration(env,0.9,0.001)
    #env.visualise(get_visualisation_values(policy))
    #env.render()
    #wait_n_or_s()
    print('Policy:', policy)
    #utils = init_utils(env)

    env.visualise(get_visualisation_values(policy))
    env.render()
    time.sleep(5)
