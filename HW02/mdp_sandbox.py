#!/usr/bin/env python3

import kuimaze
import random
import os
import time
import sys


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
    if dictvalues is None:
        return None
    ret = []
    for key, value in dictvalues.items():
        # ret.append({'x': key[0], 'y': key[1], 'value': [value, value, value, value]})
        ret.append({'x': key[0], 'y': key[1], 'value': value})
    return ret

# the init functions are provided for your convenience, modify, use ...
def init_policy(problem):
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


def find_policy_via_policy_iteration(problem,discount_factor):
    policy = init_policy(problem)
    return(policy)

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
    # policy1 = find_policy_via_value_iteration(env)
    policy = find_policy_via_policy_iteration(env,0.9999)
    env.visualise(get_visualisation_values(policy))
    env.render()
    wait_n_or_s()
    print('Policy:', policy)
    utils = init_utils(env)
    env.visualise(get_visualisation_values(utils))
    env.render()
    time.sleep(5)
