#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
Reinforcement learning agent made for HW05 on course B3B36KUI @CTU
@author: Matej Jurcik, code was inspired by Tomas Svoboda's rl_sandbox.py
@contact: jurcimat@fel.cvut.cz
You can find Tomas Svoboda's code rl_sandbox.py at:
https://cw.fel.cvut.cz/wiki/courses/b3b33kui/cviceni/sekvencni_rozhodovani/rl
'''

import kuimaze
import numpy as np
import sys
import os
import gym
import time
import random

MAP = 'maps/normal/normal3.bmp'
# Constants that were used in example code rl_sandbox.py
# MAP = 'maps/easy/easy2.bmp'
# MAP = 'maps/normal/normal12.bmp'
MAP = os.path.join(os.path.dirname(os.path.abspath(__file__)), MAP)
PROBS = [1, 0, 0, 0]
GRAD = (0, 0)
SKIP = False
VERBOSITY = 0
GRID_WORLD3 = [[[255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 0, 0]],
               [[255, 255, 255], [0, 0, 0], [255, 255, 255], [0, 255, 0]],
               [[0, 0, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255]]]

# My added constants
MAX_STEPS_PER_EPISODE = 1000                # Maximum allowed number of steps in one episode

LEARNING_RATE = 0.1
DISCOUNT_RATE = 0.99


MAX_EXPLORATION_RATE = 1
MIN_EXPLORATION_RATE = 0.01
EXPLORATION_DECAY_RATE = 0.01               # Rate at which algorithm shifts from exploration to exploitation
TIME_LIMIT = 18

# MAP = GRID_WORLD3

def wait_n_or_s():
    # Function used in debugging created by Tomas Svoboda
    def wait_key():
        '''
        returns key pressed ... works only in terminal! NOT in IDE!
        '''
        result = None
        if os.name == 'nt':
            import msvcrt
            # https://cw.felk.cvut.cz/forum/thread-3766-post-14959.html#pid14959
            result = chr(msvcrt.getch()[0])
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


def get_visualisation(table):
    # Function used when visualizing the problem created by Tomas Svoboda
    ret = []
    for i in range(len(table[0])):
        for j in range(len(table)):
            ret.append({'x': j, 'y': i, 'value': [table[j][i][0], table[j][i][1], table[j][i][2], table[j][i][3]]})
    return ret


def initialize_q_table(env):
    # Returns initialized table of q values
    # This function was based on Tomas Svoboda's code in rl_sandbox.py
    # Maze size
    x_dims = env.observation_space.spaces[0].n
    y_dims = env.observation_space.spaces[1].n
    maze_size = tuple((x_dims, y_dims))
    # Number of discrete actions
    num_actions = env.action_space.n

    # Q-table:
    # numpy function np.zeroes creates matrix of arrays where one elemnt represents state which contains array
    # of actions' rewards.
    q_table = np.zeros([maze_size[0], maze_size[1], num_actions], dtype=float)
    return q_table


def choose_max_action(list_of_possible_actions):
    # Returns index of the most rewarded action
    index_of_max = 0
    max_value = -float("inf")
    for index in range(len(list_of_possible_actions)):
        if list_of_possible_actions[index] > max_value:
            max_value = list_of_possible_actions[index]
            index_of_max = index
    return index_of_max


def make_policy(q_table):
    # Generates policy from existing q-value table based on the most rewarded actions
    policy = {}
    for y in range(len(q_table)):
        for x in range(len(q_table[0])):
            best = choose_max_action(q_table[y][x])
            policy[(y, x)] = best
    return policy


def learn_policy(env):
    # Main algorithm that generates policy using reinforcement learning
    # Based on Q-learning algorithm using epsilon greedy strategy
    start_time = time.time()
    num_of_episode = 1            # Counts number of episodes done
    exploration_rate = 1          # Set exploration rate to 1 to explore the most at start of learning
    q_table = initialize_q_table(env)
    while time.time() < (start_time + TIME_LIMIT):  # Learn as much as it is allowed in time restriction
        obv = env.reset()                           # reset environment at start of new episode
        state = obv[0:2]
        for step in range(MAX_STEPS_PER_EPISODE):    # Cycle for one episode
            exploration_rate_decision_var = random.uniform(0, 1)
            if exploration_rate_decision_var > exploration_rate:            # random decision whether explore or exploit
                action = choose_max_action(q_table[state[0]][state[1]])     # exploit
            else:
                action = env.action_space.sample()                          # explore
            obv, reward, is_done, _ = env.step(action)
            new_state = obv[0:2]                                    # move to different state based on previous action
            q_table[state][action] = q_table[state][action]*(1 - LEARNING_RATE) + \
                LEARNING_RATE*(reward + DISCOUNT_RATE*max(q_table[new_state[0]][new_state[1]]))     # update q_value
            state = new_state
            if is_done:                   # episode terminates in goal state or after maximum amount of steps was used
                break
        # Change exploration rate towards exploitation exponentially after each finished episode
        exploration_rate = MIN_EXPLORATION_RATE + \
            (MAX_EXPLORATION_RATE - MIN_EXPLORATION_RATE) * np.exp(-EXPLORATION_DECAY_RATE * num_of_episode)

        num_of_episode += 1
    return make_policy(q_table)




if __name__ == "__main__":
    # Initialize the maze environment and q_table
    env = kuimaze.HardMaze(map_image=MAP, probs=PROBS, grad=GRAD)
    q_table = initialize_q_table(env)

    if VERBOSITY > 0:
        print('====================')
        print('works only in terminal! NOT in IDE!')
        print('press n - next')
        print('press s - skip to end')
        print('====================')

    if VERBOSITY > 0:
        env.visualise(get_visualisation(q_table))
        env.render()
    learn_policy(env)

    if VERBOSITY > 0:
        #global SKIP
        SKIP = True
        env.visualise(get_visualisation(q_table))
        env.render()
        # wait_n_or_s()
        time.sleep(5)
        env.save_path()
        env.save_eps()
