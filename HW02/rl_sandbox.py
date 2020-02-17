#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
A sandbox for playing with the HardMaze
@author: Tomas Svoboda
@contact: svobodat@fel.cvut.cz
@copyright: (c) 2017, 2018
'''


import kuimaze
import numpy as np
import sys
import os
import gym


# MAP = 'maps/normal/normal3.bmp'
MAP = 'maps/easy/easy2.bmp'
MAP = os.path.join(os.path.dirname(os.path.abspath(__file__)), MAP)
# PROBS = [0.8, 0.1, 0.1, 0]
PROBS = [1, 0, 0, 0]
GRAD = (0, 0)
SKIP = False
VERBOSITY = 2

GRID_WORLD3 = [[[255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 0, 0]],
               [[255, 255, 255], [0, 0, 0], [255, 255, 255], [0, 255, 0]],
               [[0, 0, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255]]]

# MAP = GRID_WORLD3

def wait_n_or_s():

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
    ret = []
    for i in range(len(table[0])):
        for j in range(len(table)):
            ret.append({'x': j, 'y': i, 'value': [table[j][i][0], table[j][i][1], table[j][i][2], table[j][i][3]]})
    return ret


def walk_randomly(env):
    # use gym.spaces.prng.seen(number) if you want to fix the randomness between experiments
    gym.spaces.prng.seed()  # without parameter it uses machine tima
    obv = env.reset()
    state = obv[0:2]
    total_reward = 0
    is_done = False
    MAX_T = 1000 # max trials (for one episode)
    t = 0
    while not is_done and t<MAX_T:
        t += 1
        action = env.action_space.sample()

        obv, reward, is_done, _ = env.step(action)
        nextstate = obv[0:2]
        total_reward += reward

        # this is perhaps not correct, just to show something
        q_table[state[0]][state[1]][action] = reward
        # another silly idea
        # q_table[state[0]][state[1]][action] = t

        if VERBOSITY>0:
            print(state, action, nextstate, reward)
            env.visualise(get_visualisation(q_table))
            env.render()
            wait_n_or_s()

        state = nextstate

    if not is_done:
        print('Timed out')

    print('total_reward:', total_reward)



if __name__ == "__main__":
    # Initialize the maze environment
    env = kuimaze.HardMaze(map_image=MAP, probs=PROBS, grad=GRAD)

    if VERBOSITY>0:
        print('====================')
        print('works only in terminal! NOT in IDE!')
        print('press n - next')
        print('press s - skip to end')
        print('====================')
    
    '''
    Define constants:
    '''
    # Maze size
    x_dims = env.observation_space.spaces[0].n
    y_dims = env.observation_space.spaces[1].n
    maze_size = tuple((x_dims, y_dims))

    # Number of discrete actions
    num_actions = env.action_space.n
    # Q-table:
    q_table = np.zeros([maze_size[0], maze_size[1], num_actions], dtype=float)
    if VERBOSITY>0:
        env.visualise(get_visualisation(q_table))
        env.render()
    walk_randomly(env)

    if VERBOSITY>0:
        global SKIP
        SKIP = False
        env.visualise(get_visualisation(q_table))
        env.render()
        wait_n_or_s()

        env.save_path()
        env.save_eps()
