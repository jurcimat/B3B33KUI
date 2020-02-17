#!/usr/bin/python3
'''
Very simple example how to use gym_wrapper and BaseAgent class for state space search 
@author: Zdeněk Rozsypálek, and the KUI-2018 team
@contact: svobodat@fel.cvut.cz
@copyright: (c) 2017, 2018
'''

import time
import kuimaze
import os
import random

class Agent(kuimaze.SearchAgent):
    '''
    Simple example of agent class that inherits kuimaze.BaseAgent class 
    '''
    def __init__(self, environment):
        self.environment = environment




if __name__ == '__main__':

    MAP = 'maps/easy/easy3.bmp'
    MAP = os.path.join(os.path.dirname(os.path.abspath(__file__)), MAP)
    GRAD = (0, 0)
    SAVE_PATH = False
    SAVE_EPS = False

    env = kuimaze.InfEasyMaze(map_image=MAP, grad=GRAD)       # For using random map set: map_image=None
    agent = Agent(env) 

    path = agent.find_path()
    env.set_path(path)          # set path it should go from the init state to the goal state
    if SAVE_PATH:
        env.save_path()         # save path of agent to current directory
    if SAVE_EPS:
        env.save_eps()          # save rendered image to eps
    env.render(mode='human')
    time.sleep(3)
