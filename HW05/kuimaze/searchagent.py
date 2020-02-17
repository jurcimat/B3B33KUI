#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Contains class BaseAgent from which all of players must inherit.
@author: Zdeněk Rozsypálek, and the KUI-2018 team
@contact: svobodat@fel.cvut.cz
@copyright: (c) 2017, 2018
'''

import collections
import random

import kuimaze.maze
from kuimaze.baseagent import BaseAgent

class SearchAgent(BaseAgent):
    '''
    Base class for all search agents, which extends BaseAgent Class. All student solutions must inherit from this class.
    '''

    def heuristic_function(self, position, goal):
        '''
        Method that must be implemented by you. Otherwise raise NotImplementedError. We are expecting that you will implement some admissible heuristic function. 

        @return: value
        '''
        raise NotImplementedError('Not implemented yet')
