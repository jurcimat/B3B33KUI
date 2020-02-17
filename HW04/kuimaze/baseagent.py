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


class BaseAgent:
    '''
    Base class for players. All student solutions must inherit from this class.
    '''

    def __init__(self, problem):
        '''
        All inherited players must call this method. Expects problem to be instance of L{kuimaze.Maze}.
        If problem has L{show_level<kuimaze.SHOW>} other than L{kuimaze.SHOW.NONE}, it will start a GUI automatically.

        @param problem: Maze to associate your player with:
        @type problem: L{Maze}
        @raise AssertionError: if problem is not an instance of L{Maze}
        '''
        assert(isinstance(problem, kuimaze.maze.Maze))
        self.problem = problem
        self.problem.set_player(self) # mainly for visualisazion


    def find_path(self):
        '''
        Method that must be implemented. Otherwise raise NotImplementedError. Expects to return a path_section as a list of positions [(x1, y1), (x2, y2), ... ].

        @return: path_section as a list of positions [(x1, y1), (x2, y2), ... ]. Must 

        '''
        raise NotImplementedError('Not implemented yet')
