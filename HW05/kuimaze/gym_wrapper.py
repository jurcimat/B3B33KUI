# -*- coding: utf-8 -*-

'''
File wrapping maze.py functionality into few methods defined by OpenAI GYM
@author: Zdeněk Rozsypálek, Tomas Svoboda
@contact: rozsyzde(at)fel.cvut.cz, svobodat@fel.cvut.cz
@copyright: (c) 2017, 2018
'''

import collections
import os
import sys
import numpy as np
import gym
import copy
from gym import spaces
from gym.utils import seeding

import kuimaze
from .map_generator import maze as mapgen_maze

path_section = collections.namedtuple('Path', ['state_from', 'state_to', 'cost', 'action'])
state = collections.namedtuple('State', ['x', 'y'])

class MazeEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    _path = []
    _visited = []
    MAP = '../maps/easy/easy3.bmp'

    def __init__(self, informed, gym_compatible, deter, map_image_dir=None, grad=(0, 0), node_rewards=None):
        '''
        Class wrapping Maze into gym enviroment.
        @param informed: boolean
        @param gym_compatible: boolean - T = HardMaze, F = EasyMaze
        @param deter: boolean - T = deterministic maze, F = probabilistic maze
        @param map_image_dir: string - path to image of map
        @param grad: tuple - vector tuning the tilt of maze`
        '''
        if map_image_dir is None:
            '''
            If there is no map in parameter, it will be generated, with following setup
            '''
            x_size = 6
            y_size = 6                                 # not 100% accurate size, could have smaller dimensions
            complexity = 0.1                            # in interval (0, 1]
            density = 0.25                              # in interval [0, 1]
            self.MAP = mapgen_maze(x_size, y_size, complexity, density)
        else:
            self.MAP = map_image_dir
        if grad is None:
            self._grad = (0, 0)
        else:
            self._grad = grad
        self._problem = kuimaze.Maze(self.MAP, self._grad, node_rewards=node_rewards)
        self._player = EnvAgent(self._problem)
        self._curr_state = self._problem.get_start_state()
        self._informed = informed
        self._gym_compatible = gym_compatible
        self._deter = deter
        self._gui_disabled = True
        self._set = False
        # set action and observation space
        self._xsize = self._problem.get_dimensions()[0]
        self._ysize = self._problem.get_dimensions()[1]
        self.action_space = self._get_action_space()
        self.observation_space = spaces.Tuple((spaces.Discrete(self._xsize), spaces.Discrete(self._ysize)))
        self.seed()
        self.reset()

    def step(self, action):
        assert self._set, "reset() must be called first!"
        last_state = self._curr_state
        assert(0 <= action <= 3)
        if not self._deter:
            action = self._problem.non_det_result(action)
        self._curr_state = self._problem.result(self._curr_state, action)
        self._path.append(self._curr_state)
        if self._curr_state not in self._visited:
            self._visited.append(self._curr_state)
        reward, done = self._get_reward(self._curr_state, last_state)
        # reward = self._problem.get_state_reward(self._curr_state)
        return self._get_observation(), reward, done, None

    def get_all_states(self):
        '''
        auxiliary function for MDPs - where states (map) are supposed to be known
        :return: list of states
        '''
        return self._problem.get_all_states()

    def reset(self):
        self._set = True
        self._gui_disabled = True
        self._path = []
        self._visited = []
        self._problem.clear_player_data()
        self._problem.set_player(self._player)
        if self._gym_compatible:
            self._path.append(self._problem.get_start_state())
        self._visited.append(self._problem.get_start_state())
        self._curr_state = self._problem.get_start_state()
        return self._get_observation()

    def render(self, mode='human', close=False, visited=None, explored=None):
        assert self._set, "reset() must be called first!"
        self._gui_disabled = False
        if visited is None:
            self._problem.set_visited(self._visited)
        else:
            self._problem.set_visited(self._visited)
        if explored is None:
            self._problem.set_explored([self._curr_state])
        else:
            self._problem.set_explored(explored)

        self._problem.show_and_break()
        self._gui_on = True

    def close(self):
        self._gui_disabled = True
        self._problem.close_gui()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def save_path(self):
        '''
        Method for saving path of the agent into the file named 'saved_path.txt' into the directory where was the script
        runned from.
        @return: None
        '''
        assert len(self._path) > 0, "Path length must be greater than 0, for easy enviroment call set_path first"
        # at the moment it assumes the output directory exists
        pathfname = os.path.join(os.path.dirname(os.path.dirname(sys.argv[0])), "saved_path.txt")
        with open(pathfname, 'wt') as f:
            # then go backwards throught the path restored by bactracking
            if (type(self._path[0]) == tuple or type(self._path[0]) == list) and not self._gym_compatible:
                for pos in self._path:
                    f.write("x:{}, y:{}, z:{}\n".format(pos[0], pos[1], self._get_depth(state(pos[0], pos[1]))))
            if self._gym_compatible:
                for pos in self._path:
                    f.write("x:{}, y:{}, z:{}\n".format(pos.x, pos.y, self._get_depth(pos)))

    def save_eps(self):
        '''
        Save last rendered image into directory where the script was runned from.
        @return: None
        '''
        assert not self._gui_disabled, "render() must be called before save_eps"
        self._problem.save_as_eps(self._gui_disabled)

    def visualise(self, dictionary=None):
        '''
        Visualise input. If visualise is called before GUI opening, render() is called first
        @param dictionary: input to visualise, can be None -> visulise depth, or dictionary:
        {'x': x_coord, 'y': y_coord, 'value': value_to_visualise} where value can be scalar
        or 4 dimensional vector (tuple or list).
        @return: none
        '''
        assert self._set, "reset() must be called before any visualisation setting!"
        if self._gui_disabled:
            self.render()
        self._problem.visualise(dictionary)

    def _get_observation(self):
        '''
        method to generate observation - current state, finish states
        @return: tuple
        '''
        if self._informed:
            ret = [(self._curr_state.x, self._curr_state.y, self._get_depth(self._curr_state))]
            for n in self._problem.get_goal_nodes():
                ret.append((n.x, n.y, self._get_depth(n)))
        else:
            ret = [self._curr_state.x, self._curr_state.y, self._get_depth(self._curr_state)]
        return tuple(ret)

    def _get_action_space(self):
        '''
        method to get action space - all available actions in enviroment
        @return: spaces
        '''
        if self._gym_compatible:
            return spaces.Discrete(4)
        else:
            return spaces.Tuple(spaces.Tuple((spaces.Discrete(self._xsize), spaces.Discrete(self._ysize))))

    def __get_reward_curr_state(self):
        return self._problem.__node_rewards[self._curr_state.x, self._curr_state.y]

    def _get_reward(self, curr, last):
        '''
        returns reward and indication of goal state
        @param curr: new state
        @param last: last state
        @return: float, boolean
        '''
        reward = -2
        done = False
        vector = [curr.x - last.x, curr.y - last.y]
        z_axis = vector[0] * self._grad[0] + vector[1] * self._grad[1]
        if curr != last:
            reward = -(abs(vector[0]) + abs(vector[1]) + z_axis)
        if self._problem.is_goal_state(curr):
            reward = 100.0
            done = True
            if self._gym_compatible:
                self._player.set_path(self._path)
                self._player.find_path()
        return reward, done

    def _get_depth(self, state):
        '''
        Get depth (z coordinate) of state based on gradient. Start state of map has depth 0.
        @param state: namedtuple state
        @return: float
        '''
        start = self._problem.get_start_state()
        vector = [state.x - start.x, state.y - start.y]
        ret = self._grad[0] * vector[0] + self._grad[1] * vector[1]
        return float(format(ret, '.3f'))


class EnvAgent(kuimaze.BaseAgent):
    '''
    Class necessary for wrapping maze
    '''
    __path = []

    def set_path(self, path):
        self.__path = path

    def find_path(self):
        '''
        visualise path of the agent, path must be set before visualising!
        @return:
        '''
        ret = []
        for i in range(len(self.__path) - 1):
            ret.append(path_section(self.__path[i], self.__path[i + 1], 1, None))
        try:
            self.problem.show_path(ret)
        except:
            pass # do nothing if no graphics is prepared
        return self.__path


class EasyMazeEnv(MazeEnv):
    '''
    EasyMazeEnv is version of maze closer to graph search. It is possible to move agent from any state to
    different one already visited or neighbour state of current one. EasyMaze has all methods of HardMaze.
    Unlike the HardMaze, EasyMaze has additional method set_path - which can set different path than agent movement.
    '''

    def __init__(self, informed, map_image_dir=None, grad=(0, 0)):
        super(EasyMazeEnv, self).__init__(informed, False, True, map_image_dir, grad)
        self._gui_on = False

    def step(self, action):
        last_state = self._curr_state
        assert (type(action) == list or type(action) == tuple) and len(action) == 2
        self._curr_state = self._easy_result(action)
        if self._curr_state not in self._visited:
            self._visited.append(self._curr_state)
        reward, done = self._get_reward(self._curr_state, last_state)
        return self._get_observation(), reward, done, None

    # def render(self, mode='human', close=False):
    #    super(EasyMazeEnv, self).render(mode, close)
    #    self._gui_on = True

    def set_path(self, path):
        '''
        This method sets enviroment to visualize your found path. Method render, must be called afterwards.
        @param path: list of lists in format: [[x1, y1], [x2, y2], ... ]
        @return: None
        '''
        ret = []
        self._path = path
        if self._gui_on:
            assert (type(path[0]) == list or type(path[0]) == tuple) and (len(path[0]) == 2 or len(path[0]) == 3)
            previus_state = None
            for state_list in path:
                if previus_state != None:
                    if (abs(state_list[0]-previus_state[0]) + abs(state_list[1]-previus_state[1]) != 1):
                        raise AssertionError('The path is not continuous - distance between neighbouring path segments should be 1')
                ret.append(state(state_list[0], state_list[1]))
                previus_state = copy.copy(state_list)

            self._player.set_path(ret)
            self._player.find_path()

    def _is_available(self, new_state):
        '''
        returns true if new state is available
        @param new_state:
        @return: boolean
        '''
        tmp = []
        tmp.extend(self._visited)
        tmp.extend([self._problem.result(self._curr_state, 0), self._problem.result(self._curr_state, 1),
                    self._problem.result(self._curr_state, 2), self._problem.result(self._curr_state, 3)])
        return new_state in tmp

    def _easy_result(self, state_list):
        '''
        Gives result of desired action in parameter
        @param state_list: list or tuple of coordinates [x, y]
        @return: state - new position of agent
        '''
        new_state = state(state_list[0], state_list[1])
        if self._is_available(new_state):
            return new_state
        else:
            # print('UNAVAILABLE ' + str(new_state) + ' from ' + str(self._curr_state))
            return self._curr_state

    def _get_cost(self, curr, last):
        '''
        returns cost of movement from last to curr
        @param curr: new state
        @param last: last state
        @return: float
        '''
        reward = 0
        vector = [curr.x - last.x, curr.y - last.y]
        # TODO rewrite cost function
        z_axis = vector[0] * self._grad[0] + vector[1] * self._grad[1]
        addition_cost = 0
        if curr in self._problem.hard_places:
            addition_cost = 5
        
        if curr != last:
            reward = abs(vector[0]) + abs(vector[1]) + z_axis + addition_cost
        return reward 

    def expand(self,position):
        '''
        returns tuple of positions with associated costs that can be visited from "position"
        @param position: position in the maze defined by coordinates (x,y)

        @return: tuple of coordinates [x, y] with "cost" for movement to these positions: [[[x1, y1], cost1], [[x2, y2], cost2], ... ] 
        '''
        expanded_nodes = []
        maze_pose = state(position[0], position[1])
        tmp = [self._problem.result(maze_pose, 0), self._problem.result(maze_pose, 1),
               self._problem.result(maze_pose, 2), self._problem.result(maze_pose, 3)]
        for new_state in tmp: 
            if new_state.x == maze_pose.x and new_state.y == maze_pose.y:
                continue
            if new_state not in self._visited:
                self._visited.append(new_state)
            reward = self._get_cost(maze_pose, new_state)
            expanded_nodes.append([(new_state.x, new_state.y), reward])
        return expanded_nodes




'''
Final set of classes to use. As defined in OpenAI gym, all without any params needed in constructor.
Main method of wrapper is function step, which returns three values:

Observations:
For informed search is observation in format: ((current position coords), (finish_1 coords), (finish_2 coords), ...)
For Uninformed only (current position coords)

Rewards:
When agent moves to different place it gets reward -1 - depth.
When agent reaches finish it gets reward +100.
If unavailible action is called, agent stays in same position and reward is 0.

Done:
True when agent reaches the finish.

Input (parameter) of step method is defined by action space:
Easy maze action space is list [x_coordinate, y_coordinate].
Hard maze action space is integer from 0 to 3.
'''


class InfEasyMaze(EasyMazeEnv):
    '''
    informed easy maze, suitable for A* implementation
    step([x, y])
    '''
    def __init__(self, map_image=None, grad=(0, 0)):
        super(InfEasyMaze, self).__init__(True, map_image, grad)


class EasyMaze(EasyMazeEnv):
    '''
    uninformed easy maze, suitable for BFS, DFS ...
    step([x, y])
    '''
    def __init__(self, map_image=None, grad=(0, 0)):
        super(EasyMaze, self).__init__(False, map_image, grad)


class MDPMaze(MazeEnv):
    '''
    maze for solving MDP problems
    '''
    def __init__(self, map_image=None, grad=(0, 0), probs=None, node_rewards=None):
        if probs is not None:
            super().__init__(False, True, False, map_image, grad, node_rewards=node_rewards)
            self._problem.set_probs_table(probs[0], probs[1], probs[2], probs[3])    # set probabilities here
        else:
            super().__init__(False, True, True, map_image, grad)

    def _get_reward(self, curr, last):
        '''
        returns reward and indication of goal state
        @param curr: new state
        @param last: last state
        @return: float, boolean
        '''
        reward = self._problem.get_state_reward(curr)
        done = False
        vector = [curr.x - last.x, curr.y - last.y]
        z_axis = vector[0] * self._grad[0] + vector[1] * self._grad[1]
        if curr != last:
            reward = z_axis + self._problem.get_state_reward(last)
        if self._problem.is_goal_state(curr):
            reward = self._problem.get_state_reward(curr)
            done = True
            if self._gym_compatible:
                self._player.set_path(self._path)
                self._player.find_path()
        return reward, done

    def get_actions(self, state):
        return self._problem.get_actions(state)

    def is_goal_state(self, state):
        return self._problem.is_goal_state(state)

    def is_terminal_state(self, state):
        return self._problem.is_goal_state(state) or self._problem.is_danger_state(state)

    def get_next_states_and_probs(self, state, action):
        return self._problem.get_next_states_and_probs(state, action)

    def get_state_reward(self,curr):
        return self._problem.get_state_reward(curr)


class HardMaze(MazeEnv):
    '''
    Uninformed hard maze, suitable for reinforcement learning
    step(param) where param is integer; 0 <= param <= 3
    '''
    def __init__(self, map_image=None, grad=(0, 0), probs=None, node_rewards=None):
        if probs is not None:
            super(HardMaze, self).__init__(False, True, False, map_image, grad, node_rewards=node_rewards)
            self._problem.set_probs(probs[0], probs[1], probs[2], probs[3])    # set probabilities here
        else:
            super(HardMaze, self).__init__(False, True, True, map_image, grad)

    def _get_reward(self, curr, last):
        '''
        returns reward and indication of goal state
        @param curr: new state
        @param last: last state
        @return: float, boolean
        '''
        reward = self._problem.get_state_reward(last) # energy consumption
        done = False
        vector = [curr.x - last.x, curr.y - last.y]
        z_axis = vector[0] * self._grad[0] + vector[1] * self._grad[1]
        if curr != last:
            reward = reward  - z_axis # more or less depending on the gradient; going up costs more
        if self._problem.is_goal_state(curr):
            reward = reward + self._problem.get_state_reward(curr)
            done = True
            if self._gym_compatible:
                self._player.set_path(self._path)
                self._player.find_path()
        return reward, done

class InfHardMaze(MazeEnv):
    '''
    Informed hard maze, suitable for reinforcement learning
    step(param) where param is integer; 0 <= param <= 3
    '''
    def __init__(self, map_image=None, grad=(0, 0), probs=None):
        if probs is not None:
            super(InfHardMaze, self).__init__(True, True, False, map_image, grad)
            self._problem.set_probs(probs[0], probs[1], probs[2], probs[3])    # set probabilities here
        else:
            super(InfHardMaze, self).__init__(True, True, True, map_image, grad)
