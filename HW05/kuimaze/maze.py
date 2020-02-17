#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Main part of kuimaze - framework for working with mazes. Contains class Maze (capable of displaying it) and couple helper classes
@author: Otakar Jašek, Tomas Svoboda
@contact: jasekota(at)fel.cvut.cz, svobodat@fel.cvut.cz
@copyright: (c) 2017, 2018
'''

import collections
import enum
import numpy as np
import os
import random
import warnings
from PIL import Image, ImageTk
import sys

import tkinter

import kuimaze

# nicer warnings
fw_orig = warnings.formatwarning
warnings.formatwarning = lambda msg, categ, fname, lineno, line=None: fw_orig(msg, categ, fname, lineno, '')

# some named tuples to be used throughout the package - notice that state and weighted_state are essentially the same
#: Namedtuple to hold state position with reward. Interchangeable with L{state}
weighted_state = collections.namedtuple('State', ['x', 'y', 'reward'])
#: Namedtuple to hold state position. Mostly interchangeable with L{weighted_state}
state = collections.namedtuple('State', ['x', 'y'])
#: Namedtuple to hold path_section from state A to state B. Expects C{state_from} and C{state_to} to be of type L{state} or L{weighted_state}
path_section = collections.namedtuple('Path', ['state_from', 'state_to', 'cost', 'action'])

# constants used for GUI drawing
#: Maximum size of one cell in GUI in pixels. If problem is too large to fit on screen, the cell size will be smaller
MAX_CELL_SIZE = 200
#: Maximal percentage of smaller screen size, GUI window can occupy.
MAX_WINDOW_PERCENTAGE = 0.85
#: Border size of canvas from border of GUI window, in pixels.
BORDER_SIZE = 0
#: Percentage of actuall cell size that specifies thickness of line size used in show_path. Line thickness is then determined by C{max(1, int(LINE_SIZE_PERCENTAGE * cell_size))}
LINE_SIZE_PERCENTAGE = 0.02
#: Draw the x,y labels
# Todo: add a possibility of having the maze to the borders
DRAW_LABELS = True

LINE_COLOR = "#FFF555333"
WALL_COLOR = "#000000000"
EMPTY_COLOR = "#FFFFFFFFF"
EXPLORED_COLOR = "#000BBB000"
SEEN_COLOR = "#BBBFFFBBB"
START_COLOR = "#000000FFF"
# FINISH_COLOR = "#FFF000000"
FINISH_COLOR = "#000FFFFFF"
DANGER_COLOR = "#FFF000000"

#: Font family used in GUI
FONT_FAMILY = "Helvetica"

#: Text size in GUI (not on Canvas itself)
FONT_SIZE = round(12*MAX_CELL_SIZE/50)

REWARD_NORMAL = -0.04 # e.g. energy consumption
REWARD_DANGER = -1
REWARD_GOAL = 1

class SHOW(enum.Enum):
    '''
    Enum class used for storing what is displayed in GUI - everything higher includes everything lower (except NONE, of course).
    So if SHOW is NODE_REWARDS, it automatically means, that it will display FULL_MAZE (and EXPLORED), however it won't display ACTION_COSTS
    '''
    NONE = 0
    EXPLORED = 1
    FULL_MAZE = 2


class ACTION(enum.Enum):
    '''
    Enum class to represent actions in a grid-world.
    '''
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    def __str__(self):
        if self == ACTION.UP:
            return "/\\"
        if self == ACTION.RIGHT:
            return ">"
        if self == ACTION.DOWN:
            return "\\/"
        if self == ACTION.LEFT:
            return "<"


class ProbsRoulette:
    '''
    Class for probabilistic maze - implements roulette wheel with intervals
    '''

    def __init__(self, obey=0.8, confusionL=0.1, confusionR=0.1, confusion180=0):
        # create bounds -> use defined method 'set_probs' outside init
        self._obey = None
        self._confusionLeft = None
        self._confusionRight = None
        self.set_probs(obey, confusionL, confusionR, confusion180)

    def set_probs(self, obey, confusionL, confusionR, confusion180):
        assert obey + confusionL + confusionR + confusion180 == 1
        assert 0 <= obey <= 1
        assert 0 <= confusionL <= 1
        assert 0 <= confusionR <= 1
        assert 0 <= confusion180 <= 1
        self._obey = obey
        self._confusionLeft = self._obey + confusionL
        self._confusionRight = self._confusionLeft + confusionR

    def confuse_action(self, action):
        roulette = random.uniform(0.0, 1.0)
        if 0 <= roulette < self._obey:
            return action
        else:
            # Confused left
            if self._obey <= roulette < self._confusionLeft:
                return (action - 1) % 4
            else:
                # Confused right
                if self._confusionLeft <= roulette < self._confusionRight:
                    return (action + 1) % 4
                else:
                    # Confused back
                    return (action + 2) % 4

    def __str__(self):
        return str(self.probtable)


class ActionProbsTable:
    def __init__(self, obey=0.8, confusionL=0.1, confusionR=0.1, confusion180=0):
        assert abs(1-(obey+confusionR+confusionL+confusion180)) < 0.00001
        # self.obey = obey
        # self.confusion90 = confusion90
        # self.confusion180 = confusion180
        self.probtable = dict()
        self.probtable[ACTION.UP, ACTION.LEFT] = confusionL
        self.probtable[ACTION.UP, ACTION.UP] = obey
        self.probtable[ACTION.UP, ACTION.RIGHT] = confusionR
        self.probtable[ACTION.UP, ACTION.DOWN] = confusion180

        self.probtable[ACTION.RIGHT, ACTION.LEFT] = confusion180
        self.probtable[ACTION.RIGHT, ACTION.UP] = confusionL
        self.probtable[ACTION.RIGHT, ACTION.RIGHT] = obey
        self.probtable[ACTION.RIGHT, ACTION.DOWN] = confusionR

        self.probtable[ACTION.DOWN, ACTION.LEFT] = confusionR
        self.probtable[ACTION.DOWN, ACTION.UP] = confusion180
        self.probtable[ACTION.DOWN, ACTION.RIGHT] = confusionL
        self.probtable[ACTION.DOWN, ACTION.DOWN] = obey

        self.probtable[ACTION.LEFT, ACTION.LEFT] = obey
        self.probtable[ACTION.LEFT, ACTION.UP] = confusionR
        self.probtable[ACTION.LEFT, ACTION.RIGHT] = confusion180
        self.probtable[ACTION.LEFT, ACTION.DOWN] = confusionL

    def __getitem__(self, item):
        return self.probtable[item]

    def __str__(self):
        return str(self.probtable)




class Maze:
    '''
    Maze class takes care of GUI and interaction functions.
    '''
    __deltas = [[0, -1], [1, 0], [0, 1], [-1, 0]]
    __ACTIONS = [ACTION.UP, ACTION.RIGHT, ACTION.DOWN, ACTION.LEFT]

    def __init__(self, image, grad, node_rewards=None, path_costs=None, trans_probs=None, show_level=SHOW.FULL_MAZE,
                 start_node=None, goal_nodes=None, ):
        '''
        Parameters node_rewards, path_costs and trans_probs are meant for defining more complicated mazes. Parameter start_node redefines start state completely, parameter goal_nodes will add nodes to a list of goal nodes.

        @param image: path_section to an image file describing problem. Expects to find RGB image in given path_section

            white color - empty space

            black color - wall space

            red color - goal state

            blue color - start state
        @type image: string
        @keyword node_rewards: optional setting of state rewards. If not set, or incorrect input, it will be set to default value - all nodes have reward of zero.
        @type node_rewards: either string pointing to stored numpy.ndarray or numpy.ndarray itself or None for default value. Shape of numpy.ndarray must be (x, y) where (x, y) is shape of problem.
        @keyword path_costs: optional setting of path_section costs. If not set, or incorrect input, it will be set to default value - all paths have cost of one.
        @type path_costs: either string pointing to stored numpy.ndarray or numpy.ndarray itself or None for default value. Shape of numpy.ndarray must be (x, y, 2) where (x, y) is shape of problem.
        @keyword trans_probs: optional setting of transition probabilities for modelling MDP. If not set, or incorrect input, it will be set to default value - actions have probability of 1 for itself and 0 for any other.
        @type trans_probs: either string pointing to stored numpy.ndarray or numpy.ndarray itself or None for default value. Shape of numpy.ndarray must be (x, y, 4, 4) where (x, y) is shape of problem.
        @keyword show_level: Controlling level of displaying in GUI.
        @type show_level: L{kuimaze.SHOW}
        @keyword start_node: Redefining start state. Must be a valid state inside a problem without a wall.
        @type start_node: L{namedtuple state<state>} or None for default start state loaded from image.
        @keyword goal_nodes: Appending to a list of goal nodes. Must be valid nodes inside a problem without a wall.
        @type goal_nodes: iterable of L{namedtuples state<state>} or None for default set of goal nodes loaded from image.

        @raise AssertionError: When image is not RGB image or if show is not of type L{kuimaze.SHOW} or if initialization didn't finish correctly.
        '''
        try:
            im_data = Image.open(image)
            self.__filename = image
        except:
            im_data = image
            self.__filename = 'given'
        maze = np.array(im_data, dtype=int)
        assert (len(maze.shape) == 3 and maze.shape[2] == 3)
        self.__maze = maze.sum(axis=2, dtype=bool).T
        self.__start = None
        self.__finish = None
        self.hard_places = []
        self.__node_rewards = None
        self.__node_utils = None
        self.__path_costs = None
        self.__trans_probs = None
        self.__i = 0
        self.__till_end = False
        self.__gui_root = None
        self.__gui_lock = False
        self.__player = None
        self.__gui_setup = False
        self.__running_find = False
        self.__eps_folder = os.getcwd()
        self.__eps_prefix = ""

        assert type(grad) == tuple or type(grad) == list
        assert len(grad) == 2 and -1 < grad[0] < 1 and -1 < grad[1] < 1
        self.__grad = grad
        self.__set_grad_data()

        self.__has_triangles = False

        maze = maze.tolist()
        finish = []
        if start_node is None or goal_nodes is None:
            for y, col in enumerate(maze):
                for x, cell in enumerate(col):
                    if cell == [255, 0, 0]:
                        finish.append(state(x, y))
                    if cell == [0, 0, 255]:
                        self.__start = state(x, y)
                    if cell == [0, 255, 0]:
                        self.hard_places.append(state(x, y))
                        finish.append(state(x,y))
            self.__finish = frozenset(finish)

        if start_node is not None:
            if self.__is_inside_valid(start_node):
                if self.__start is not None:
                    warnings.warn('Replacing start state as there could be only one!')
                self.__start = state(start_node.x, start_node.y)

        if goal_nodes is not None:
            finish = list(self.__finish)
            warnings.warn('Adding to list of goal nodes!')
            for point in goal_nodes:
                if self.__is_inside_valid(point):
                    finish.append(point)
            self.__finish = frozenset(finish)

        if node_rewards is not None:
            if isinstance(node_rewards, str):
                node_rewards = np.load(node_rewards)
            else: # array provided directly
                node_rewards = np.array(node_rewards)
                node_rewards = np.transpose(node_rewards)
            print(node_rewards.shape, self.__maze.shape)
            if node_rewards.shape == self.__maze.shape:
                self.__node_rewards = node_rewards
            print(self.__node_rewards)

        if self.__node_rewards is None:
            self.__node_rewards = np.zeros(self.__maze.shape, dtype=float)
            for y, col in enumerate(maze):
                for x, cell in enumerate(col):
                    pos = state(x,y)
                    self.__node_rewards[x, y] = REWARD_NORMAL # implicit
                    if pos in self.__finish:
                        self.__node_rewards[x,y] = REWARD_GOAL
                    if pos in self.hard_places:
                        self.__node_rewards[x,y] = REWARD_DANGER
            print(self.__node_rewards)

        if self.__node_utils is None:
            self.__node_utils = np.zeros(self.__maze.shape, dtype=float)

        if path_costs is not None:
            if isinstance(path_costs, str):
                path_costs = np.load(path_costs)
            if path_costs.shape == (self.__maze.shape[0], self.__maze.shape[1], 2):
                self.__path_costs = path_costs
        if self.__path_costs is None:
            self.__path_costs = np.ones((self.__maze.shape[0], self.__maze.shape[1], 2), dtype=int)

        if trans_probs is not None:
            self.__trans_probs = ProbsRoulette()# trans_probs
        if self.__trans_probs is None:
            self.__trans_probs = ProbsRoulette(0.8, 0.1, 0.1, 0)

        assert (isinstance(show_level, SHOW))
        self.show_level = show_level
        self.__backup_show = show_level
        self.__clear_player_data()

        assert (self.__start is not None)
        assert (self.__finish is not None)
        assert (self.__node_rewards is not None)
        assert (self.__path_costs is not None)
        assert (self.__trans_probs is not None)
        print('maze init done')

    def get_state_reward(self, state):
        return self.__node_rewards[state.x, state.y]

    def get_start_state(self):
        '''
        Returns a start state
        @return: start state
        @rtype: L{namedtuple state<state>}
        '''
        return self.__start

    def close_gui(self):
        self.__destroy_gui()

    def set_node_utils(self,utils):
        '''
        a visualisation method - sets an interal variable for displaying utilities
        @param utils: dictionary of utilities, indexed by tuple - state coordinates
        @return: None
        '''
        for position in utils.keys():
            self.__node_utils[position] = utils[position]

    def is_goal_state(self, current_state):
        '''
        Check whether a C{current_node} is goal state or not
        @param current_state: state to check.
        @type current_state: L{namedtuple state<state>}
        @return: True if state is a goal state, False otherwise
        @rtype: boolean
        '''
        return state(current_state.x, current_state.y) in self.__finish

    def is_danger_state(self, current_state):
        return state(current_state.x, current_state.y) in self.hard_places

    def get_goal_nodes(self):
        '''
        Returns a list of goal nodes
        @return: list of goal nodes
        @rtype: list
        '''
        return list(self.__finish)

    def get_all_states(self):
        '''
        Returns a list of all the problem states
        @return: list of all states
        @rtype: list of L{namedtuple weighted_state<weighted_state>}
        '''
        dims = self.get_dimensions()
        states = []
        for x in range(dims[0]):
            for y in range(dims[1]):
                if self.__maze[x, y]:  # do not include walls
                    states.append(weighted_state(x, y, self.__node_rewards[x, y]))
        return states

    def get_dimensions(self):
        '''
        Returns dimensions of problem
        @return: x and y dimensions of problem. Note that state indices are zero-based so if returned dimensions are (5, 5), state (5, 5) is B{not} inside problem.
        @rtype: tuple
        '''
        return self.__maze.shape

    def get_actions(self, current_state):
        '''
        Generate (yield) actions possible for the current_state
        It does not check the outcome this is left to the result method
        @param current_state:
        @return: action (relevant for the problem - problem in this case)
        @rtype: L{action from ACTION<ACTION>}
        '''
        for action in ACTION:
            yield action

    def result(self, current_state, action):
        '''
        Apply the action and get the state; deterministic version
        @param current_state: state L{namedtuple state<state>}
        @param action: L{action from ACTION<ACTION>}
        @return: state (result of the action applied at the current_state)
        @rtype: L{namedtuple state<state>}
        '''
        x, y = self.__deltas[action]
        nx = current_state.x + x  # yet to be change as this is not probabilistic
        ny = current_state.y + y
        if self.__is_inside(state(nx, ny)) and self.__maze[nx, ny]:
            nstate = weighted_state(nx, ny, self.__node_rewards[nx, ny])
        else:  # no outcome, just stay, thing about bouncing back, should be handled by the search agent
            nstate = weighted_state(current_state.x, current_state.y,
                                    self.__node_rewards[current_state.x, current_state.y])
        #return nstate, self.__get_path_cost(current_state, nstate)
        return state(nstate.x, nstate.y)

    def get_next_states_and_probs(self, curr, action):
        '''
        For the commanded action it generates all posiible outcomes with associated probabilities
        @param state: state L{namedtuple state<state>}
        @param action: L{action from ACTION<ACTION>}
        @return: list of tuples (next_state, probability_of_ending_in_the_next_state)
        @rtype: list of tuples
        '''
        states_probs = []
        for out_action in ACTION:
            next_state = self.result(curr, out_action.value)
            states_probs.append((next_state, self.__trans_probs[action, out_action]))
        return states_probs

    def set_explored(self, states):
        '''
        sets explored states list, preparation for visualisation
        @param states: iterable of L{state<state>}
        '''
        self.__explored = np.zeros(self.__maze.shape, dtype=bool)
        for state in states:
            self.__explored[state.x, state.y] = True
            if self.__changed_cells is not None:
                self.__changed_cells.append(state)

    def set_probs(self, obey, confusionL, confusionR, confusion180):
        self.__trans_probs.set_probs(obey, confusionL, confusionR, confusion180)

    def set_probs_table(self, obey, confusionL, confusionR, confusion180):
        self.__trans_probs = ActionProbsTable(obey, confusionL, confusionR, confusion180)

    def set_visited(self, states):
        '''
        sets seen states list, preparation for visualisation
        @param states: iterable of L{state<state>}
        '''
        for state in states:
            self.__seen[state.x, state.y] = True
            if self.__changed_cells is not None:
                self.__changed_cells.append(state)

    def non_det_result(self, action):
        real_action = self.__trans_probs.confuse_action(action)
        return real_action

    def __is_inside(self, current_state):
        '''
        Check whether a state is inside a problem
        @param current_state: state to check
        @type current_state: L{namedtuple state<state>}
        @return: True if state is inside problem, False otherwise
        @rtype: boolean
        '''
        dims = self.get_dimensions()
        return current_state.x >= 0 and current_state.y >= 0 and current_state.x < dims[0] and current_state.y < dims[1]

    def __is_inside_valid(self, current_state):
        '''
        Check whether a state is inside a problem and is not a wall
        @param current_state: state to check
        @type current_state: L{namedtuple state<state>}
        @return: True if state is inside problem and is not a wall, False otherwise
        @rtype: boolean
        '''
        return self.__is_inside(current_state) and self.__maze[current_state.x, current_state.y]

    def clear_player_data(self):
        '''
        Clear player data for using with different player or running another find_path
        '''
        self.__seen = np.zeros(self.__maze.shape, dtype=bool)
        self.__seen[self.__start.x, self.__start.y] = True
        self.__explored = np.zeros(self.__maze.shape, dtype=bool)
        self.__explored[self.__start.x, self.__start.y] = True
        self.__i = 0
        self.__running_find = False
        self.__renew_gui()
        self.__changed_cells = None
        # self.show_and_break()
        self.__clear_lines()

    def __clear_player_data(self):
        '''
        Clear player data for using with different player or running another find_path
        '''
        self.__seen = np.zeros(self.__maze.shape, dtype=bool)
        self.__seen[self.__start.x, self.__start.y] = True
        self.__explored = np.zeros(self.__maze.shape, dtype=bool)
        self.__explored[self.__start.x, self.__start.y] = True
        self.__i = 0
        self.__running_find = False

    def set_player(self, player):
        '''
        Set player associated with this problem.
        @param player: player to be used for association
        @type player: L{BaseAgent<kuimaze.BaseAgent>} or its descendant
        @raise AssertionError: if player is not instance of L{BaseAgent<kuimaze.BaseAgent>} or its descendant
        '''
        assert (isinstance(player, kuimaze.baseagent.BaseAgent))
        self.__player = player
        self.__clear_player_data()
        #self.__renew_gui()
        #self.show_and_break()
        '''
        if self.__gui_root is not None:
            self.__gui_root.mainloop()
            '''

    def show_and_break(self, drawed_nodes=None):
        '''
        Main GUI function - call this from L{C{BaseAgent.find_path()}<kuimaze.BaseAgent.find_path()>} to update GUI and
        break at this point to be able to step your actions.
        Example of its usage can be found at L{C{BaseAgent.find_path()}<kuimaze.BaseAgent.find_path()>}

        Don't use it too often as it is quite expensive and rendering after single exploration might be slowing your
        code down a lot.

        You can optionally set parameter C{drawed_nodes} to a list of lists of dimensions corresponding to dimensions of
        problem and if show_level is higher or equal to L{SHOW.NODE_REWARDS}, it will plot those in state centers
        instead of state rewards.
        If this parameter is left unset, no redrawing of texts in center of nodes is issued, however, it can be set to
        True which will draw node_rewards saved in the problem.

        If show_level is L{SHOW.NONE}, thisets function has no effect

        @param drawed_nodes: custom objects convertible to string to draw to center of nodes or True or None
        @type drawed_nodes: list of lists of the same dimensions as problem or boolean or None
        '''
        assert (self.__player is not None)
        if self.show_level is not SHOW.NONE:
            first_run = False
            if not self.__gui_setup:
                self.__setup_gui()
                first_run = True
            if self.show_level.value >= SHOW.FULL_MAZE.value:
                self.__gui_update_map(explored_only=False)
            else:
                if self.show_level.value == SHOW.EXPLORED.value:
                    self.__gui_update_map(explored_only=True)
            if first_run:
                #self.__gui_canvas.create_image(self.__cell_size + BORDER_SIZE, self.__cell_size + BORDER_SIZE
                #                               , anchor=tkinter.NW, image=self._image)
                first_run = False
            if not self.__till_end and self.__running_find:
                self.__gui_lock = True
            self.__changed_cells = []
            self.__gui_canvas.update()
            '''
            while self.__gui_lock:
                time.sleep(0.01)
                self.__gui_root.update()
            '''

    def show_path(self, full_path):
        '''
        Show resulting path_section given as a list of consecutive L{namedtuples path_section<path_section>} to show in GUI.
        Example of such usage can be found in L{C{BaseAgent.find_path()}<kuimaze.BaseAgent.find_path()>}

        @param full_path: path_section in a form of list of consecutive L{namedtuples path_section<path_section>}
        @type full_path: list of consecutive L{namedtuples path_section<path_section>}
        '''
        if self.show_level is not SHOW.NONE and len(full_path) is not 0:
            def coord_gen(paths):
                paths.append(path_section(paths[-1].state_to, None, None, None))
                for item in paths:
                    for j in range(2):
                        num = item.state_from.x if j == 0 else item.state_from.y
                        yield (num + 1.5) * self.__cell_size + BORDER_SIZE
            size = int(self.__cell_size/3)
            coords = list(coord_gen(full_path))
            full_path = full_path[:-1]
            self.__drawn_lines.append((self.__gui_canvas.create_line(
                *coords, width=self.__line_size, capstyle='round', fill=LINE_COLOR, # stipple='gray75',
                arrow=tkinter.LAST, arrowshape=(size, size, int(size/2.5))), coords))
            self.__text_to_top()

    def set_show_level(self, show_level):
        '''
        Set new show level. It will redraw whole GUI, so it takes a while.
        @param show_level: new show_level to set
        @type show_level: L{SHOW}
        @raise AssertionError: if show_level is not an instance of L{SHOW}
        '''
        assert (isinstance(show_level, SHOW))
        self.__backup_show = show_level
        self.__changed_cells = None
        if self.show_level is not show_level:
            self.__destroy_gui(unblock=False)
            self.show_level = show_level
            if self.show_level is SHOW.NONE:
                self.__gui_lock = False
            self.__show_tkinter.set(show_level.value)
            coords = [c for i, c in self.__drawn_lines]
            self.show_and_break()
            if self.show_level is not SHOW.NONE:
                self.__drawn_lines = []
                for coord in coords:
                    self.__drawn_lines.append((self.__gui_canvas.create_line(
                        *coord, width=self.__line_size, capstyle='round', fill=LINE_COLOR), coord))

    def set_eps_folder(self):
        '''
        Set folder where the EPS files will be saved.
        @param folder: folder to save EPS files
        @type folder: string with a valid path_section
        '''
        folder = os.path.join(os.path.dirname(os.path.dirname(sys.argv[0])))
        self.__save_name = os.path.join(folder, "%04d.eps" % (self.__i,))

    def __setup_gui(self):
        '''
        Setup and draw basic GUI. Imports tkinter.
        '''
        self.__gui_root = tkinter.Tk()
        self.__gui_root.title('KUI - Maze')
        self.__gui_root.protocol('WM_DELETE_WINDOW', self.__destroy_gui)
        self.__gui_root.resizable(0, 0)
        w = (self.__gui_root.winfo_screenwidth() / (self.get_dimensions()[0] + 2)) * MAX_WINDOW_PERCENTAGE
        h = (self.__gui_root.winfo_screenheight() / (self.get_dimensions()[1] + 2)) * MAX_WINDOW_PERCENTAGE
        use_font = FONT_FAMILY + str(FONT_SIZE)
        self.__cell_size = min(w, h, MAX_CELL_SIZE)
        self.__show_tkinter = tkinter.IntVar()
        self.__show_tkinter.set(self.show_level)
        top_frame = tkinter.Frame(self.__gui_root)
        top_frame.pack(expand=False, side=tkinter.TOP)
        width_pixels = (self.__cell_size * (self.get_dimensions()[0] + 2) + 2 * BORDER_SIZE)
        height_pixels = (self.__cell_size * (self.get_dimensions()[1] + 2) + 2 * BORDER_SIZE)
        self.__gui_canvas = tkinter.Canvas(top_frame, width=width_pixels, height=height_pixels)
        self.__gui_canvas.pack(expand=False, side=tkinter.LEFT)
        self.__color_handles = (-np.ones(self.get_dimensions(), dtype=int)).tolist()
        self.__text_handles = (-np.ones(self.get_dimensions(), dtype=int)).tolist()
        self.__text_handles_four = (-np.ones([self.get_dimensions()[0], self.get_dimensions()[1], 4], dtype=int)).tolist()
        font_size = max(2, int(0.2 * self.__cell_size))
        font_size_small = max(1, int(0.14 * self.__cell_size))
        self.__font = FONT_FAMILY + " " + str(font_size)
        self.__font_small = FONT_FAMILY + " " + str(font_size_small)
        self.__line_size = max(1, int(self.__cell_size * LINE_SIZE_PERCENTAGE))
        self.__drawn_lines = []
        self.__changed_cells = None
        for x in range(self.get_dimensions()[0]):
            draw_num = DRAW_LABELS
            if font_size == 1 and ((x % int(self.get_dimensions()[0] / 5)) != 0 and x != self.get_dimensions()[0] - 1):
                draw_num = False
            if draw_num:
                self.__gui_canvas.create_text(self.__get_cell_center(x), (BORDER_SIZE + self.__cell_size) / 2,
                                              text=str(x), font=self.__font)
                self.__gui_canvas.create_text(self.__get_cell_center(x),
                                              BORDER_SIZE + self.__cell_size * (self.get_dimensions()[1] + 1) + (
                                              BORDER_SIZE + self.__cell_size) / 2, text=str(x), font=self.__font)
        for y in range(self.get_dimensions()[1]):
            draw_num = DRAW_LABELS
            if font_size == 1 and ((y % int(self.get_dimensions()[1] / 5)) != 0 and y != self.get_dimensions()[1] - 1):
                draw_num = False
            if draw_num:
                self.__gui_canvas.create_text((BORDER_SIZE + self.__cell_size) / 2, self.__get_cell_center(y),
                                              text=str(y), font=self.__font)
                self.__gui_canvas.create_text(BORDER_SIZE + self.__cell_size * (self.get_dimensions()[0] + 1) + (
                BORDER_SIZE + self.__cell_size) / 2, self.__get_cell_center(y), text=str(y), font=self.__font)
        box_size = (
        int(self.__cell_size * self.get_dimensions()[0] + 2), int(self.__cell_size * self.get_dimensions()[1] + 2))
        self.__gui_setup = True

    def __destroy_gui(self, unblock=True):
        '''
        Safely destroy GUI. It is possible to pass an argument whether to unblock
        L{find_path()<kuimaze.BaseAgent.find_path()>}
        method, by default it is unblocking.

        @param unblock: Whether to unblock L{find_path()<kuimaze.BaseAgent.find_path()>} method by calling this method
        @type unblock: boolean
        '''
        if unblock:
            self.__gui_lock = False
        if self.__gui_root is not None:
            self.__gui_root.update()
            self.__gui_root.destroy()
        self.__gui_root = None
        self.show_level = SHOW.NONE
        self.__gui_setup = False

    def __renew_gui(self):
        '''
        Renew GUI if a new player connects to a problem object.
        '''
        #self.__destroy_gui()
        self.__has_triangles = False
        self.show_level = self.__backup_show

    def __set_show_level_cb(self):
        '''
        Just a simple callback for tkinter radiobuttons for selecting show level
        '''
        self.set_show_level(SHOW(self.__show_tkinter.get()))

    def __clear_lines(self):
        '''
        Clear path_section lines if running same player twice.
        '''
        if self.__gui_setup:
            for line, _ in self.__drawn_lines:
                self.__gui_canvas.delete(line)
            self.__drawn_lines = []

    def __set_cell_color(self, current_node, color):
        '''
        Set collor at position given by current position. Code inspired by old implementation of RPH Maze (predecessor of kuimaze)
        @param current_node: state at which to set a color
        @type current_node: L{namedtuple state<state>}
        @param color: color string recognized by tkinter (see U{http://wiki.tcl.tk/37701})
        @type color: string
        '''
        assert (self.__gui_setup)
        x, y = current_node.x, current_node.y
        if self.__color_handles[x][y] > 0:
            if self.__gui_canvas.itemcget(self.__color_handles[x][y], "fill") is not color:
                self.__gui_canvas.itemconfigure(self.__color_handles[x][y], fill=color)
        else:
            left = self.__get_cell_center(x) - self.__cell_size / 2
            right = left + self.__cell_size
            up = self.__get_cell_center(y) - self.__cell_size / 2
            down = up + self.__cell_size
            self.__color_handles[x][y] = self.__gui_canvas.create_rectangle(left, up, right, down, fill=color)

    def save_as_eps(self, disabled):
        '''
        Save canvas as color EPS - response for third button.
        '''
        self.set_eps_folder()
        if not disabled:
            self.__gui_canvas.postscript(file=self.__save_name, colormode="color")
            self.__i += 1
        else:
            raise EnvironmentError('Maze must be rendered before saving to eps!')

    def __get_cell_center_coords(self, x, y):
        '''
        Mapping from problem coordinates to GUI coordinates.
        @param x: x coord in problem
        @param y: y coord in problem
        @return: (x, y) coordinates in GUI (centers of cells)
        '''
        return self.__get_cell_center(x), self.__get_cell_center(y)

    def __get_cell_center(self, x):
        '''
        Mapping from problem coordinate to GUI coordinate, only one coord.
        @param x: coord in problem (could be either x or y)
        @return: center of cell corresponding to such coordinate in GUI
        '''
        return BORDER_SIZE + self.__cell_size * (x + 1.5)

    def __gui_update_map(self, explored_only=True):
        '''
        Updating cell colors depending on what has been already explored.

        @param explored_only: if True, update only explored position and leave unexplored black. if False, draw everything
        @type explored_only: boolean
        '''
        assert (self.__gui_setup)

        def get_cells():
            dims = self.get_dimensions()
            if self.__changed_cells is None:
                for x in range(dims[0]):
                    for y in range(dims[1]):
                        yield x, y
            else:
                for item in self.__changed_cells:
                    yield item.x, item.y

        for x, y in get_cells():
            n = state(x, y)
            if not self.__maze[x, y]:
                self.__set_cell_color(n, self.__color_string_depth(WALL_COLOR, x, y))
            else:
                if self.is_goal_state(n) and not self.is_danger_state(n):
                    self.__set_cell_color(n, self.__color_string_depth(FINISH_COLOR, x, y))
                    if self.__explored[x, y]:
                        self.__set_cell_color(n, self.__color_string_depth(EXPLORED_COLOR, x, y))
                else:
                    if self.__explored[x, y]:
                        self.__set_cell_color(n, self.__color_string_depth(EXPLORED_COLOR, x, y))
                    else:
                        if self.__seen[x, y]:
                            self.__set_cell_color(n, self.__color_string_depth(SEEN_COLOR, x, y))
                        else:
                            if explored_only:
                                self.__set_cell_color(n, self.__color_string_depth(WALL_COLOR, x, y))
                            else:
                                self.__set_cell_color(n, self.__color_string_depth(EMPTY_COLOR, x, y))
                        if n == self.__start:
                            self.__set_cell_color(n, self.__color_string_depth(START_COLOR, x, y))
                        if self.is_danger_state(n):
                            self.__set_cell_color(n, self.__color_string_depth(DANGER_COLOR, x, y))

    def visualise(self, dictionary):
        '''
        Update state rewards in GUI. If drawed_nodes is passed and is not None, it is expected to be list of lists of objects with string representation of same dimensions as the problem. Might fail on IndexError if passed list is smaller.
        if one of these objects in list is None, then no text is printed.

        If drawed_nodes is None, then node_rewards saved in Maze objects are printed instead

        @param drawed_nodes: list of lists of objects to be printed in GUI instead of state rewards
        @type drawed_nodes: list of lists of appropriate dimensions or None
        @raise IndexError: if drawed_nodes parameter doesn't match dimensions of problem
        '''
        dims = self.get_dimensions()

        def get_cells():
            for x in range(dims[0]):
                for y in range(dims[1]):
                    yield x, y

        if dictionary is None:
            for x, y in get_cells():
                if self.__maze[x, y]:
                    n = state(x, y)
                    vector = (n.x - self.__start.x, n.y - self.__start.y)
                    ret = self.__grad[0] * vector[0] + self.__grad[1] * vector[1]
                    self.__draw_text(n, format(ret, '.2f'))
            return

        assert type(dictionary[0]) == dict, "ERROR: Visualisation input must be dictionary"
        # assert len(dictionary) == dims[0]*dims[1], "ERROR: Visualisation input must have same size as maze!"
        if type(dictionary[0]['value']) == tuple or type(dictionary[0]['value']) == list:
            assert len(dictionary[0]['value']) == 4, "ERROR: When visualising list or tuple, length must be 4!"
            if not self.__has_triangles:
                # create triangles
                for x, y in get_cells():
                    if self.__maze[x, y]:
                        center = self.__get_cell_center_coords(x, y)
                        size = int(self.__cell_size/2)
                        point1 = [center[0] - size, center[1] - size]
                        point2 = [center[0] + size, center[1] + size]
                        point3 = [center[0] + size, center[1] - size]
                        point4 = [center[0] - size, center[1] + size]
                        self.__gui_canvas.create_line(point1[0], point1[1], point2[0], point2[1], width=1.4)
                        self.__gui_canvas.create_line(point3[0], point3[1], point4[0], point4[1], width=1.4)
                self.__has_triangles = True
            for element in dictionary:
                x = element['x']
                y = element['y']
                if self.__maze[x, y]:
                    n = state(x, y)
                    index = y * dims[0] + x
                    # self.__draw_text_four(n, dictionary[index]['value'])
                    self.__draw_text_four(n, element['value'])
            return

        # if type(dictionary[0]['value']) == int or type(dictionary[0]['value']) == float:
        if True: # at the moment for everything else
            for element in dictionary:
                x = element['x']
                y = element['y']
                if self.__maze[x, y]:
                    n = state(x, y)
                    index = y * dims[0] + x
                    # self.__draw_text(n, format(dictionary[index]['value'], '.2f'))
                    try:
                        string_to_print = format(element['value'], '.2f')
                    except:
                        string_to_print = str(element['value'])
                    self.__draw_text(n, string_to_print)


    def __draw_text(self, current_node, string):
        '''
        Draw text in the center of cells in the same manner as draw colors is done.

        @param current_node: position on which the text is to be printed in Maze coordinates
        @type current_node: L{namedtuple state<state>}
        @param string: string to be drawn
        @type string: string
        '''

        x, y = current_node.x, current_node.y
        assert self.__gui_setup
        if self.__text_handles[x][y] > 0:
            if self.__gui_canvas.itemcget(self.__text_handles[x][y], "text") != string:
                self.__gui_canvas.itemconfigure(self.__text_handles[x][y], text=string)
        else:
            self.__text_handles[x][y] = self.__gui_canvas.create_text(*self.__get_cell_center_coords(x, y), text=string,
                                                                      font=self.__font)

    def __text_to_top(self):
        '''
        Move text fields to the top layer of the canvas - to cover arrow
        :return:
        '''
        if self.__has_triangles:
            for x in range(self.get_dimensions()[0]):
                for y in range(self.get_dimensions()[1]):
                    for i in range(4):
                        if self.__text_handles_four[x][y][i] > 0:
                            self.__gui_canvas.tag_raise(self.__text_handles_four[x][y][i])
        else:
            for x in range(self.get_dimensions()[0]):
                for y in range(self.get_dimensions()[1]):
                    if self.__text_handles[x][y] > 0:
                        self.__gui_canvas.tag_raise(self.__text_handles[x][y])

    def __draw_text_four(self, current_node, my_list):
        '''
        Draw four text cells into one square

        @param current_node: position on which the text is to be printed in Maze coordinates
        @param my_list: list to be drawn
        @type my_list: list of floats or ints
        '''

        x, y = current_node.x, current_node.y
        format_string = '.2f'
        assert self.__gui_setup
        for i in range(4):
            if self.__text_handles_four[x][y][i] > 0:
                if self.__gui_canvas.itemcget(self.__text_handles_four[x][y][i], "text") != format(my_list[i], format_string):
                    self.__gui_canvas.itemconfigure(self.__text_handles_four[x][y][i], text=format(my_list[i], format_string))
            else:
                center = self.__get_cell_center_coords(x, y)
                size = self.__cell_size/2
                if i == 0:
                    self.__text_handles_four[x][y][i] = self.__gui_canvas.create_text([center[0], center[1] - int(0.7*size)],
                                                                              text=format(my_list[i], format_string), font=self.__font_small)
                elif i == 1:
                    self.__text_handles_four[x][y][i] = self.__gui_canvas.create_text([center[0] + int(0.565*size), center[1]],
                                                                              text=format(my_list[i], format_string), font=self.__font_small)
                elif i == 2:
                    self.__text_handles_four[x][y][i] = self.__gui_canvas.create_text([center[0], center[1] + int(0.7*size)],
                                                                              text=format(my_list[i], format_string), font=self.__font_small)
                elif i == 3:
                    self.__text_handles_four[x][y][i] = self.__gui_canvas.create_text([center[0] - int(0.565*size), center[1]],
                                                                              text=format(my_list[i], format_string), font=self.__font_small)

    def __color_string_depth(self, color, x, y):
        '''
        Method adjust color due to depth of square in maze
        :param color: color string in hexadecimal ... for example "#FFF000000" for red
        :param x: index of square
        :param y: index of square
        :return: new color string
        '''
        assert len(color) == 10
        rgb = [int(color[1:4], 16), int(color[4:7], 16), int(color[7:10], 16)]
        tmp = self.__koef * (x * self.__grad[0] + y * self.__grad[1] + self.__offset)
        strings = []
        for i in range(3):
            rgb[i] = rgb[i] - abs(int(tmp) - self.__max_minus)
            if rgb[i] < 0:
                rgb[i] = 0
            strings.append(hex(rgb[i])[2:])
        for i in range(3):
            while len(strings[i]) < 3:
                strings[i] = "0" + strings[i]
        ret = "#" + strings[0] + strings[1] + strings[2]
        return ret

    def __set_grad_data(self):
        '''
        Sets data needed for rendering 3D ilusion
        :return: None
        '''
        self.__max_minus = 2048
        lt = 0
        lb = self.get_dimensions()[1] * self.__grad[1]
        rt = self.get_dimensions()[0] * self.__grad[0]
        rb = self.get_dimensions()[0] * self.__grad[0] + self.get_dimensions()[1] * self.__grad[1]
        tmp = [lt, lb, rt, rb]
        maxi = max(tmp)
        mini = min(tmp)
        self.__offset = 0 - mini
        if self.__grad[0] != 0 or self.__grad[1] != 0:
            self.__koef = self.__max_minus / (maxi - mini)
        else:
            self.__koef = 0
            self.__max_minus = 0
