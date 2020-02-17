#!/usr/bin/python3
# Homework solution of HW03 - state space search from subject B3B33KUI - Kybernetika a umělá inteligence
# at FEL CTU in Prague
# @author: Matej Jurčík
# @contact: jurcimat@fel.cvut.cz

import time
import kuimaze
import os
from Node import Node as Node       # Import custom made file containing object representing single node in graph


class Agent(kuimaze.BaseAgent):
    # Class with implemented A* search
    # attributes of Agent class:
    # ==========================
    # environment = instance of kuimaze.InfEasyMaze
    # start = tuple containing x and y position of start node
    # goal = tuple containing x and y position of goal node

    def __init__(self, environment):
        # Initialize attributes of Agent class
        self.environment = environment
        observation = self.environment.reset()  # must be called, it is necessary for maze initialization
        self.goal = observation[1][0:2]
        self.start = observation[0][0:2]

    def heuristic_manhattan(self, node):
        # Return heuristic value of node using Manhattan distance
        distance_between_x = abs(node[0] - self.goal[0])
        distance_between_y = abs(node[1] - self.goal[1])
        const = 1
        return const*(distance_between_x + distance_between_y)

    def update_nodes(self, processed_set):
        # Updates value property of nodes in chosen set
        for node in processed_set:
            node.h_value = self.heuristic_manhattan(node.position)
            node.update_value()
        return processed_set

    def minimal_node(self, processed_set):
        # Returns node with minimal value property in chosen set
        min_value = processed_set[0].value
        min_node = processed_set[0]
        for node in processed_set:
            if node.value <= min_value:
                min_value = node.value
                min_node = node
        return min_node

    def reconstruct_path(self, current):
        # Returns path between goal and start node
        path = [current.position]
        while current.parent is not None:
            path.append(current.parent.position)
            current = current.parent
        path.reverse()  # reverse path to be in direction start node -> goal node
        return path

    def remove_node(self, r_node, processed_set):
        # Removes node (r_node) from chosen set (processed set)
        # Returns False on failure
        for node in range(len(processed_set)):
            if processed_set[node].position == r_node.position:
                del processed_set[node]
                return True
        return False

    def find_node(self, wanted, processed_set):
        # Returns wanted node from processed set
        for node in processed_set:
            if node.position == wanted.position:
                return True
        return False

    def find_index_node(self, wanted, processed_set):
        # Returns index of wanted node in processed set
        for node in range(len(processed_set)):
            if processed_set[node].position == wanted.position:
                return node

    def new_positions(self, current_node):
        # Method creating list of neighbouring nodes from current_node
        # :param new_position = list of new position generate from method expand
        # :return = list of neighbouring nodes
        #           in format [neighbouring node,distance between current and neighbouring node]
        new_positions = self.environment.expand(current_node.position)
        list_of_nodes = []
        for node in new_positions:
            new_node = Node(node, current_node)
            list_of_nodes.append([new_node, node[1]])
        return list_of_nodes

    def find_path(self):
        # Implementation of A* search algorithm
        # :param cost_so_far = dictionary storing g_values of nodes which are not final
        # :param current = currently processed node
        # :param open_list = list of nodes which algorithm manipulates with
        # :param possible_g = possible g_value of neighbouring in path from current node
        # :return = optimal path between start and goal nodes in form of list containing these nodes
        open_list = []

        position = self.start
        open_list.append(Node([position, 0], None))
        cost_so_far = {position: 0}
        while len(open_list) != 0:
            self.update_nodes(open_list)
            current = self.minimal_node(open_list)
            if current.position == self.goal:
                return self.reconstruct_path(current)
            self.remove_node(current, open_list)
            new_positions = self.new_positions(current)
            for neighbour in new_positions:
                possible_g = current.g_value + neighbour[1]
                if neighbour[0].position not in cost_so_far or possible_g < cost_so_far[neighbour[0].position]:
                    cost_so_far[neighbour[0].position] = possible_g
                    neighbour[0].g_value = possible_g
                    neighbour[0].parent = current
                    neighbour[0].h_value = self.heuristic_manhattan(neighbour[0].position)
                    open_list.append(neighbour[0])
            self.environment.render()               # show enviroment's GUI       DO NOT FORGET TO COMMENT THIS LINE BEFORE FINAL SUBMISSION!
            time.sleep(0.2)

        return None


if __name__ == '__main__':
    """
    for i in range(1, 12):
        MAP = 'maps/normal/normal' + str(i) + '.bmp'
        MAP = os.path.join(os.path.dirname(os.path.abspath(__file__)), MAP)
        GRAD = (0, 0)
        SAVE_PATH = False
        SAVE_EPS = False

        env = kuimaze.InfEasyMaze(map_image=MAP, grad=GRAD)       # For using random map set: map_image=None
        agent = Agent(env)

        path = agent.find_path()
        print(path)
        env.set_path(path)          # set path it should go from the init state to the goal state
        if SAVE_PATH:
            env.save_path()         # save path of agent to current directory
        if SAVE_EPS:
            env.save_eps()          # save rendered image to eps
        env.render(mode='human')
        time.sleep(2)"""""
    MAP = 'maps/normal/normal1.bmp'
    MAP = os.path.join(os.path.dirname(os.path.abspath(__file__)), MAP)
    GRAD = (0, 0)
    SAVE_PATH = False
    SAVE_EPS = False

    env = kuimaze.InfEasyMaze(map_image=MAP, grad=GRAD)       # For using random map set: map_image=None
    agent = Agent(env)

    path = agent.find_path()
    print(path)
    env.set_path(path)          # set path it should go from the init state to the goal state
    if SAVE_PATH:
        env.save_path()         # save path of agent to current directory
    if SAVE_EPS:
        env.save_eps()          # save rendered image to eps
    env.render(mode='human')
    time.sleep(10)