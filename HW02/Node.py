class Node:
    # Custom made class representing node in graph
    # attributes of class:
    # ====================
    # position = tuple containing x and y position in graph
    # parent = reference to instance of node which is root of current node
    # h_value = heuristic value of this node
    # g_value = price of path between start and this node
    # value = sum of g_value and f_value -> value of this node

    def __init__(self, node, parent):
        # Initializer of Node class
        # parameters:
        # node = gets input in this form: [(x,y),g_value between this node and its parent]
        # parent = instance of node class
        self.position = node[0]
        self.parent = parent
        self.h_value = 0
        self.g_value = 0
        self.value = 0

    def return_parent(self):
        return self.parent

    def update_value(self):
        # This method recalculates and updates value of Node
        self.value = self.g_value + self.h_value
