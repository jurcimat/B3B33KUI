from random import randint
import copy
class MyPlayer(object):
    """
    Docstring for MyPlayer
    """

    def __init__(self, my_color, opponent_color):
        self.name = 'bieleluk'                   # TODO: Fill in your username
        self.my_color = my_color
        self.opponent_color = opponent_color

    def move(self, board):


        possible_moves = self.get_all_valid_moves(board, active_color=self.my_color)
        #print(self.get_all_valid_moves(board, active_color=self.my_color))
        dx = [-1, -1, -1, 0, 1, 1, 1, 0]
        dy = [-1, 0, 1, 1, 1, 0, -1, -1]
        score_of_possible_moves=[]
        for i in possible_moves:
            score_of_possible_moves.append(0)
            xposition=i[0]
            yposition=i[1]
            score_of_possible_moves[len(score_of_possible_moves)-1]=\
                self.weighted_board(board)[xposition][yposition]
            for j in range(8):
                score_of_possible_moves[len(score_of_possible_moves)-1]+=\
                    self.evaluate_in_one_way(board, xposition, yposition, dx[j], dy[j])
        #print(score_of_possible_moves)
        #print('final move is ',possible_moves[score_of_possible_moves.index(max(score_of_possible_moves))])
        return possible_moves[score_of_possible_moves.index(max(score_of_possible_moves))]


    def evaluate_in_one_way(self, board, x_position, y_position, x_way, y_way):
        score_in_one_way = 0
        actual_x_position = x_position
        actual_y_position = y_position
        while 1:
            actual_x_position+=x_way
            actual_y_position+=y_way
            if actual_x_position>=(len(board)) or actual_y_position>=(len(board)) or actual_x_position<0\
                    or actual_y_position<0:
                score_in_one_way = 0
                break
            elif board[actual_x_position][actual_y_position]==self.my_color:
                break
            elif board[actual_x_position][actual_y_position]==self.opponent_color:
                score_in_one_way+=2*self.weighted_board(board)[actual_x_position][actual_y_position]
            else:
                score_in_one_way=0
                break
        return score_in_one_way

    def weighted_board(self, board):
        weight_of_positions = copy.deepcopy(board)
        for i in range(len(board)):
            for j in range(len(board)):
                    if ((i==0 or i==(len(board)-1)) and (j==0 or j==(len(board)-1))):
                        weight_of_positions[i][j]=20
                    elif len(board)>=3 and( ((i==0 or i==(len(board)-1)) and (j==1 or j==(len(board)-2)))
                                            or ((i==1 or i==(len(board)-2)) and (j==0 or j==1 or
                                            j==(len(board)-2) or j==(len(board)-1))) ):
                        weight_of_positions[i][j] = -20
                    elif i==0 or j==0 or i==(len(board)-1) or j==(len(board)-1):
                        weight_of_positions[i][j] = 10
                    elif len(board) >= 3 and (i == 1 or j == 1 or i == (len(board) - 2) or
                                              j == (len(board) - 2)):
                        weight_of_positions[i][j] = -10
                    elif len(board) >= 5 and (i == 2 or j == 2 or i == (len(board) - 3) or
                                              j == (len(board) - 3)):
                        weight_of_positions[i][j] = 3
                    else:
                        weight_of_positions[i][j] = 1
        return weight_of_positions


    def __is_correct_move(self, move, board, board_size, active_color):
        dx = [-1, -1, -1, 0, 1, 1, 1, 0]
        dy = [-1, 0, 1, 1, 1, 0, -1, -1]
        for i in range(len(dx)):
            if self.__confirm_direction(move, dx[i], dy[i], board, board_size, active_color):
                return True
        return False

    def __confirm_direction(self, move, dx, dy, board, board_size, active_color):
        posx = move[0]+dx
        posy = move[1]+dy
        passive_color = (active_color+1) % 2
        if (posx >= 0) and (posx < board_size) and (posy >= 0) and (posy < board_size):
            if board[posx][posy] == passive_color:
                while (posx >= 0) and (posx <= (board_size-1)) and (posy >= 0) and (posy <= (board_size-1)):
                    posx += dx
                    posy += dy
                    if (posx >= 0) and (posx < board_size) and (posy >= 0) and (posy < board_size):
                        if board[posx][posy] == -1:
                            return False
                        if board[posx][posy] == active_color:
                            return True

        return False

    def get_all_valid_moves(self, board, active_color):
        """
        Returns all valid moves for a given board setup with active_color player
        trying to do the move.
        :param board: board setup list
        :param active_color: 0 or 1
        :return: list or None
        """
        board_size = len(board)
        valid_moves = []
        for x in range(board_size):
            for y in range(board_size):
                if (board[x][y] == -1) and self.__is_correct_move([x, y], board, board_size, active_color):
                    valid_moves.append((x, y))

        if len(valid_moves) <= 0:
            print('No possible move for ' + str(active_color))
            return None
        return valid_moves

