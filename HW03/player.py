class MyPlayer:
    """
    Greedy strategy
    """

    def __init__(self, my_color, opponent_color, board_size=8):
        self.name = 'jurcimat'                  # my username filled
        self.my_color = my_color
        self.opponent_color = opponent_color
        self.board_size = board_size
        self.board_heuristic = self.calculate_heuristic()

    def move(self, board):
        # Method to move player
        possible_moves = self.get_all_valid_moves(board)            # get all possible moves
        dx = [-1, -1, -1, 0, 1, 1, 1, 0]                            # lists used to check neighbouring positions
        dy = [-1, 0, 1, 1, 1, 0, -1, -1]
        print(self.board_heuristic)
        possible_moves_scores = []
        for move in possible_moves:
            possible_moves_scores.append(0)                         # set default score to zero
            x_pos = move[0]
            y_pos = move[1]
            last_at_score = len(possible_moves_scores) - 1
            possible_moves_scores[last_at_score] = self.board_heuristic[x_pos][y_pos]   # update value based on heuristic boards
            for num in range(8):
                possible_moves_scores[last_at_score] += self.value_at_direction(board, x_pos, y_pos, dx[num], dy[num]) # add value based on opponent pieces position
        return possible_moves[possible_moves_scores.index(max(possible_moves_scores))]      # returns max option based on heuristics

    def value_at_direction(self, board, x_pos, y_pos, x_dir, y_dir):
        # This method calculates additional heuristics based on deleting opponent pieces
        score_at_dir = 0
        current_x = x_pos
        current_y = y_pos
        board_s = self.board_size
        while True:
            current_x += x_dir          # expand in input direction
            current_y += y_dir
            if current_x >= board_s or current_y >= board_s or current_x < 0 or current_y < 0:
                score_at_dir = 0
                break                   # stop if out of board
            elif board[current_x][current_y] == self.my_color:
                break                   # stop if my figures in way
            elif board[current_x][current_y] == self.opponent_color:
                score_at_dir += 2 * self.board_heuristic[current_x][current_y]  # add value if opponent pieces are in way
            else:
                score_at_dir = 0
                break
        return score_at_dir

    def calculate_heuristic(self):
        # This method initialize m x m matrix where each position on board has its heuristic value
        board_size = self.board_size
        weight_of_positions = [[x for x in range(board_size)] for y in range(board_size)] # syntax sugar to initialize matrix with size of game board
        for i in range(board_size):
            for j in range(board_size):
                    if (i == 0 or i == (board_size - 1)) and (j == 0 or j == (board_size - 1)):
                        weight_of_positions[i][j] = 20
                    elif board_size >= 3 and(((i == 0 or i == (board_size-1)) and (j == 1 or j == (board_size-2)))
                                            or ((i == 1 or i == (board_size-2)) and (j == 0 or j == 1 or
                                            j == (board_size-2) or j == (board_size-1)))):
                        weight_of_positions[i][j] = -20
                    elif i == 0 or j == 0 or i == (board_size-1) or j == (board_size-1):
                        weight_of_positions[i][j] = 10
                    elif board_size >= 3 and ((i == 1 or j == 1 or i == board_size - 2) or j == (board_size - 2)):
                        weight_of_positions[i][j] = -10
                    elif board_size >= 5 and (i == 2 or j == 2 or i == (board_size - 3) or
                                              j == (board_size - 3)):
                        weight_of_positions[i][j] = 3
                    else:
                        weight_of_positions[i][j] = 1
        return weight_of_positions

    def __is_correct_move(self, move, board):
        dx = [-1, -1, -1, 0, 1, 1, 1, 0]
        dy = [-1, 0, 1, 1, 1, 0, -1, -1]
        for i in range(len(dx)):
            if self.__confirm_direction(move, dx[i], dy[i], board)[0]:
                return True, 
        return False

    def __confirm_direction(self, move, dx, dy, board):
        posx = move[0]+dx
        posy = move[1]+dy
        opp_stones_inverted = 0
        if (posx >= 0) and (posx < self.board_size) and (posy >= 0) and (posy < self.board_size):
            if board[posx][posy] == self.opponent_color:
                opp_stones_inverted += 1
                while (posx >= 0) and (posx <= (self.board_size-1)) and (posy >= 0) and (posy <= (self.board_size-1)):
                    posx += dx
                    posy += dy
                    if (posx >= 0) and (posx < self.board_size) and (posy >= 0) and (posy < self.board_size):
                        if board[posx][posy] == -1:
                            return False, 0
                        if board[posx][posy] == self.my_color:
                            return True, opp_stones_inverted
                    opp_stones_inverted += 1

        return False, 0

    def get_all_valid_moves(self, board):
        valid_moves = []
        for x in range(self.board_size):
            for y in range(self.board_size):
                if (board[x][y] == -1) and self.__is_correct_move([x, y], board):
                    valid_moves.append( (x, y) )

        if len(valid_moves) <= 0:
            print('No possible move!')
            return None
        return valid_moves
