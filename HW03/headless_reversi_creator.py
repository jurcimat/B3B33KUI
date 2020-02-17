# import random_player
from game_board import GameBoard
import time, getopt, sys
from player_creator import create_player


class HeadlessReversiCreator(object):
    '''
    Creator of the Reversi game without the GUI.
    '''

    def __init__(self, player1, player1_color, player2, player2_color, board_size=8):
        '''
        :param player1: Instance of first player
        :param player1_color: color of player1
        :param player2: Instance of second player
        :param player1_color: color of player2
        :param boardSize: Board will have size [boardSize x boardSize]
        '''
        self.board = GameBoard(board_size, player1_color, player2_color)
        self.player1 = player1
        self.player2 = player2
        self.current_player = self.player1
        self.current_player_color = player1_color
        self.player1_color = player1_color
        self.player2_color = player2_color

    def play_game(self):
        '''
        This function contains game loop that plays the game.
        '''
        # TODO: Time limit for move
        correct_finish = True
        while self.board.can_play(self.current_player_color):
            startTime = time.time()
            move = self.current_player.move(self.board.get_board_copy())
            endTime = time.time()
            moveTime = (endTime - startTime) * 1000
            if move is None:
                print('Player %d reurns None istead of a valid move. Move takes %.3f ms.' % (self.current_player_color, moveTime))
                correct_finish = False
                break
            else:
                print('Player %d wants move [%d,%d]. Move takes %.3f ms.' % (self.current_player_color, move[0], move[1], moveTime))
            
            move = (int(move[0]),int(move[1]))
            if self.board.is_correct_move(move, self.current_player_color):
                print('Move is correct')
                self.board.play_move(move, self.current_player_color)

            else:
                print('Player %d made the wrong move [%d,%d]' % (self.current_player_color, move[0], move[1]))
                correct_finish = False
                break

            self.change_player()
            if not self.board.can_play(self.current_player_color):
                print('No possible move for Player %d' % (self.current_player_color))
                self.change_player()
                if self.board.can_play(self.current_player_color):
                    print('Player %d plays again ' % (self.current_player_color))
                else:
                    print('Game over')


            self.board.print_board()
        if correct_finish:
            self.printFinalScore()
        else:
            print('Game over.')
            if self.current_player_color == self.player1_color:
                print('Winner is player %d.' % (self.player2_color))
            else:
                print('Winner is player %d.' % (self.player1_color))


    def change_player(self):
        '''
        Change the current_player
        '''
        if self.current_player == self.player1:
            self.current_player = self.player2
            self.current_player_color = self.player2_color
        else:
            self.current_player = self.player1
            self.current_player_color = self.player1_color

    def printFinalScore(self):
        p1Stones = 0
        p2Stones = 0
        for x in range(self.board.board_size):
            for y in range(self.board.board_size):
                if self.board.board[x][y] == 0:
                    p1Stones += 1
                if self.board.board[x][y] == 1:
                    p2Stones += 1

        print('\n\n-----------------------------\n')
        print('Final score:\n\nPlayer%d:Player%d\n\t[%d:%d]\n' % (self.player1_color, self.player2_color, p1Stones, p2Stones))
        if p1Stones > p2Stones:
            print('Player %d wins!' % (self.player1_color))
        elif p2Stones > p1Stones:
            print('Player %d wins!' % (self.player2_color))
        else:
            print('Draw')
        print('\n-----------------------------\n\n')

if __name__ == "__main__":
    (choices,args) = getopt.getopt(sys.argv[1:],"")
    p1_color = 0
    p2_color = 1
    board_size = 8

    importsCorrect = True
    colors = [p1_color, p2_color]
    players = []
    # if only one player specified, it will play againts itself
    for i in [0,-1]:
        try:
            to_import = args[i]
            if ".py" in args[i]:
                to_import = args[i].replace(".py","")
            print('importing', to_import)
            player_module = __import__(to_import)
            players.append(create_player(player_module.MyPlayer, colors[i],colors[i+1], board_size))
        except ImportError:
            importsCorrect = False
            print('Error: Cannot import given player: %s.' %(args[i]))

    if importsCorrect:
        game = HeadlessReversiCreator(players[0], p1_color, players[1], p2_color, board_size)
        game.play_game()

