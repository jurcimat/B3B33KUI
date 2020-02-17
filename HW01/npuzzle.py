import random


def tile_to_string(value):
    '''
    a small helper, converts symbols to proper strings for displaying
    :param value: int or None expected
    :return: {:2d} for number e.g. ' 1' for 1, '15' for 15
             '  ' (two spaces) for None
    '''
    if value is None:
        return '  '
    else:
        return '{:2d}'.format(value)


class NPuzzle:
    '''
    sliding puzzle class of a general size
    https://en.wikipedia.org/wiki/15_puzzle
    '''
    def __init__(self,size):
        '''
        create the list of symbols, typically from 1 to 8 or 15. Empty tile
        is represented by None
        :param size: the board will be size x size,
                     size=3 - 8-puzzle; 4 - 15 puzzle
        '''
        self.__size = size
        self.__tiles = [x for x in range(1,size**2)]
        self.__tiles.append(None)

    def reset(self):
        '''
        initialize the board by a random shuffle of symbols
        :return: None
        '''
        random.shuffle(self.__tiles)

    def __str__(self):
        '''
        create a string visualisizng the board
        :return:  the string
        '''
        symbols = [tile_to_string(x) for x in self.__tiles]
        msg = '-'*self.__size*6+'\n'
        for row in range(self.__size):
            msg += str(symbols[row*self.__size:row*self.__size+self.__size])
            msg += '\n'
        msg += '-'*self.__size*6
        return msg

    def visualise(self):
        '''
        just print itself to the standard output
        :return: None
        '''
        print(self)

    def __is_inside(self, row, col):
        return row>=0 and col>=0 and row<self.__size and col<self.__size

    def read_tile(self, row, col):
        '''
        returns a symbol on row, col position
        :param row: index of the row
        :param column: index of the column
        :return: value of the tile - int 1 to size^2-1, None for empty tile
        The function raises IndexError exception if outside the board
        '''
        if self.__is_inside(row, col):
            return self.__tiles[row*self.__size + col]
        else:
            raise IndexError


if __name__ == "__main__":
    # demonstrate basic usage of the NPuzzle class
    size = 3
    env = NPuzzle(size)
    for i in range(3):
        env.reset()
        env.visualise()
    for row in range(size+1):
        for col in range(size+1):
            try:
                val = env.read_tile(row, col)
                print('Tile at position:',row, col, 'has value', env.read_tile(row,col))
            except:
                print('Ouch! position:', row, col, 'not on the desk!')

