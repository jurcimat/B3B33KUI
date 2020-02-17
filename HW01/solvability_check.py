import npuzzle


def get_size(env):
    # Function gets size of n-puzzle
    # param: width = width of row (or column) of n-puzzle
    # return: value n of n-puzzle
    width = 0
    while True:
        try:
            env.read_tile(0, width)
        except IndexError:
            break
        width += 1
    return width * width - 1


def get_array_representation(env, width):
    # Generates 1-dimensional representation of n-puzzle
    # param: arr = 1-dimensional array representation of n-puzzle
    # param: x,y = x and y positions of tiles in n-puzzle
    arr = []
    for x in range(width):
        for y in range(width):
            arr.append(env.read_tile(x, y))
    return arr


def count_inversions(arr, size):
    # Counts number of inversions in n-puzzle
    # arr.remove(None) -> Removes blank tile from array representation
    # param: x,y = x and y positions of tiles in n-puzzle
    # param: inversions = number of inversions in n-puzzle
    inversions = 0
    arr.remove(None)
    for x in range(size):
        for y in range(x + 1, size):
            if arr[x] > arr[y]:
                inversions += 1
    return inversions


def get_blank_position(env, width):
    # Calculates position of "blank" tile
    # param: x,y = x and y positions of tiles in n-puzzle
    # param: width = value of width (or row) of n-puzzle
    # return: blank tile's row position counting from bottom
    for x in range(width - 1, -1, -1):
        for y in range(width - 1, -1, -1):
            if env.read_tile(x, y) is None:
                return width - x


def is_solvable(env):
    # This is the main function to check whether is puzzle solvable
    # param: width = calculates width of row(or column) in n-puzzle
    # return: True or False based on solvability conditions for n-puzzle
    size_of_puzzle = get_size(env)
    width = int((size_of_puzzle + 1)**(1/2))
    array = get_array_representation(env, width)
    blank_pos = get_blank_position(env, width)
    inversions = count_inversions(array, size_of_puzzle)
    if width % 2 == 0:
        if blank_pos % 2 == 1 and inversions % 2 == 0:
            return True
        if blank_pos % 2 == 0 and inversions % 2 == 1:
            return True
    if width % 2 == 1 and inversions % 2 == 0:
        return True
    return False


if __name__ == "__main__":
    env = npuzzle.NPuzzle(3)
    env.reset()
    env.visualise()
    # just check
    print(is_solvable(env))
