# for backward compatibility with 2-parameters players
from inspect import signature

def create_player(player_class, my_color, opp_color, board_size):
    num_params = len(signature(player_class).parameters)
    if num_params == 2:
        player = player_class(my_color, opp_color)
    elif num_params == 3:
        player = player_class(my_color, opp_color, board_size)
    else:
        raise NotImplementedError
    return player
