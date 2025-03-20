import pickle
import numpy as np
import torch


def read_data(file_name: str):
    with open(f"data/{file_name}.pickle", 'rb') as f:
        data_file = pickle.load(f)
    if isinstance(data_file, dict):
        return data_file['data']
    else:
        return data_file


def food_direction(data):
    food = data[0].get('food')
    head = data[0].get('snake_body')[-1]
    data_dict = {'food_up': 0, 'food_right': 0, 'food_down': 0, 'food_left': 0}

    if head[1] - food[1] > 0:
        data_dict.update({'food_up': 1})
    if head[0] - food[0] < 0:
        data_dict.update({'food_right': 1})
    if head[1] - food[1] < 0:
        data_dict.update({'food_down': 1})
    if head[0] - food[0] > 0:
        data_dict.update({'food_left': 1})
    
    return data_dict

def collision(data):
    head = data[0].get('snake_body')[-1]
    data_dict = {'collision_up': 0, 'collision_right': 0, 'collision_down': 0, 'collision_left': 0}

    if head[1] == 0:
        data_dict.update({'collision_up': 1})
    if head[0] == 270:
        data_dict.update({'collision_right': 1})
    if head[1] == 270:
        data_dict.update({'collision_down': 1})
    if head[0] == 0:
        data_dict.update({'collision_left': 1})
    
    return data_dict

def tail(data):
    gs = data[0]
    tail_forward = 0
    tail_right  = 0
    tail_left = 0 
    body_values = list()
    head = gs["snake_body"][-1]
    for element in gs["snake_body"][:-1]:
        body_values.append((element[0], element[1]))
    state1 = head[0] + 30
    state2 = head[0] - 30
    state3 = head[1] + 30
    state4 = head[1] - 30

    if gs["snake_direction"].value == 0: # up
        if (head[0], state4) in body_values:
            tail_forward = 1
        if (state1, head[1]) in body_values:
            tail_right = 1
        if (state2, head[1]) in body_values:
            tail_left = 1
    
    if gs["snake_direction"].value == 1: # right
        if (state1, head[1]) in body_values:
            tail_forward = 1
        if (head[0], state3) in body_values:
            tail_right = 1
        if (head[0], state4) in body_values:
            tail_left = 1

    if gs["snake_direction"].value == 2: # down
        if (head[0], state3) in body_values:
            tail_forward = 1
        if (state1, head[1]) in body_values:
            tail_left = 1
        if (state2, head[1]) in body_values:
            tail_right = 1

    if gs["snake_direction"].value == 3: # left
        if (state2, head[1]) in body_values:
            tail_forward = 1
        if (head[0], state3) in body_values:
            tail_left = 1
        if (head[0], state4) in body_values:
            tail_right = 1

    return {'tail_forward': tail_forward, 'tail_right': tail_right, 'tail_left': tail_left}

def y_data_prepare(data):
    data_dict = {'up': 0, 'right': 0, 'down': 0, 'left': 0}
    if data == 0:
        data_dict.update({'up': 1})
    elif data == 1:
        data_dict.update({'right': 1})
    elif data == 2:
        data_dict.update({'down': 1})
    elif data == 3:
        data_dict.update({'left': 1})
    return data_dict


def get_data(file_name: str):
    raw_data = read_data(file_name)
    row_list = list()
    list_of_data = list()
    y_data = list()
    # {food_up: 0, food_right: 0, food_down: 0, food_left: 0, collision_up: 1, collision_right: 0, collision_down: 0, collision_left: 0}
    for game_state in raw_data:
        row_list += list(food_direction(game_state).values())
        row_list += list(collision(game_state).values())
        list_of_data.append(row_list)
        row_list = list()
        row_list += list(y_data_prepare(game_state[1].value).values())
        y_data.append(row_list)
        row_list = list()
        
    length_x = len(list_of_data)
    length_y = len(y_data)
    list_of_data = np.array(list_of_data)
    y_data = np.array(y_data)
    return np.ndarray(shape=(length_x, 8), dtype=int, buffer=list_of_data), np.ndarray(shape=(length_y, 4), dtype=int, buffer=y_data)

def game_state_to_data_sample(game_state):
    row_list = list()
    list_of_data = list()
    # {food_up: 0, food_right: 0, food_down: 0, food_left: 0, collision_up: 1, collision_right: 0, collision_down: 0, collision_left: 0}
    row_list += list(food_direction(game_state).values())
    row_list += list(collision(game_state).values())
    list_of_data.append(row_list)
    row_list = list()
        
    length_x = len(list_of_data)
    list_of_data = np.array(list_of_data)
    
    data =  np.ndarray(shape=(length_x, 8), dtype=int, buffer=list_of_data)
    return torch.from_numpy(data).float()


def get_data_with_tail(file_name: str):
    raw_data = read_data(file_name)
    row_list = list()
    list_of_data = list()
    y_data = list()
    # {food_up: 0, food_right: 0, food_down: 0, food_left: 0, collision_up: 1, collision_right: 0, collision_down: 0, collision_left: 0}
    for game_state in raw_data:
        row_list += list(food_direction(game_state).values())
        row_list += list(collision(game_state).values())
        row_list += list(tail(game_state).values())
        list_of_data.append(row_list)
        row_list = list()
        row_list += list(y_data_prepare(game_state[1].value).values())
        y_data.append(row_list)
        row_list = list()
        
    length_x = len(list_of_data)
    length_y = len(y_data)
    list_of_data = np.array(list_of_data)
    y_data = np.array(y_data)
    return np.ndarray(shape=(length_x, 11), dtype=int, buffer=list_of_data), np.ndarray(shape=(length_y, 4), dtype=int, buffer=y_data)

def game_state_to_data_sample_tail(game_state):
    row_list = list()
    list_of_data = list()
    
    row_list += list(food_direction(game_state).values())
    row_list += list(collision(game_state).values())
    row_list += list(tail(game_state).values())
    list_of_data.append(row_list)
    row_list = list()
        
    length_x = len(list_of_data)
    list_of_data = np.array(list_of_data)
    
    data =  np.ndarray(shape=(length_x,11), dtype=int, buffer=list_of_data)
    return torch.from_numpy(data).float()