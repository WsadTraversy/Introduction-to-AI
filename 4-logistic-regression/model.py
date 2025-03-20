import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

"""Implement your model, training code and other utilities here. Please note, you can generate multiple 
pickled data files and merge them into a single data list."""

def read_data(file_name: str):
    with open(f"data/{file_name}.pickle", 'rb') as f:
        data_file = pickle.load(f)
    return data_file['data']


def food_direction(data):
    food = data[0].get('food')
    head = data[0].get('snake_body')[-1]
    data_dict = {'food_up': 0, 'food_right': 0, 'food_down': 0, 'food_left': 0}

    if head[1] - food[1] < 0:
        data_dict.update({'food_up': 1})
    elif head[0] - food[0] > 0:
        data_dict.update({'food_right': 1})
    elif head[1] - food[1] > 0:
        data_dict.update({'food_down': 1})
    elif head[0] - food[0] < 0:
        data_dict.update({'food_left': 1})
    
    return data_dict


def collision(data):
    head = data[0].get('snake_body')[-1]
    data_dict = {'collision_up': 0, 'collision_right': 0, 'collision_down': 0, 'collision_left': 0}

    if head[1] == 30:
        data_dict.update({'collision_up': 1})
    elif head[0] == 270:
        data_dict.update({'collision_right': 1})
    elif head[1] == 270:
        data_dict.update({'collision_down': 1})
    elif head[0] == 30:
        data_dict.update({'collision_left': 1})
    
    return data_dict


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


class LogisticRegression():

    def __init__(self, learning_rate=0.1, n_iter=1000):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.weights = None
        self.bias= None
    
    def _sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iter):
            linear_pred = np.dot(X, self.weights) + self.bias
            prediction = self._sigmoid(linear_pred)
            
            dw = (1/n_samples) * np.dot(X.T, (prediction-y))
            db = (1/n_samples) * np.sum(prediction-y)
            
            self.weights = self.weights - self.learning_rate*dw        
            self.bias = self.bias - self.learning_rate*db

    def predict(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = self._sigmoid(linear_pred)
        return float(y_pred)
        #class_pred = [0 if y<=0.5 else 1 for y in y_pred]
        #return class_pred


def teached_models():
    X, y_list = get_data('testowy')
    y_up_list = list()
    y_right_list = list()
    y_down_list = list()
    y_left_list = list()

    # y = []
    # for val in y_list:
    #     if val[0] == 1:
    #         y.append(0)
    #     if val[1] == 1:
    #         y.append(1)
    #     if val[2] == 1:
    #         y.append(2)
    #     if val[3] == 1:
    #         y.append(3)

    # model = LogisticRegression(multi_class='multinomial', solver='lbfgs')  
    # model.fit(X, y)

    # return model

    clf_up, clf_right, clf_down, clf_left = LogisticRegression(), LogisticRegression(), LogisticRegression(), LogisticRegression()
    for matrix in y_list:
        y_up_list.append(matrix[0])
        y_right_list.append(matrix[1])
        y_down_list.append(matrix[2])
        y_left_list.append(matrix[3])

    clf_up.fit(X, y_up_list)
    clf_right.fit(X, y_right_list)
    clf_down.fit(X, y_down_list)
    clf_left.fit(X, y_left_list)

    return clf_up, clf_right, clf_down, clf_left


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
    
    return np.ndarray(shape=(length_x, 8), dtype=int, buffer=list_of_data)


if __name__ == "__main__":
    pass