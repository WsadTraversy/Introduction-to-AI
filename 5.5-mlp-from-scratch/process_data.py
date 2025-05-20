from ucimlrepo import fetch_ucirepo 
import numpy as np 
import pandas as pd 
import pickle


def save_data_1():
    # fetch dataset 
    heart_disease = fetch_ucirepo(id=45) 
  
    # data (as pandas dataframes) 
    X = heart_disease.data.features.to_numpy() 
    y = heart_disease.data.targets.to_numpy()

    transformed_data = []
    for i, x in enumerate(X):
        if not np.any(np.isnan(x)):
            x = np.append(x, y[i])
            transformed_data.append(x)
    transformed_data = np.array(transformed_data)
    with open("data1.pickle", 'wb') as f:
       pickle.dump((transformed_data), f)

def get_data_1():
    with open("data/data1.pickle", 'rb') as f:
        data = pickle.load(f)
    
    np.random.shuffle(data)
    X = []
    y = []
    y = np.zeros((297, 5))
    for i, el in enumerate(data):
        X.append(el[:-1])
        y[i][int(el[-1])] = 1 

    x_train = np.array(X[:250])
    y_train = np.array(y[:250])
    x_test = np.array(X[250:])
    y_test = np.array(y[250:])

    return (x_train, y_train), (x_test, y_test)

if __name__ == "__main__":
    save_data_1()
    # pass