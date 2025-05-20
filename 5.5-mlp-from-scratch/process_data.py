from ucimlrepo import fetch_ucirepo 
import numpy as np 
import pandas as pd 
import pickle


def save_data_2():
    adult = fetch_ucirepo(id=2) 

    X = adult.data.features.to_numpy() 
    y = adult.data.targets.to_numpy()

    workclass = np.array(['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'])
    education = np.array(['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'])
    marital_status = np.array(['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'])
    occupation = np.array(['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'])
    relationship = np.array(['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'])
    race = np.array(['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'])
    gender = np.array(['Female', 'Male'])
    country = np.array(['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands'])
    workclass_mapping = {category: idx for idx, category in enumerate(workclass)}
    education_mapping = {category: idx for idx, category in enumerate(education)}
    marital_status_mapping = {category: idx for idx, category in enumerate(marital_status)}
    occupation_mapping = {category: idx for idx, category in enumerate(occupation)}
    relationship_mapping = {category: idx for idx, category in enumerate(relationship)}
    race_mapping = {category: idx for idx, category in enumerate(race)}
    gender_mapping = {category: idx for idx, category in enumerate(gender)}
    country_mapping = {category: idx for idx, category in enumerate(country)}

    transformed_data = []
    holder = []
    for i, x in enumerate(X):
        holder.append([x[0], workclass_mapping.get(x[1], None), x[2], education_mapping.get(x[3], None), x[4],  marital_status_mapping.get(x[5], None),  occupation_mapping.get(x[6], None),  relationship_mapping.get(x[7], None), race_mapping.get(x[8], None),  gender_mapping.get(x[9], None), x[10],  x[11], x[12], country_mapping.get(x[13], None)])
        if not None in holder[0]:
            if y[i][0] == '<=50K.' or y[i][0] == '<=50K':
                holder[0].append(0)
            elif y[i][0] == '>50K.' or y[i][0] == '>50K':
                holder[0].append(1)
            transformed_data.append(holder[0])
        holder = []
    transformed_data = np.array(transformed_data)

    with open("data2.pickle", 'wb') as f:
       pickle.dump((transformed_data), f)

def get_data_2():
    with open("data/data2.pickle", 'rb') as f:
        data = pickle.load(f)
    
    np.random.shuffle(data)
    X = []
    y = []
    y = np.zeros((754, 1))
    for i, el in enumerate(data[::60]):
        X.append(el[:-1])
        y[i] = 1 if int(el[-1]) == 1 else 0

    x_train = np.array(X[:700])
    y_train = np.array(y[:700])
    x_test = np.array(X[700:])
    y_test = np.array(y[700:])

    return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":
    # (x_train, y_train), (x_test, y_test) = get_data_2()
    pass