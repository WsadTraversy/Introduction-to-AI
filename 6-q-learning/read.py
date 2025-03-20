import pickle

# Replace 'your_file.pkl' with the path to your pickle file
file_path = 'data/data1.pickle'

# Open the pickle file in read-binary mode
with open(file_path, 'rb') as file:
    data = pickle.load(file)

# Print the contents of the pickle file
print(data)