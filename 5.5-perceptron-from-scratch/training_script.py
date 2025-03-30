from model import Perceptron
from process_data import get_data
import torch
import numpy as np

train_dataset, validation_dataset = get_data()
features = train_dataset[0][0].shape[0]
targets = train_dataset[0][2].shape[0]

model = Perceptron(input=features, output=targets, hid_layers=1, neurons=16)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

for epoch in range(100):
    for data, _, target in train_loader:
        data = np.array(data)
        target = np.array(target)
        model.training(data[0], target[0])
    mean_loss = model.loss().mean()
    print(f'{mean_loss:.3f}')
