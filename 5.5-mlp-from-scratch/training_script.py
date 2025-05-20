from model import MLP
from dataset import MLPDataset
import torch
import numpy as np


train_dataset = MLPDataset('train')
test_dataset = MLPDataset('test')
sample_X, sample_y = train_dataset[0]
features = sample_X.shape[0]
targets = sample_y.shape[0]
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)

model = MLP(input=features, output=targets, hid_layers=1, neurons=4)


for epoch in range(30):
    for input, target in train_loader:
        model.training(input, target)
    mean_loss = model.loss().mean()
    print(f'{mean_loss:.6f}')

correct = 0
for input, target in test_loader:
    if np.argmax(model.evaluate(input)) == np.argmax(target):
        correct += 1
acc = correct / test_dataset.__len__()
print(acc)
