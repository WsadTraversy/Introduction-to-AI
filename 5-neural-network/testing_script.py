from BCDataset import BCDataset
from torch.utils.data import DataLoader
from MLP import MLP
import torch
import torch.nn as nn
from torchmetrics.classification import BinaryAccuracy

test_dataset = BCDataset('test')
test_loader = DataLoader(dataset=test_dataset, batch_size=test_dataset.__len__(), shuffle=True, num_workers=2)
metric = BinaryAccuracy()
model = MLP(1, 4)
loss_fn = nn.CrossEntropyLoss()
model.load_state_dict(torch.load('data/state_dict.pickle'))
model.eval()
for inputs, targets in test_loader:
    with torch.no_grad():
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
print(f'{loss.item():.3f}, {metric(outputs, targets).item():.3f}')