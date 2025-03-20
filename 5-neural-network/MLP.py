from BCDataset import BCDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch


class MLP(nn.Module):
    def __init__(self, h_size, neuron_number, dropout=False):
        super().__init__()

        # self.hidden = nn.Linear(8, 64)
        self.hidden = nn.ModuleList()
        self.hidden.append(nn.Linear(8, neuron_number))
        for _ in range(h_size):
            self.hidden.append(nn.Linear(neuron_number, neuron_number))

        self.act = nn.ReLU()

        self.output = nn.Linear(neuron_number, 4)
        self.dropout_state = False
        if dropout:
            self.dropout_state = True
            self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        for layer in self.hidden:
            x = self.act(layer(x))
            if self.dropout_state:
                x = self.dropout(x)
        x = self.output(x)
        return x
