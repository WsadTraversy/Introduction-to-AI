from torch.utils.data import Dataset
from process_data import get_data_1
import torch

class MLPDataset():
    def __init__(self, set):
        xy_train, xy_test = get_data_1()
        if set == 'train':
            xy = xy_train
        elif set == 'test':
            xy = xy_test
        else:
            raise ValueError('Choose beetwen train, test sets')
        self.X = xy[0]
        self.y = xy[1]
        self.n_samples = len(xy[0])

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return self.X[index], self.y[index]