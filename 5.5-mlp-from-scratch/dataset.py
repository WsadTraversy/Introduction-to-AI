from torch.utils.data import Dataset
from process_data import get_data
import torch

class MLPDataset(Dataset):
    def __init__(self, set):
        xy_train, xy_test = get_data()
        if set == 'train':
            xy = xy_train
        elif set == 'test':
            xy = xy_test
        else:
            raise ValueError('Choose beetwen train, test sets')
        self.X = [item[0] for item in xy]
        self.y = [item[1] for item in xy]
        self.n_samples = len(self.X)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return self.X[index], self.y[index]