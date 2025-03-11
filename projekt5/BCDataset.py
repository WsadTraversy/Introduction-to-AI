from torch.utils.data import Dataset, DataLoader
import torch
import math
from read_data import get_data, get_data_with_tail

class BCDataset(Dataset):
    def __init__(self, file_name):
        # data loading
        xy = get_data(file_name)
        self.x = torch.from_numpy(xy[0]).float()
        self.y = torch.from_numpy(xy[1]).float()
        self.n_samples = len(xy[0])
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.n_samples

if __name__ == "__main__":
    pass
    # dataset = BCDataset()
    # dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=2)

    # dataiter = iter(dataloader)
    # data = dataiter._next_data()
    # features, labels = data
    # print(features, labels)

    # training loop
    # num_epochs = 2
    # total_samples = len(dataset)
    # n_iterations = math.ceil(total_samples/4)
    # print(total_samples, n_iterations)

    # for epoch in range(num_epochs):
    #     for i, (inputs, labels) in enumerate(dataloader):
    #         # forward backward, pass
    #         if (i + 1) % 5 == 0:
    #             print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{n_iterations}, inputs {inputs.shape}')