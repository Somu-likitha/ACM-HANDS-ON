import torch
import pandas as pd
from json import load

class Train(torch.utils.data.Dataset):
    def __init__(self, length = 50000):
        self.length = length
        base_dir = "mnist"
        data = torch.tensor(pd.read_csv(base_dir+"/train.csv", header=None).values)
        self.x, self.y = (data[:, 1:]/255).float(), data[:, 0].long()

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.x[index], self.y[index].float()

class Val(torch.utils.data.Dataset):
    def __init__(self, length = 10000):
        self.length = length
        base_dir = "mnist"
        test = torch.tensor(pd.read_csv(base_dir+"/test.csv", header=None).values)
        self.x_test, self.y_test = (test[:, 1:]/255).float(), test[:, 0].long()

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.x_test[index], self.y_test[index].float()

if __name__ == "__main__":
    from tqdm import tqdm
    train_data = Train()
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True,  num_workers=16, pin_memory=True)
    for x_data, y_data in tqdm(train_loader):
        # print(x_data)
        # break
        pass
