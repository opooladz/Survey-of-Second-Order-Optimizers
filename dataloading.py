import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
from sklearn import datasets

transform_MNIST = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])


class MNIST_small(Dataset):
    def __init__(self, train=True):
        digits = datasets.load_digits()
        data_num = digits['target'].shape[0]
        if train == True:
            self.data, self.target = digits['data'][:int(0.8*data_num)], digits['target'][:int(0.8*data_num)]
        else:
            self.data, self.target = digits['data'][int(0.8*data_num):], digits['target'][int(0.8*data_num):]
        
    def __len__(self):
        return self.target.shape[0]
    
    def __getitem__(self, idx):
        inputs = torch.from_numpy(self.data[idx]).float()
        targets = torch.from_numpy(np.asarray(self.target[idx]))
        
        return (inputs, targets)