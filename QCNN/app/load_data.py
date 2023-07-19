from torch.utils.data import Dataset
import numpy as np
import torch


class MyCSVDatasetReader(Dataset):

    def __init__(self, csv_path):
        print(csv_path)
        self.DATA = np.genfromtxt(csv_path, delimiter = ',')
        self.X = self.DATA[:, 0:-1]
        self.Y = self.DATA[:, -1]
        self.X = np.pi*self.X/255
        
        
    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        X = torch.FloatTensor(self.X[idx, :])
        Y = self.Y[idx]
        sample = {'feature': X, 'label': Y}
        return sample

    def _get_labels(self):
        return self.Y