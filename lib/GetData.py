import numpy as np
import pandas as pd
import torch
from torch import Generator, FloatTensor
from torch.utils.data import DataLoader, random_split, TensorDataset
from pytorch_lightning import LightningDataModule

def make_lag(n_seq, y):
    n = y.shape[0]
    X = torch.empty((n-n_seq, n_seq))
    for i in range(n_seq):
        X[:, i] = y[i:n-n_seq+i]
    y = y[n_seq:]
    return X, torch.unsqueeze(y, -1)

class BaseData(LightningDataModule):
    def __init__(self, dataset, val_split=0.2, test_split=0.2, random_seed=7, **kwargs):
        super().__init__()
        n = len(dataset)
        n_val = int(n*val_split)
        n_test = int(n*test_split)
        n_split = [n-n_val-n_test, n_val, n_test]
        gen = Generator().manual_seed(random_seed)
        self.train_data, self.val_data, self.test_data = random_split(dataset, n_split, generator=gen)
        self.kwargs = kwargs

    def train_dataloader(self):
        return DataLoader(self.train_data, **self.kwargs)
    
    def val_dataloader(self):
        return DataLoader(self.val_data, **self.kwargs)

    def test_dataloader(self):
        return DataLoader(self.test_data, **self.kwargs)

class SinData(BaseData):
    def __init__(self, n_seq, **kwargs):
        t = np.linspace(0, 300, num=3000, endpoint=False) 
        X, y = make_lag(n_seq, FloatTensor(np.sin(t)))
        super().__init__(TensorDataset(X, y), **kwargs)
        
class SunspotData(BaseData):
    def __init__(self, n_seq, **kwargs):
        df = pd.read_csv('../rawdata/sunspots.csv', index_col='Unnamed: 0')
        df.columns = ['Date', 'Sunspot']
        y = df['Sunspot'].to_numpy()
        X, y = make_lag(n_seq, FloatTensor(y))
        X = torch.unsqueeze(X, -1)
        super().__init__(TensorDataset(X, y), **kwargs)