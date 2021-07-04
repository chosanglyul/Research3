import numpy as np
import pandas as pd
from torch import empty, unsqueeze, FloatTensor
from torch.utils.data import Subset, DataLoader, TensorDataset
from pytorch_lightning import LightningDataModule

def make_lag(n_X, n_y, y):
    n = y.shape[0]+1-n_X-n_y
    if n <= 0: raise ValueError
    X = empty((n, n_X))
    Y = empty((n, n_y))
    for i in range(n_X):
        X[:, i] = y[i:n+i]
    for i in range(n_y):
        Y[:, i] = y[n_X+i:n+n_X+i]
    return X, Y

def split_data(dataset, permutation, n_split):
    if (permutation.ndim != 1) or (len(dataset) != permutation.size):
        raise ValueError
    train = Subset(dataset, permutation[:n_split[0]])
    val = Subset(dataset, permutation[n_split[0]:-n_split[2]])
    test = Subset(dataset, permutation[-n_split[2]:])
    return train, val, test
        
class BaseData(LightningDataModule):
    def __init__(self, dataset, val_split=0.2, test_split=0.2, random_seed=7, shuffle=True, **kwargs):
        super().__init__()
        if (val_split <= 0) or (test_split <= 0) or (test_split+val_split >= 1):
            raise ValueError
        n = len(dataset)
        n_val = int(n*val_split)
        n_test = int(n*test_split)
        n_split = [n-n_val-n_test, n_val, n_test]
        rng = np.random.default_rng(random_seed)
        if shuffle:
            per = rng.permutation(n)
        else:
            per = np.arange(n)
            rng.shuffle(per[:n_split[0]])
        self.train_data, self.val_data, self.test_data = split_data(dataset, per, n_split)
        self.kwargs = kwargs
            
    def train_dataloader(self):
        return DataLoader(self.train_data, **self.kwargs)
    
    def val_dataloader(self):
        return DataLoader(self.val_data, **self.kwargs)

    def test_dataloader(self):
        return DataLoader(self.test_data, **self.kwargs)

class SinData(BaseData):
    def __init__(self, n_X, n_y, **kwargs):
        t = np.linspace(0, 300, num=3000, endpoint=False) 
        X, y = make_lag(n_X, n_y, FloatTensor(np.sin(t)))
        super().__init__(TensorDataset(X, y), **kwargs)
        
class SunspotData(BaseData):
    def __init__(self, n_X, n_y, **kwargs):
        df = pd.read_csv('../rawdata/sunspots.csv', index_col='Unnamed: 0')
        df.columns = ['Date', 'Sunspot']
        y = df['Sunspot'].to_numpy()
        X, y = make_lag(n_X, n_y, FloatTensor(y))
        super().__init__(TensorDataset(unsqueeze(X, -1), y), **kwargs)

class ElectricData(BaseData):
    def __init__(self, n_X, n_y, **kwargs):
        df = pd.read_csv('../rawdata/electric.csv', index_col='Date')
        y = df['Demand'].to_numpy()
        X, y = make_lag(n_X, n_y, FloatTensor(y))
        super().__init__(TensorDataset(unsqueeze(X, -1), y), **kwargs)