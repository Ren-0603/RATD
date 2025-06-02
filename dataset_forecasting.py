import pickle
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

class Dataset_Electricity(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='electricity.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len, dim]
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.dim = size[3]

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(self.root_path + '/' + self.data_path, index_col='date', parse_dates=True)
        self.scaler = StandardScaler()

        df_raw = pd.DataFrame(df_raw)

        num_train = int(len(df_raw) * 0.7) - self.pred_len - self.seq_len + 1
        num_test = int(len(df_raw) * 0.2)
        num_valid = int(len(df_raw) * 0.1)
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_valid, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_data = df_raw.values

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(df_data)
        else:
            data = df_data

        target_col_idx = list(df_raw.columns).index(self.target)  # ← インデント正しく修正！

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2, target_col_idx:target_col_idx+1]
        self.mask_data = np.ones_like(self.data_x)


    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len + self.pred_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        target_mask = self.mask_data[s_begin:s_end].copy()
        target_mask[-self.pred_len:] = 0.

        s = {
            'observed_data': seq_x,
            'observed_mask': self.mask_data[s_begin:s_end],
            'gt_mask': target_mask,
            'timepoints': np.arange(self.seq_len + self.pred_len) * 1.0,
            'feature_id': np.arange(self.dim) * 1.0,
            'reference': np.zeros((3 * self.pred_len, self.dim)),
        }

        return s

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
