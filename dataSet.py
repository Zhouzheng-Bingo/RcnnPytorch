import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pywt


class DataSet():
    def __init__(self, res_data_path, sample_data_path):
        self.all_res_data = np.load(res_data_path)
        self.all_sample_data = np.load(sample_data_path)
        self.flute_1_wear = self.all_res_data[:, 0]
        self.flute_2_wear = self.all_res_data[:, 1]
        self.flute_3_wear = self.all_res_data[:, 2]
        self.wear = np.mean(self.all_res_data, axis=1)

    def gen_data(self):
        # print('Wavelet is used, db4 and 2 level')
        A2, D2, D1 = pywt.wavedec(self.all_sample_data, 'db4', mode='symmetric', level=2, axis=2)
        # print(self.all_sample_data.shape, A2.shape, D2.shape, D1.shape)
        return A2, D2, D1


class Data(Dataset):
    def __init__(self, data, start, end):
        sample_data, d2, d1 = data.gen_data()
        sample_data = sample_data[start:end]
        labels = data.wear
        labels = labels[start:end]
        self.dataset = list(zip(sample_data, labels))

    def __getitem__(self, idx):
        assert idx < len(self.dataset)
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)


class TCNData(Dataset):
    def __init__(self, data1, data2):
        self.dataset = list(zip(data1, data2))

    def __getitem__(self, idx):
        assert idx < len(self.dataset)
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)


if __name__ == '__main__':
    res_dat = '.cache/all_res_dat_num_5000.npy'
    sample_dat = '.cache/all_sample_dat_num_5000.npy'
    rul_x = '.cache/RUL_X.npy'
    rul_y = '.cache/RUL_Y.npy'
    res_data = np.load(res_dat)
    sample_data = np.load(sample_dat)
    x = np.load(rul_x)
    y = np.load(rul_y)
    print(res_data.shape, sample_data.shape, x.shape, y.shape)