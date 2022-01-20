import torch
import torch.utils.data as tordata
import os.path as osp
import numpy as np
from functools import partial


class CTPatchDataset(tordata.Dataset):
    def __init__(self, npy_root, hu_range, transforms=None):
        self.transforms = transforms
        hu_min, hu_max = hu_range
        self.hu_min = hu_min
        self.hu_max = hu_max
        self.data = np.load(npy_root, mmap_mode="r")
        # data = torch.from_numpy(np.load(npy_root, mmap_mode="r").astype(np.float32) - 1024)
        # normalize to [0, 1]
        # data = (torch.clamp(data, hu_min, hu_max) - hu_min) / (hu_max - hu_min)
        # self.low_doses, self.full_doses = data[0], data[1]

    def normalizer(self, data):
        data = torch.from_numpy(data.astype(np.float32) - 1024)
        beneath = self.hu_max - self.hu_min
        data = (torch.clamp(data, self.hu_min, self.hu_max) - self.hu_min) / beneath
        return data

    def __getitem__(self, index):
        low_dose = self.normalizer(self.data[0][index])
        full_dose = self.normalizer(self.data[1][index])
        if self.transforms is not None:
            low_dose = self.transforms(low_dose)
            full_dose = self.transforms(full_dose)
        return low_dose, full_dose

    def __len__(self):
        return len(self.data[0])


data_root = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), 'dataset')
dataset_dict = {
    'cmayo_train_64': partial(CTPatchDataset, npy_root=osp.join(data_root, 'cmayo/train_64.npy')),
    'cmayo_test_512': partial(CTPatchDataset, npy_root=osp.join(data_root, 'cmayo/test_512.npy')),
}
