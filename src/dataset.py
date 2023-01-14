from torch.utils.data import Dataset
import torch
import random
import cv2
import numpy as np

class Step1_Dataset(Dataset):
    def __init__(self, data_txt_path, transform=None):
        self.data_txt_path = data_txt_path
        self.transform = transform
        with open(self.data_txt_path, 'r') as f:
            self.data_info = f.readlines()

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, item):
        img_pth, label = self.data_info[item].split()
        img = cv2.imread(img_pth)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            img = self.transform(img)

        return img, torch.tensor(int(label), dtype=torch.long)

class Step2_Dataset(Dataset):
    def __init__(self, data_txt_path, transform=None):
        self.data_txt_path = data_txt_path
        self.transform = transform
        with open(self.data_txt_path, 'r') as f:
            self.data_info = f.readlines()

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, item):
        img_pth, prefix = self.data_info[item].split()
        img = cv2.imread(img_pth)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            img = self.transform(img)

        label = np.asarray(list(map(int, list(prefix))), dtype=float)
        return img, label


