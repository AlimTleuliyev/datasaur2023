import torch
import torch.nn as nn
from torch.utils.data import Dataset
import PIL
from glob import glob
import os


class InferenceDataset(Dataset):
    def __init__(self, images_path, transforms=None, images_format='jpeg'):
        self.images_path = list(glob(os.path.join(images_path, f'*.{images_format}')))
        if transforms:
            self.transforms = transforms

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, index):
        img_path = self.images_path[index]
        img = PIL.Image.open(img_path)

        if self.transforms:
            img = self.transforms(img)

        return img
