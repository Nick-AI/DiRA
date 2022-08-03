
import os
import torch
import random
import copy
from glob import glob
from PIL import Image
import numpy as np
from scipy import ndimage
from torch.utils.data import Dataset
import numpy as np


class ChestX_ray14(Dataset):

    def __init__(self, pathImageDirectory, pathDatasetFile, augment):

        self.img_list = []
        self.augment = augment

        with open(pathDatasetFile, "r") as fileDescriptor:
            line = True

            while line:
                line = fileDescriptor.readline()

                if line:
                    lineItems = line.split()
                    imagePath = os.path.join(pathImageDirectory, lineItems[0])
                    self.img_list.append(imagePath)

    def __getitem__(self, index):
        imagePath = self.img_list[index]
        image_data = np.load(imagePath).astype(np.float32)
        image_data = np.moveaxis(image_data, -1, 0)
        image_data /= image_data.max()
        image_data = torch.from_numpy(image_data)
        return self.augment(image_data)

    def __len__(self):
        return len(self.img_list)
