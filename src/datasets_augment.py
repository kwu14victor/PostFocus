"""
Dataset object to read OOF images
Modified from dataset used for training
"""
import os

import numpy as np
from PIL import Image
from skimage import io,exposure
from torch.utils.data import Dataset
from torchvision import transforms

class DeblurDs(Dataset):
    """
    Dataset class to infer/restore OOF images
    """
    def __init__(self, blur_image_files, sharp_image_files, root_dir,\
                crop_size=256, multi_scale=False,\
                transform=None,rescale=True):
        blur_file = open(blur_image_files, 'r')
        self.blur_image_files = blur_file.readlines()
        sharp_file = open(sharp_image_files, 'r')
        self.sharp_image_files = sharp_file.readlines()
        self.root_dir = root_dir
        self.transform = transform
        self.multi_scale = multi_scale
        self.rescale = rescale
        self.crop_size = crop_size

    def __len__(self):
        return len(self.blur_image_files)

    def __getitem__(self, idx):

        image_name1 = self.blur_image_files[idx][:-1]
        image_name2 = self.sharp_image_files[idx][:-1]
        blur_image = io.imread(os.path.join(self.root_dir, image_name1))
        sharp_image = io.imread(os.path.join(self.root_dir, image_name2))
        blur_image = exposure.rescale_intensity(blur_image,  out_range=np.uint8)
        sharp_image = exposure.rescale_intensity(sharp_image,  out_range=np.uint8)
        blur_image = Image.fromarray(blur_image)
        sharp_image = Image.fromarray(sharp_image)
        trf_fun_1 = transforms.Compose([transforms.Resize(size = (self.crop_size, self.crop_size))])
        trf_fun_2 = transforms.Compose([transforms.Resize(size = (self.crop_size, self.crop_size))])
        blur_image = trf_fun_1(blur_image)
        sharp_image = trf_fun_2(sharp_image)
        if self.transform:
            blur_image = self.transform(blur_image)
            sharp_image = self.transform(sharp_image)
