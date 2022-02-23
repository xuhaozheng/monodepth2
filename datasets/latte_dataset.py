from __future__ import absolute_import, division, print_function
from fileinput import filename

import os
import random
import numpy as np
import copy
from PIL import Image
import PIL.Image as pil

import torch
import torch.utils.data as data
from torchvision import transforms

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

class latte_dataset(data.Dataset):
    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 acquire_gt=False,
                 is_train=False,
                 img_ext='.jpg'):
        super(latte_dataset, self).__init__()

        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.interp = Image.ANTIALIAS
        self.acquire_gt = acquire_gt

        self.frame_idxs = frame_idxs

        self.is_train = is_train
        self.img_ext = img_ext

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()
        self.K = np.array([[0.9285097, 0, 0.50113267, 0],
                           [0, 1.1606371, 0.5078349, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = {}
        for i in range(self.num_scales):
            s = 2**i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)

    def preprocess(self, inputs, color_aug):
        for k in list(inputs):
            if 'color' in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i-1)])

        for k in list(inputs):
            f = inputs[k]
            if 'color' in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))
    #

    def get_color(self, image_name, frame_index, side):
        color = self.loader(self.get_image_path(image_name, frame_index, side))

        return color

    def get_image_path(self, folder, frame_index, side):
            side_id = {'l': 'left', 'r': 'right'}[side]
            image_path = os.path.join(self.data_path, side_id, frame_index)
            return image_path


    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        inputs = {}

        # do_color_aug = self.is_train and random.random() > 0.5
        do_color_aug = False

        frame_idx = self.filenames[index] # Round10/00043.jpg
        folder = ""###include in the frame_idx, no longer use
        side = "l"
        for i in self.frame_idxs:
            if i == "s":
                other_side = {"r": "l", "l": "r"}[side]
                inputs[("color", i, -1)] = self.get_color(folder, other_side, frame_idx)
            else:
                if self.acquire_gt == True:
                    inputs[("gt_depth")] = self.get_depth(folder, frame_idx, side)
                inputs[("color", i, -1)] = self.get_color(folder, frame_idx, side)
        
        for scale in range(self.num_scales):
            K = self.K.copy()

            K[0,:] *= self.width // 2**scale
            K[1,:] *= self.height // 2**scale

            K_inv = np.linalg.pinv(K)

            inputs[('K', scale)] = torch.from_numpy(K)
            # inputs[('K_inv'), scale] = torch.from_numpy(K_inv)
            inputs[('inv_K'), scale] = torch.from_numpy(K_inv)
        
        if do_color_aug:
            color_aug = transforms.ColorJitter(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)

        for i in self.frame_idxs:
            del inputs[('color', i, -1)]
            del inputs[('color_aug', i, -1)]
        
        if "s" in self.frame_idxs:
            stereo_T = np.eye(4, dtype=np.float32)
            # baseline_sign = -1 if do_flip else 1
            side_sign = -1 if side == "l" else 1
            stereo_T[0, 3] = side_sign * 0.1

            inputs[("stereo_T")] = torch.from_numpy(stereo_T)

        return inputs


    def check_depth(self):
        raise NotImplementedError

    def get_depth(self, image_name, frame_index, side):
        depth = pil.open(self.get_image_path(image_name, frame_index, side)).convert('L')
        depth = np.array(depth)
        depth = torch.tensor(depth,dtype=torch.float)

        return depth







# def main():
#     img = pil_loader('/Users/xuchi/Desktop/MRes_medical_robotics/individual/dataset/daVinci/train/image_jpg_%d/000001.jpg' %1)
#     img = transforms.ToTensor()(img)
#     f_str = "{:06d}{}".format(1, '.jpg')
#     print(f_str)

# if __name__ == '__main__':
#     main()