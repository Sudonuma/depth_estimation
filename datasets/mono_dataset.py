# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import random
import numpy as np
import copy
from PIL import Image  # using pillow-simd for increased speed

import torch
import torch.utils.data as data
from torchvision import transforms


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def npy_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class MonoDataset(data.Dataset):
    """Superclass for monocular dataloaders

    Args:
        data_path
        filenames
        height
        width
        frame_idxs
        num_scales
        is_train
        img_ext
    """
    def __init__(self,
                 data_path,
                 filenames,
                 gt_filenames,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 is_train=False,
                 img_ext='.jpg'):
        super(MonoDataset, self).__init__()

        self.data_path = data_path
        self.filenames = filenames

        self.gt_filenames = gt_filenames

        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.interp = Image.ANTIALIAS
        # self.folder = 0
        self.frame_idxs = frame_idxs

        self.is_train = is_train
        self.img_ext = img_ext

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
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
        self.resize1 = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)

            # self.resize1[i] = transforms.Resize((288, 512), interpolation=0)

            self.resize1[i] = transforms.Resize((self.height // s, self.width // s), interpolation=0)

        # self.load_depth = self.check_depth()
        self.load_depth = True

    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            frame = inputs[k]
            # print('aaa',k)
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        for k in list(inputs):
            frame = inputs[k]
            # print(k)
            if "ground_truth" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    # x = inputs[(n, im, i)]
                    # x = Image.fromarray(inputs[(n, im, i - 1)])
                    inputs[(n, im, i)] = self.resize1[i](inputs[(n, im, i - 1)])

                    # inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])
        for k in list(inputs):
            frame = inputs[k]
            # print(k)
            if "mask" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    # x = inputs[(n, im, i)]
                    # x = Image.fromarray(inputs[(n, im, i - 1)])
                    inputs[(n, im, i)] = self.resize1[i](inputs[(n, im, i - 1)])


        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))
                # torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        for k in list(inputs):
            f = inputs[k]
            if "ground_truth" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                # inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))


        for k in list(inputs):
            f = inputs[k]
            if "mask" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5
        line = self.filenames[index].split() 
        #print("index",index)
        #print("self.gt_filenames",len(self.gt_filenames))
        #print("self.filenames",len(self.filenames))
        line_gt = self.gt_filenames[index].split()
        filenames_list = []
        for idx, i in enumerate(self.gt_filenames):
            filenames_list.append(i.split())

        files = np.asarray(filenames_list)
        list_of_files = files[:,1]
        folder = line[0]
        folder_gt = line_gt[0]
        ####################################gt_folder = gt_line[0]
        self.file_number = int(line[1])
        self.file_number_gt = int(line_gt[1])
        self.folder = folder
        self.folder_gt = folder_gt

        if len(line) == 3:
            frame_index = int(line[1])
            frame_index_gt = int(line_gt[1])
        else:
            frame_index = 0
            frame_index_gt = 0

        if len(line) == 3:
            side = line[2]
        else:
            side = None

        for i in self.frame_idxs:
            if i == "s":
                other_side = {"r": "l", "l": "r"}[side]
                inputs[("color", i, -1)] = self.get_color(folder, frame_index, other_side, do_flip)
            else:
                inputs[("color", i, -1)] = self.get_color(folder, frame_index + i, side, do_flip)

        
        for i in self.frame_idxs:
            value = int(line[1])+int(i) 
            compvalue = [folder, frame_index + i, side]
            l =  compvalue == files
            if (l.all(axis=1)).any():
                inputs[("ground_truth", i, -1)] = self.get_gtdepth(folder, frame_index + i, side, do_flip)
            else:
                x = np.zeros((1920,1080))
                inputs[("ground_truth", i, -1)] = Image.fromarray(x)
         
        for i in self.frame_idxs:
                inputs[("mask", i, -1)] = self.get_seg_mask(folder, frame_index + i, side, do_flip)

        for scale in range(self.num_scales):
            K = self.K.copy()
            K[:, 0, :] *= self.width // (2 ** scale)
            K[:, 1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if do_color_aug:
            color_aug = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)

        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]
            del inputs[("ground_truth", i, -1)]
            del inputs[("mask", i, -1)]

        if "s" in self.frame_idxs:
            stereo_T = np.eye(4, dtype=np.float32)
            baseline_sign = -1 if do_flip else 1
            side_sign = -1 if side == "l" else 1
            stereo_T[0, 3] = side_sign * baseline_sign * 0.1

            inputs["stereo_T"] = torch.from_numpy(stereo_T)
        inputs["target_folder"] = folder
        inputs["target_file"] = self.file_number

        return inputs

    def return_folder(self, folder):
        return self.folder

    def get_color(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def check_depth(self):
        raise NotImplementedError

    def get_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError
