from __future__ import print_function
import os
import cv2
import numpy as np
import random
import torch
from torch.utils import data
from torchvision import transforms


def read_list(list_file):
    rgb_depth_list = []
    with open(list_file) as f:
        lines = f.readlines()
        for line in lines:
            rgb_depth_list.append(line.strip().split(" "))
    return rgb_depth_list


class Matterport3D(data.Dataset):
    """The Matterport3D Dataset"""

    def __init__(self, root_dir, list_file, equi_info, height=256, width=512, max_depth_meters=32,
                 disable_color_augmentation=False,
                 disable_LR_filp_augmentation=False, disable_yaw_rotation_augmentation=False, is_training=False,
                 ):
        """
        Args:
            root_dir (string): Directory of the Stanford2D3D Dataset.
            list_file (string): Path to the txt file contain the list of image and depth files.
            height, width: input size.
            disable_color_augmentation, disable_LR_filp_augmentation,
            disable_yaw_rotation_augmentation: augmentation options.
            is_training (bool): True if the dataset is the training set.
            equi_info:equi_info
        """
        self.root_dir = root_dir
        self.depth_Up_list = read_list(os.path.join(list_file, 'Matterport3D_Up_depth.txt'))
        self.normal_Up_list = read_list(os.path.join(list_file, 'Matterport3D_Up_rgb.txt'))
        self.depth_Center_list = read_list(os.path.join(list_file, 'Matterport3D_Center_depth.txt'))
        self.normal_Center_list = read_list(os.path.join(list_file, 'Matterport3D_Center_rgb.txt'))
        self.w = width
        self.h = height
        self.equi_info = equi_info
        self.max_depth_meters = max_depth_meters

        self.color_augmentation = not disable_color_augmentation
        self.LR_filp_augmentation = not disable_LR_filp_augmentation
        self.yaw_rotation_augmentation = not disable_yaw_rotation_augmentation

        self.is_training = is_training

        if self.color_augmentation:
            try:
                self.brightness = (0.8, 1.2)
                self.contrast = (0.8, 1.2)
                self.saturation = (0.8, 1.2)
                self.hue = (-0.1, 0.1)
                self.color_aug = transforms.ColorJitter(
                    self.brightness, self.contrast, self.saturation, self.hue)
            except TypeError:
                self.brightness = 0.2
                self.contrast = 0.2
                self.saturation = 0.2
                self.hue = 0.1
                self.color_aug = transforms.ColorJitter(
                    self.brightness, self.contrast, self.saturation, self.hue)

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.depth_Up_list) + len(self.normal_Up_list) + len(self.depth_Center_list) + len(
            self.normal_Center_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        inputs = {}
        rgb_name_Up = os.path.join(self.root_dir, "Up/Matterport3D", str(self.normal_Up_list[idx][0]))
        rgb_name_Up = cv2.imread(rgb_name_Up)
        rgb_name_Up = cv2.cvtColor(rgb_name_Up, cv2.COLOR_BGR2RGB)
        rgb_name_Up = cv2.resize(rgb_name_Up, dsize=(self.w, self.h), interpolation=cv2.INTER_CUBIC)

        depth_name_Up = os.path.join(self.root_dir, "Up/Matterport3D", self.depth_Up_list[idx][0])
        depth_name_Up = cv2.imread(depth_name_Up, cv2.IMREAD_UNCHANGED)
        depth_name_Up = cv2.resize(depth_name_Up, dsize=(self.w, self.h), interpolation=cv2.INTER_NEAREST)
        depth_name_Up = depth_name_Up.astype(np.float)
        depth_name_Up[depth_name_Up > self.max_depth_meters + 1] = self.max_depth_meters + 1

        rgb_name_Center = os.path.join(self.root_dir, "Center/Matterport3D", self.normal_Center_list[idx][0])
        rgb_name_Center = cv2.imread(rgb_name_Center)
        rgb_name_Center = cv2.cvtColor(rgb_name_Center, cv2.COLOR_BGR2RGB)
        rgb_name_Center = cv2.resize(rgb_name_Center, dsize=(self.w, self.h), interpolation=cv2.INTER_CUBIC)

        if self.is_training:
            rgb_name_Up = np.concatenate([rgb_name_Up, self.equi_info], 2)
            depth_name_Up = np.concatenate([depth_name_Up, self.equi_info], 2)
            rgb_name_Center = np.concatenate([rgb_name_Center, self.equi_info], 2)

        rgb_name_Up = self.to_tensor(rgb_name_Up.copy())
        rgb_name_Center = self.to_tensor(rgb_name_Center.copy())

        inputs["rgb_name_Up"] = rgb_name_Up
        inputs["rgb_name_Center"] = rgb_name_Center

        inputs["depth_name_Up"] = torch.from_numpy(np.expand_dims(depth_name_Up, axis=0))
        inputs["val_mask"] = ((inputs["depth_name_Up"] > 0) & (inputs["depth_name_Up"] <= self.max_depth_meters)
                              & (~ torch.isnan(inputs["depth_name_Up"])))
        return inputs
