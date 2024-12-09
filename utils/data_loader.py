"""
Description: build [train / valid / test] dataloader
Author: Xiongjun Guan
Date: 2023-03-01 19:41:05
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2023-03-20 21:01:47

Copyright (C) 2023 by Xiongjun Guan, Tsinghua University. All rights reserved.
"""
import copy
import logging
import os.path as osp
import random
from glob import glob
from random import randint

import cv2
import numpy as np
import scipy.io as scio
import torch
from scipy.ndimage import rotate, shift, zoom
from torch.utils.data import DataLoader, Dataset

from .noise import GaussianNoise, dryness, heavypress, sensor_noise


class load_dataset_test(Dataset):

    def __init__(
        self,
        info_lst: list,
        img_dir: str,
    ):
        self.info_lst = info_lst
        self.img_dir = img_dir

    def __len__(self):
        return len(self.info_lst)

    def __getitem__(self, idx):
        ftitle = self.info_lst[idx]
        img = cv2.imread(osp.join(self.img_dir, ftitle + ".png"), 0)

        img = (255.0 - img) / 255.0

        img = np.ascontiguousarray(img)[None, :, :]

        return img, ftitle


class load_dataset_train(Dataset):

    def __init__(
        self,
        info_lst: list,
        img_dir: str,
        bimg_dir: str,
        patch_size=128,
        need_name=False,
    ):
        self.info_lst = info_lst
        self.img_dir = img_dir
        self.bimg_dir = bimg_dir
        self.patch_size = patch_size
        self.need_name = need_name

    def __len__(self):
        return len(self.info_lst)

    def __getitem__(self, idx):
        ftitle = self.info_lst[idx]
        img = cv2.imread(osp.join(self.img_dir, ftitle + ".png"), 0)
        bimg = cv2.imread(osp.join(self.bimg_dir, ftitle + ".png"), 0)

        h, w = img.shape
        hc, wc = h // 2, w // 2

        hc += random.randint(-150, 150)
        wc += random.randint(-150, 150)
        ps = self.patch_size // 2

        img = img[hc - ps:hc + ps, wc - ps:wc + ps] * 1.0
        bimg = bimg[hc - ps:hc + ps, wc - ps:wc + ps] * 1.0

        # ---- fliplr
        if np.random.rand() < 0.5:
            img, bimg = np.fliplr(img), np.fliplr(bimg)

        # ---- rotate
        ang = np.random.choice([0, 90, 180, 270],
                               1,
                               p=[0.25, 0.25, 0.25, 0.25])[0]
        img = rotate(img, ang, reshape=False, mode="constant", cval=255)
        bimg = rotate(bimg, ang, reshape=False, mode="constant", cval=255)

        # ---- sensor noise: perlin
        if np.random.rand() < 0.1:
            strength = np.random.uniform(60, 120)
            if np.random.rand() < 0.5:
                img, _ = sensor_noise(img,
                                      stride=8,
                                      do_wet=False,
                                      pL=0.1,
                                      tB=20,
                                      strength=strength)
            else:
                img, _ = sensor_noise(img,
                                      stride=8,
                                      do_wet=True,
                                      pL=0.1,
                                      tB=20,
                                      strength=strength)
        # ---- sensor noise: gaussian
        if np.random.rand() < 0.5:
            noise_mode = np.random.choice([1, 2, 3], 1, p=[0.33, 0.33,
                                                           0.34])[0]
            if noise_mode == 1:
                img = dryness(img)
            elif noise_mode == 2:
                img = heavypress(img)
            elif noise_mode == 3:
                blur_core = np.random.choice([3, 5], 1, p=[0.5, 0.5])[0]
                img = cv2.GaussianBlur(img, (blur_core, blur_core), 3)
                gaussian_sigma = np.random.choice([5, 10], 1, p=[0.5, 0.5])[0]
                img = GaussianNoise(img, 0, gaussian_sigma, 0.1)

        # ---- clip
        img = np.clip(img, 0, 255)
        bimg = np.clip(bimg, 0, 255)

        img = (255.0 - img) / 255.0
        bimg = (255.0 - bimg) / 255.0

        img = np.ascontiguousarray(img)[None, :, :]
        bimg = np.ascontiguousarray(bimg)[None, :, :]

        if self.need_name is True:
            return img, bimg, ftitle
        else:
            return img, bimg


def get_dataloader_train(
    info_lst: list,
    img_dir: str,
    bimg_dir: str,
    patch_size=128,
    batch_size=1,
    shuffle=True,
):
    # Create dataset
    try:
        dataset = load_dataset_train(
            info_lst,
            img_dir=img_dir,
            bimg_dir=bimg_dir,
            patch_size=patch_size,
        )
    except Exception as e:
        logging.error("Error in DataLoader: ", repr(e))
        return

    train_loader = DataLoader(dataset,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              num_workers=16,
                              pin_memory=True)
    logging.info(f"n_train:{len(dataset)}")

    return train_loader


def get_dataloader_valid(
    info_lst: list,
    img_dir: str,
    bimg_dir: str,
    patch_size=128,
    batch_size=1,
    need_name=False,
):
    # Create dataset
    try:
        dataset = load_dataset_train(
            info_lst,
            img_dir=img_dir,
            bimg_dir=bimg_dir,
            patch_size=patch_size,
            need_name=need_name,
        )
    except Exception as e:
        logging.error("Error in DataLoader: ", repr(e))
        return

    valid_loader = DataLoader(dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=16,
                              pin_memory=True)
    logging.info(f"n_valid:{len(dataset)}")

    return valid_loader


def get_dataloader_test(
    info_lst: list,
    img_dir: str,
    batch_size=1,
):
    # Create dataset
    try:
        dataset = load_dataset_test(
            info_lst,
            img_dir=img_dir,
        )
    except Exception as e:
        logging.error("Error in DataLoader: ", repr(e))
        return

    test_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=12,
                             pin_memory=True)
    logging.info(f"n_test:{len(dataset)}")

    return test_loader
