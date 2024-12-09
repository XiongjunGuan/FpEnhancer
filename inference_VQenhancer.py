'''
Description: 
Author: Xiongjun Guan
Date: 2023-12-04 15:28:01
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2024-12-09 19:57:52

Copyright (C) 2023 by Xiongjun Guan, Tsinghua University. All rights reserved.
'''
import argparse
import datetime
import logging
import os
import os.path as osp
import random
from glob import glob

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from easydict import EasyDict as edict
from torch.optim.lr_scheduler import (CosineAnnealingLR, ReduceLROnPlateau,
                                      StepLR)
from tqdm import tqdm

from loss.loss import FinalLoss
from models.Enhancer import VQEnhancer
from utils.data_loader import get_dataloader_test


def edict_to_dict(ed):
    if isinstance(ed, edict):
        return {k: edict_to_dict(v) for k, v in ed.items()}
    elif isinstance(ed, dict):
        return {k: edict_to_dict(v) for k, v in ed.items()}
    else:
        return ed


def save_model(model, save_path):
    if hasattr(model, "module"):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()

    torch.save(
        model_state,
        osp.join(save_path),
    )
    return


def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train(
    model,
    test_dataloader,
    device,
    save_dir=None,
):

    model.eval()
    with torch.no_grad():

        pbar = tqdm(test_dataloader, desc=f"")

        for imgs, ftitle_lst in pbar:
            imgs = imgs.float().to(device)

            decoded_images, _, _ = model(imgs)

            for i in range(imgs.shape[0]):
                ftitle_i = ftitle_lst[i]
                pred_i = decoded_images[
                    i, :, :, :].detach().squeeze().cpu().numpy()

                pred_i = np.clip(255 * (1 - pred_i), 0, 255)
                cv2.imwrite(osp.join(save_dir, f"{ftitle_i}.png"), pred_i)

    return


if __name__ == "__main__":
    set_seed(seed=7)

    # ------------ settings ------------ #
    model_dir = "/disk3/guanxiongjun/backup_clean/FpEnhancer/ckpts/VQEnhancer/"
    parser = argparse.ArgumentParser(description="settings for training")
    parser.add_argument(
        "--yaml",
        "-y",
        type=str,
        default="auto",
        help="yaml name",
    )
    parser.add_argument(
        "--cuda-ids",
        "-c",
        type=int,
        nargs='+',
        default=[0],
    )
    args = parser.parse_args()

    args.yaml = osp.basename(glob(osp.join(model_dir,
                                           "*.yaml"))[0]).replace(".yaml", "")

    # ------------ load yaml ------------ #
    yaml_path = osp.join(model_dir, args.yaml + ".yaml")
    with open(yaml_path, "r") as config:
        cfg = edict(yaml.safe_load(config))

    cfg.update(vars(args))

    # ------------ set save dir ------------ #
    save_dir = "./data/result/VQEnhancer/"

    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    # ------------ load database ------------ #
    img_dir = "./data/imgs/"
    ftitle_lst = glob(osp.join(img_dir, "*.png"))
    ftitle_lst = [osp.basename(x).replace(".png", "") for x in ftitle_lst]
    ftitle_lst = sorted(ftitle_lst)

    test_info = {"info_lst": ftitle_lst, "img_dir": img_dir}

    test_loader = get_dataloader_test(
        info_lst=test_info["info_lst"],
        img_dir=test_info["img_dir"],
        batch_size=16,
    )

    # ------------ load model ------------ #
    logging.info("load model: {}".format(cfg.MODEL.name))
    if cfg.MODEL.name == "VQEnhancer":
        model = VQEnhancer(
            img_channel=cfg.MODEL.img_channel,
            width=cfg.MODEL.width,
            mid_blk_num=cfg.MODEL.mid_blk_num,
            enc_blk_nums=cfg.MODEL.enc_blk_nums,
            dec_blk_nums=cfg.MODEL.dec_blk_nums,
            dw_expand=cfg.MODEL.dw_expand,
            ffn_expand=cfg.MODEL.ffn_expand,
            num_codebook_vectors=cfg.MODEL.num_codebook_vectors,
        )
    pth_path = osp.join(model_dir, "best.pth")
    model.load_state_dict(
        torch.load(pth_path, map_location=f'cuda:{cfg.cuda_ids[0]}'))
    device = torch.device("cuda:{}".format(str(cfg.cuda_ids[0])) if torch.cuda.
                          is_available() else "cpu")
    model = torch.nn.DataParallel(
        model,
        device_ids=cfg.cuda_ids,
        output_device=cfg.cuda_ids[0],
    ).to(device)

    # ------------ test ------------ #
    logging.info("******** begin testing ********")
    train(
        model=model,
        test_dataloader=test_loader,
        device=device,
        save_dir=save_dir,
    )
