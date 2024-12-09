'''
Description: 
Author: Xiongjun Guan
Date: 2023-12-04 15:28:01
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2024-12-09 11:52:40

Copyright (C) 2023 by Xiongjun Guan, Tsinghua University. All rights reserved.
'''
import argparse
import datetime
import logging
import os
import os.path as osp
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from easydict import EasyDict as edict
from loss.loss import FinalLoss
from models.Enhancer import Enhancer
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import (CosineAnnealingLR, ReduceLROnPlateau,
                                      StepLR)
from tqdm import tqdm
from utils.data_loader import get_dataloader_train, get_dataloader_valid


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
    train_dataloader,
    valid_dataloader,
    device,
    cfg,
    save_dir=None,
    save_checkpoint=4,
    writer=None,
):
    # ------------ init settings ------------ #
    lr = cfg.OPT.lr
    end_lr = cfg.OPT.end_lr
    optim = cfg.OPT.optimizer
    scheduler_type = cfg.OPT.scheduler_type
    num_epoch = cfg.epochs
    save_epoch = cfg.epochs - save_checkpoint
    min_loss = None

    # ------------ select optimizer ------------ #
    if optim == "adamW":
        opt = torch.optim.AdamW(
            (param for param in model.parameters() if param.requires_grad),
            lr=lr,
            weight_decay=1e-2,
        )

    # ------------ select scheduler ------------ #
    if scheduler_type == "CosineAnnealingLR":
        scheduler = CosineAnnealingLR(opt,
                                      T_max=np.round(num_epoch),
                                      eta_min=end_lr)

    # ------------ select loss ------------ #
    criterion = FinalLoss()

    # ------------ run epoch ------------ #
    for epoch in range(num_epoch):
        # -------------- train phase ------------ #
        model.train()

        train_losses = {}

        logging.info("epoch: {}, lr:{:.8f}".format(
            epoch,
            opt.state_dict()["param_groups"][0]["lr"]))

        pbar = tqdm(train_dataloader, desc=f"epoch:{epoch}, train")

        for i, (imgs, target) in enumerate(pbar):
            imgs = imgs.float().to(device)
            target = target.float().to(device)

            decoded_images = model(imgs)

            # --- loss
            total_loss = criterion(target, decoded_images)

            items = {
                "total_loss": total_loss.item(),
            }

            klist = items.keys()
            for k in klist:
                if k not in train_losses:
                    train_losses[k] = items[k] / len(train_dataloader)
                else:
                    train_losses[k] += items[k] / len(train_dataloader)

            pbar.set_postfix(**{"loss": total_loss.item()})

            opt.zero_grad()
            total_loss.backward(retain_graph=True)
            opt.step()

        pbar.close()

        if writer is not None:
            for k in train_losses:
                writer.add_scalar("train-{}".format(k), train_losses[k], epoch)
            writer.add_scalar("lr".format(k),
                              opt.state_dict()["param_groups"][0]["lr"], epoch)

        klist = train_losses.keys()
        logging_info = "\tTRAIN: ".format(epoch)
        k = "total_loss"
        logging_info += "{}:{:.4f}, ".format(k, train_losses[k])
        for k in klist:
            if k == "total_loss":
                continue
            logging_info = logging_info + "{}:{:.4f}, ".format(
                k, train_losses[k])
        logging.info(logging_info)

        scheduler.step()

        if save_dir is not None and epoch >= save_epoch:
            save_model(model, osp.join(save_dir, f"epoch_{epoch}.pth"))

        # -------------- valid phase ------------ #
        if valid_dataloader is None:
            continue

        model.eval()
        with torch.no_grad():
            valid_losses = {}
            pbar = tqdm(valid_dataloader, desc=f"epoch:{epoch}, val")

            for i, (imgs, target) in enumerate(pbar):
                imgs = imgs.float().to(device)
                target = target.float().to(device)

                decoded_images = model(imgs)

                # --- loss
                total_loss = criterion(target, decoded_images)

                items = {
                    "total_loss": total_loss.item(),
                }

                klist = items.keys()
                for k in klist:
                    if k not in valid_losses:
                        valid_losses[k] = items[k] / len(valid_dataloader)
                    else:
                        valid_losses[k] += items[k] / len(valid_dataloader)

                pbar.set_postfix(**{"loss": total_loss.item()})

            ####### show
            for i in range(min(20, imgs.shape[0])):
                input_i = imgs[i, :, :, :].detach().squeeze().cpu().numpy()
                target_i = target[i, :, :, :].detach().squeeze().cpu().numpy()
                pred_i = decoded_images[
                    i, :, :, :].detach().squeeze().cpu().numpy()
                show_img_i = np.concatenate([target_i, input_i, pred_i],
                                            axis=1)
                show_img_i = np.clip(255 * (1 - show_img_i), 0, 255)
                cv2.imwrite(osp.join(save_dir, f"show-{i}.png"), show_img_i)
            #######

            pbar.close()

            klist = valid_losses.keys()
            logging_info = "\tVALID: ".format(epoch)
            k = "total_loss"
            logging_info += "{}:{:.4f}, ".format(k, valid_losses[k])
            for k in klist:
                if k == "total_loss":
                    continue
                logging_info = logging_info + "{}:{:.4f}, ".format(
                    k, valid_losses[k])
            logging.info(logging_info)

        if min_loss is None or min_loss > valid_losses["total_loss"]:
            min_loss = valid_losses["total_loss"]
            save_model(model, osp.join(save_dir, f"best.pth"))

        if writer is not None:
            for k in valid_losses:
                writer.add_scalar("valid-{}".format(k), valid_losses[k], epoch)

    writer.close()
    return


if __name__ == "__main__":
    set_seed(seed=7)

    # ------------ settings ------------ #
    parser = argparse.ArgumentParser(description="settings for training")
    parser.add_argument(
        "--yaml",
        "-y",
        type=str,
        default="config_enhancer",
        help="yaml name",
    )
    parser.add_argument(
        "--cuda-ids",
        "-c",
        type=int,
        nargs='+',
        default=[0, 1],
    )
    args = parser.parse_args()

    # ------------ load yaml ------------ #
    current_path = os.path.abspath(__file__)
    yaml_path = osp.join(osp.dirname(current_path), "configs",
                         args.yaml + ".yaml")
    with open(yaml_path, "r") as config:
        cfg = edict(yaml.safe_load(config))

    cfg.update(vars(args))

    # ------------ set save dir ------------ #
    save_path = osp.join(cfg.SAVE.save_basedir,
                         "{}_p{}".format(cfg.MODEL.name, cfg.AUG.patch_size))
    if cfg.SAVE.save_title == "time":
        now = datetime.datetime.now()
        save_dir = osp.join(save_path, now.strftime("%Y-%m-%d-%H-%M-%S"))
    else:
        save_dir = osp.join(save_path, cfg.SAVE.save_title)

    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    # ------------ save config as yaml ------------ #
    with open(osp.join(save_dir, cfg.yaml + ".yaml"), 'w') as file:
        yaml.safe_dump(edict_to_dict(cfg), file, default_flow_style=False)

    # ------------ save tensorboard visualization ------------ #
    writer = SummaryWriter(osp.join(save_dir, "runs"))

    # ------------ save logging information as log ------------ #
    logging_path = osp.join(save_dir, "info.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
        filename=logging_path,
        filemode="w",
    )
    logging.info(f"loading training profile from {yaml_path}")

    # ------------ load database ------------ #
    train_info_path = cfg.DATASET.train_info_path
    valid_info_path = cfg.DATASET.valid_info_path

    train_info = np.load(train_info_path, allow_pickle=True).item()
    valid_info = np.load(valid_info_path, allow_pickle=True).item()

    train_loader = get_dataloader_train(
        info_lst=train_info["info_lst"],
        img_dir=train_info["img_dir"],
        bimg_dir=train_info["bimg_dir"],
        patch_size=cfg.AUG.patch_size,
        batch_size=cfg.batch_size,
        shuffle=True,
    )

    valid_loader = get_dataloader_valid(
        info_lst=valid_info["info_lst"],
        img_dir=train_info["img_dir"],
        bimg_dir=train_info["bimg_dir"],
        patch_size=cfg.AUG.patch_size,
        batch_size=cfg.batch_size,
    )

    # ------------ load model ------------ #
    logging.info("load model: {}".format(cfg.MODEL.name))
    if cfg.MODEL.name == "Enhancer":
        model = Enhancer(
            img_channel=cfg.MODEL.img_channel,
            width=cfg.MODEL.width,
            mid_blk_num=cfg.MODEL.mid_blk_num,
            enc_blk_nums=cfg.MODEL.enc_blk_nums,
            dec_blk_nums=cfg.MODEL.dec_blk_nums,
            dw_expand=cfg.MODEL.dw_expand,
            ffn_expand=cfg.MODEL.ffn_expand,
        )
    device = torch.device("cuda:{}".format(str(cfg.cuda_ids[0])) if torch.cuda.
                          is_available() else "cpu")
    model = torch.nn.DataParallel(
        model,
        device_ids=cfg.cuda_ids,
        output_device=cfg.cuda_ids[0],
    ).to(device)
    # ------------ train ------------ #
    logging.info("******** begin training ********")
    train(
        model=model,
        train_dataloader=train_loader,
        valid_dataloader=valid_loader,
        device=device,
        cfg=cfg,
        save_dir=save_dir,
        save_checkpoint=4,
        writer=writer,
    )
