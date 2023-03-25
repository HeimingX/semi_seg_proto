import copy
import math
import os
import os.path
import random

import numpy as np

import mindspore
from mindspore import ops
import mindspore.dataset as ds

from . import augmentation as psp_trsform
from .base import BaseDataset


class city_dset(BaseDataset):
    def __init__(self, data_root, data_list, trs_form, seed, n_sup, split="val", cp=False, paste_trs=None):
        # cp: copy paste
        super(city_dset, self).__init__(data_list)
        self.data_root = data_root
        self.transform = trs_form
        random.seed(seed)
        if len(self.list_sample) >= n_sup and split == "train":
            self.list_sample_new = random.sample(self.list_sample, n_sup)
        elif len(self.list_sample) < n_sup and split == "train":
            num_repeat = math.ceil(n_sup / len(self.list_sample))
            self.list_sample = self.list_sample * num_repeat

            self.list_sample_new = random.sample(self.list_sample, n_sup)
        else:
            self.list_sample_new = self.list_sample

        self.cp = cp
        self.prob = 0.5
        self.paste_trs = paste_trs

    def __getitem__(self, index):
        # load image and its label
        image_path = os.path.join(self.data_root, self.list_sample_new[index][0])
        label_path = os.path.join(self.data_root, self.list_sample_new[index][1])
        image = self.img_loader(image_path, "RGB")
        label = self.img_loader(label_path, "L")
        image, label = self.transform(image, label)
        if self.cp:
            # print("enter cp")
            # import pdb; pdb.set_trace()
            if random.random() > self.prob:
                paste_idx = random.randint(0, self.__len__() - 1)
                paste_img_path = os.path.join(
                    self.data_root, self.list_sample_new[paste_idx][0]
                )
                paste_img = self.img_loader(paste_img_path, "RGB")
                paste_label_path = os.path.join(
                    self.data_root, self.list_sample_new[paste_idx][1]
                )
                paste_label = self.img_loader(paste_label_path, "L")
                paste_img, paste_label = self.paste_trs(paste_img, paste_label)
                return ops.Concat(0)((image[0], paste_img[0])), ops.Concat(0)(
                    [label[0, 0].long(), paste_label[0, 0].long()]
                )
            else:
                # paste_img, paste_label = None, None
                h, w = image[0].shape[1], image[0].shape[2]
                paste_img = ops.zeros((3, h, w), mindspore.float32)
                paste_label = ops.zeros((h, w), mindspore.float32)
                return ops.Concat(0)((image[0], paste_img)), ops.Concat(0)(
                    [label[0, 0].long(), paste_label.long()], dim=0
                )
            
        return image[0], label[0, 0].long()

    def __len__(self):
        return len(self.list_sample_new)


def build_transfrom(cfg, acp=False):
    trs_form = []
    mean, std, ignore_label = cfg["mean"], cfg["std"], cfg["ignore_label"]
    trs_form.append(psp_trsform.ToTensor())
    trs_form.append(psp_trsform.Normalize(mean=mean, std=std))
    if cfg.get("resize", False):
        trs_form.append(psp_trsform.Resize(cfg["resize"]))
    if cfg.get("rand_resize", False):
        if not acp:
            trs_form.append(psp_trsform.RandResize(cfg["rand_resize"]))
        else:
            trs_form.append(psp_trsform.RandResize(cfg["acp"]["rand_resize"]))
    if cfg.get("rand_rotation", False):
        rand_rotation = cfg["rand_rotation"]
        trs_form.append(
            psp_trsform.RandRotate(rand_rotation, ignore_label=ignore_label)
        )
    if cfg.get("GaussianBlur", False) and cfg["GaussianBlur"]:
        trs_form.append(psp_trsform.RandomGaussianBlur())
    if cfg.get("flip", False) and cfg.get("flip"):
        trs_form.append(psp_trsform.RandomHorizontalFlip())
    if cfg.get("crop", False):
        crop_size, crop_type = cfg["crop"]["size"], cfg["crop"]["type"]
        trs_form.append(
            psp_trsform.Crop(crop_size, crop_type=crop_type, ignore_label=ignore_label)
        )
    if cfg.get("cutout", False):
        n_holes, length = cfg["cutout"]["n_holes"], cfg["cutout"]["length"]
        trs_form.append(psp_trsform.Cutout(n_holes=n_holes, length=length))
    if cfg.get("cutmix", False):
        n_holes, prop_range = cfg["cutmix"]["n_holes"], cfg["cutmix"]["prop_range"]
        trs_form.append(psp_trsform.Cutmix(prop_range=prop_range, n_holes=n_holes))

    return psp_trsform.Compose(trs_form)


def build_cityloader(split, all_cfg, seed=0, distributed_flag=True):
    cfg_dset = all_cfg["dataset"]
    cfg_trainer = all_cfg["trainer"]

    cfg = copy.deepcopy(cfg_dset)
    cfg.update(cfg.get(split, {}))

    workers = cfg.get("workers", 2)
    batch_size = cfg.get("batch_size", 1)
    n_sup = cfg.get("n_sup", 2975)

    # build transform
    trs_form = build_transfrom(cfg)
    dset = city_dset(cfg["data_root"], cfg["data_list"], trs_form, seed, n_sup, split)

    # build sampler
    if distributed_flag:
        sample = ds.DistributedSampler(dset)
    else:
        sample = ds.RandomSampler(dset)
    
    loader = ds.GeneratorDataset(
        dset,
        ["data", "label"],
        num_parallel_workers=workers,
        sampler=sample,
        shuffle=False,
    )
    loader = loader.batch(batch_size=batch_size, drop_remainder=False)
    return loader


def build_city_semi_loader(split, all_cfg, seed=0, distributed_flag=True):
    cfg_dset = all_cfg["dataset"]

    cfg = copy.deepcopy(cfg_dset)
    cfg.update(cfg.get(split, {}))

    workers = cfg.get("workers", 2)
    batch_size = cfg.get("batch_size", 1)
    n_sup = 2975 - cfg.get("n_sup", 2975)

    # build transform
    trs_form = build_transfrom(cfg)
    trs_form_unsup = build_transfrom(cfg)
    # import pdb; pdb.set_trace()
    if cfg.get("acp", False):
        paste_trs = build_transfrom(cfg, acp=True)
        dset = city_dset(cfg["data_root"], cfg["data_list"], trs_form, seed, n_sup, split, cp=True, paste_trs=paste_trs)
    else:
        dset = city_dset(cfg["data_root"], cfg["data_list"], trs_form, seed, n_sup, split)
    # dset = city_dset(cfg["data_root"], cfg["data_list"], trs_form, seed, n_sup, split)

    if split == "val":
        # build sampler
        if distributed_flag:
            sample = ds.DistributedSampler(dset)
        else:
            sample = ds.RandomSampler(dset)
        
        loader = ds.GeneratorDataset(
            dset,
            ["data", "label"],
            num_parallel_workers=workers,
            sampler=sample,
            shuffle=False,
        )
        loader = loader.batch(batch_size=batch_size, drop_remainder=False)
        return loader

    else:
        # build sampler for unlabeled set
        data_list_unsup = cfg["data_list"].replace("labeled.txt", "unlabeled.txt")
        dset_unsup = city_dset(
            cfg["data_root"], data_list_unsup, trs_form_unsup, seed, n_sup, split
        )
        if distributed_flag:
            sample_sup = ds.DistributedSampler(dset)
        else:
            sample_sup = ds.RandomSampler(dset)
        
        loader_sup = ds.GeneratorDataset(
            dset,
            ["data", "label"],
            num_parallel_workers=workers,
            sampler=sample,
            shuffle=False,
        )
        loader_sup = loader_sup.batch(batch_size=batch_size, drop_remainder=True)

        
        if distributed_flag:
            sample_unsup = ds.DistributedSampler(dset_unsup)
        else:
            sample_unsup = ds.RandomSampler(dset_unsup)
        
        loader_unsup = ds.GeneratorDataset(
            dset_unsup,
            ["data", "label"],
            num_parallel_workers=workers,
            sampler=sample,
            shuffle=False,
        )
        loader_unsup = loader_unsup.batch(batch_size=batch_size, drop_remainder=True)
        return loader_sup, loader_unsup
