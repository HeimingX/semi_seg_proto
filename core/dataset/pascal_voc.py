import copy
import math
import os
import os.path
import random

import numpy as np

import mindspore.dataset as ds

from . import augmentation as psp_trsform
from .base import BaseDataset


class voc_dset(BaseDataset):
    def __init__(
        self, data_root, data_list, trs_form, seed=0, n_sup=10582, split="val"
    ):
        super(voc_dset, self).__init__(data_list)
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

    def __getitem__(self, index):
        # load image and its label
        image_path = os.path.join(self.data_root, self.list_sample_new[index][0])
        label_path = os.path.join(self.data_root, self.list_sample_new[index][1])
        image = self.img_loader(image_path, "RGB")
        label = self.img_loader(label_path, "L")
        image, label = self.transform(image, label)
        return image[0], label[0, 0].long()

    def __len__(self):
        return len(self.list_sample_new)


def build_transfrom(cfg):
    trs_form = []
    mean, std, ignore_label = cfg["mean"], cfg["std"], cfg["ignore_label"]
    trs_form.append(psp_trsform.ToTensor())
    trs_form.append(psp_trsform.Normalize(mean=mean, std=std))
    if cfg.get("resize", False):
        trs_form.append(psp_trsform.Resize(cfg["resize"]))
    if cfg.get("rand_resize", False):
        trs_form.append(psp_trsform.RandResize(cfg["rand_resize"]))
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
    return psp_trsform.Compose(trs_form)


def build_vocloader(split, all_cfg, seed=0, distributed_flag=True):
    cfg_dset = all_cfg["dataset"]

    cfg = copy.deepcopy(cfg_dset)
    cfg.update(cfg.get(split, {}))

    workers = cfg.get("workers", 2)
    batch_size = cfg.get("batch_size", 1)
    n_sup = cfg.get("n_sup", 10582)
    # build transform
    trs_form = build_transfrom(cfg)
    dset = voc_dset(cfg["data_root"], cfg["data_list"], trs_form, seed, n_sup)

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


def build_voc_semi_loader(split, all_cfg, seed=0, distributed_flag=True):
    cfg_dset = all_cfg["dataset"]

    cfg = copy.deepcopy(cfg_dset)
    cfg.update(cfg.get(split, {}))

    workers = cfg.get("workers", 2)
    batch_size = cfg.get("batch_size", 1)
    n_sup = 10582 - cfg.get("n_sup", 10582)

    # build transform
    trs_form = build_transfrom(cfg)
    trs_form_unsup = build_transfrom(cfg)
    dset = voc_dset(cfg["data_root"], cfg["data_list"], trs_form, seed, n_sup, split)

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
        dset_unsup = voc_dset(
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
