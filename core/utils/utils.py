import logging
import os
import random
from collections import OrderedDict

import numpy as np
from PIL import Image

from skimage.measure import label, regionprops


def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val, num=1):
        if self.length > 0:
            # currently assert num==1 to avoid bad usage, refine when there are some explict requirements
            assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val * num
            self.count += num
            self.avg = self.sum / self.count


logs = set()


def init_log(name, level=logging.INFO):
    if (name, level) in logs:
        return
    logs.add((name, level))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        logger.addFilter(lambda record: rank == 0)
    else:
        rank = 0
    format_str = "[%(asctime)s][%(levelname)8s] %(message)s"
    formatter = logging.Formatter(format_str)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def convert_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict
    

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def colorize(mask, colormap):
    color_mask = np.zeros([mask.shape[0], mask.shape[1], 3])
    for i in np.unique(mask):
        color_mask[mask == i] = colormap[i]

    return Image.fromarray(np.uint8(color_mask))


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def check_makedirs(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def get_palette(num_cls):
    """Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= ((lab >> 0) & 1) << (7 - i)
            palette[j * 3 + 1] |= ((lab >> 1) & 1) << (7 - i)
            palette[j * 3 + 2] |= ((lab >> 2) & 1) << (7 - i)
            i += 1
            lab >>= 3
    return palette


def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.ndim in [1, 2, 3]
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K + 1))
    area_output, _ = np.histogram(output, bins=np.arange(K + 1))
    area_target, _ = np.histogram(target, bins=np.arange(K + 1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def load_state(path, model, optimizer=None, key="state_dict", distributed_flag=True, self_define_ignore_keys=[]):
    if distributed_flag:
        rank = dist.get_rank()
    else:
        rank = 0

    def map_func(storage, location):
        return storage.cuda()

    if os.path.isfile(path):
        if rank == 0:
            print("=> loading checkpoint '{}'".format(path))

        checkpoint = torch.load(path, map_location=map_func)

        # fix size mismatch error
        ignore_keys = []
        state_dict = checkpoint[key]

        if not distributed_flag:
            for key in list(state_dict.keys()):
                if 'module' in key:
                    state_dict[key.replace('module.', '')] = state_dict.pop(key)

        for k, v in state_dict.items():
            continue_flag = False
            for _key in self_define_ignore_keys:
                if _key in k:
                    ignore_keys.append(k)
                    continue_flag = True
            if continue_flag:
                continue
            if k in model.state_dict().keys():
                v_dst = model.state_dict()[k]
                if v.shape != v_dst.shape:
                    ignore_keys.append(k)
                    if rank == 0:
                        print(
                            "caution: size-mismatch key: {} size: {} -> {}".format(
                                k, v.shape, v_dst.shape
                            )
                        )
        # import pdb; pdb.set_trace()
        for k in ignore_keys:
            # checkpoint.pop(k)
            state_dict.pop(k)

        model.load_state_dict(state_dict, strict=False)

        if rank == 0:
            ckpt_keys = set(state_dict.keys())
            own_keys = set(model.state_dict().keys())
            missing_keys = own_keys - ckpt_keys
            for k in missing_keys:
                print("caution: missing keys from checkpoint {}: {}".format(path, k))

        if optimizer is not None:
            best_metric = checkpoint["best_miou"]
            last_iter = checkpoint["epoch"]
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            if rank == 0:
                print(
                    "=> also loaded optimizer from checkpoint '{}' (epoch {})".format(
                        path, last_iter
                    )
                )
            return best_metric, last_iter
    else:
        if rank == 0:
            print("=> no checkpoint found at '{}'".format(path))


def create_cityscapes_label_colormap():
    """Creates a label colormap used in CityScapes segmentation benchmark.
    Returns:
        A colormap for visualizing segmentation results.
    """
    colormap = np.zeros((256, 3), dtype=np.uint8)
    colormap[0] = [128, 64, 128]
    colormap[1] = [244, 35, 232]
    colormap[2] = [70, 70, 70]
    colormap[3] = [102, 102, 156]
    colormap[4] = [190, 153, 153]
    colormap[5] = [153, 153, 153]
    colormap[6] = [250, 170, 30]
    colormap[7] = [220, 220, 0]
    colormap[8] = [107, 142, 35]
    colormap[9] = [152, 251, 152]
    colormap[10] = [70, 130, 180]
    colormap[11] = [220, 20, 60]
    colormap[12] = [255, 0, 0]
    colormap[13] = [0, 0, 142]
    colormap[14] = [0, 0, 70]
    colormap[15] = [0, 60, 100]
    colormap[16] = [0, 80, 100]
    colormap[17] = [0, 0, 230]
    colormap[18] = [119, 11, 32]

    return colormap


def create_pascal_label_colormap():
    """Creates a label colormap used in Pascal segmentation benchmark.
    Returns:
        A colormap for visualizing segmentation results.
    """
    colormap = 255 * np.ones((256, 3), dtype=np.uint8)
    colormap[0] = [0, 0, 0]
    colormap[1] = [128, 0, 0]
    colormap[2] = [0, 128, 0]
    colormap[3] = [128, 128, 0]
    colormap[4] = [0, 0, 128]
    colormap[5] = [128, 0, 128]
    colormap[6] = [0, 128, 128]
    colormap[7] = [128, 128, 128]
    colormap[8] = [64, 0, 0]
    colormap[9] = [192, 0, 0]
    colormap[10] = [64, 128, 0]
    colormap[11] = [192, 128, 0]
    colormap[12] = [64, 0, 128]
    colormap[13] = [192, 0, 128]
    colormap[14] = [64, 128, 128]
    colormap[15] = [192, 128, 128]
    colormap[16] = [0, 64, 0]
    colormap[17] = [128, 64, 0]
    colormap[18] = [0, 192, 0]
    colormap[19] = [128, 192, 0]
    colormap[20] = [0, 64, 128]

    return colormap


def create_pascal_label_colormap_cluster():
    """Creates a label colormap used in Pascal segmentation benchmark.
    Returns:
        A colormap for visualizing segmentation results.
    """
    colormap = 255 * np.ones((256, 3), dtype=np.uint8)
    colormap[0] = [0, 0, 0]
    colormap[1] = [0, 0, 100]
    colormap[2] = [0, 100, 0]
    colormap[3] = [0, 100, 100]

    colormap[4] = [128, 0, 0]
    colormap[5] = [128, 0, 100]
    colormap[6] = [128, 100, 0]
    colormap[7] = [128, 100, 100]

    colormap[8] = [0, 128, 0]
    colormap[9] = [0, 128, 50]
    colormap[10] = [50, 128, 0]
    colormap[11] = [50, 128, 50]

    colormap[12] = [128, 128, 0]
    colormap[13] = [128, 128, 50]
    colormap[14] = [128, 128, 100]
    colormap[15] = [128, 128, 200]

    colormap[16] = [0, 0, 128]
    colormap[17] = [50, 0, 128]
    colormap[18] = [50, 50, 128]
    colormap[19] = [50, 100, 128]

    colormap[20] = [128, 0, 128]
    colormap[21] = [128, 50, 128]
    colormap[22] = [128, 100, 128]
    colormap[23] = [128, 150, 128]

    colormap[24] = [0, 128, 128]
    colormap[25] = [50, 128, 128]
    colormap[26] = [100, 128, 128]
    colormap[27] = [200, 128, 128]

    colormap[28] = [128, 128, 128]
    colormap[29] = [128, 0, 128]
    colormap[30] = [128, 200, 128]
    colormap[31] = [128, 150, 128]

    colormap[32] = [64, 0, 0]
    colormap[33] = [64, 50, 0]
    colormap[34] = [64, 0, 100]
    colormap[35] = [64, 100, 0]

    colormap[36] = [192, 0, 0]
    colormap[37] = [192, 0, 50]
    colormap[38] = [192, 100, 0]
    colormap[39] = [192, 0, 150]

    colormap[40] = [64, 128, 0]
    colormap[41] = [64, 128, 50]
    colormap[42] = [64, 128, 100]
    colormap[43] = [64, 128, 200]

    colormap[44] = [192, 128, 0]
    colormap[45] = [192, 128, 50]
    colormap[46] = [192, 128, 100]
    colormap[47] = [192, 128, 150]

    colormap[48] = [64, 0, 128]
    colormap[49] = [64, 50, 128]
    colormap[50] = [64, 100, 128]
    colormap[51] = [64, 150, 128]

    colormap[52] = [192, 0, 128]
    colormap[53] = [192, 50, 128]
    colormap[54] = [192, 100, 128]
    colormap[55] = [192, 150, 128]

    colormap[56] = [64, 128, 128]
    colormap[57] = [100, 128, 128]
    colormap[58] = [200, 128, 128]
    colormap[59] = [250, 128, 128]

    colormap[60] = [192, 128, 128]
    colormap[61] = [192, 128, 0]
    colormap[62] = [192, 128, 50]
    colormap[63] = [192, 128, 200]

    colormap[64] = [0, 64, 0]
    colormap[65] = [0, 64, 50]
    colormap[66] = [100, 64, 0]
    colormap[67] = [0, 64, 100]

    colormap[68] = [128, 64, 0]
    colormap[69] = [128, 64, 50]
    colormap[70] = [128, 64, 100]
    colormap[71] = [128, 64, 200]

    colormap[72] = [0, 192, 0]
    colormap[73] = [50, 192, 0]
    colormap[74] = [0, 192, 100]
    colormap[75] = [100, 192, 0]

    colormap[76] = [128, 192, 0]
    colormap[77] = [128, 192, 50]
    colormap[78] = [128, 192, 100]
    colormap[79] = [128, 192, 200]

    colormap[80] = [0, 64, 128]
    colormap[81] = [50, 64, 128]
    colormap[82] = [100, 64, 128]
    colormap[83] = [200, 64, 128]

    return colormap
