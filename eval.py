import logging
import os
import time
from argparse import ArgumentParser

import numpy as np

import mindspore
from mindspore import Tensor
from mindspore import ops

import yaml
from PIL import Image

from core.models.model_helper import ModelBuilder
from core.utils.utils import (
    AverageMeter, 
    check_makedirs, 
    colorize, 
    convert_state_dict, 
    create_cityscapes_label_colormap, 
    create_pascal_label_colormap, 
    intersectionAndUnion
)


# Setup Parser
def get_parser():
    parser = ArgumentParser(description="Mindspore Evaluation")
    parser.add_argument(
        "--base_size", type=int, default=2048, help="based size for scaling"
    )
    parser.add_argument(
        "--scales", type=float, default=[1.0], nargs="+", help="evaluation scales"
    )
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument(
        "--model_path",
        type=str,
        default="checkpoints/psp_best.pth",
        help="evaluation model path",
    )
    parser.add_argument(
        "--save_folder",
        type=str,
        default="checkpoints/results/",
        help="results save folder",
    )
    parser.add_argument(
        "--names_path",
        type=str,
        default="../../vis_meta/cityscapes/cityscapesnames.mat",
        help="path of dataset category names",
    )
    parser.add_argument(
        "--crop", action="store_true", default=False, help="whether use crop evaluation"
    )
    parser.add_argument("--machine_name", default='cloud', type=str)
    return parser


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def main():
    global args, logger, cfg, colormap
    args = get_parser().parse_args()
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    logger = get_logger()
    logger.info(args)

    if args.machine_name == 'YOUR_MACHINE_NAME':
        output_basic_path = 'YOUR_OUTPUT_BASIC_PATH'
        dataset_basic_path = 'YOUR_DATASET_BASIC_PATH'
    else:
        output_basic_path = '../../../../'
        dataset_basic_path = '../../../../'
    cfg["exp_path"] = os.path.join(output_basic_path, cfg["saver"]["main_dir"])
    cfg["dataset"]["train"]["data_root"] = os.path.join(dataset_basic_path, cfg["dataset"]["train"]["data_root"])
    cfg["dataset"]["val"]["data_root"] = os.path.join(dataset_basic_path, cfg["dataset"]["val"]["data_root"])
    if cfg["net"].get('relation_net_head', False):
        cfg["net"]["relation_net_head"]["init_proto_path"] = os.path.join(output_basic_path, cfg["net"]["relation_net_head"]["init_proto_path"])

    cfg["save_path"] = os.path.join(cfg["exp_path"], cfg["saver"]["snapshot_dir"])
    cfg["saver"]["snapshot_dir"] = os.path.join(cfg["exp_path"], cfg["saver"]["snapshot_dir"])
    args.model_path = os.path.join(cfg["saver"]["snapshot_dir"], args.model_path)

    cfg_dset = cfg["dataset"]
    mean, std = cfg_dset["mean"], cfg_dset["std"]
    num_classes = cfg["net"]["num_classes"]
    crop_size = cfg_dset["val"]["crop"]["size"]
    crop_h, crop_w = crop_size

    assert num_classes > 1
    
    gray_folder = os.path.join(cfg["exp_path"], "gray")
    color_folder = os.path.join(cfg["exp_path"], "color")
    os.makedirs(gray_folder, exist_ok=True)
    os.makedirs(color_folder, exist_ok=True)

    # mindspore.context.set_context(mode=mindspore.context.GRAPH_MODE, device_target="CPU")
    mindspore.context.set_context(mode=mindspore.context.PYNATIVE_MODE, device_target="GPU")

    cfg_dset = cfg["dataset"]
    data_root, f_data_list = cfg_dset["val"]["data_root"], cfg_dset["val"]["data_list"]
    data_list = []

    if "cityscapes" in data_root:
        colormap = create_cityscapes_label_colormap()
        for line in open(f_data_list, "r"):
            arr = [
                line.strip(),
                "gtFine/" + line.strip()[12:-15] + "gtFine_labelTrainIds.png",
            ]
            arr = [os.path.join(data_root, item) for item in arr]
            data_list.append(arr)

    else:
        colormap = create_pascal_label_colormap()
        for line in open(f_data_list, "r"):
            arr = [
                "JPEGImages/{}.jpg".format(line.strip()),
                "SegmentationClassAug/{}.png".format(line.strip()),
            ]
            arr = [os.path.join(data_root, item) for item in arr]
            data_list.append(arr)

    # Create network.
    args.use_auxloss = True if cfg["net"].get("aux_loss", False) else False
    logger.info("=> creating model from '{}' ...".format(args.model_path))

    cfg["net"]["sync_bn"] = False
    model = ModelBuilder(cfg["net"], args.machine_name)

    checkpoint = mindspore.load_checkpoint(args.model_path)
    logger.info("=> load teacher checkpoint")
    saved_state_dict = convert_state_dict(checkpoint)
    param_not_load = mindspore.load_param_into_net(model, saved_state_dict)
    # logger.info(f"param not load: {param_not_load}")
    logger.info("Load Model Done!")

    if "cityscapes" in cfg["dataset"]["type"]:
        validate_city(
            model,
            num_classes,
            data_list,
            mean,
            std,
            args.base_size,
            crop_h,
            crop_w,
            args.scales,
            gray_folder,
            color_folder,
        )
    else:
        valiadte_whole(
            model,
            num_classes,
            data_list,
            mean,
            std,
            args.scales,
            gray_folder,
            color_folder,
        )


def net_process(model, image, output_proto=False):
    b, c, h, w = image.shape
    output = model(image, eval=True, output_proto=output_proto)
    linear_clf_output = output["pred"]
    linear_clf_output = ops.interpolate(linear_clf_output, roi=None, scales=None, sizes=(h, w), coordinate_transformation_mode="align_corners", mode="bilinear")
    if output_proto:
        if "proto" in output:
            proto_clf_output = output["proto"]
        else:
            proto_clf_output = None
        return ops.Squeeze(0)(linear_clf_output), ops.Squeeze(0)(proto_clf_output)
    else:
        return ops.Squeeze(0)(linear_clf_output)


def scale_crop_process(model, image, classes, crop_h, crop_w, h, w, stride_rate=2 / 3):
    ori_h, ori_w = image.shape[-2:]
    pad_h = max(crop_h - ori_h, 0)
    pad_w = max(crop_w - ori_w, 0)
    pad_h_half = int(pad_h / 2)
    pad_w_half = int(pad_w / 2)
    if pad_h > 0 or pad_w > 0:
        border = (pad_w_half, pad_w - pad_w_half, pad_h_half, pad_h - pad_h_half)
        image = ops.pad(image, border, mode="constant", value=0.0)
    new_h, new_w = image.shape[-2:]
    stride_h = int(np.ceil(crop_h * stride_rate))
    stride_w = int(np.ceil(crop_w * stride_rate))
    grid_h = int(np.ceil(float(new_h - crop_h) / stride_h) + 1)
    grid_w = int(np.ceil(float(new_w - crop_w) / stride_w) + 1)
    prediction_crop =  ops.zeros((1, classes, new_h, new_w), mindspore.float32)
    count_crop =  ops.zeros((new_h, new_w), mindspore.float32)
    for index_h in range(0, grid_h):
        for index_w in range(0, grid_w):
            s_h = index_h * stride_h
            e_h = min(s_h + crop_h, new_h)
            s_h = e_h - crop_h
            s_w = index_w * stride_w
            e_w = min(s_w + crop_w, new_w)
            s_w = e_w - crop_w
            image_crop = image[:, :, s_h:e_h, s_w:e_w]
            count_crop[s_h:e_h, s_w:e_w] += 1
            prediction_crop[:, :, s_h:e_h, s_w:e_w] += net_process(model, image_crop)

    prediction_crop /= count_crop
    prediction_crop = prediction_crop[
        :, :, pad_h_half : pad_h_half + ori_h, pad_w_half : pad_w_half + ori_w
    ]
    prediction = ops.interpolate(prediction_crop, roi=None, scales=None, sizes=(h, w), coordinate_transformation_mode="align_corners", mode="bilinear")
    return ops.Squeeze(0)(prediction)


def scale_whole_process(model, image, h, w):
    linear_clf_output, proto_clf_output = net_process(model, image, output_proto=True)
    return linear_clf_output, proto_clf_output


def validate_city(
    model,
    classes,
    data_list,
    mean,
    std,
    base_size,
    crop_h,
    crop_w,
    scales,
    gray_folder,
    color_folder,
):
    global colormap
    logger.info(">>>>>>>>>>>>>>>> Start Crop Evaluation >>>>>>>>>>>>>>>>")
    data_time = AverageMeter()
    batch_time = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()

    model.set_train(False)
    end = time.time()
    for i, (input_pth, label_path) in enumerate(data_list):
        data_time.update(time.time() - end)
        image = Image.open(input_pth).convert("RGB")
        image = np.asarray(image).astype(np.float32)
        label = Image.open(label_path).convert("L")
        label = np.asarray(label).astype(np.uint8)

        image = (image - mean) / std
        image = ops.Transpose()(Tensor(image, mindspore.float32), (2, 0, 1))
        image = ops.expand_dims(image, 0)
        
        h, w = image.shape[-2:]
        prediction = ops.zeros((classes, h, w), mindspore.float32)
        for scale in scales:
            long_size = round(scale * base_size)
            new_h = long_size
            new_w = long_size
            if h > w:
                new_w = round(long_size / float(h) * w)
            else:
                new_h = round(long_size / float(w) * h)
            # image_scale = ops.interpolate(image, roi=None, scales=None, sizes=(new_h, new_w), coordinate_transformation_mode="align_corners", mode="bilinear")
            image_scale = ops.ResizeBilinear((new_h, new_w), align_corners=True)(image)
            prediction += scale_crop_process(
                model, image_scale, classes, crop_h, crop_w, h, w
            )
        prediction = ops.ArgMaxWithValue(axis=0)(prediction)[0].asnumpy()
        batch_time.update(time.time() - end)
        end = time.time()
        if (i + 1) % 10 == 0:
            logger.info(
                "Test: [{}/{}] "
                "Data {data_time.val:.3f} ({data_time.avg:.3f}) "
                "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}).".format(
                    i + 1, len(data_list), data_time=data_time, batch_time=batch_time
                )
            )
        gray = np.uint8(prediction)
        color = colorize(gray, colormap)

        image_path, _ = data_list[i]
        image_name = image_path.split("/")[-1].split(".")[0]
        color_path = os.path.join(color_folder, image_name + ".png")
        # color.save(color_path)

        intersection, union, target = intersectionAndUnion(gray, label, classes)
        intersection_meter.update(intersection)
        union_meter.update(union)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    for i, iou in enumerate(iou_class):
        logger.info(" * class [{}] IoU {:.2f}".format(i, iou * 100))
    logger.info(" * mIoU {:.2f}".format(np.mean(iou_class) * 100))
    logger.info("<<<<<<<<<<<<<<<<< End Crop Evaluation <<<<<<<<<<<<<<<<<")


def valiadte_whole(
    model, classes, data_list, mean, std, scales, gray_folder, color_folder
):
    logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
    data_time = AverageMeter()
    batch_time = AverageMeter()
    intersection_meter_linear = AverageMeter()
    intersection_meter_proto = AverageMeter()
    union_meter_linear = AverageMeter()
    union_meter_proto = AverageMeter()

    model.set_train(False)
    end = time.time()
    for i, (input_pth, label_path) in enumerate(data_list):
        data_time.update(time.time() - end)
        image = Image.open(input_pth).convert("RGB")
        image = np.asarray(image).astype(np.float32)
        label = Image.open(label_path).convert("L")
        label = np.asarray(label).astype(np.uint8)

        image = (image - mean) / std
        image = ops.Transpose()(Tensor(image, mindspore.float32), (2, 0, 1))
        image = ops.expand_dims(image, 0)

        h, w = image.shape[-2:]
        prediction_linear = ops.zeros((classes, h, w), mindspore.float32)
        prediction_proto = ops.zeros((classes, h, w), mindspore.float32)
        for scale in scales:
            new_h = round(h * scale)
            new_w = round(w * scale)
            # image_scale = ops.interpolate(image, roi=None, scales=None, sizes=(new_h, new_w), coordinate_transformation_mode="align_corners", mode="bilinear")
            image_scale = ops.ResizeBilinear((new_h, new_w), align_corners=True)(image)
            linear_clf_output, proto_clf_output = scale_whole_process(model, image_scale, h, w)
            prediction_linear += linear_clf_output
            prediction_proto += proto_clf_output
        prediction_linear = ops.ArgMaxWithValue(axis=0)(prediction_linear)[0].asnumpy()
        prediction_proto = ops.ArgMaxWithValue(axis=0)(prediction_proto)[0].asnumpy()
        ##############attention###############
        batch_time.update(time.time() - end)
        end = time.time()
        if (i + 1) % 10 == 0:
            logger.info(
                "Test: [{}/{}] "
                "Data {data_time.val:.3f} ({data_time.avg:.3f}) "
                "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}).".format(
                    i + 1, len(data_list), data_time=data_time, batch_time=batch_time
                )
            )
        check_makedirs(gray_folder)
        check_makedirs(color_folder)

        gray_linear = np.uint8(prediction_linear)
        intersection_linear, union_linear, target = intersectionAndUnion(gray_linear, label, classes)
        intersection_meter_linear.update(intersection_linear)
        union_meter_linear.update(union_linear)

        gray_proto = np.uint8(prediction_proto)
        intersection_proto, union_proto, target = intersectionAndUnion(gray_proto, label, classes)
        intersection_meter_proto.update(intersection_proto)
        union_meter_proto.update(union_proto)

        color = colorize(gray_linear, colormap)
        image_path, _ = data_list[i]
        image_name = image_path.split("/")[-1].split(".")[0]
        gray_path = os.path.join(gray_folder, image_name + ".png")
        color_path = os.path.join(color_folder, image_name + ".png")
        gray = Image.fromarray(gray_linear)
        # gray.save(gray_path)
        # color.save(color_path)
    
    iou_class_linear = intersection_meter_linear.sum / (union_meter_linear.sum + 1e-10)
    iou_class_proto = intersection_meter_proto.sum / (union_meter_proto.sum + 1e-10)
    for i, iou in enumerate(iou_class_linear):
        logger.info(" * class [{}] IoU {:.2f}".format(i, iou * 100))
    logger.info(" * linear head mIoU {:.2f}".format(np.mean(iou_class_linear) * 100))
    logger.info(" * proto head mIoU {:.2f}".format(np.mean(iou_class_proto) * 100))
    logger.info("<<<<<<<<<<<<<<<<< End  Evaluation <<<<<<<<<<<<<<<<<")


if __name__ == "__main__":
    main()
