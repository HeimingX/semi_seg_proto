from os import stat
from random import sample
from turtle import forward
import numpy as np
import pickle

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.ops import operations as P
from mindspore import Tensor, Parameter
import mindspore.numpy as ms_np

from .base import ASPP, get_syncbn


class dec_deeplabv3_plus(nn.Cell):
    def __init__(
        self,
        in_planes,
        num_classes=19,
        inner_planes=256,
        sync_bn=False,
        dilations=(12, 24, 36),
        rep_head=True,
        rep_clf=False,
    ):
        super(dec_deeplabv3_plus, self).__init__()

        norm_layer = get_syncbn() if sync_bn else nn.BatchNorm2d
        self.rep_head = rep_head
        self.rep_clf = rep_clf

        self.low_conv = nn.SequentialCell(
            nn.Conv2d(256, 256, kernel_size=1, has_bias=True, weight_init='xavier_uniform',), 
            norm_layer(256), nn.ReLU()
        )

        self.aspp = ASPP(
            in_planes, inner_planes=inner_planes, sync_bn=sync_bn, dilations=dilations
        )

        self.head = nn.SequentialCell(
            nn.Conv2d(
                self.aspp.get_outplanes(),
                256,
                kernel_size=3,
                padding=1,
                pad_mode="pad",
                dilation=1,
                has_bias=False,
                weight_init='xavier_uniform',
            ),
            norm_layer(256),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.num_classes = num_classes
        self.classifier = nn.SequentialCell(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, pad_mode="pad", has_bias=True, weight_init='xavier_uniform'),
            norm_layer(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, pad_mode="pad", has_bias=True, weight_init='xavier_uniform'),
            norm_layer(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, has_bias=True, weight_init='xavier_uniform'),
        )

        if self.rep_head:
            self.representation = nn.SequentialCell(
                nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, pad_mode="pad", has_bias=True, weight_init='xavier_uniform'),
                norm_layer(256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, pad_mode="pad", has_bias=True, weight_init='xavier_uniform'),
                norm_layer(256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, has_bias=True, weight_init='xavier_uniform'),
            )
        
        self.concat = P.Concat(axis=1)
        self.shape = P.Shape()

    def construct(self, x):
        x1, x2, x3, x4 = x
        aspp_out = self.aspp(x4)
        low_feat = self.low_conv(x1)
        aspp_out = self.head(aspp_out)
        h, w = self.shape(low_feat)[-2:]
        aspp_out = ops.ResizeBilinear((h, w), align_corners=True)(aspp_out)
        aspp_out = self.concat((low_feat, aspp_out))

        logits = self.classifier(aspp_out)
        res = {"pred": logits}

        if self.rep_head:
            res["rep"] = self.representation(aspp_out)
        
        if self.rep_clf:
            res["rep_clf"] = aspp_out

        return res


class Aux_Module(nn.Cell):
    def __init__(self, in_planes, num_classes=19, sync_bn=False):
        super(Aux_Module, self).__init__()

        norm_layer = get_syncbn() if sync_bn else nn.BatchNorm2d
        self.aux = nn.SequentialCell(
            nn.Conv2d(in_planes, 256, kernel_size=3, stride=1, padding=1, pad_mode="pad", weight_init='xavier_uniform'),
            norm_layer(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, has_bias=True, weight_init='xavier_uniform'),
        )
    
    def construct(self, x):
        res = self.aux(x)
        return res


class CosProto_Module(nn.Cell):
    def __init__(self, in_planes, num_classes, num_micro_proto, init_proto_path, proto_unpdate_momentum=0.99):
        super(CosProto_Module, self).__init__()
        self.in_planes = in_planes
        self.num_classes = num_classes
        self.num_micro_proto = num_micro_proto
        self.init_proto_path = init_proto_path
        self.temp = 0.1

        init_proto = np.random.randn(num_micro_proto * num_classes, in_planes).astype('f')
        self.proto_list = Parameter(init_proto, requires_grad=False)
        self.momentum = proto_unpdate_momentum
        # TODO: val does not need to load
        # self.init_proto()
        
        # mindspre operations
        self.shape = P.Shape()
        self.transpose = ops.Transpose()
        self.reshape = ops.Reshape()
        self.einsum = ops.Einsum("nd,md->nm")
    
    @staticmethod
    def l2_normalize(x):
        return ops.L2Normalize(axis=-1)(x)

    def construct(self, x, select_mask):
        bs, in_channel, h_origin, w_origin = self.shape(x)
        
        # x->pixel2sample: bs, C, H, W --> bs*H*W, C
        satisfied_loc = (select_mask == 1).astype(mindspore.int32)
        # TODO: mindspore 1.10 version does not support mask-based slice on tensor
        # x_p2s = self.reshape(self.transpose(x, (0, 2, 3, 1))[:, satisfied_loc], (-1, self.in_planes))
        x_p2s = self.reshape(self.transpose(x, (0, 2, 3, 1)), (-1, self.in_planes))
        x_p2s = self.l2_normalize(x_p2s)

        cur_proto = self.proto_list.clone()
        cur_proto.requires_grad = False
        cur_proto = self.l2_normalize(cur_proto)
        
        masks = self.einsum((x_p2s, cur_proto))
        fuse_res = self.reshape(masks, (-1, self.num_classes, self.num_micro_proto))
        res_idx, res = ops.ArgMaxWithValue(axis=2)(fuse_res)

        unsatisfied_loc = (select_mask == 0).astype(mindspore.float32)
        res = self.transpose(self.reshape(res, (bs, h_origin, w_origin, self.num_classes)), (0, 3, 1, 2)) / self.temp
        # if(ops.ReduceSum(keep_dims=False)(unsatisfied_loc, [0,1]) == 0):
        #     res = self.transpose(self.reshape(res, (bs, h_origin, w_origin, self.num_classes)), (0, 3, 1, 2)) / self.temp
        # else:
        #     h_out = int(np.sqrt(self.shape(res)[0]/bs))
        #     res = self.transpose(self.reshape(res, (bs, h_out, h_out, self.num_classes)), (0, 3, 1, 2)) / self.temp
        
        return res, res_idx

    def init_proto(self):
        print(f"Load proto from: '{self.init_proto_path}'")
        with open(self.init_proto_path, 'rb') as handle:
            init_protos = pickle.load(handle)
        num_class = len(init_protos)
        all_protos = list()
        for cls_id in range(num_class):
            all_protos.append(Tensor(np.stack(init_protos[cls_id], 0)))
        proto_tensor = ops.stack(all_protos, 0)
        self.proto_list = self.reshape(proto_tensor, (-1, self.in_plances))

    def update_proto(self, rep, cls_id, proto_id):
        self.proto_list[cls_id*self.num_micro_proto+proto_id, :] = self.proto_list[cls_id*self.num_micro_proto+proto_id, :] * self.momentum + (1 - self.momentum) * rep