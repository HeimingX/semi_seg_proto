import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.ops import operations as P

def get_syncbn():
    # return nn.BatchNorm2d
    return nn.SyncBatchNorm


class ASPP(nn.Cell):
    """
    Reference:
        Chen, Liang-Chieh, et al. *"Rethinking Atrous Convolution for Semantic Image Segmentation."*
    """

    def __init__(
        self, in_planes, inner_planes=256, sync_bn=False, dilations=(12, 24, 36)
    ):
        super(ASPP, self).__init__()

        norm_layer = get_syncbn() if sync_bn else nn.BatchNorm2d
        self.conv1 = nn.SequentialCell(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(
                in_planes,
                inner_planes,
                kernel_size=1,
                padding=0,
                dilation=1,
                has_bias=False,
                weight_init='xavier_uniform',
            ),
            norm_layer(inner_planes),
            nn.ReLU(),
        )
        self.conv2 = nn.SequentialCell(
            nn.Conv2d(
                in_planes,
                inner_planes,
                kernel_size=1,
                padding=0,
                dilation=1,
                has_bias=False,
                weight_init='xavier_uniform',
            ),
            norm_layer(inner_planes),
            nn.ReLU(),
        )
        self.conv3 = nn.SequentialCell(
            nn.Conv2d(
                in_planes,
                inner_planes,
                kernel_size=3,
                padding=dilations[0],
                pad_mode="pad",
                dilation=dilations[0],
                has_bias=False,
                weight_init='xavier_uniform',
            ),
            norm_layer(inner_planes),
            nn.ReLU(),
        )
        self.conv4 = nn.SequentialCell(
            nn.Conv2d(
                in_planes,
                inner_planes,
                kernel_size=3,
                padding=dilations[1],
                pad_mode="pad",
                dilation=dilations[1],
                has_bias=False,
                weight_init='xavier_uniform',
            ),
            norm_layer(inner_planes),
            nn.ReLU(),
        )
        self.conv5 = nn.SequentialCell(
            nn.Conv2d(
                in_planes,
                inner_planes,
                kernel_size=3,
                padding=dilations[2],
                pad_mode="pad",
                dilation=dilations[2],
                has_bias=False,
                weight_init='xavier_uniform',
            ),
            norm_layer(inner_planes),
            nn.ReLU(),
        )

        self.out_planes = (len(dilations) + 2) * inner_planes

        self.concat = P.Concat(axis=1)
        self.shape = P.Shape()

    def get_outplanes(self):
        return self.out_planes

    def construct(self, x):
        _, _, h, w = self.shape(x)
        feat1 = ops.ResizeBilinear((h, w), align_corners=True)(self.conv1(x))
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)
        aspp_out = self.concat((feat1, feat2, feat3, feat4, feat5))
        return aspp_out
