from typing import Tuple, Sequence, Optional, Iterable

from torch import nn

from fpn.containers import (Parallel, SequentialMultiInputMultiOutput)
from fpn.layers import (Interpolate, Reverse, Sum)
from typing import Tuple, Optional
import torch
from torch import nn
import torchvision as tv
from .backbone import ResNetFeatureMapsExtractor
from .layers import Interpolate, SelectOne
from .fpn import FPN, PanopticFPN
from .utils import _get_shapes

class FPNWithoutHiearchy(nn.Sequential):
    """
       Implementation of the architecture described in the paper
       "Feature Pyramid Networks for Object Detection" by Lin et al.,
       https://arxiv.com/abs/1612.03144.

       Takes in an n-tuple of feature maps in reverse order
       (1st feature map, 2nd feature map, ..., nth feature map), where
       the 1st feature map is the one produced by the earliest layer in the
       backbone network.

       The feature maps are passed through the architecture shown below, producing
       n outputs, such that the height and width of the ith output is equal to
       that of the corresponding input feature map and the number of channels
       is equal to out_channels.

       Returns all outputs as a tuple like so: (1st out, 2nd out, ..., nth out)

       Architecture diagram:

       nth feat. map ────────[nth in_conv]──────────┐────────[nth out_conv]────> nth out
                                                    │
                                                [upsample]
                                                    │
                                                    V
       (n-1)th feat. map ────[(n-1)th in_conv]────>(+)────[(n-1)th out_conv]────> (n-1)th out
                                                    │
                                                [upsample]
                                                    │
                                                    V
               .                     .                           .                    .
               .                     .                           .                    .
               .                     .                           .                    .
                                                    │
                                                [upsample]
                                                    │
                                                    V
       1st feat. map ────────[1st in_conv]────────>(+)────────[1st out_conv]────> 1st out

       """
    def __init__(self,
                 in_feats_shapes: Sequence[Tuple[int, ...]],
                 hidden_channels: int = 256,
                 out_channels: int = 2):
        """Constructor.
        Args:
            in_feats_shapes (Sequence[Tuple[int, ...]]): Shapes of the feature
                maps that will be fed into the network. These are expected to
                be tuples of the form (., C, H, ...).
            hidden_channels (int, optional): The number of channels to which
                all feature maps are convereted before being added together.
                Defaults to 256.
            out_channels (int, optional): Number of output channels. This will
                normally be the number of classes. Defaults to 2.
        """
        # reverse so that the deepest (i.e. produced by the deepest layer in
        # the backbone network) feature map is first.
        in_feats_shapes = in_feats_shapes[::-1]
        in_feats_channels = [s[1] for s in in_feats_shapes]
        # 1x1 conv to make the channels of all feature maps the same
        in_convs = Parallel([
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
            for in_channels in in_feats_channels
        ])

        # yapf: disable
        def resize_and_add(to_size):
            return nn.Sequential(
                Parallel([nn.Identity(), Interpolate(size=to_size)]),
                # Sum()
                SelectOne(idx=0)
            )

        top_down_layer = SequentialMultiInputMultiOutput(
            nn.Identity(),
            *[resize_and_add(shape[-2:]) for shape in in_feats_shapes[1:]]
        )

        out_convs = Parallel([
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1)
            for _ in in_feats_shapes
        ])
        layers = [
            Reverse(),
            in_convs,
            top_down_layer,
            out_convs,
            Reverse()
        ]
        # yapf: enable
        super().__init__(*layers)


def make_spot_fpn_resnet(name: str = 'resnet18',
                         fpn_type: str = 'fpn',
                         out_size: Tuple[int, int] = (224, 224),
                         fpn_channels: int = 256,
                         num_classes: int = 1000,
                         in_channels: int = 500):
    assert in_channels > 0
    assert num_classes > 0

    resnet = tv.models.resnet.__dict__[name](pretrained=False)

    old_conv = resnet.conv1
    old_conv_args = {
        'out_channels': old_conv.out_channels,
        'kernel_size': old_conv.kernel_size,
        'stride': old_conv.stride,
        'padding': old_conv.padding,
        'dilation': old_conv.dilation,
        'groups': old_conv.groups,
        'bias': old_conv.bias
    }
    # just replace the first conv layer
    new_conv = nn.Conv2d(in_channels=in_channels, **old_conv_args)
    resnet.conv1 = new_conv
    backbone = ResNetFeatureMapsExtractor(resnet)

    feat_shapes = _get_shapes(backbone, channels=in_channels, size=out_size)
    if fpn_type == 'fpn':
        fpn = nn.Sequential(
            FPNWithoutHiearchy(feat_shapes,
                hidden_channels=fpn_channels,
                out_channels=num_classes),
            SelectOne(idx=0))
    else:
        raise NotImplementedError()

    # yapf: disable
    model = nn.Sequential(
        backbone,
        fpn,
        Interpolate(size=out_size))
    # yapf: enable
    return model