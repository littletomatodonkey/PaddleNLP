import numpy as np
import paddle
import paddle.nn.functional as F
from paddle import nn
from paddle import ParamAttr

from abc import ABCMeta, abstractmethod

from collections import namedtuple


class Conv2d(nn.Conv2D):
    def __init__(self, *args, **kwargs):
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.stride = self._stride
        self.padding = self._updated_padding
        self.dilation = self._dilation
        self.groups = self._groups

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding,
                     self.dilation, self.groups)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class CNNBlockBase(nn.Layer):
    def __init__(self, in_channels, out_channels, stride):
        """
        The `__init__` method of any subclass should also contain these arguments.
        Args:
            in_channels (int):
            out_channels (int):
            stride (int):
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

    def freeze(self):
        for p in self.parameters():
            p.stop_gradient = True


class ShapeSpec(
        namedtuple("_ShapeSpec", ["channels", "height", "width", "stride"])):
    def __new__(cls, channels=None, height=None, width=None, stride=None):
        return super().__new__(cls, channels, height, width, stride)


def get_norm(norm, out_channels):
    """
    Args:
        norm (str or callable): either one of BN, SyncBN, FrozenBN, GN;
            or a callable that takes a channel number and returns
            the normalization layer as a nn.Layer.
    Returns:
        nn.Layer or None: the normalization layer
    """
    if norm is None:
        return None
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "BN": nn.BatchNorm,
            "SyncBN": nn.SyncBatchNorm,
            "FrozenBN": FrozenBatchNorm,
        }[norm]
    return norm(out_channels)


class FrozenBatchNorm(nn.BatchNorm):
    def __init__(self, num_channels):
        param_attr = ParamAttr(learning_rate=0.0, trainable=False)
        bias_attr = ParamAttr(learning_rate=0.0, trainable=False)
        super().__init__(
            num_channels,
            param_attr=param_attr,
            bias_attr=bias_attr,
            use_global_stats=True)


class Backbone(nn.Layer):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self):
        pass

    @property
    def size_divisibility(self) -> int:
        return 0

    def output_shape(self):
        # this is a backward-compatible default
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name])
            for name in self._out_features
        }
