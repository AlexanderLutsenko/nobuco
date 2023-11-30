import numbers
from typing import Optional, Union, List, Tuple, Sequence, Any

from tensorflow.python.ops.image_ops_impl import ResizeMethod
from torch import Tensor

import tensorflow as tf
import torch
import torch.nn.functional as F

from nobuco.commons import TF_TENSOR_CLASSES
from nobuco.converters.node_converter import converter


@converter(F.interpolate)
def converter_interpolate(input: Tensor, size: Optional[int] = None, scale_factor: Optional[List[float]] = None, mode: str = 'nearest',
                align_corners: Optional[bool] = None, recompute_scale_factor: Optional[bool] = None, antialias: bool = False):

    if mode == 'bilinear':
        method = ResizeMethod.BILINEAR
    elif mode == 'nearest':
        method = ResizeMethod.NEAREST_NEIGHBOR
    else:
        raise Exception('Unsupported mode: ', mode)

    def func(input, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None, antialias=False):
        assert not (align_corners and antialias), "'align_corners' and 'antialias' cannot both be True"

        try:
            scale_factor[0], scale_factor[1]
        except Exception:
            scale_factor = (scale_factor, scale_factor)

        if size is None:
            _, h, w, _ = tf.cast(tf.shape(input), dtype=tf.float32)
            size = (scale_factor[0] * h, scale_factor[1] * w)
        else:
            try:
                size[0], size[1]
            except Exception:
                size = (size, size)

        if align_corners:
            return tf.compat.v1.image.resize(input, size=size, method=method, align_corners=align_corners)
        else:
            return tf.image.resize(input, size=size, method=method, antialias=antialias)
    return func
