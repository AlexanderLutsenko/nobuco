from nobuco.converters.node_converter import converter
from nobuco.commons import ChannelOrderingStrategy

import torch
from torch import Tensor
import torchvision.ops

import tensorflow as tf
from tensorflow import keras


@converter(torchvision.ops.StochasticDepth, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_StochasticDepth(self, input: Tensor):
    p = self.p
    mode = self.mode
    import keras.src.applications
    return keras.src.applications.convnext.StochasticDepth(p)


@converter(torchvision.ops.FrozenBatchNorm2d)
def converter_FrozenBatchNorm(self, input: Tensor):
    epsilon = self.eps
    weight = self.weight.cpu().detach().numpy()
    bias = self.bias.cpu().detach().numpy()
    running_mean = self.running_mean.cpu().detach().numpy()
    running_var = self.running_var.cpu().detach().numpy()

    layer = keras.layers.BatchNormalization(epsilon=epsilon, weights=[weight, bias, running_mean, running_var])
    return layer

    # def func(input, *args, **kwargs):
    #     return (input - running_mean) / (tf.sqrt(running_var + epsilon)) * weight + bias
    # return func


@converter(torchvision.ops.boxes._upcast, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter__upcast(t: Tensor):
    def func(t):
        if t.dtype.is_floating:
            return t if t.dtype in (tf.float32, tf.float64) else tf.cast(t, tf.float32)
        else:
            return t if t.dtype in (tf.int32, tf.int64) else tf.cast(t, tf.int32)
    return func


@converter(torchvision.ops.nms, torch.ops.torchvision.nms, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def converter_nms(boxes: Tensor, scores: Tensor, iou_threshold: float):
    def func(boxes, scores, iou_threshold: float):
        return tf.image.non_max_suppression(boxes, scores, max_output_size=tf.dtypes.int32.max, iou_threshold=iou_threshold)
    return func


@converter(torchvision.ops.batched_nms, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def converter_batched_nms(boxes: Tensor, scores: Tensor, idxs: Tensor, iou_threshold: float):
    def func(boxes, scores, idxs, iou_threshold):
        # strategy: in order to perform NMS independently per class,
        # we add an offset to all the boxes. The offset is dependent
        # only on the class idx, and is large enough so that boxes
        # from different classes do not overlap
        max_coordinate = tf.reduce_max(boxes)
        offsets = tf.cast(idxs, boxes.dtype) * (max_coordinate + 1)
        boxes_for_nms = boxes + offsets[:, None]
        keep = tf.image.non_max_suppression(boxes_for_nms, scores, max_output_size=tf.dtypes.int32.max, iou_threshold=iou_threshold)
        return keep
    return func
