import torch
from torch import nn
import torch.nn.functional as F

import tensorflow_addons
from tensorflow import keras

import nobuco
from nobuco import ChannelOrderingStrategy


class DenseImageWarp(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, image, flow):
        _, _, h, w = nobuco.shape(flow)

        lin_h = torch.linspace(start=0, end=(h - 1), steps=h)[:, None].repeat(1, w)
        lin_w = torch.linspace(start=0, end=(w - 1), steps=w)[None, :].repeat(h, 1)
        lin = torch.stack([lin_h, lin_w], dim=0)[None, ...]

        scale = torch.stack([h, w], dim=0)[None, :, None, None]
        scale = (scale - 1) / 2

        '''
        grid = lin - flow
        grid = grid / scale - 1
        '''
        grid = lin / scale - 1 - (flow / scale)

        grid = torch.roll(grid, shifts=1, dims=1)
        grid = grid.permute(0, 2, 3, 1)
        return F.grid_sample(image, grid, mode='bilinear', padding_mode='border', align_corners=True)


@nobuco.converter(DenseImageWarp, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_TENSORFLOW_ORDER)
def dense_image_warp(self, image: torch.Tensor, flow: torch.Tensor):
    def func(image, flow):
        return keras.layers.Lambda(lambda args: tensorflow_addons.image.dense_image_warp(args[0], args[1]))([image, flow])
    return func
