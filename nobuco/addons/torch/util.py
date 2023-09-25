import torch


def channel_interleave2d(x: torch.Tensor, block_size: int, reverse: bool) -> torch.Tensor:
    b, c, h, w = x.shape
    n_blocks = block_size ** 2

    if reverse:
        x = x.view(b, n_blocks, c // n_blocks, h, w)
    else:
        x = x.view(b, c // n_blocks, n_blocks, h, w)

    x = x.transpose(1, 2).reshape(b, c, h, w)
    return x
