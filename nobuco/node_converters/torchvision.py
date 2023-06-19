import torchvision.ops

from nobuco.commons import ChannelOrderingStrategy
from torch import Tensor

from nobuco.converters.node_converter import converter


@converter(torchvision.ops.StochasticDepth, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_StochasticDepth(self, input: Tensor):
    p = self.p
    mode = self.mode
    import keras.applications
    return keras.applications.convnext.StochasticDepth(p)
