from nobuco.converters.node_converter import converter
from nobuco.commons import ChannelOrderingStrategy

from torch import Tensor
import torchvision.ops

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