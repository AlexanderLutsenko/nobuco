import os

from transformers.models.detr.modeling_detr import DetrFrozenBatchNorm2d

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from transformers import DetrForObjectDetection

import nobuco
from nobuco import ChannelOrder, ChannelOrderingStrategy
from nobuco.layers.weight import WeightLayer

import torch
import keras


device = 'cpu'

pytorch_module = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").eval().to(device)


@nobuco.converter(DetrFrozenBatchNorm2d)
def converter_DetrFrozenBatchNorm(self, input):
    weight = self.weight.cpu().detach().numpy()
    bias = self.bias.cpu().detach().numpy()
    running_mean = self.running_mean.cpu().detach().numpy()
    running_var = self.running_var.cpu().detach().numpy()

    layer = keras.layers.BatchNormalization(epsilon=1e-5, weights=[weight, bias, running_mean, running_var])
    return layer


x = torch.normal(0, 1, size=(1, 3, 256, 256)).to(device)

keras_model = nobuco.pytorch_to_keras(
    pytorch_module,
    args=[x],
    inputs_channel_order=ChannelOrder.TENSORFLOW,
)

model_path = 'detr'
keras_model.save(model_path + '.h5')
print('Model saved')

custom_objects = {'WeightLayer': WeightLayer}

keras_model_restored = keras.models.load_model(model_path + '.h5', custom_objects=custom_objects)
print('Model loaded')
