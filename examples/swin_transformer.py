import nobuco
from nobuco import ChannelOrder, ChannelOrderingStrategy
from nobuco.layers.weight import WeightLayer

import torch
from torchvision import models


device = 'cuda'

pytorch_module = models.swin_v2_t().eval().to(device)

x = torch.normal(0, 1, size=(1, 3, 224, 224)).to(device)

keras_model = nobuco.pytorch_to_keras(
    pytorch_module,
    args=[x],
    inputs_channel_order=ChannelOrder.TENSORFLOW,
)

model_path = 'swin_transformer'
keras_model.save(model_path + '.h5')
print('Model saved')

import keras.src.applications
custom_objects = {'WeightLayer': WeightLayer, 'StochasticDepth': keras.src.applications.convnext.StochasticDepth}

keras_model_restored = keras.models.load_model(model_path + '.h5', custom_objects=custom_objects)
print('Model loaded')
