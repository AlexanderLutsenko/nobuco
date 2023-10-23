import nobuco
from nobuco import ChannelOrder, ChannelOrderingStrategy
from nobuco.layers.weight import WeightLayer

import torch
from torchvision import models

import tensorflow as tf
import keras


device = 'cuda'

pytorch_module = models.vit_l_32().eval().to(device)

x = torch.normal(0, 1, size=(1, 3, 224, 224)).to(device)

# Disable GPU for Tensorflow
# tf.config.experimental.set_visible_devices([], 'GPU')
keras_model = nobuco.pytorch_to_keras(
    pytorch_module,
    args=[x],
    inputs_channel_order=ChannelOrder.TENSORFLOW,
    # full_validation=False
)

model_path = 'vision_transformer'
keras_model.save(model_path + '.h5')
print('Model saved')

custom_objects = {'WeightLayer': WeightLayer}

keras_model_restored = keras.models.load_model(model_path + '.h5', custom_objects=custom_objects)
print('Model loaded')
