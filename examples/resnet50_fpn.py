import urllib
import PIL.Image
from torchvision.models.detection import FCOS_ResNet50_FPN_Weights

import nobuco
from nobuco import ChannelOrder, ChannelOrderingStrategy
from nobuco.layers.weight import WeightLayer

import torch
import torchvision

import tensorflow as tf
import keras


model = torchvision.models.detection.fcos_resnet50_fpn(weights="DEFAULT")
model.eval()

categoties = FCOS_ResNet50_FPN_Weights.DEFAULT.meta['categories']
print(categoties)

urllib.request.urlretrieve('https://pytorch.org/tutorials/_images/cat_224x224.jpg', "cat_224x224.jpg")
image_pil = PIL.Image.open("cat_224x224.jpg").resize((640, 640))

image_pt = torchvision.transforms.functional.pil_to_tensor(image_pil).to(torch.float32)[None] / 255

keras_model, outputs_pt = nobuco.pytorch_to_keras(
    model,
    args=[image_pt],
    kwargs=None,
    inputs_channel_order=ChannelOrder.TENSORFLOW,
    outputs_channel_order=ChannelOrder.TENSORFLOW,
    enable_torch_tracing=True,
    return_outputs_pt=True
)
print('Output classes:', [categoties[i] for i in outputs_pt[1][0]["labels"]])

model_path = 'resnet50_fpn'
keras_model.save(model_path)
print('Model saved')

custom_objects = {'WeightLayer': WeightLayer}
keras_model_restored = keras.models.load_model(model_path, custom_objects=custom_objects)
print('Model loaded')

image_np = tf.keras.utils.img_to_array(image_pil)[None] / 255
_, _, output_classes = keras_model_restored(image_np)
print('Output classes:', [categoties[i] for i in output_classes])
