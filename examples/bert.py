import torch
import keras
import tensorflow as tf
from tensorflow.lite.python.lite import TFLiteConverter

import nobuco
from nobuco import ChannelOrder, ChannelOrderingStrategy
from nobuco.layers.weight import WeightLayer


tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-cased')

text_1 = "Who was Jim Henson ?"
text_2 = "Jim Henson was a puppeteer"

# Tokenized input with special tokens around it (for BERT: [CLS] at the beginning and [SEP] at the end)
indexed_tokens = tokenizer.encode(text_1, text_2, add_special_tokens=True)

# Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
segments_ids = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]

device = 'cuda'

# Convert inputs to PyTorch tensors
segments_tensors = torch.tensor([segments_ids]).to(device)
tokens_tensor = torch.tensor([indexed_tokens]).to(device)

pytorch_module = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-cased').eval().to(device)

# tf.config.experimental.set_visible_devices([], 'GPU')
keras_model = nobuco.pytorch_to_keras(
    pytorch_module,
    args=[tokens_tensor], kwargs={'token_type_ids': segments_tensors},
    inputs_channel_order=ChannelOrder.TENSORFLOW,
    trace_shape=True,
)

model_path = 'bert'
keras_model.save(model_path + '.h5')
print('Model saved')

custom_objects = {'WeightLayer': WeightLayer}

keras_model_restored = keras.models.load_model(model_path + '.h5', custom_objects=custom_objects)
print('Model loaded')

converter = TFLiteConverter.from_keras_model_file(model_path + '.h5', custom_objects=custom_objects)
converter.target_ops = [tf.lite.OpsSet.SELECT_TF_OPS, tf.lite.OpsSet.TFLITE_BUILTINS]
tflite_model = converter.convert()
with open(model_path + '.tflite', 'wb') as f:
    f.write(tflite_model)
