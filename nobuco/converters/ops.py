import tensorflow as tf


def hard_sigmoid_pytorch_compatible(x):
  x = tf.clip_by_value(x/6 + 1/2, clip_value_min=0, clip_value_max=1)
  return x


def hard_swish_pytorch_compatible(x):
  x = x * hard_sigmoid_pytorch_compatible(x)
  return x


def hard_tanh_pytorch_compatible(x, min_val, max_val):
  x = tf.clip_by_value(x, clip_value_min=min_val, clip_value_max=max_val)
  return x