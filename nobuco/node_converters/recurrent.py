import tensorflow as tf
from tensorflow import keras
from torch import nn

import numpy as np

from nobuco.commons import ChannelOrder, ChannelOrderingStrategy
from nobuco.converters.node_converter import converter


@converter(nn.GRU, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def converter_GRU(self: nn.GRU, input, hx=None):
    assert not self.bidirectional

    def reorder(param):
        assert param.shape[-1] % 3 == 0
        p1, p2, p3 = np.split(param, 3, axis=-1)
        return np.concatenate([p2, p1, p3], axis=-1)

    grus = []
    for i in range(self.num_layers):
        weight_ih = self.__getattr__(f'weight_ih_l{i}').detach().numpy().transpose((1, 0))
        weight_hh = self.__getattr__(f'weight_hh_l{i}').detach().numpy().transpose((1, 0))
        bias_ih = self.__getattr__(f'bias_ih_l{i}').detach().numpy()
        bias_hh = self.__getattr__(f'bias_hh_l{i}').detach().numpy()

        weight_ih = reorder(weight_ih)
        weight_hh = reorder(weight_hh)
        bias_ih = reorder(bias_ih)
        bias_hh = reorder(bias_hh)

        gru = keras.layers.GRU(
            units=self.hidden_size,
            activation='tanh',
            recurrent_activation='sigmoid',
            use_bias=self.bias,
            dropout=self.dropout,
            return_sequences=True,
            return_state=True,
            time_major=not self.batch_first,
            reset_after=True,
            unroll=True,
            weights=[weight_ih, weight_hh, tf.stack([bias_ih, bias_hh], axis=0)],
        )
        grus.append(gru)

    def func(input, hx=None):
        x = input
        hxs = []
        for i in range(len(grus)):
            initial_state = hx[i] if hx is not None else None
            x, hxo = grus[i](x, initial_state=initial_state)
            hxs.append(hxo)
        hxs = tf.stack(hxs, axis=0)
        return x, hxs
    return func


@converter(nn.LSTM, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def converter_LSTM(self: nn.LSTM, input, hx=None):
    assert not self.bidirectional

    lstms = []
    for i in range(self.num_layers):
        weight_ih = self.__getattr__(f'weight_ih_l{i}').detach().numpy().transpose((1, 0))
        weight_hh = self.__getattr__(f'weight_hh_l{i}').detach().numpy().transpose((1, 0))
        bias_ih = self.__getattr__(f'bias_ih_l{i}').detach().numpy()
        bias_hh = self.__getattr__(f'bias_hh_l{i}').detach().numpy()

        lstm = keras.layers.LSTM(
            units=self.hidden_size,
            activation='tanh',
            recurrent_activation='sigmoid',
            use_bias=self.bias,
            dropout=self.dropout,
            return_sequences=True,
            return_state=True,
            time_major=not self.batch_first,
            unroll=True,
            weights=[weight_ih, weight_hh, bias_ih + bias_hh],
        )
        lstms.append(lstm)

    def func(input, hx=None):
        x = input
        hxs = []
        cxs = []
        for i in range(len(lstms)):
            initial_state = (hx[0][i], hx[1][i]) if hx is not None else None
            x, hxo, cxo = lstms[i](x, initial_state=initial_state)
            hxs.append(hxo)
            cxs.append(cxo)
        hxs = tf.stack(hxs, axis=0)
        cxs = tf.stack(cxs, axis=0)
        return x, (hxs, cxs)
    return func
