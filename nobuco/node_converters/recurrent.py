import torch
from torch import nn

import tensorflow as tf
from tensorflow import keras

import numpy as np

from nobuco.commons import ChannelOrder, ChannelOrderingStrategy
from nobuco.converters.node_converter import converter


class Bidirectional:
    def __init__(self, layer, backward_layer):
        self.layer = layer
        self.backward_layer = backward_layer

    def __call__(self, x, initial_state=None):
        if initial_state is not None:
            half = len(initial_state) // 2
            state_f = initial_state[:half]
            state_b = initial_state[half:]
        else:
            state_f = None
            state_b = None

        ret_f = self.layer(x, state_f)
        ret_b = self.backward_layer(x, state_b)
        y_f, h_f = ret_f[0], ret_f[1:]
        y_b, h_b = ret_b[0], ret_b[1:]
        y_b = tf.reverse(y_b, axis=(1,))
        y_cat = tf.concat([y_f, y_b], axis=-1)
        return y_cat, *h_f, *h_b


@converter(nn.GRU, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def converter_GRU(self: nn.GRU, input, hx=None):
    assert self.batch_first, 'Only batch_first mode is supported at the moment'

    bidirectional = self.bidirectional
    num_layers = self.num_layers

    def create_layer(i, reverse):

        def reorder(param):
            assert param.shape[-1] % 3 == 0
            p1, p2, p3 = np.split(param, 3, axis=-1)
            return np.concatenate([p2, p1, p3], axis=-1)

        suffix = '_reverse' if reverse else ''
        weight_ih = self.__getattr__(f'weight_ih_l{i}{suffix}').cpu().detach().numpy().transpose((1, 0))
        weight_hh = self.__getattr__(f'weight_hh_l{i}{suffix}').cpu().detach().numpy().transpose((1, 0))
        weight_ih = reorder(weight_ih)
        weight_hh = reorder(weight_hh)
        weights = [weight_ih, weight_hh]

        if self.bias:
            bias_ih = self.__getattr__(f'bias_ih_l{i}{suffix}').cpu().detach().numpy()
            bias_hh = self.__getattr__(f'bias_hh_l{i}{suffix}').cpu().detach().numpy()
            bias_ih = reorder(bias_ih)
            bias_hh = reorder(bias_hh)
            weights += [np.stack([bias_ih, bias_hh], axis=0)]

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
            unroll=False,
            go_backwards=reverse,
            weights=weights,
        )
        return gru

    def convert_initial_states(hx):
        if hx is not None:
            h0 = tf.reshape(hx, (num_layers, -1, *hx.shape[1:]))
            initial_states = []
            for i in range(num_layers):
                if bidirectional:
                    state = (h0[i][0], h0[i][1])
                else:
                    state = h0[i][0]
                initial_states.append(state)
            return initial_states
        else:
            return None

    layers = []
    for i in range(self.num_layers):
        layer = create_layer(i, reverse=False)
        if bidirectional:
            layer_reverse = create_layer(i, reverse=True)
            # layer = keras.layers.Bidirectional(layer=layer, backward_layer=layer_reverse)
            layer = Bidirectional(layer=layer, backward_layer=layer_reverse)
        layers.append(layer)

    no_batch = input.dim() == 2

    def func(input, hx=None):
        x = input

        if no_batch:
            x = x[None, :, :]
            if hx is not None:
                hx = hx[:, None, :]

        initial_states = convert_initial_states(hx)

        hxs = []
        for i in range(num_layers):
            state = initial_states[i] if initial_states else None
            ret = layers[i](x, initial_state=state)
            x, hxo = ret[0], ret[1:]
            hxs += hxo
        hxs = tf.stack(hxs, axis=0)

        if no_batch:
            x = x[0, :, :]
            hxs = hxs[:, 0, :]

        return x, hxs
    return func


@converter(nn.LSTM, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def converter_LSTM(self: nn.LSTM, input, hx=None):
    assert self.batch_first, 'Only batch_first mode is supported at the moment'

    bidirectional = self.bidirectional
    num_layers = self.num_layers

    def create_layer(i, reverse):
        suffix = '_reverse' if reverse else ''
        weight_ih = self.__getattr__(f'weight_ih_l{i}{suffix}').cpu().detach().numpy().transpose((1, 0))
        weight_hh = self.__getattr__(f'weight_hh_l{i}{suffix}').cpu().detach().numpy().transpose((1, 0))
        weights = [weight_ih, weight_hh]

        if self.bias:
            bias_ih = self.__getattr__(f'bias_ih_l{i}{suffix}').cpu().detach().numpy()
            bias_hh = self.__getattr__(f'bias_hh_l{i}{suffix}').cpu().detach().numpy()
            weights += [bias_ih + bias_hh]

        lstm = keras.layers.LSTM(
            units=self.hidden_size,
            activation='tanh',
            recurrent_activation='sigmoid',
            use_bias=self.bias,
            dropout=self.dropout,
            return_sequences=True,
            return_state=True,
            time_major=not self.batch_first,
            unroll=False,
            go_backwards=reverse,
            weights=weights,
        )
        return lstm

    def convert_initial_states(hx):
        if hx is not None:
            h0, c0 = tuple(tf.reshape(h, (num_layers, -1, *h.shape[1:])) for h in hx)
            initial_states = []
            for i in range(num_layers):
                if bidirectional:
                    state = (h0[i][0], c0[i][0], h0[i][1], c0[i][1])
                else:
                    state = (h0[i][0], c0[i][0])
                initial_states.append(state)
            return initial_states
        else:
            return None

    layers = []
    for i in range(self.num_layers):
        layer = create_layer(i, reverse=False)
        if bidirectional:
            layer_reverse = create_layer(i, reverse=True)
            # layer = keras.layers.Bidirectional(layer=layer, backward_layer=layer_reverse)
            layer = Bidirectional(layer=layer, backward_layer=layer_reverse)
        layers.append(layer)

    no_batch = input.dim() == 2

    def func(input, hx=None):
        x = input

        if no_batch:
            x = x[None, :, :]
            if hx is not None:
                hxs, cxs = hx
                hxs = hxs[:, None, :]
                cxs = cxs[:, None, :]
                hx = (hxs, cxs)

        initial_states = convert_initial_states(hx)

        hxs = []
        cxs = []
        for i in range(num_layers):
            x, *rec_o = layers[i](x, initial_state=initial_states[i] if initial_states else None)
            hxs.append(rec_o[0::2])
            cxs.append(rec_o[1::2])
        hxs = tf.concat(hxs, axis=0)
        cxs = tf.concat(cxs, axis=0)

        if no_batch:
            x = x[0, :, :]
            hxs = hxs[:, 0, :]
            cxs = cxs[:, 0, :]

        return x, (hxs, cxs)
    return func
