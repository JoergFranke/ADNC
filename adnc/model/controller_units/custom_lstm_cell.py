# Copyright 2018 Jörg Franke
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import numpy as np
import tensorflow as tf

from adnc.model.utils import layer_norm, get_activation

"""
A implementation of the LSTM unit, it performs a bit faster as the TF implementation and implements layer norm.
"""

class CustomLSTMCell():
    def __init__(self, num_units, layer_norm=False, activation='tanh', seed=100, reuse=False, trainable=True,
                 dtype=None, name='lstm'):

        self.num_units = num_units
        self.layer_norm = layer_norm

        self.rng = np.random.RandomState(seed=seed)
        self.seed = seed
        self.reuse = reuse
        self.name = name
        self.dtype = dtype

        self._forget_bias = -1.0

        self.act = get_activation(activation)

    @property
    def state_size(self):
        return self.num_units

    @property
    def output_size(self):
        return self.num_units

    @property
    def trainable_variables(self):
        return tf.get_collection('recurrent_unit')

    @property
    def parameter_amount(self):
        var_list = self.trainable_variables
        parameters = 0
        for variable in var_list:
            shape = variable.get_shape()
            variable_parametes = 1
            for dim in shape:
                variable_parametes *= dim.value
            parameters += variable_parametes
        return parameters

    def zero_state(self, batch_size, dtype=tf.float32):
        zero_state = tf.zeros([batch_size, self.num_units], dtype=dtype)
        return zero_state

    def __call__(self, inputs, cell_state, scope=None):

        if self.layer_norm:
            lstm_layer = self._lnlstm_layer
        else:
            lstm_layer = self._lstm_layer

        with tf.variable_scope("{}".format(self.name), reuse=self.reuse):

            outputs, cell_states = lstm_layer(inputs, cell_state, name="{}".format(self.name))

        return outputs, cell_states

    def _lstm_cell(self, inputs, pre_cell_state, cell_size, w_ifco, b_ifco):

        ifco = tf.matmul(inputs, w_ifco) + b_ifco

        gates = tf.sigmoid(ifco[:, 0 * cell_size:3 * cell_size])
        cell_state = tf.add(tf.multiply(gates[:, 0:cell_size], pre_cell_state),
                            tf.multiply(gates[:, cell_size:2 * cell_size],
                                        self.act(ifco[:, 3 * cell_size:4 * cell_size])))
        output = gates[:, 2 * cell_size:3 * cell_size] * self.act(cell_state)

        return output, cell_state

    def _lstm_layer(self, inputs, pre_cell_state, name=0):

        inputs_shape = inputs.get_shape()
        if inputs_shape.__len__() != 2:
            raise UserWarning("invalid shape: inputs at _lstm_layer {}".format(name))
        input_size = inputs_shape[1].value

        cell_shape = pre_cell_state.get_shape()
        if cell_shape.__len__() != 2:
            raise UserWarning("invalid shape: cell_shape at _lstm_layer {}".format(name))
        cell_size = cell_shape[1].value

        with tf.variable_scope("cell_{}".format(name), reuse=self.reuse):
            w_ifco = tf.get_variable("w_ifco_{}".format(name), (input_size, 4 * cell_size),
                                     initializer=tf.contrib.layers.xavier_initializer(seed=self.seed),
                                     collections=['recurrent_unit', tf.GraphKeys.GLOBAL_VARIABLES], dtype=self.dtype)
            b_ifco = tf.get_variable("b_ifco_{}".format(name), (4 * cell_size,),
                                     initializer=tf.constant_initializer(0.),
                                     collections=['recurrent_unit', tf.GraphKeys.GLOBAL_VARIABLES], dtype=self.dtype)

            output, cell_state = self._lstm_cell(inputs, pre_cell_state, cell_size, w_ifco, b_ifco)

        return output, cell_state

    def _lnlstm_cell(self, inputs, pre_cell_state, cell_size, w_ifco, b_ifco):

        ifco = layer_norm(tf.matmul(inputs, w_ifco), name="w_ifco", dtype=self.dtype,
                          collection='recurrent_unit') + b_ifco
        gates = tf.sigmoid(ifco[:, 0 * cell_size:3 * cell_size])
        cell_state = tf.add(tf.multiply(gates[:, 0:cell_size], pre_cell_state),
                            tf.multiply(gates[:, cell_size:2 * cell_size],
                                        self.act(ifco[:, 3 * cell_size:4 * cell_size])))
        output = gates[:, 2 * cell_size:3 * cell_size] * self.act(
            layer_norm(cell_state, name="out_act", dtype=self.dtype, collection='recurrent_unit'))

        return output, cell_state

    def _lnlstm_layer(self, inputs, pre_cell_state, name):

        cell_state = pre_cell_state

        input_size = inputs.get_shape()[1].value

        print(pre_cell_state)

        cell_shape = cell_state.get_shape()
        cell_size = cell_shape[1].value

        with tf.variable_scope("{}".format(name)):
            w_ifco = tf.get_variable("w_ifco_{}".format(name), (input_size, 4 * cell_size),
                                     initializer=tf.contrib.layers.xavier_initializer(seed=self.seed),
                                     collections=['recurrent_unit', tf.GraphKeys.GLOBAL_VARIABLES], dtype=self.dtype)

            b_ifco = tf.get_variable("b_ifco_ln_{}".format(name), (4 * cell_size,),
                                     initializer=tf.constant_initializer(0.),
                                     collections=['recurrent_unit', tf.GraphKeys.GLOBAL_VARIABLES], dtype=self.dtype)

            output, cell_state = self._lnlstm_cell(inputs, cell_state, cell_size, w_ifco, b_ifco)

        return output, cell_state
