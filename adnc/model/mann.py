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
from tensorflow.contrib.rnn import MultiRNNCell
from tensorflow.python.ops import variable_scope as vs

from adnc.model.controller_units.controller import get_rnn_cell_list
from adnc.model.memory_units.memory_unit import get_memory_unit

from adnc.model.utils import HolisticMultiRNNCell
from adnc.model.utils import WordEmbedding

"""
The memory augmented neural network (MANN) model object contains the controller and the memory unit as well as the
loss function and connects everything.
"""

class MANN():
    def __init__(self, config, analyse=False, reuse=False, name='mann', dtype=tf.float32):
        """
        Args:
            config:     dict, configuration of the whole model
            analyse:    bool, is analyzer is used or not
            reuse:      bool, reuse model or not
        """

        self.seed = config["seed"]
        self.rng = np.random.RandomState(seed=self.seed)
        self.dtype = dtype
        self.analyse = analyse

        self.input_size = config["input_size"]
        self.output_size = config["output_size"]
        self.batch_size = config["batch_size"]

        self.input_embedding = config["input_embedding"]
        self.architecture = config['architecture']
        self.controller_config = config["controller_config"]
        self.memory_unit_config = config["memory_unit_config"]
        self.output_function = config["output_function"]
        self.output_mask = config["output_mask"]
        self.loss_function = config['loss_function']

        self.reuse = reuse
        self.name = name

        self.mask = tf.placeholder(self.dtype, [None, self.batch_size], name='mask')
        self.target = tf.placeholder(self.dtype, [None, self.batch_size, self.output_size], name='y')

        if self.input_embedding:
            word_idx_dict = self.input_embedding['word_idx_dict']
            embedding_size = self.input_embedding['embedding_size']
            if 'tmp_dir' in self.input_embedding:
                tmp_dir = self.input_embedding['tmp_dir']
            else:
                tmp_dir = "data_tmp"
            glove = WordEmbedding(embedding_size, word_idx_dict=word_idx_dict, initialization='glove', tmp_dir=tmp_dir)

            self._data = tf.placeholder(tf.int64, [None, self.batch_size], name='x')
            self.data = glove.embed(self._data)
        else:
            self.data = tf.placeholder(tf.float32, [None, self.batch_size, self.input_size], name='x')

        if self.architecture in ['uni', 'unidirectional']:
            unweighted_outputs, states = self.unidirectional(self.data, self.controller_config, self.memory_unit_config,
                                                             reuse=self.reuse)
        elif self.architecture in ['bi', 'bidirectional']:
            unweighted_outputs, states = self.bidirectional(self.data, self.controller_config, self.memory_unit_config,
                                                            reuse=self.reuse)
        else:
            raise UserWarning("Unknown architecture, use unidirectional or bidirectional")

        if self.analyse:
            with tf.device('/cpu:0'):
                if self.architecture in ['uni', 'unidirectional']:
                    analyse_outputs, analyse_states = self.unidirectional(self.data, self.controller_config,
                                                                          self.memory_unit_config, analyse=True,
                                                                          reuse=True)
                    analyse_outputs, analyse_signals = analyse_outputs
                    self.analyse = (analyse_outputs, analyse_signals, analyse_states)
                elif self.architecture in ['bi', 'bidirectional']:
                    analyse_outputs, analyse_states = self.bidirectional(self.data, self.controller_config,
                                                                         self.memory_unit_config, analyse=True,
                                                                         reuse=True)
                    analyse_outputs, analyse_signals = analyse_outputs
                    self.analyse = (analyse_outputs, analyse_signals, analyse_states)

        self.unweighted_outputs = unweighted_outputs
        self.prediction, self.outputs = self._output_layer(unweighted_outputs)
        self.loss = self.get_loss(self.prediction)

    def _output_layer(self, outputs):
        """
        Calculates the weighted and activated output of the MANN model
        Args:
            outputs: TF tensor, concatenation of memory units output and controller output

        Returns: TF tensor, predictions; TF tensor, unactivated predictions

        """
        with tf.variable_scope("output_layer"):
            output_size = outputs.get_shape()[-1].value

            weights_concat = tf.get_variable("weights_concat", (output_size, self.output_size),
                                             initializer=tf.contrib.layers.xavier_initializer(seed=self.seed),
                                             collections=['mann', tf.GraphKeys.GLOBAL_VARIABLES], dtype=self.dtype)
            bias_merge = tf.get_variable("bias_merge", (self.output_size,), initializer=tf.constant_initializer(0.),
                                         collections=['mann', tf.GraphKeys.GLOBAL_VARIABLES], dtype=self.dtype)

            output_flat = tf.reshape(outputs, [-1, output_size])
            output_flat = tf.matmul(output_flat, weights_concat) + bias_merge

        if self.output_function == 'softmax':
            predictions_flat = tf.nn.softmax(output_flat)
        elif self.output_function == 'tanh':
            predictions_flat = tf.tanh(output_flat)
        elif self.output_function == 'linear':
            predictions_flat = output_flat
        else:
            raise UserWarning("Unknown output function, use softmax, tanh or linear")

        predictions = tf.reshape(predictions_flat, [-1, self.batch_size, self.output_size])
        weighted_outputs = tf.reshape(output_flat, [-1, self.batch_size, self.output_size])

        return predictions, weighted_outputs

    def get_loss(self, prediction):
        """
        Args:
            prediction:     TF tensor, activated prediction of the model

        Returns:            TF scalar,  loss of the current forward set

        """
        if self.loss_function == 'cross_entropy':
            if self.output_mask:
                cost = tf.reduce_sum(
                    -1 * self.target * tf.log(tf.clip_by_value(prediction, 1e-12, 10.0)) - (1 - self.target) * tf.log(
                        tf.clip_by_value(1 - prediction, 1e-12, 10.0)), axis=2)
                cost *= self.mask
                loss = tf.reduce_sum(cost) / tf.reduce_sum(self.mask)
            else:
                loss = tf.reduce_mean(
                    -1 * self.target * tf.log(tf.clip_by_value(prediction, 1e-12, 10.0)) - (1 - self.target) * tf.log(
                        tf.clip_by_value(1 - prediction, 1e-12, 10.0)))

        elif self.loss_function == 'mse':
            clipped_prediction = tf.clip_by_value(prediction, 1e-12, 10.0)
            mse = tf.square(self.target - clipped_prediction)
            mse = tf.reduce_mean(mse, axis=2)

            if self.output_mask:
                cost = mse * self.mask
                loss = tf.reduce_sum(cost) / tf.reduce_sum(self.mask)
            else:
                loss = tf.reduce_mean(mse)
        else:
            raise UserWarning("Unknown loss function, use cross_entropy or mse")
        return loss

    def unidirectional(self, inputs, controller_config, memory_unit_config, analyse=False, reuse=False):
        """
        Connects unidirectional controller and memory unit and performs scan over sequence
        Args:
            inputs:                 TF tensor, input sequence
            controller_config:      dict, configuration of the controller
            memory_unit_config:     dict, configuration of the memory unit
            analyse:                bool, do analysis
            reuse:                  bool, reuse

        Returns:        TF tensor, output sequence; TF tensor, hidden states

        """

        with tf.variable_scope("controller"):
            controller_list = get_rnn_cell_list(controller_config, name='controller', reuse=reuse, seed=self.seed,
                                                dtype=self.dtype)

        if controller_config['connect'] == 'sparse':
            memory_input_size = controller_list[-1].output_size
            mu_cell = get_memory_unit(memory_input_size, memory_unit_config, 'memory_unit', analyse=analyse,
                                      reuse=reuse)
            cell = MultiRNNCell(controller_list + [mu_cell])
        else:
            controller_cell = HolisticMultiRNNCell(controller_list)
            memory_input_size = controller_cell.output_size
            mu_cell = get_memory_unit(memory_input_size, memory_unit_config, 'memory_unit', analyse=analyse,
                                      reuse=reuse)
            cell = MultiRNNCell([controller_cell, mu_cell])

        batch_size = inputs.get_shape()[1].value
        cell_init_states = cell.zero_state(batch_size, dtype=self.dtype)
        output_init = tf.zeros([batch_size, cell.output_size], dtype=self.dtype)

        if analyse:
            output_init = (output_init, mu_cell.analyse_state(batch_size, dtype=self.dtype))

        init_states = (output_init, cell_init_states)

        def step(pre_states, inputs):
            pre_rnn_output, pre_rnn_states = pre_states

            if analyse:
                pre_rnn_output = pre_rnn_output[0]

            controller_inputs = tf.concat([inputs, pre_rnn_output], axis=-1)
            rnn_output, rnn_states = cell(controller_inputs, pre_rnn_states)
            return (rnn_output, rnn_states)

        outputs, states = tf.scan(step, inputs, initializer=init_states, parallel_iterations=32)

        return outputs, states

    def bidirectional(self, inputs, controller_config, memory_unit_config, analyse=False, reuse=False):
        """
        Connects bidirectional controller and memory unit and performs scan over sequence
        Args:
            inputs:                 TF tensor, input sequence
            controller_config:      dict, configuration of the controller
            memory_unit_config:     dict, configuration of the memory unit
            analyse:                bool, do analysis
            reuse:                  bool, reuse

        Returns:        TF tensor, output sequence; TF tensor, hidden states

        """

        with tf.variable_scope("controller"):
            list_fw = get_rnn_cell_list(controller_config, name='con_fw', reuse=reuse, seed=self.seed, dtype=self.dtype)
            list_bw = get_rnn_cell_list(controller_config, name='con_bw', reuse=reuse, seed=self.seed, dtype=self.dtype)
        if controller_config['connect'] == 'sparse':
            cell_fw = MultiRNNCell(list_fw)
            cell_bw = MultiRNNCell(list_bw)
        else:
            cell_fw = HolisticMultiRNNCell(list_fw)
            cell_bw = HolisticMultiRNNCell(list_bw)

        memory_input_size = cell_fw.output_size + cell_bw.output_size
        cell_mu = get_memory_unit(memory_input_size, memory_unit_config, 'memory_unit', analyse=analyse, reuse=reuse)

        with vs.variable_scope("bw") as bw_scope:
            inputs_reverse = tf.reverse(inputs, axis=[0])
            output_bw, output_state_bw = tf.nn.dynamic_rnn(cell=cell_bw, inputs=inputs_reverse, dtype=self.dtype,
                                                           parallel_iterations=32, time_major=True, scope=bw_scope)
            output_bw = tf.reverse(output_bw, axis=[0])

        batch_size = inputs.get_shape()[1].value
        cell_fw_init_states = cell_fw.zero_state(batch_size, dtype=self.dtype)
        cell_mu_init_states = cell_mu.zero_state(batch_size, dtype=self.dtype)
        output_init = tf.zeros([batch_size, cell_mu.output_size], dtype=self.dtype)

        if analyse:
            output_init = (output_init, cell_mu.analyse_state(batch_size, dtype=self.dtype))

        init_states = (output_init, cell_fw_init_states, cell_mu_init_states)
        coupled_inputs = (inputs, output_bw)

        with vs.variable_scope("fw") as fw_scope:

            def step(pre_states, coupled_inputs):
                inputs, output_bw = coupled_inputs
                pre_outputs, pre_states_fw, pre_states_mu = pre_states

                if analyse:
                    pre_outputs = pre_outputs[0]

                controller_inputs = tf.concat([inputs, pre_outputs], axis=-1)
                output_fw, states_fw = cell_fw(controller_inputs, pre_states_fw)

                mu_inputs = tf.concat([output_fw, output_bw], axis=-1)
                output_mu, states_mu = cell_mu(mu_inputs, pre_states_mu)

                return (output_mu, states_fw, states_mu)

            outputs, states_fw, states_mu = tf.scan(step, coupled_inputs, initializer=init_states,
                                                    parallel_iterations=32)

        states = states_fw, states_mu
        return outputs, states

    @property
    def feed(self):
        """
        Returns:    TF placeholder for data, target and mask inout to the model
        """
        return self.data, self.target, self.mask

    @property
    def controller_trainable_variables(self):
        return tf.get_collection('recurrent_unit')

    @property
    def memory_unit_trainable_variables(self):
        return tf.get_collection('memory_unit')

    @property
    def mann_trainable_variables(self):
        return tf.get_collection('mann')

    @property
    def trainable_variables(self):
        return tf.trainable_variables()

    @property
    def controller_parameter_amount(self):
        return self.count_parameter_amount(self.controller_trainable_variables)

    @property
    def memory_unit_parameter_amount(self):
        return self.count_parameter_amount(self.memory_unit_trainable_variables)

    @property
    def mann_parameter_amount(self):
        return self.count_parameter_amount(self.mann_trainable_variables)

    @staticmethod
    def count_parameter_amount(var_list):
        parameters = 0
        for variable in var_list:
            shape = variable.get_shape()
            variable_parametes = 1
            for dim in shape:
                variable_parametes *= dim.value
            parameters += variable_parametes
        return parameters

    @property
    def parameter_amount(self):
        var_list = tf.trainable_variables()
        return self.count_parameter_amount(var_list)
