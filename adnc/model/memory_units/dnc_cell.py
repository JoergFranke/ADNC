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

from adnc.model.memory_units.base_cell import BaseMemoryUnitCell
from adnc.model.utils import oneplus, layer_norm, unit_simplex_initialization

"""
The vanilla DNC memory unit.
"""

class DNCMemoryUnitCell(BaseMemoryUnitCell):
    def __init__(self, input_size, memory_length, memory_width, read_heads, bypass_dropout=False, dnc_norm=False,
                 seed=100, reuse=False, analyse=False, dtype=tf.float32, name='dnc_mu'):

        super().__init__(input_size, memory_length, memory_width, read_heads, bypass_dropout, dnc_norm, seed, reuse,
                         analyse, dtype, name)


    @property
    def state_size(self):
        init_memory = tf.TensorShape([self.h_N, self.h_W])
        init_usage_vector = tf.TensorShape([self.h_N])
        init_write_weighting = tf.TensorShape([self.h_N])
        init_precedence_weightings = tf.TensorShape([self.h_N])
        init_link_mat = tf.TensorShape([self.h_N, self.h_N])
        init_read_weighting = tf.TensorShape([self.h_RH, self.h_N])
        return (init_memory, init_usage_vector, init_write_weighting, init_precedence_weightings,
                init_link_mat, init_read_weighting)

    def zero_state(self, batch_size, dtype=tf.float32):

        init_memory = tf.fill([batch_size, self.h_N, self.h_W], tf.cast(1 / (self.h_N * self.h_W), dtype=dtype))
        init_usage_vector = tf.zeros([batch_size, self.h_N], dtype=dtype)
        init_write_weighting = unit_simplex_initialization(self.rng, batch_size, [self.h_N], dtype=dtype)
        init_precedence_weightings = tf.zeros([batch_size, self.h_N], dtype=dtype)

        init_link_mat = tf.zeros([batch_size, self.h_N, self.h_N], dtype=dtype)
        init_read_weighting = unit_simplex_initialization(self.rng, batch_size, [self.h_RH, self.h_N], dtype=dtype)
        zero_states = (init_memory, init_usage_vector, init_write_weighting, init_precedence_weightings,
                       init_link_mat, init_read_weighting,)

        return zero_states

    def analyse_state(self, batch_size, dtype=tf.float32):

        alloc_gate = tf.zeros([batch_size, 1], dtype=dtype)
        free_gates = tf.zeros([batch_size, self.h_RH, 1], dtype=dtype)
        write_gate = tf.zeros([batch_size, 1], dtype=dtype)
        write_keys = tf.zeros([batch_size, 1, self.h_W], dtype=dtype)
        write_strengths = tf.zeros([batch_size, 1], dtype=dtype)
        write_vector = tf.zeros([batch_size, 1, self.h_W], dtype=dtype)
        erase_vector = tf.zeros([batch_size, 1, self.h_W], dtype=dtype)
        read_keys = tf.zeros([batch_size, self.h_RH, self.h_W], dtype=dtype)
        read_strengths = tf.zeros([batch_size, self.h_RH, 1], dtype=dtype)
        read_modes = tf.zeros([batch_size, self.h_RH, 3], dtype=dtype)

        analyse_states = alloc_gate, free_gates, write_gate, write_keys, write_strengths, write_vector, \
                         erase_vector, read_keys, read_strengths, read_modes

        return analyse_states

    def __call__(self, inputs, pre_states, scope=None):

        self.h_B = inputs.get_shape()[0].value

        link_matrix_inv_eye, memory_ones, batch_memory_range = self._create_constant_value_tensors(self.h_B, self.dtype)
        self.const_link_matrix_inv_eye = link_matrix_inv_eye
        self.const_memory_ones = memory_ones
        self.const_batch_memory_range = batch_memory_range

        pre_memory, pre_usage_vector, pre_write_weightings, pre_precedence_weighting, pre_link_matrix, pre_read_weightings = pre_states

        weighted_input = self._weight_input(inputs)

        control_signals = self._create_control_signals(weighted_input)
        alloc_gate, free_gates, write_gate, write_keys, write_strengths, write_vector, \
        erase_vector, read_keys, read_strengths, read_modes = control_signals

        alloc_weightings, usage_vector = self._update_alloc_and_usage_vectors(pre_write_weightings, pre_read_weightings,
                                                                              pre_usage_vector, free_gates)
        write_content_weighting = self._calculate_content_weightings(pre_memory, write_keys, write_strengths)
        write_weighting = self._update_write_weighting(alloc_weightings, write_content_weighting, write_gate,
                                                       alloc_gate)
        memory = self._update_memory(pre_memory, write_weighting, write_vector, erase_vector)
        link_matrix, precedence_weighting = self._update_link_matrix(pre_link_matrix, write_weighting,
                                                                     pre_precedence_weighting)
        forward_weightings, backward_weightings = self._make_read_forward_backward_weightings(link_matrix,
                                                                                              pre_read_weightings)
        read_content_weightings = self._calculate_content_weightings(memory, read_keys, read_strengths)
        read_weightings = self._make_read_weightings(forward_weightings, backward_weightings, read_content_weightings,
                                                     read_modes)
        read_vectors = self._read_memory(memory, read_weightings)
        read_vectors = tf.reshape(read_vectors, [self.h_B, self.h_W * self.h_RH])

        if self.bypass_dropout:
            input_bypass = tf.nn.dropout(inputs, self.bypass_dropout)
        else:
            input_bypass = inputs

        output = tf.concat([read_vectors, input_bypass], axis=-1)

        if self.analyse:
            output = (output, control_signals)

        return output, (memory, usage_vector, write_weighting, precedence_weighting, link_matrix, read_weightings)

    def _create_constant_value_tensors(self, batch_size, dtype):

        link_matrix_inv_eye = 1 - tf.constant(np.identity(self.h_N), dtype=dtype, name="link_matrix_inv_eye")
        memory_ones = tf.ones([batch_size, self.h_N, self.h_W], dtype=dtype, name="memory_ones")

        batch_range = tf.range(0, batch_size, delta=1, dtype=tf.int32, name="batch_range")
        repeat_memory_length = tf.fill([self.h_N], tf.constant(self.h_N, dtype=tf.int32), name="repeat_memory_length")
        batch_memory_range = tf.matmul(tf.expand_dims(batch_range, -1), tf.expand_dims(repeat_memory_length, 0),
                                       name="batch_memory_range")
        return link_matrix_inv_eye, memory_ones, batch_memory_range

    def _weight_input(self, inputs):

        input_size = inputs.get_shape()[1].value
        total_signal_size = (3 + self.h_RH) * self.h_W + 5 * self.h_RH + 3

        with tf.variable_scope('{}'.format(self.name), reuse=self.reuse):
            w_x = tf.get_variable("mu_w_x", (input_size, total_signal_size),
                                  initializer=tf.contrib.layers.xavier_initializer(seed=self.seed),
                                  collections=['memory_unit', tf.GraphKeys.GLOBAL_VARIABLES], dtype=self.dtype)

            b_x = tf.get_variable("mu_b_x", (total_signal_size,), initializer=tf.constant_initializer(0.),
                                  collections=['memory_unit', tf.GraphKeys.GLOBAL_VARIABLES], dtype=self.dtype)

            weighted_input = tf.matmul(inputs, w_x) + b_x

            if self.dnc_norm:
                weighted_input = layer_norm(weighted_input, name='layer_norm', dtype=self.dtype,
                                            collection='memory_unit')
        return weighted_input

    def _create_control_signals(self, weighted_input):

        write_keys = weighted_input[:, :         self.h_W]  # W
        write_strengths = weighted_input[:, self.h_W:         self.h_W + 1]  # 1
        erase_vector = weighted_input[:, self.h_W + 1:       2 * self.h_W + 1]  # W
        write_vector = weighted_input[:, 2 * self.h_W + 1:       3 * self.h_W + 1]  # W
        alloc_gates = weighted_input[:, 3 * self.h_W + 1:       3 * self.h_W + 2]  # 1
        write_gates = weighted_input[:, 3 * self.h_W + 2:       3 * self.h_W + 3]  # 1
        read_keys = weighted_input[:, 3 * self.h_W + 3: (self.h_RH + 3) * self.h_W + 3]  # R * W
        read_strengths = weighted_input[:,
                         (self.h_RH + 3) * self.h_W + 3: (self.h_RH + 3) * self.h_W + 3 + 1 * self.h_RH]  # R
        read_modes = weighted_input[:, (self.h_RH + 3) * self.h_W + 3 + 1 * self.h_RH: (
                                                                                           self.h_RH + 3) * self.h_W + 3 + 4 * self.h_RH]  # 3R
        free_gates = weighted_input[:, (self.h_RH + 3) * self.h_W + 3 + 4 * self.h_RH: (
                                                                                           self.h_RH + 3) * self.h_W + 3 + 5 * self.h_RH]  # R

        alloc_gates = tf.sigmoid(alloc_gates, 'alloc_gates')
        free_gates = tf.sigmoid(free_gates, 'free_gates')
        free_gates = tf.expand_dims(free_gates, 2)
        write_gates = tf.sigmoid(write_gates, 'write_gates')

        write_keys = tf.expand_dims(write_keys, axis=1)
        write_strengths = oneplus(write_strengths)
        # write_strengths = tf.expand_dims(write_strengths, axis=2)
        write_vector = tf.reshape(write_vector, [self.h_B, 1, self.h_W])
        erase_vector = tf.sigmoid(erase_vector, 'erase_vector')
        erase_vector = tf.reshape(erase_vector, [self.h_B, 1, self.h_W])

        read_keys = tf.reshape(read_keys, [self.h_B, self.h_RH, self.h_W])
        read_strengths = oneplus(read_strengths)
        read_strengths = tf.expand_dims(read_strengths, axis=2)
        read_modes = tf.reshape(read_modes, [self.h_B, self.h_RH, 3])  # 3 read modes
        read_modes = tf.nn.softmax(read_modes, dim=2)

        return alloc_gates, free_gates, write_gates, write_keys, write_strengths, write_vector, \
               erase_vector, read_keys, read_strengths, read_modes

    def _update_alloc_and_usage_vectors(self, pre_write_weightings, pre_read_weightings, pre_usage_vector, free_gates):

        retention_vector = tf.reduce_prod(1 - free_gates * pre_read_weightings, axis=1, keepdims=False,
                                          name='retention_prod')
        usage_vector = (
                           pre_usage_vector + pre_write_weightings - pre_usage_vector * pre_write_weightings) * retention_vector

        sorted_usage, free_list = tf.nn.top_k(-1 * usage_vector, self.h_N)
        sorted_usage = -1 * sorted_usage

        cumprod_sorted_usage = tf.cumprod(sorted_usage, axis=1, exclusive=True)
        corrected_free_list = free_list + self.const_batch_memory_range

        cumprod_sorted_usage_re = [tf.reshape(cumprod_sorted_usage, [-1, ]), ]
        corrected_free_list_re = [tf.reshape(corrected_free_list, [-1]), ]

        stitched_usage = tf.dynamic_stitch(corrected_free_list_re, cumprod_sorted_usage_re, name=None)

        stitched_usage = tf.reshape(stitched_usage, [self.h_B, self.h_N])

        alloc_weighting = (1 - usage_vector) * stitched_usage

        return alloc_weighting, usage_vector

    @staticmethod
    def _update_write_weighting(alloc_weighting, write_content_weighting, write_gate, alloc_gate):

        write_weighting = write_gate * (alloc_gate * alloc_weighting + (1 - alloc_gate) * write_content_weighting)

        return write_weighting

    def _update_memory(self, pre_memory, write_weighting, write_vector, erase_vector):

        write_w = tf.expand_dims(write_weighting, 2)
        erase_matrix = tf.multiply(pre_memory, (self.const_memory_ones - tf.matmul(write_w, erase_vector)))
        write_matrix = tf.matmul(write_w, write_vector)
        return erase_matrix + write_matrix

    def _update_link_matrix(self, pre_link_matrix, write_weighting, pre_precedence_weighting):

        precedence_weighting = (1 - tf.reduce_sum(write_weighting, 1,
                                                  keepdims=True)) * pre_precedence_weighting + write_weighting

        add_mat = tf.matmul(tf.expand_dims(write_weighting, axis=2),
                            tf.expand_dims(pre_precedence_weighting, axis=1))
        erase_mat = 1 - tf.expand_dims(write_weighting, 1) - tf.expand_dims(write_weighting, 2)

        updated_link_mat = erase_mat * pre_link_matrix + add_mat
        link_matrix = self.const_link_matrix_inv_eye * updated_link_mat

        return link_matrix, precedence_weighting

    @staticmethod
    def _make_read_forward_backward_weightings(link_matrix, pre_read_weightings):

        forward_weightings = tf.matmul(pre_read_weightings, link_matrix)
        backward_weightings = tf.matmul(pre_read_weightings, link_matrix, adjoint_b=True)

        return forward_weightings, backward_weightings

    @staticmethod
    def _make_read_weightings(forward_weightings, backward_weightings, read_content_weightings, read_modes):

        read_weighting = tf.expand_dims(read_modes[:, :, 0], 2) * backward_weightings + \
                         tf.expand_dims(read_modes[:, :, 1], 2) * read_content_weightings + \
                         tf.expand_dims(read_modes[:, :, 2], 2) * forward_weightings

        return read_weighting
