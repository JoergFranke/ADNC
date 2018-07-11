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
from adnc.model.utils import layer_norm
from adnc.model.utils import oneplus
from adnc.model.utils import unit_simplex_initialization

"""
The vanilla DNC memory unit with multi write heads.
"""

class MWDNCMemoryUnitCell(BaseMemoryUnitCell):
    def __init__(self, input_size, memory_length, memory_width, read_heads, write_heads, bypass_dropout=False,
                 dnc_norm=False, seed=100, reuse=False, analyse=False, dtype=tf.float32, name='mwdnc_mu'):

        self.h_WH = write_heads
        super().__init__(input_size, memory_length, memory_width, read_heads, bypass_dropout, dnc_norm, seed, reuse,
                         analyse, dtype, name)

    @property
    def state_size(self):
        init_memory = tf.TensorShape([self.h_N, self.h_W])
        init_usage_vector = tf.TensorShape([self.h_N])
        init_write_weighting = tf.TensorShape([self.h_WH, self.h_N])
        init_precedence_weightings = tf.TensorShape([self.h_WH, self.h_N])
        init_link_mat = tf.TensorShape([self.h_WH, self.h_N, self.h_N])
        init_read_weighting = tf.TensorShape([self.h_RH, self.h_N])
        return (init_memory, init_usage_vector, init_write_weighting, init_precedence_weightings,
                init_link_mat, init_read_weighting)

    def zero_state(self, batch_size, dtype=tf.float32):

        init_memory = tf.fill([batch_size, self.h_N, self.h_W], tf.cast(1 / (self.h_N * self.h_W), dtype=dtype))
        init_usage_vector = tf.zeros([batch_size, self.h_N], dtype=dtype)
        init_write_weighting = unit_simplex_initialization(self.rng, batch_size, [self.h_WH, self.h_N], dtype=dtype)
        init_precedence_weightings = tf.zeros([batch_size, self.h_WH, self.h_N], dtype=dtype)
        init_link_mat = tf.zeros([batch_size, self.h_WH, self.h_N, self.h_N], dtype=dtype)
        init_read_weighting = unit_simplex_initialization(self.rng, batch_size, [self.h_RH, self.h_N], dtype=dtype)
        zero_states = (init_memory, init_usage_vector, init_write_weighting, init_precedence_weightings,
                       init_link_mat, init_read_weighting,)
        return zero_states

    def analyse_state(self, batch_size, dtype=tf.float32):

        alloc_gate = tf.zeros([batch_size, self.h_WH, 1], dtype=dtype)  # WH
        free_gates = tf.zeros([batch_size, self.h_RH, 1], dtype=dtype)
        write_gate = tf.zeros([batch_size, self.h_WH, 1], dtype=dtype)
        write_keys = tf.zeros([batch_size, self.h_WH, self.h_W], dtype=dtype)
        write_strengths = tf.zeros([batch_size, self.h_WH, 1], dtype=dtype)
        write_vector = tf.zeros([batch_size, self.h_WH, self.h_W], dtype=dtype)
        erase_vector = tf.zeros([batch_size, self.h_WH, self.h_W], dtype=dtype)
        read_keys = tf.zeros([batch_size, self.h_RH, self.h_W], dtype=dtype)
        read_strengths = tf.zeros([batch_size, self.h_RH, 1], dtype=dtype)
        read_modes = tf.zeros([batch_size, self.h_RH, 1 + 2 * self.h_WH], dtype=dtype)

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
                                                                              pre_usage_vector, free_gates, write_gate)
        write_content_weighting = self._calculate_content_weightings(pre_memory, write_keys, write_strengths)
        write_weighting = self._update_write_weightings(alloc_weightings, write_content_weighting, write_gate,
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
        link_matrix_inv_eye = tf.stack([link_matrix_inv_eye, ] * self.h_WH, axis=0)
        link_matrix_inv_eye = tf.stack([link_matrix_inv_eye, ] * batch_size, axis=0)

        memory_ones = tf.ones([batch_size, self.h_N, self.h_W], dtype=dtype, name="memory_ones")

        batch_range = tf.range(0, batch_size, delta=1, dtype=tf.int32, name="batch_range")
        repeat_memory_length = tf.fill([self.h_N], tf.constant(self.h_N, dtype=tf.int32), name="repeat_memory_length")
        batch_memory_range = tf.matmul(tf.expand_dims(batch_range, -1), tf.expand_dims(repeat_memory_length, 0),
                                       name="batch_memory_range")
        return link_matrix_inv_eye, memory_ones, batch_memory_range

    def _weight_input(self, inputs):

        input_size = inputs.get_shape()[1].value
        total_signal_size = self.h_RH * (3 + 2 * self.h_WH + self.h_W) + self.h_WH * (3 + 3 * self.h_W)

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

        alloc_gates = weighted_input[:, :                  self.h_WH]
        free_gates = weighted_input[:, self.h_WH:                  self.h_WH + self.h_RH]
        write_gates = weighted_input[:, self.h_WH + self.h_RH:                2 * self.h_WH + self.h_RH]
        write_keys = weighted_input[:, 2 * self.h_WH + self.h_RH:   (self.h_W + 2) * self.h_WH + self.h_RH]
        write_strengths = weighted_input[:,
                          (self.h_W + 2) * self.h_WH + self.h_RH:   (self.h_W + 3) * self.h_WH + self.h_RH]
        write_vectors = weighted_input[:,
                        (self.h_W + 3) * self.h_WH + self.h_RH: (2 * self.h_W + 3) * self.h_WH + self.h_RH]
        erase_vectors = weighted_input[:,
                        (2 * self.h_W + 3) * self.h_WH + self.h_RH: (3 * self.h_W + 3) * self.h_WH + self.h_RH]
        read_keys = weighted_input[:, (3 * self.h_W + 3) * self.h_WH + self.h_RH: (3 * self.h_W + 3) * self.h_WH +
                                      (self.h_W + 1) * self.h_RH]
        read_strengths = weighted_input[:,
                         (3 * self.h_W + 3) * self.h_WH + (self.h_W + 1) * self.h_RH: (3 * self.h_W + 3) * self.h_WH +
                         (self.h_W + 2) * self.h_RH]
        read_modes = weighted_input[:, (3 * self.h_W + 3) * self.h_WH + (self.h_W + 2) * self.h_RH:]

        alloc_gates = tf.sigmoid(alloc_gates, 'alloc_gates')
        alloc_gates = tf.expand_dims(alloc_gates, 2)
        free_gates = tf.sigmoid(free_gates, 'free_gates')
        free_gates = tf.expand_dims(free_gates, 2)
        write_gates = tf.sigmoid(write_gates, 'write_gates')
        write_gates = tf.expand_dims(write_gates, 2)

        write_keys = tf.reshape(write_keys, [self.h_B, self.h_WH, self.h_W])
        write_strengths = oneplus(write_strengths)
        write_strengths = tf.expand_dims(write_strengths, axis=2)
        write_vectors = tf.reshape(write_vectors, [self.h_B, self.h_WH, self.h_W])
        erase_vectors = tf.reshape(erase_vectors, [self.h_B, self.h_WH, self.h_W])
        erase_vectors = tf.sigmoid(erase_vectors, 'erase_vector')

        read_keys = tf.reshape(read_keys, [self.h_B, self.h_RH, self.h_W])
        read_strengths = oneplus(read_strengths)
        read_strengths = tf.expand_dims(read_strengths, axis=2)
        read_modes = tf.reshape(read_modes, [self.h_B, self.h_RH, 1 + 2 * self.h_WH])
        read_modes = tf.nn.softmax(read_modes, axis=2)

        return alloc_gates, free_gates, write_gates, write_keys, write_strengths, write_vectors, \
               erase_vectors, read_keys, read_strengths, read_modes

    def _update_alloc_and_usage_vectors(self, pre_write_weightings, pre_read_weightings, pre_usage_vector, free_gates,
                                        write_gates):

        # usage update after write from last time step
        pre_write_weighting = 1 - tf.reduce_prod(1 - pre_write_weightings, [1], keepdims=False)
        usage_vector = pre_usage_vector + pre_write_weighting - pre_usage_vector * pre_write_weighting

        # usage update after read
        retention_vector = tf.reduce_prod(1 - free_gates * pre_read_weightings, axis=1, keepdims=False,
                                          name='retention_prod')
        usage_vector = usage_vector * retention_vector

        usage_vector_cp = tf.identity(usage_vector)

        alloc_list = []
        for w in range(self.h_WH):
            sorted_usage, free_list = tf.nn.top_k(-1 * usage_vector_cp, self.h_N)
            sorted_usage = -1 * sorted_usage

            cumprod_sorted_usage = tf.cumprod(sorted_usage, axis=1, exclusive=True)
            corrected_free_list = free_list + self.const_batch_memory_range

            corrected_free_list_un = [tf.reshape(corrected_free_list, [-1, ]), ]
            cumprod_sorted_usage_un = [tf.reshape(cumprod_sorted_usage, [-1, ]), ]

            stitched_usage = tf.dynamic_stitch(corrected_free_list_un, cumprod_sorted_usage_un, name=None)
            stitched_usage = tf.reshape(stitched_usage, [self.h_B, self.h_N])

            alloc_weighting = (1 - usage_vector_cp) * stitched_usage

            alloc_list.append(alloc_weighting)
            usage_vector_cp = usage_vector_cp + ((1 - usage_vector_cp) * write_gates[:, w, :] * alloc_weighting)

        alloc_weighting = tf.stack(alloc_list, 1)

        return alloc_weighting, usage_vector

    @staticmethod
    def _update_write_weightings(alloc_weighting, write_content_weighting, write_gate, alloc_gate):

        write_weighting = write_gate * (alloc_gate * alloc_weighting + (1 - alloc_gate) * write_content_weighting)

        return write_weighting

    @staticmethod
    def _update_memory(pre_memory, write_weighting, write_vector, erase_vector):

        write_w = tf.expand_dims(write_weighting, 3)
        erase_vector = tf.expand_dims(erase_vector, 2)
        erase_matrix = tf.reduce_prod(1 - write_w * erase_vector, axis=1, keepdims=False)
        write_matrix = tf.matmul(write_weighting, write_vector, adjoint_a=True)

        return pre_memory * erase_matrix + write_matrix

    def _update_link_matrix(self, pre_link_matrices, write_weightings, pre_precedence_weightings):

        precedence_weightings = (1 - tf.reduce_sum(write_weightings, 2,
                                                   keepdims=True)) * pre_precedence_weightings + write_weightings

        add_mat = tf.expand_dims(write_weightings, axis=3) * tf.expand_dims(pre_precedence_weightings, axis=2)
        erase_mat = 1 - tf.expand_dims(write_weightings, 2) - tf.expand_dims(write_weightings, 3)

        updated_link_mat = erase_mat * pre_link_matrices + add_mat

        link_matrices = self.const_link_matrix_inv_eye * updated_link_mat

        return link_matrices, precedence_weightings

    def _make_read_forward_backward_weightings(self, link_matrix, pre_read_weightings):

        read_weightings_stacked = tf.stack([pre_read_weightings, ] * self.h_WH, axis=1)

        forward_weightings = tf.matmul(read_weightings_stacked, link_matrix)
        backward_weightings = tf.matmul(read_weightings_stacked, link_matrix, adjoint_b=True)

        return tf.transpose(forward_weightings, (0, 2, 1, 3)), tf.transpose(backward_weightings, (0, 2, 1, 3))

    def _make_read_weightings(self, forward_weightings, backward_weightings, read_content_weightings, read_modes):

        read_weighting = tf.reduce_sum(tf.expand_dims(read_modes[:, :, :self.h_WH], 3) * backward_weightings, axis=2) + \
                         tf.expand_dims(read_modes[:, :, self.h_WH], 2) * read_content_weightings + \
                         tf.reduce_sum(tf.expand_dims(read_modes[:, :, self.h_WH + 1:], 3) * forward_weightings, axis=2)

        return read_weighting
