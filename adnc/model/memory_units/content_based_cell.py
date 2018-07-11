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
import tensorflow as tf

from adnc.model.memory_units.dnc_cell import DNCMemoryUnitCell
from adnc.model.utils import oneplus, layer_norm, unit_simplex_initialization

"""
The content-based memory unit.
"""

class ContentBasedMemoryUnitCell(DNCMemoryUnitCell):

    @property
    def state_size(self):
        init_memory = tf.TensorShape([self.h_N, self.h_W])
        init_usage_vector = tf.TensorShape([self.h_N])
        init_write_weighting = tf.TensorShape([self.h_N])
        init_read_weighting = tf.TensorShape([self.h_RH, self.h_N])
        return (init_memory, init_usage_vector, init_write_weighting, init_read_weighting)

    def zero_state(self, batch_size, dtype=tf.float32):

        init_memory = tf.fill([batch_size, self.h_N, self.h_W], tf.cast(1 / (self.h_N * self.h_W), dtype=dtype))
        init_usage_vector = tf.zeros([batch_size, self.h_N], dtype=dtype)
        init_write_weighting = unit_simplex_initialization(self.rng, batch_size, [self.h_N], dtype=dtype)
        init_read_weighting = unit_simplex_initialization(self.rng, batch_size, [self.h_RH, self.h_N], dtype=dtype)
        zero_states = (init_memory, init_usage_vector, init_write_weighting, init_read_weighting,)

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

        analyse_states = alloc_gate, free_gates, write_gate, write_keys, write_strengths, write_vector, \
                         erase_vector, read_keys, read_strengths

        return analyse_states

    def _weight_input(self, inputs):

        input_size = inputs.get_shape()[1].value
        total_signal_size = (3 + self.h_RH) * self.h_W + 2 * self.h_RH + 3

        with tf.variable_scope('{}'.format(self.name), reuse=self.reuse):
            w_x = tf.get_variable("mu_w_x", (input_size, total_signal_size),
                                  initializer=tf.contrib.layers.xavier_initializer(seed=self.seed),
                                  collections=['memory_unit', tf.GraphKeys.GLOBAL_VARIABLES], dtype=self.dtype)

            b_x = tf.get_variable("mu_b_x", (total_signal_size,), initializer=tf.constant_initializer(0.),
                                  collections=['memory_unit', tf.GraphKeys.GLOBAL_VARIABLES], dtype=self.dtype)

            weighted_input = tf.matmul(inputs, w_x) + b_x

            if self.dnc_norm:
                weighted_input = layer_norm(weighted_input, name='layer_norm', dtype=self.dtype)

        return weighted_input

    def __call__(self, inputs, pre_states, scope=None):

        self.h_B = inputs.get_shape()[0].value

        memory_ones, batch_memory_range = self._create_constant_value_tensors(self.h_B, self.dtype)
        self.const_memory_ones = memory_ones
        self.const_batch_memory_range = batch_memory_range

        pre_memory, pre_usage_vector, pre_write_weightings, pre_read_weightings = pre_states

        weighted_input = self._weight_input(inputs)

        control_signals = self._create_control_signals(weighted_input)
        alloc_gate, free_gates, write_gate, write_keys, write_strengths, write_vector, \
        erase_vector, read_keys, read_strengths = control_signals

        alloc_weightings, usage_vector = self._update_alloc_and_usage_vectors(pre_write_weightings, pre_read_weightings,
                                                                              pre_usage_vector, free_gates)
        write_content_weighting = self._calculate_content_weightings(pre_memory, write_keys, write_strengths)
        write_weighting = self._update_write_weighting(alloc_weightings, write_content_weighting, write_gate,
                                                       alloc_gate)
        memory = self._update_memory(pre_memory, write_weighting, write_vector, erase_vector)
        read_content_weightings = self._calculate_content_weightings(memory, read_keys, read_strengths)
        read_vectors = self._read_memory(memory, read_content_weightings)

        read_vectors = tf.reshape(read_vectors, [self.h_B, self.h_W * self.h_RH])

        if self.bypass_dropout:
            input_bypass = tf.nn.dropout(inputs, self.bypass_dropout)
        else:
            input_bypass = inputs

        output = tf.concat([read_vectors, input_bypass], axis=-1)

        if self.analyse:
            output = (output, control_signals)

        return output, (memory, usage_vector, write_weighting, read_content_weightings)

    def _create_constant_value_tensors(self, batch_size, dtype):

        memory_ones = tf.ones([batch_size, self.h_N, self.h_W], dtype=dtype, name="memory_ones")

        batch_range = tf.range(0, batch_size, delta=1, dtype=tf.int32, name="batch_range")
        repeat_memory_length = tf.fill([self.h_N], tf.constant(self.h_N, dtype=tf.int32), name="repeat_memory_length")
        batch_memory_range = tf.matmul(tf.expand_dims(batch_range, -1), tf.expand_dims(repeat_memory_length, 0),
                                       name="batch_memory_range")
        return memory_ones, batch_memory_range

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
        free_gates = weighted_input[:, (self.h_RH + 3) * self.h_W + 3 + 1 * self.h_RH: (
                                                                                           self.h_RH + 3) * self.h_W + 3 + 2 * self.h_RH]

        alloc_gates = tf.sigmoid(alloc_gates, 'alloc_gates')
        free_gates = tf.sigmoid(free_gates, 'free_gates')
        free_gates = tf.expand_dims(free_gates, 2)
        write_gates = tf.sigmoid(write_gates, 'write_gates')

        write_keys = tf.expand_dims(write_keys, axis=1)
        write_strengths = oneplus(write_strengths)
        write_vector = tf.reshape(write_vector, [self.h_B, 1, self.h_W])
        erase_vector = tf.sigmoid(erase_vector, 'erase_vector')
        erase_vector = tf.reshape(erase_vector, [self.h_B, 1, self.h_W])

        read_keys = tf.reshape(read_keys, [self.h_B, self.h_RH, self.h_W])
        read_strengths = oneplus(read_strengths)
        read_strengths = tf.expand_dims(read_strengths, axis=2)

        return alloc_gates, free_gates, write_gates, write_keys, write_strengths, write_vector, \
               erase_vector, read_keys, read_strengths
