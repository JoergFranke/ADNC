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
import pytest
import time

from adnc.model.memory_units.content_based_cell import ContentBasedMemoryUnitCell


@pytest.fixture(
    params=[{"seed": 123, "input_size": 13, "batch_size": 3, "memory_length": 4, "memory_width": 4, "read_heads": 3,
             "dnc_norm": True, "bypass_dropout": False},
            {"seed": 124, "input_size": 11, "batch_size": 3, "memory_length": 256, "memory_width": 23, "read_heads": 2,
             "dnc_norm": False, "bypass_dropout": False},
            {"seed": 125, "input_size": 5, "batch_size": 3, "memory_length": 4, "memory_width": 11, "read_heads": 8,
             "dnc_norm": True, "bypass_dropout": True},
            {"seed": 126, "input_size": 2, "batch_size": 3, "memory_length": 56, "memory_width": 9, "read_heads": 11,
             "dnc_norm": False, "bypass_dropout": True}
            ])
def memory_config(request):
    config = request.param
    return ContentBasedMemoryUnitCell(input_size=config['input_size'], memory_length=config["memory_length"],
                                      memory_width=config["memory_width"],
                                      read_heads=config["read_heads"], seed=config["seed"],
                                      reuse=False, name='test_mu'), config


@pytest.fixture()
def session():
    with tf.Session() as sess:
        yield sess
    tf.reset_default_graph()


@pytest.fixture()
def np_rng():
    seed = np.random.randint(1, 999)
    return np.random.RandomState(seed)


class TestContentBasedMemoryUnitCell():
    def test_zero_state(self, memory_config, session, np_rng):
        memory_unit, config = memory_config

        init_tuple = memory_unit.zero_state(batch_size=config['batch_size'], dtype=tf.float32)

        # test init_tuple
        init_memory, init_usage_vector, init_write_weighting, init_read_weighting = init_tuple
        assert init_memory.eval().shape == (config['batch_size'], config['memory_length'], config['memory_width'])
        assert init_usage_vector.eval().shape == (config['batch_size'], config['memory_length'])
        assert init_write_weighting.eval().shape == (config['batch_size'], config['memory_length'])
        assert init_read_weighting.eval().shape == (config['batch_size'], config["read_heads"], config['memory_length'])

    def test_parameter_amount(self, memory_config, session, np_rng):
        memory_unit, config = memory_config

        total_signal_size = (config['memory_width'] * (3 + config["read_heads"]) + 2 * config['read_heads'] + 3)

        inputs = np.ones([config['batch_size'], config['input_size']])
        tf_input = tf.constant(inputs, tf.float32)

        memory_unit._weight_input(tf_input)
        parameter_amount = memory_unit.parameter_amount

        assert parameter_amount == (config['input_size'] + 1) * total_signal_size

    def test_create_constant_value_tensors(self, memory_config, session, np_rng):
        memory_unit, config = memory_config
        memory_ones, batch_memory_range = memory_unit._create_constant_value_tensors(batch_size=config['batch_size'],
                                                                                     dtype=tf.float32)

        np_memory_ones = np.ones([config['batch_size'], config['memory_length'], config['memory_width']])
        assert np.array_equal(memory_ones.eval(), np_memory_ones)

        np_batch_range = np.arange(0, config['batch_size'])
        np_repeat_memory_length = np.repeat(config['memory_length'], config['memory_length'])
        np_batch_memory_range = np.matmul(np.expand_dims(np_batch_range, axis=-1),
                                          np.expand_dims(np_repeat_memory_length, 0))
        assert np.array_equal(batch_memory_range.eval(), np_batch_memory_range)

    def test_weight_input(self, memory_config, session, np_rng):
        memory_unit, config = memory_config

        inputs = np.ones([config['batch_size'], config['input_size']])
        tf_input = tf.placeholder(tf.float32, [config['batch_size'], config['input_size']], name='x')

        weight_inputs = memory_unit._weight_input(tf_input)
        session.run(tf.global_variables_initializer())
        np_weight_inputs = weight_inputs.eval(session=session, feed_dict={tf_input: inputs})

        total_signal_size = (config['memory_width'] * (3 + config["read_heads"]) + 2 * config['read_heads'] + 3)
        assert np_weight_inputs.shape == (config['batch_size'], total_signal_size)

    def test_create_control_signals(self, memory_config, session, np_rng):
        memory_unit, config = memory_config

        total_signal_size = (config['memory_width'] * (3 + config["read_heads"]) + 5 * config['read_heads'] + 3)
        np_weighted_input = np.array([np.arange(1, 1 + total_signal_size)] * config['batch_size'])

        weighted_input = tf.constant(np_weighted_input, dtype=tf.float32)

        memory_unit.h_B = config['batch_size']
        control_signals = memory_unit._create_control_signals(weighted_input)
        control_signals = session.run(control_signals)

        alloc_gates, free_gates, write_gates, write_keys, write_strengths, write_vector, \
        erase_vector, read_keys, read_strengths = control_signals

        assert alloc_gates.shape == (config['batch_size'], 1)
        assert 0 <= alloc_gates.min() and alloc_gates.max() <= 1
        assert free_gates.shape == (config['batch_size'], config['read_heads'], 1)
        assert 0 <= free_gates.min() and free_gates.max() <= 1
        assert write_gates.shape == (config['batch_size'], 1)
        assert 0 <= write_gates.min() and write_gates.max() <= 1

        assert write_keys.shape == (config['batch_size'], 1, config['memory_width'])
        assert write_strengths.shape == (config['batch_size'], 1)
        assert 1 <= write_strengths.min()
        assert write_vector.shape == (config['batch_size'], 1, config['memory_width'])
        assert erase_vector.shape == (config['batch_size'], 1, config['memory_width'])
        assert 0 <= erase_vector.min() and erase_vector.max() <= 1
        # comment
        assert read_keys.shape == (config['batch_size'], config['read_heads'], config['memory_width'])
        assert read_strengths.shape == (config['batch_size'], config['read_heads'], 1)
        assert 1 <= read_strengths.min()

    def test_update_alloc_weightings_and_usage_vectors(self, memory_config, session, np_rng):
        memory_unit, config = memory_config

        np_pre_write_weightings = np_rng.uniform(0, 1 / config['memory_length'],
                                                 [config['batch_size'], config['memory_length']])
        np_pre_usage_vectors = np_rng.uniform(0, 1 / config['memory_length'],
                                              [config['batch_size'], config['memory_length']])
        np_free_gates = np.ones([config['batch_size'], config['read_heads'], 1]) * 0.5

        pre_write_weightings = tf.constant(np_pre_write_weightings, dtype=tf.float32)
        pre_usage_vectors = tf.constant(np_pre_usage_vectors, dtype=tf.float32)
        free_gates = tf.constant(np_free_gates, dtype=tf.float32)

        np_inputs = np_rng.normal(0, 1, [config['batch_size'], config['input_size']])
        np_pre_memory = np_rng.normal(0, 1, [config['batch_size'], config['memory_length'], config['memory_width']])
        np_pre_usage_vector = np_rng.uniform(0, 1 / config['memory_length'],
                                             [config['batch_size'], config['memory_length']])
        np_pre_write_weighting = np_rng.uniform(0, 1 / config['memory_length'],
                                                [config['batch_size'], config['memory_length']])
        np_pre_read_weightings = np_rng.uniform(0, 1 / config['memory_length'],
                                                [config['batch_size'], config['read_heads'], config['memory_length']])

        inputs = tf.constant(np_inputs, dtype=tf.float32)
        pre_memory = tf.constant(np_pre_memory, dtype=tf.float32)
        pre_usage_vector = tf.constant(np_pre_usage_vector, dtype=tf.float32)
        pre_write_weighting = tf.constant(np_pre_write_weighting, dtype=tf.float32)
        pre_read_weightings = tf.constant(np_pre_read_weightings, dtype=tf.float32)

        pre_states = (pre_memory, pre_usage_vector, pre_write_weighting, pre_read_weightings)

        memory_unit.zero_state(config['batch_size'])
        memory_unit(inputs, pre_states)  # just for initialization

        alloc_weightings, usage_vectors = memory_unit._update_alloc_and_usage_vectors(pre_write_weightings,
                                                                                      pre_read_weightings,
                                                                                      pre_usage_vectors, free_gates)
        alloc_weightings, usage_vectors = session.run([alloc_weightings, usage_vectors])

        np_retention_vector = np.prod(1 - np_free_gates * np_pre_read_weightings, axis=1, keepdims=False)
        np_usage_vectors = (
                           np_pre_usage_vectors + np_pre_write_weightings - np_pre_usage_vectors * np_pre_write_weightings) * np_retention_vector

        assert usage_vectors.shape == (config['batch_size'], config['memory_length'])
        assert usage_vectors.min() >= 0 and usage_vectors.max() <= 1
        assert np.allclose(usage_vectors, np_usage_vectors)

        free_list = np.argsort(np_usage_vectors).astype(int)

        np_alloc_weightings = np.zeros([config['batch_size'], config['memory_length']])

        for b in range(config['batch_size']):
            for j in range(config['memory_length']):
                fj = free_list[b, j]
                np_alloc_weightings[b, fj] = (1 - np_usage_vectors[b, fj]) * np.prod(
                    [np_usage_vectors[b, free_list[b, i]] for i in range(j)])

        assert alloc_weightings.shape == (config['batch_size'], config['memory_length'])
        assert np.allclose(alloc_weightings, np_alloc_weightings)

    def test_update_write_weighting(self, memory_config, session, np_rng):
        memory_unit, config = memory_config

        np_alloc_weighting = np_rng.uniform(0, 1 / config['memory_length'],
                                            [config['batch_size'], config['memory_length']])
        np_write_content_weighting = np_rng.uniform(0, 1 / config['memory_length'],
                                                    [config['batch_size'], config['memory_length']])
        np_write_gate = np.ones([config['batch_size'], 1]) * 0.5
        np_alloc_gate = np.ones([config['batch_size'], 1]) * 0.5

        alloc_weighting = tf.constant(np_alloc_weighting, dtype=tf.float32)
        write_content_weighting = tf.constant(np_write_content_weighting, dtype=tf.float32)
        write_gate = tf.constant(np_write_gate, dtype=tf.float32)
        alloc_gate = tf.constant(np_alloc_gate, dtype=tf.float32)

        write_weighting = memory_unit._update_write_weighting(alloc_weighting, write_content_weighting, write_gate,
                                                              alloc_gate)
        write_weighting = write_weighting.eval()

        np_write_weighting = np_write_gate * (
        np_alloc_gate * np_alloc_weighting + (1 - np_alloc_gate) * np_write_content_weighting)

        assert write_weighting.shape == (config['batch_size'], config['memory_length'])
        assert 0 <= write_weighting.min() and write_weighting.max() <= 1 and write_weighting.sum(axis=1).all() <= 1
        assert np.allclose(write_weighting, np_write_weighting)

    def test_update_memory(self, memory_config, session, np_rng):
        memory_unit, config = memory_config

        np_write_weighting = np_rng.uniform(0, 1 / config['memory_length'],
                                            [config['batch_size'], config['memory_length']])
        np_write_vector = np_rng.normal(0, 2, [config['batch_size'], 1, config['memory_width']])
        np_erase_vector = np_rng.uniform(0, 1, [config['batch_size'], 1, config['memory_width']])

        write_weighting = tf.constant(np_write_weighting, dtype=tf.float32)
        write_vector = tf.constant(np_write_vector, dtype=tf.float32)
        erase_vector = tf.constant(np_erase_vector, dtype=tf.float32)

        np_inputs = np_rng.normal(0, 1, [config['batch_size'], config['input_size']])
        np_pre_memory = np_rng.normal(0, 1, [config['batch_size'], config['memory_length'], config['memory_width']])
        np_pre_usage_vector = np_rng.uniform(0, 1 / config['memory_length'],
                                             [config['batch_size'], config['memory_length']])
        np_pre_write_weighting = np_rng.uniform(0, 1 / config['memory_length'],
                                                [config['batch_size'], config['memory_length']])
        np_pre_read_weightings = np_rng.uniform(0, 1 / config['memory_length'],
                                                [config['batch_size'], config['read_heads'], config['memory_length']])

        inputs = tf.constant(np_inputs, dtype=tf.float32)
        pre_memory = tf.constant(np_pre_memory, dtype=tf.float32)
        pre_usage_vector = tf.constant(np_pre_usage_vector, dtype=tf.float32)
        pre_write_weighting = tf.constant(np_pre_write_weighting, dtype=tf.float32)
        pre_read_weightings = tf.constant(np_pre_read_weightings, dtype=tf.float32)

        pre_states = (pre_memory, pre_usage_vector, pre_write_weighting, pre_read_weightings)

        memory_unit.zero_state(config['batch_size'])
        memory_unit(inputs, pre_states)  # just for initialization

        memory = memory_unit._update_memory(pre_memory, write_weighting, write_vector, erase_vector)
        memory = memory.eval()

        write_w = np.expand_dims(np_write_weighting, 2)
        np_erase_memory = (1 - (write_w * np_erase_vector))
        np_add_memory = np.matmul(write_w, np_write_vector)
        np_memory = np_pre_memory * np_erase_memory + np_add_memory

        assert memory.shape == (config['batch_size'], config['memory_length'], config['memory_width'])
        assert np.allclose(memory, np_memory, atol=1e-06)

    def test_read_memory(self, memory_config, session, np_rng):
        memory_unit, config = memory_config

        np_memory = np_rng.normal(0, 1, [config['batch_size'], config['memory_length'], config['memory_width']])
        np_read_weightings = np_rng.uniform(0, 1 / config['memory_length'],
                                            [config['batch_size'], config['read_heads'], config['memory_length']])

        memory = tf.constant(np_memory, dtype=tf.float32)
        read_weightings = tf.constant(np_read_weightings, dtype=tf.float32)

        read_vectors = memory_unit._read_memory(memory, read_weightings)
        read_vectors = read_vectors.eval()

        np_read_vectors = np.empty([config['batch_size'], config['read_heads'], config['memory_width']])
        for b in range(config['batch_size']):
            for r in range(config['read_heads']):
                np_read_vectors[b, r, :] = np.matmul(np.expand_dims(np_read_weightings[b, r, :], 0), np_memory[b, :, :])

        assert read_vectors.shape == (config['batch_size'], config['read_heads'], config['memory_width'])
        assert np.allclose(read_vectors, np_read_vectors, atol=1e-06)

    def test_call(self, memory_config, session, np_rng):
        memory_unit, config = memory_config

        np_inputs = np_rng.normal(0, 1, [config['batch_size'], config['input_size']])
        np_pre_memory = np_rng.normal(0, 1, [config['batch_size'], config['memory_length'], config['memory_width']])
        np_pre_usage_vector = np_rng.uniform(0, 1 / config['memory_length'],
                                             [config['batch_size'], config['memory_length']])
        np_pre_write_weighting = np_rng.uniform(0, 1 / config['memory_length'],
                                                [config['batch_size'], config['memory_length']])
        np_pre_read_weightings = np_rng.uniform(0, 1 / config['memory_length'],
                                                [config['batch_size'], config['read_heads'], config['memory_length']])

        inputs = tf.constant(np_inputs, dtype=tf.float32)
        pre_memory = tf.constant(np_pre_memory, dtype=tf.float32)
        pre_usage_vector = tf.constant(np_pre_usage_vector, dtype=tf.float32)
        pre_write_weighting = tf.constant(np_pre_write_weighting, dtype=tf.float32)
        pre_read_weightings = tf.constant(np_pre_read_weightings, dtype=tf.float32)

        pre_states = (pre_memory, pre_usage_vector, pre_write_weighting, pre_read_weightings)

        memory_unit.zero_state(config['batch_size'])
        read_vectors, states = memory_unit(inputs, pre_states)

        session.run(tf.global_variables_initializer())
        read_vectors, states = session.run([read_vectors, states])

        # test const initialization
        np_memory_ones = np.ones([config['batch_size'], config['memory_length'], config['memory_width']])
        assert np.array_equal(memory_unit.const_memory_ones.eval(), np_memory_ones)
        np_batch_range = np.arange(0, config['batch_size'])
        np_repeat_memory_length = np.repeat(config['memory_length'], config['memory_length'])
        np_batch_memory_range = np.matmul(np.expand_dims(np_batch_range, axis=-1),
                                          np.expand_dims(np_repeat_memory_length, 0))
        assert np.array_equal(memory_unit.const_batch_memory_range.eval(), np_batch_memory_range)

        assert read_vectors.shape == (
        config['batch_size'], config['memory_width'] * config['read_heads'] + config['input_size'])
