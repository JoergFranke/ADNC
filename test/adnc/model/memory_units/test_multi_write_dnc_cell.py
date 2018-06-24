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

from adnc.model.memory_units.multi_write_dnc_cell import MWDNCMemoryUnitCell


@pytest.fixture(
    params=[{"seed": 123, "input_size": 13, "batch_size": 3, "memory_length": 4, "memory_width": 4, "read_heads": 3,
             "write_heads": 3, "dnc_norm": True, "bypass_dropout": False},
            {"seed": 124, "input_size": 11, "batch_size": 3, "memory_length": 256, "memory_width": 23, "read_heads": 2,
             "write_heads": 2, "dnc_norm": False, "bypass_dropout": False},
            {"seed": 125, "input_size": 5, "batch_size": 3, "memory_length": 4, "memory_width": 11, "read_heads": 8,
             "write_heads": 5, "dnc_norm": True, "bypass_dropout": True},
            {"seed": 126, "input_size": 2, "batch_size": 3, "memory_length": 56, "memory_width": 9, "read_heads": 11,
             "write_heads": 9, "dnc_norm": False, "bypass_dropout": True}
            ])
def memory_config(request):
    config = request.param
    return MWDNCMemoryUnitCell(input_size=config['input_size'], memory_length=config["memory_length"],
                               memory_width=config["memory_width"], write_heads=config["write_heads"],
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


class TestMWDNCMemoryUnit():
    def test_init(self, memory_config, session, np_rng):
        memory_unit, config = memory_config
        assert isinstance(memory_unit, object)
        assert isinstance(memory_unit.rng, np.random.RandomState)

        assert memory_unit.h_N == config["memory_length"]
        assert memory_unit.h_W == config["memory_width"]
        assert memory_unit.h_RH == config["read_heads"]
        assert memory_unit.h_WH == config["write_heads"]

    def test_parameter_amount(self, memory_config, session, np_rng):
        memory_unit, config = memory_config

        total_signal_size = (
            config['memory_width'] * (3 * config["write_heads"] + config["read_heads"]) + 3 * config['read_heads'] + 3 *
            config["write_heads"] + 2 * config['read_heads'] * config["write_heads"])

        inputs = np.ones([config['batch_size'], config['input_size']])
        tf_input = tf.constant(inputs, tf.float32)

        memory_unit._weight_input(tf_input)
        parameter_amount = memory_unit.parameter_amount

        assert parameter_amount == (config['input_size'] + 1) * total_signal_size

    def test_create_constant_value_tensors(self, memory_config, session, np_rng):
        memory_unit, config = memory_config

        link_matrix_inv_eye, memory_ones, batch_memory_range = memory_unit._create_constant_value_tensors(
            batch_size=config['batch_size'], dtype=tf.float32)

        np_link_matrix_inv_eye = np.ones([config['memory_length'], config['memory_length']]) - np.eye(
            config['memory_length'])
        np_link_matrix_inv_eye = np.stack([np_link_matrix_inv_eye, ] * config["write_heads"], axis=0)
        np_link_matrix_inv_eye = np.stack([np_link_matrix_inv_eye, ] * config['batch_size'], axis=0)

        assert np.array_equal(link_matrix_inv_eye.eval(), np_link_matrix_inv_eye)

        np_memory_ones = np.ones([config['batch_size'], config['memory_length'], config['memory_width']])
        assert np.array_equal(memory_ones.eval(), np_memory_ones)

        np_batch_range = np.arange(0, config['batch_size'])
        np_repeat_memory_length = np.repeat(config['memory_length'], config['memory_length'])
        np_batch_memory_range = np.matmul(np.expand_dims(np_batch_range, axis=-1),
                                          np.expand_dims(np_repeat_memory_length, 0))
        assert np.array_equal(batch_memory_range.eval(), np_batch_memory_range)

    def test_zero_state(self, memory_config, session, np_rng):
        memory_unit, config = memory_config

        init_tuple = memory_unit.zero_state(batch_size=config['batch_size'], dtype=tf.float32)

        # test init_tuple
        init_memory, init_usage_vector, init_write_weighting, init_precedence_weightings, init_link_mat, init_read_weighting = init_tuple
        assert init_memory.eval().shape == (config['batch_size'], config['memory_length'], config['memory_width'])
        assert init_usage_vector.eval().shape == (config['batch_size'], config['memory_length'])
        assert init_write_weighting.eval().shape == (
            config['batch_size'], config["write_heads"], config['memory_length'])
        assert init_precedence_weightings.eval().shape == (
            config['batch_size'], config['write_heads'], config['memory_length'])
        assert init_link_mat.eval().shape == (
            config['batch_size'], config['write_heads'], config['memory_length'], config['memory_length'])
        assert init_read_weighting.eval().shape == (config['batch_size'], config["read_heads"], config['memory_length'])

    def test_weight_input(self, memory_config, session, np_rng):
        memory_unit, config = memory_config

        inputs = np.ones([config['batch_size'], config['input_size']])
        tf_input = tf.placeholder(tf.float32, [config['batch_size'], config['input_size']], name='x')

        weight_inputs = memory_unit._weight_input(tf_input)
        session.run(tf.global_variables_initializer())
        np_weight_inputs = weight_inputs.eval(session=session, feed_dict={tf_input: inputs})

        total_signal_size = (
            config['memory_width'] * (3 * config["write_heads"] + config["read_heads"]) + 3 * config['read_heads'] + 3 *
            config["write_heads"] + 2 * config['read_heads'] * config["write_heads"])
        assert np_weight_inputs.shape == (config['batch_size'], total_signal_size)

    def test_create_control_signals(self, memory_config, session, np_rng):
        memory_unit, config = memory_config

        total_signal_size = (
            config['memory_width'] * (3 * config["write_heads"] + config["read_heads"]) + 3 * config['read_heads'] + 3 *
            config["write_heads"] + 2 * config['read_heads'] * config["write_heads"])
        np_weighted_input = np.array([np.arange(1, 1 + total_signal_size)] * config['batch_size'])
        weighted_input = tf.constant(np_weighted_input, dtype=tf.float32)

        memory_unit.h_B = config['batch_size']
        control_signals = memory_unit._create_control_signals(weighted_input)
        control_signals = session.run(control_signals)

        alloc_gates, free_gates, write_gates, write_keys, write_strengths, write_vectors, \
        erase_vector, read_keys, read_strengths, read_modes = control_signals

        assert alloc_gates.shape == (config['batch_size'], config['write_heads'], 1)
        assert 0 <= alloc_gates.min() and alloc_gates.max() <= 1
        assert free_gates.shape == (config['batch_size'], config['read_heads'], 1)
        assert 0 <= free_gates.min() and free_gates.max() <= 1
        assert write_gates.shape == (config['batch_size'], config['write_heads'], 1)
        assert 0 <= write_gates.min() and write_gates.max() <= 1

        assert write_keys.shape == (config['batch_size'], config['write_heads'], config['memory_width'])
        assert write_strengths.shape == (config['batch_size'], config['write_heads'], 1)
        assert 1 <= write_strengths.min()
        assert write_vectors.shape == (config['batch_size'], config['write_heads'], config['memory_width'])
        assert erase_vector.shape == (config['batch_size'], config['write_heads'], config['memory_width'])
        assert 0 <= erase_vector.min() and erase_vector.max() <= 1

        assert read_keys.shape == (config['batch_size'], config['read_heads'], config['memory_width'])
        assert read_strengths.shape == (config['batch_size'], config['read_heads'], 1)
        assert 1 <= read_strengths.min()
        assert read_modes.shape == (
            config['batch_size'], config['read_heads'], 1 + 2 * config['write_heads'])  # 3 read modes
        assert 0 <= read_modes.min() and read_modes.max() <= 1 and read_modes.sum(axis=2).all() == 1

    def test_update_alloc_weightings_and_usage_vectors(self, memory_config, session, np_rng):
        memory_unit, config = memory_config

        np_inputs = np_rng.normal(0, 1, [config['batch_size'], config['input_size']])
        np_pre_memory = np_rng.normal(0, 1, [config['batch_size'], config['memory_length'], config['memory_width']])
        np_pre_link_matrix = np.zeros(
            [config['batch_size'], config['write_heads'], config['memory_length'], config['memory_length']])
        np_pre_precedence_weighting = np_rng.uniform(0, 1 / config['memory_length'],
                                                     [config['batch_size'], config['write_heads'],
                                                      config['memory_length']])
        np_pre_write_weightings = np_rng.uniform(0, 1 / config['memory_length'],
                                                 [config['batch_size'], config['write_heads'], config['memory_length']])
        prw_rand = np.arange(0, config['memory_length']) / config['memory_length']
        np_pre_read_weightings = np.stack([prw_rand, ] * config['read_heads'], 0)
        np_pre_read_weightings = np.stack([np_pre_read_weightings, ] * config['batch_size'], 0)
        np_pre_usage_vectors = np_rng.uniform(0, 1 / config['memory_length'],
                                              [config['batch_size'], config['memory_length']])
        np_free_gates = np.ones([config['batch_size'], config['read_heads'], 1]) * 0.5
        np_write_gates = np.ones([config['batch_size'], config['write_heads'], 1]) * 0.5

        inputs = tf.constant(np_inputs, dtype=tf.float32)
        pre_link_matrix = tf.constant(np_pre_link_matrix, dtype=tf.float32)
        pre_memory = tf.constant(np_pre_memory, dtype=tf.float32)

        pre_write_weightings = tf.constant(np_pre_write_weightings, dtype=tf.float32)
        pre_read_weightings = tf.constant(np_pre_read_weightings, dtype=tf.float32)
        pre_usage_vectors = tf.constant(np_pre_usage_vectors, dtype=tf.float32)
        free_gates = tf.constant(np_free_gates, dtype=tf.float32)
        write_gates = tf.constant(np_write_gates, dtype=tf.float32)
        pre_precedence_weighting = tf.constant(np_pre_precedence_weighting, dtype=tf.float32)

        pre_states = (pre_memory, pre_usage_vectors, pre_write_weightings, pre_precedence_weighting, pre_link_matrix,
                      pre_read_weightings)
        memory_unit.zero_state(config['batch_size'])
        memory_unit(inputs, pre_states)
        alloc_weightings, usage_vectors = memory_unit._update_alloc_and_usage_vectors(pre_write_weightings,
                                                                                      pre_read_weightings,
                                                                                      pre_usage_vectors, free_gates,
                                                                                      write_gates)
        alloc_weightings, usage_vectors = session.run([alloc_weightings, usage_vectors])

        np_pre_write_weighting = 1 - np.prod(1 - np_pre_write_weightings, axis=1, keepdims=False)
        np_usage_vector = np_pre_usage_vectors + np_pre_write_weighting - np_pre_usage_vectors * np_pre_write_weighting

        np_retention_vector = np.prod(1 - np_free_gates * np_pre_read_weightings, axis=1, keepdims=False)
        np_usage_vector = np_usage_vector * np_retention_vector

        assert usage_vectors.shape == (config['batch_size'], config['memory_length'])
        assert usage_vectors.min() >= 0 and usage_vectors.max() <= 1
        assert np.allclose(usage_vectors, np_usage_vector, atol=1e-06)

        np_alloc_weightings = np.zeros([config['batch_size'], config['write_heads'], config['memory_length']])
        for b in range(config['batch_size']):
            for w in range(config['write_heads']):

                free_list = np.argsort(np_usage_vector, axis=1)

                for j in range(config['memory_length']):
                    np_alloc_weightings[b, w, free_list[b, j]] = (1 - np_usage_vector[b, free_list[b, j]]) * np.prod(
                        [np_usage_vector[b, free_list[b, i]] for i in range(j)])

                np_usage_vector[b, :] += (
                    (1 - np_usage_vector[b, :]) * np_write_gates[b, w, :] * np_alloc_weightings[b, w, :])

        assert alloc_weightings.shape == (config['batch_size'], config['write_heads'], config['memory_length'])
        assert np.allclose(alloc_weightings, np_alloc_weightings, atol=1e-06)

    def test_calculate_content_weightings(self, memory_config, session, np_rng):
        memory_unit, config = memory_config

        np_memory = np_rng.uniform(0, 1, (config['batch_size'], config['memory_length'], config['memory_width']))
        np_keys = np_rng.normal(0, 2, (config['batch_size'], config['read_heads'], config['memory_width']))
        np_strengths = np_rng.uniform(1, 10, (config['batch_size'], config['read_heads'], 1))

        memory = tf.constant(np_memory, dtype=tf.float32)
        keys = tf.constant(np_keys, dtype=tf.float32)
        strengths = tf.constant(np_strengths, dtype=tf.float32)

        content_weightings = memory_unit._calculate_content_weightings(memory, keys, strengths)
        weightings = content_weightings.eval()

        np_similarity = np.empty([config['batch_size'], config['read_heads'], config['memory_length']])
        for b in range(config['batch_size']):
            for r in range(config['read_heads']):
                for l in range(config['memory_length']):
                    np_similarity[b, r, l] = np.dot(np_memory[b, l, :], np_keys[b, r, :]) / (
                        np.sqrt(np.dot(np_memory[b, l, :], np_memory[b, l, :])) * np.sqrt(
                            np.dot(np_keys[b, r, :], np_keys[b, r, :])))
        np_weightings = np.empty([config['batch_size'], config['read_heads'], config['memory_length']])

        def _weighted_softmax(x, s):
            e_x = np.exp(x * s)
            return e_x / e_x.sum(axis=1, keepdims=True)

        for r in range(config['read_heads']):
            np_weightings[:, r, :] = _weighted_softmax(np_similarity[:, r, :], np_strengths[:, r])

        assert weightings.shape == (config['batch_size'], config['read_heads'], config['memory_length'])
        assert 0 <= weightings.min() and weightings.max() <= 1 and weightings.sum(axis=2).all() <= 1
        assert np.allclose(weightings, np_weightings)

    def test_update_write_weightings(self, memory_config, session, np_rng):
        memory_unit, config = memory_config

        np_alloc_weightings = np_rng.uniform(0, 1 / config['memory_length'],
                                             [config['batch_size'], config['write_heads'], config['memory_length']])
        np_write_content_weighting = np_rng.uniform(0, 1 / config['memory_length'],
                                                    [config['batch_size'], config['write_heads'],
                                                     config['memory_length']])
        np_write_gate = np.ones([config['batch_size'], config['write_heads'], 1]) * 0.5
        np_alloc_gate = np.ones([config['batch_size'], config['write_heads'], 1]) * 0.5

        alloc_weightings = tf.constant(np_alloc_weightings, dtype=tf.float32)
        write_content_weightings = tf.constant(np_write_content_weighting, dtype=tf.float32)
        write_gates = tf.constant(np_write_gate, dtype=tf.float32)
        alloc_gates = tf.constant(np_alloc_gate, dtype=tf.float32)

        write_weightings = memory_unit._update_write_weightings(alloc_weightings, write_content_weightings, write_gates,
                                                                alloc_gates)
        write_weightings = write_weightings.eval()

        np_write_weightings = np_write_gate * (
            np_alloc_gate * np_alloc_weightings + (1 - np_alloc_gate) * np_write_content_weighting)

        assert write_weightings.shape == (config['batch_size'], config['write_heads'], config['memory_length'])
        assert 0 <= write_weightings.min() and write_weightings.max() <= 1 and write_weightings.sum(axis=2).all() <= 1
        assert np.allclose(write_weightings, np_write_weightings)

    def test_update_memory(self, memory_config, session, np_rng):
        memory_unit, config = memory_config

        np_memory = np_rng.normal(0, 1, [config['batch_size'], config['memory_length'], config['memory_width']])
        np_write_weighting = np_rng.uniform(0, 1 / config['memory_length'],
                                            [config['batch_size'], config['write_heads'], config['memory_length']])
        np_write_vector = np_rng.normal(0, 2, [config['batch_size'], config['write_heads'], config['memory_width']])
        np_erase_vector = np_rng.uniform(0, 1, [config['batch_size'], config['write_heads'], config['memory_width']])

        pre_memory = tf.constant(np_memory, dtype=tf.float32)
        write_weighting = tf.constant(np_write_weighting, dtype=tf.float32)
        write_vector = tf.constant(np_write_vector, dtype=tf.float32)
        erase_vector = tf.constant(np_erase_vector, dtype=tf.float32)

        memory_unit.zero_state(config['batch_size'])
        memory = memory_unit._update_memory(pre_memory, write_weighting, write_vector, erase_vector)
        memory = memory.eval()

        np_erase_memory = (1 - np.expand_dims(np_write_weighting, 3) * np.expand_dims(np_erase_vector, 2))
        np_erase_memory = np.prod(np_erase_memory, axis=1, keepdims=False)

        np_add_memory = np.matmul(np.transpose(np_write_weighting, (0, 2, 1)), np_write_vector)

        np_memory = np_memory * np_erase_memory + np_add_memory

        assert memory.shape == (config['batch_size'], config['memory_length'], config['memory_width'])
        assert np.allclose(memory, np_memory, atol=1e-06)

    def test_update_link_matrix(self, memory_config, session, np_rng):
        memory_unit, config = memory_config

        np_inputs = np_rng.normal(0, 1, [config['batch_size'], config['input_size']])
        np_pre_memory = np_rng.normal(0, 1, [config['batch_size'], config['memory_length'], config['memory_width']])
        np_pre_usage_vector = np_rng.uniform(0, 1 / config['memory_length'],
                                             [config['batch_size'], config['memory_length']])
        np_pre_write_weighting = np_rng.uniform(0, 1 / config['memory_length'],
                                                [config['batch_size'], config['write_heads'], config['memory_length']])
        np_pre_read_weightings = np_rng.uniform(0, 1 / config['memory_length'],
                                                [config['batch_size'], config['read_heads'], config['memory_length']])
        np_pre_link_matrix = np.zeros(
            [config['batch_size'], config['write_heads'], config['memory_length'], config['memory_length']])
        np_write_weighting = np_rng.uniform(0, 1 / config['memory_length'],
                                            [config['batch_size'], config['write_heads'], config['memory_length']])
        np_pre_precedence_weighting = np_rng.uniform(0, 1 / config['memory_length'],
                                                     [config['batch_size'], config['write_heads'],
                                                      config['memory_length']])

        inputs = tf.constant(np_inputs, dtype=tf.float32)
        pre_link_matrix = tf.constant(np_pre_link_matrix, dtype=tf.float32)
        write_weighting = tf.constant(np_write_weighting, dtype=tf.float32)
        pre_memory = tf.constant(np_pre_memory, dtype=tf.float32)
        pre_usage_vector = tf.constant(np_pre_usage_vector, dtype=tf.float32)
        pre_write_weighting = tf.constant(np_pre_write_weighting, dtype=tf.float32)
        pre_read_weightings = tf.constant(np_pre_read_weightings, dtype=tf.float32)

        pre_precedence_weighting = tf.constant(np_pre_precedence_weighting, dtype=tf.float32)
        pre_states = (pre_memory, pre_usage_vector, pre_write_weighting, pre_precedence_weighting, pre_link_matrix,
                      pre_read_weightings)

        memory_unit.zero_state(config['batch_size'])
        memory_unit(inputs, pre_states)
        link_matrix, precedence_weighting = memory_unit._update_link_matrix(pre_link_matrix, write_weighting,
                                                                            pre_precedence_weighting)
        link_matrix, precedence_weighting = session.run([link_matrix, precedence_weighting])

        np_precedence_weighting = (1 - np.sum(np_write_weighting, axis=2,
                                              keepdims=True)) * np_pre_precedence_weighting + np_write_weighting

        for b in range(config['batch_size']):
            for w in range(config['write_heads']):
                for i in range(config['memory_length']):
                    for j in range(config['memory_length']):
                        if i == j:
                            np_pre_link_matrix[b, w, i, j] = 0
                        else:
                            np_pre_link_matrix[b, w, i, j] = (1 - np_write_weighting[b, w, i] - np_write_weighting[
                                b, w, j]) * np_pre_link_matrix[b, w, i, j] + \
                                                             np_write_weighting[b, w, i] * np_pre_precedence_weighting[
                                                                 b, w, j]
        np_link_matrix = np_pre_link_matrix

        assert precedence_weighting.shape == (config['batch_size'], config['write_heads'], config['memory_length'])
        assert 0 <= precedence_weighting.min() and precedence_weighting.max() <= 1 and precedence_weighting.sum(
            axis=1).all() <= 1
        assert np.allclose(precedence_weighting, np_precedence_weighting)

        assert link_matrix.shape == (
            config['batch_size'], config['write_heads'], config['memory_length'], config['memory_length'])
        assert np.allclose(link_matrix, np_link_matrix)

    def test_make_read_forward_backward_weightings(self, memory_config, session, np_rng):
        memory_unit, config = memory_config

        np_link_matrix = np.zeros(
            [config['batch_size'], config['write_heads'], config['memory_length'], config['memory_length']])
        np_pre_read_weightings = np_rng.uniform(0, 1 / config['memory_length'],
                                                [config['batch_size'], config['read_heads'], config['memory_length']])

        link_matrix = tf.constant(np_link_matrix, dtype=tf.float32)
        pre_read_weightings = tf.constant(np_pre_read_weightings, dtype=tf.float32)

        forward_weightings, backward_weightings = memory_unit._make_read_forward_backward_weightings(link_matrix,
                                                                                                     pre_read_weightings)
        forward_weightings, backward_weightings = session.run([forward_weightings, backward_weightings])

        np_forward_weightings = np.empty(
            [config['batch_size'], config['read_heads'], config['write_heads'], config['memory_length']])
        np_backward_weightings = np.empty(
            [config['batch_size'], config['read_heads'], config['write_heads'], config['memory_length']])

        for b in range(config['batch_size']):
            for r in range(config['read_heads']):
                for w in range(config['write_heads']):
                    np_forward_weightings[b, r, w, :] = np.matmul(np_pre_read_weightings[b, r, :],
                                                                  np_link_matrix[b, w, :, :])
                    np_backward_weightings[b, r, w, :] = np.matmul(np_pre_read_weightings[b, r, :],
                                                                   np.transpose(np_link_matrix[b, w, :, :]))

        assert forward_weightings.shape == (
            config['batch_size'], config['read_heads'], config['write_heads'], config['memory_length'])
        assert 0 <= forward_weightings.min() and forward_weightings.max() <= 1 and forward_weightings.sum(
            axis=3).all() <= 1
        assert np.allclose(forward_weightings, np_forward_weightings)

        assert backward_weightings.shape == (
            config['batch_size'], config['read_heads'], config['write_heads'], config['memory_length'])
        assert 0 <= backward_weightings.min() and backward_weightings.max() <= 1 and backward_weightings.sum(
            axis=3).all() <= 1
        assert np.allclose(backward_weightings, np_backward_weightings)

    def test_make_read_weightings(self, memory_config, session, np_rng):
        memory_unit, config = memory_config

        np_forward_weightings = np_rng.uniform(0, 1 / config['memory_length'],
                                               [config['batch_size'], config['read_heads'], config['write_heads'],
                                                config['memory_length']])
        np_backward_weightings = np_rng.uniform(0, 1 / config['memory_length'],
                                                [config['batch_size'], config['read_heads'], config['write_heads'],
                                                 config['memory_length']])
        np_read_content_weightings = np_rng.uniform(0, 1 / config['memory_length'],
                                                    [config['batch_size'], config['read_heads'],
                                                     config['memory_length']])
        np_read_modes = np.reshape(
            np.repeat([0.1, ], config['batch_size'] * config['read_heads'] * (2 * config['write_heads'] + 1)),
            [config['batch_size'], config['read_heads'], 1 + 2 * config['write_heads']])

        forward_weightings = tf.constant(np_forward_weightings, dtype=tf.float32)
        backward_weightings = tf.constant(np_backward_weightings, dtype=tf.float32)
        read_content_weightings = tf.constant(np_read_content_weightings, dtype=tf.float32)
        read_modes = tf.constant(np_read_modes, dtype=tf.float32)

        read_weightings = memory_unit._make_read_weightings(forward_weightings, backward_weightings,
                                                            read_content_weightings, read_modes)
        read_weightings = read_weightings.eval()

        np_read_weightings = np.sum(
            np_backward_weightings * np.expand_dims(np_read_modes[:, :, :  config['write_heads']], 3), axis=2) + \
                             np_read_content_weightings * np.expand_dims(np_read_modes[:, :, config['write_heads']],
                                                                         2) + \
                             np.sum(
                                 np_forward_weightings * np.expand_dims(np_read_modes[:, :, config['write_heads'] + 1:],
                                                                        3), axis=2)

        assert read_weightings.shape == (config['batch_size'], config['read_heads'], config['memory_length'])
        assert 0 <= read_weightings.min() and read_weightings.max() <= 1 and read_weightings.sum(axis=1).all() <= 1
        assert np.allclose(read_weightings, np_read_weightings)

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
                                                [config['batch_size'], config['write_heads'], config['memory_length']])
        np_pre_precedence_weighting = np_rng.uniform(0, 1 / config['memory_length'],
                                                     [config['batch_size'], config['write_heads'],
                                                      config['memory_length']])
        np_pre_link_matrix = np.zeros(
            [config['batch_size'], config['write_heads'], config['memory_length'], config['memory_length']])
        np_pre_read_weightings = np_rng.uniform(0, 1 / config['memory_length'],
                                                [config['batch_size'], config['read_heads'], config['memory_length']])

        inputs = tf.constant(np_inputs, dtype=tf.float32)
        pre_memory = tf.constant(np_pre_memory, dtype=tf.float32)
        pre_usage_vector = tf.constant(np_pre_usage_vector, dtype=tf.float32)
        pre_write_weighting = tf.constant(np_pre_write_weighting, dtype=tf.float32)
        pre_precedence_weighting = tf.constant(np_pre_precedence_weighting, dtype=tf.float32)
        pre_link_matrix = tf.constant(np_pre_link_matrix, dtype=tf.float32)
        pre_read_weightings = tf.constant(np_pre_read_weightings, dtype=tf.float32)

        pre_states = (pre_memory, pre_usage_vector, pre_write_weighting, pre_precedence_weighting, pre_link_matrix,
                      pre_read_weightings)

        memory_unit.zero_state(config['batch_size'])
        read_vectors, states = memory_unit(inputs, pre_states)

        session.run(tf.global_variables_initializer())
        read_vectors, states = session.run([read_vectors, states])

        assert read_vectors.shape == (
            config['batch_size'], config['memory_width'] * config['read_heads'] + config['input_size'])
