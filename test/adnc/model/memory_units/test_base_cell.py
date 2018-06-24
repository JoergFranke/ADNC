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

from adnc.model.memory_units.base_cell import BaseMemoryUnitCell


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
    return BaseMemoryUnitCell(input_size=config['input_size'], memory_length=config["memory_length"],
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


class TestDNCMemoryUnit():
    def test_init(self, memory_config):

        memory_unit, config = memory_config

        assert isinstance(memory_unit, object)
        assert isinstance(memory_unit.rng, np.random.RandomState)

        assert memory_unit.h_N == config["memory_length"]
        assert memory_unit.h_W == config["memory_width"]
        assert memory_unit.h_RH == config["read_heads"]

    def test_property_output_size(self, memory_config, session):
        memory_unit, config = memory_config

        output_size = memory_unit.output_size
        assert output_size == config['memory_width'] * config["read_heads"] + config['input_size']

    def test_calculate_content_weightings(self, memory_config, session, np_rng):
        memory_unit, config = memory_config

        np_memory = np_rng.normal(0, 1, (config['batch_size'], config['memory_length'], config['memory_width']))
        np_keys = np_rng.normal(0, 2, (config['batch_size'], 1, config['memory_width']))
        np_strengths = np_rng.uniform(1, 10, (config['batch_size'], 1))

        memory = tf.constant(np_memory, dtype=tf.float32)
        keys = tf.constant(np_keys, dtype=tf.float32)
        strengths = tf.constant(np_strengths, dtype=tf.float32)

        content_weightings = memory_unit._calculate_content_weightings(memory, keys, strengths)
        weightings = content_weightings.eval()

        np_similarity = np.empty([config['batch_size'], config['memory_length']])
        for b in range(config['batch_size']):
            for l in range(config['memory_length']):
                np_similarity[b, l] = np.dot(np_memory[b, l, :], np_keys[b, 0, :]) / (
                    np.sqrt(np.dot(np_memory[b, l, :], np_memory[b, l, :])) * np.sqrt(
                        np.dot(np_keys[b, 0, :], np_keys[b, 0, :])))

        def _weighted_softmax(x, s):
            e_x = np.exp(x * s)
            return e_x / e_x.sum(axis=1, keepdims=True)

        np_weightings = _weighted_softmax(np_similarity, np_strengths)

        assert weightings.shape == (config['batch_size'], config['memory_length'])
        assert 0 <= weightings.min() and weightings.max() <= 1 and weightings.sum(axis=1).all() <= 1
        assert np.allclose(weightings, np_weightings)

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
