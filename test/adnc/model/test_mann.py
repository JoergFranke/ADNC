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
import pytest
import tensorflow as tf

from adnc.model.mann import MANN

INPUT_SIZE = 22
OUTPUT_SIZE = 22
BATCH_SIZE = 31

CONFIG = {
    "seed": 123,
    "input_size": INPUT_SIZE,
    "output_size": OUTPUT_SIZE,
    "batch_size": BATCH_SIZE,
    "input_embedding": False,
    "architecture": 'uni',  # bidirectional
    "controller_config": {"num_units": [67, 63], "layer_norm": False, "activation": 'tanh', 'cell_type': 'clstm',
                          'connect': 'dense', 'attention': False},
    "memory_unit_config": {"memory_length": 96, "memory_width": 31, "read_heads": 4, "write_heads": None,
                           "dnc_norm": False, "bypass_dropout": False, 'cell_type': 'dnc'},
    "output_function": "softmax",
    "loss_function": "cross_entropy",
    "output_mask": True,
}


@pytest.fixture()
def mann():
    tf.reset_default_graph()
    return MANN(config=CONFIG)


@pytest.fixture()
def session():
    with tf.Session() as sess:
        yield sess
    tf.reset_default_graph()


@pytest.fixture()
def np_rng():
    seed = np.random.randint(1, 999)
    return np.random.RandomState(seed)


class TestMANN():
    def test_init(self, mann):
        assert isinstance(mann, object)
        assert isinstance(mann.rng, np.random.RandomState)

        assert mann.seed == CONFIG["seed"]
        assert mann.input_size == CONFIG["input_size"]
        assert mann.output_size == CONFIG["output_size"]
        assert mann.batch_size == CONFIG["batch_size"]

        assert mann.input_embedding == CONFIG["input_embedding"]
        assert mann.architecture == CONFIG["architecture"]
        assert mann.controller_config == CONFIG["controller_config"]
        assert mann.memory_unit_config == CONFIG["memory_unit_config"]
        assert mann.output_function == CONFIG["output_function"]
        assert mann.output_mask == CONFIG["output_mask"]

    def test_property_feed(self, mann):
        data, target, mask = mann.feed
        assert type(data) == tf.Tensor
        assert type(target) == tf.Tensor
        assert type(mask) == tf.Tensor

    def test_property_controller_trainable_variables(self, mann):
        assert mann.controller_trainable_variables.__len__() == CONFIG["controller_config"]['num_units'].__len__() * 2

    def test_property_controller_parameter_amount(self, mann):
        total_signal_size = (1 + INPUT_SIZE + CONFIG["memory_unit_config"]["memory_width"] *
                             CONFIG["memory_unit_config"]["read_heads"] +
                             CONFIG["controller_config"]['num_units'][0] + CONFIG["controller_config"]['num_units'][
                                 1]) * 4 * CONFIG["controller_config"]['num_units'][0] + \
                            (1 + CONFIG["controller_config"]['num_units'][0]) * 4 * \
                            CONFIG["controller_config"]['num_units'][1]
        parameter_amount = mann.controller_parameter_amount
        assert parameter_amount == total_signal_size

    def test_property_memory_unit_trainable_variables(self, mann):
        assert mann.memory_unit_trainable_variables.__len__() == 2

    def test_property_memory_unit_parameter_amount(self, mann):
        total_signal_size = (
        CONFIG["memory_unit_config"]['memory_width'] * (3 + CONFIG["memory_unit_config"]["read_heads"]) + 5 *
        CONFIG["memory_unit_config"]['read_heads'] + 3)
        parameter_amount = mann.memory_unit_parameter_amount
        assert parameter_amount == (sum(CONFIG["controller_config"]['num_units']) + 1) * total_signal_size

    def test_property_mann_trainable_variables(self, mann):
        assert mann.mann_trainable_variables.__len__() == 2  # weights and bias for softmax

    def test_property_mann_parameter_amount(self, mann):
        total_mann = ((CONFIG["memory_unit_config"]['memory_width'] * CONFIG["memory_unit_config"]["read_heads"]) + \
                      sum(CONFIG["controller_config"]['num_units']) + 1) * OUTPUT_SIZE
        parameter_amount = mann.mann_parameter_amount
        assert parameter_amount == total_mann

    def test_property_trainable_variables(self, mann):
        assert mann.trainable_variables.__len__() == CONFIG["controller_config"]['num_units'].__len__() * 2 + 2 + 2

    def test_property_parameter_amount(self, mann):
        total_mann = ((CONFIG["memory_unit_config"]['memory_width'] * CONFIG["memory_unit_config"]["read_heads"]) + \
                      sum(CONFIG["controller_config"]['num_units']) + 1) * OUTPUT_SIZE
        parameter_amount = mann.parameter_amount
        assert parameter_amount == mann.controller_parameter_amount + mann.memory_unit_parameter_amount + total_mann

    def test_property_predictions_loss(self, mann, session):
        np_inputs = np.ones([12, BATCH_SIZE, INPUT_SIZE])
        np_target = np.ones([12, BATCH_SIZE, OUTPUT_SIZE])
        np_mask = np.ones([12, BATCH_SIZE])

        data, target, mask = mann.feed
        session.run(tf.global_variables_initializer())

        prediction, loss = session.run([mann.prediction, mann.loss],
                                       feed_dict={data: np_inputs, target: np_target, mask: np_mask})

        assert prediction.shape == (12, BATCH_SIZE, OUTPUT_SIZE)
        assert 0 <= prediction.min() and prediction.max() <= 1 and prediction.sum(axis=2).all() == 1

        assert loss >= 0
        assert loss.shape == ()
