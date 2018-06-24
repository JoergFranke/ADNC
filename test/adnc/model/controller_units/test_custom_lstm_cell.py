# # Copyright 2018 Jörg Franke
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #    http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# # ==============================================================================
import numpy as np
import tensorflow as tf
import pytest

from adnc.model.controller_units.custom_lstm_cell import CustomLSTMCell


@pytest.fixture(
    params=[{"seed": 122, "cell_size": 10, "input_size": 20, "batch_size": 1, "layer_norm": True, "activation": 'tanh'},
            {"seed": 123, "cell_size": 11, "input_size": 18, "batch_size": 2, "layer_norm": False,
             "activation": 'tanh'},
            {"seed": 124, "cell_size": 12, "input_size": 22, "batch_size": 3, "layer_norm": True, "activation": 'relu'},
            {"seed": 125, "cell_size": 13, "input_size": 16, "batch_size": 4, "layer_norm": False, "activation": 'elu'},
            {"seed": 126, "cell_size": 14, "input_size": 24, "batch_size": 5, "layer_norm": True,
             "activation": 'softsign'}])
def lstm_config(request):
    config = request.param
    return CustomLSTMCell(num_units=config['cell_size'], layer_norm=config['layer_norm'],
                          activation=config["activation"], seed=config['seed'], reuse=False, name='lstm'), config


@pytest.fixture()
def session():
    with tf.Session() as sess:
        yield sess
    tf.reset_default_graph()


# random seed for RandomStateGenerator pytest -s
seed = np.random.randint(1, 999)
print("TEST SEED LSTM: {}".format(seed))
np_rng = np.random.RandomState(seed)


class TestCustomLSTM():
    def test_init(self, lstm_config):

        lstm, config = lstm_config

        assert isinstance(lstm, object)
        assert isinstance(lstm.rng, np.random.RandomState)
        assert lstm.seed == config['seed']
        assert lstm.num_units == config["cell_size"]

    def test_property_output_size(self, lstm_config):
        lstm, config = lstm_config
        output_size = lstm.output_size
        assert output_size == config['cell_size']

    def test_property_state_size(self, lstm_config):
        lstm, config = lstm_config
        output_size = lstm.state_size
        assert output_size == config['cell_size']

    def test_zero_state(self, lstm_config, session):
        lstm, config = lstm_config
        init_state = lstm.zero_state(batch_size=config['batch_size'], dtype=tf.float32)
        assert init_state.eval(session=session).shape == (config['batch_size'], config['cell_size'])

    def test_lstm_cell(self, session, lstm_config):
        lstm, config = lstm_config
        np_input = np_rng.normal(0, 2, [config['batch_size'], config['input_size']])
        np_pre_cell_state = np_rng.normal(0, 1, [config['batch_size'], config['cell_size']])

        np_weights = np_rng.normal(0, 1, [config['input_size'], 4 * config['cell_size']])
        np_bias = np_rng.normal(0, 1, [4 * config['cell_size']])

        inputs = tf.constant(np_input, dtype=tf.float32)
        pre_cell_state = tf.constant(np_pre_cell_state, dtype=tf.float32)
        weights = tf.constant(np_weights, dtype=tf.float32)
        bias = tf.constant(np_bias, dtype=tf.float32)

        cell_size = config['cell_size']

        output, cell_state = lstm._lstm_cell(inputs, pre_cell_state, cell_size, weights, bias)
        session.run(tf.global_variables_initializer())
        output, cell_state = session.run([output, cell_state])

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        if config['activation'] == 'tanh':
            act = np.tanh
        elif config['activation'] == 'relu':
            act = np.vectorize(lambda x: np.maximum(x, 0))
        elif config['activation'] == 'elu':
            act = np.vectorize(lambda x: np.exp(x) - 1. if (x < 0) else x)
        elif config['activation'] == 'softsign':
            act = np.vectorize(lambda x: x / (1. + np.abs(x)))

        np_ifco = np.matmul(np_input, np_weights) + np_bias
        np_fg = sigmoid(np_ifco[:, : cell_size])
        np_ig = sigmoid(np_ifco[:, cell_size: 2 * cell_size])
        np_og = sigmoid(np_ifco[:, 2 * cell_size: 3 * cell_size])
        np_in = act(np_ifco[:, 3 * cell_size:])

        np_cell_state = np_fg * np_pre_cell_state + np_ig * np_in
        np_output = np_og * act(np_cell_state)

        assert output.shape == (config['batch_size'], config['cell_size'])
        assert cell_state.shape == (config['batch_size'], config['cell_size'])
        assert np.allclose(output, np_output, atol=1e-06)
        assert np.allclose(cell_state, np_cell_state, atol=1e-06)

    def test_lstm_layer(self, session, lstm_config):
        lstm, config = lstm_config

        np_input = np_rng.normal(0, 2, [config['batch_size'], config['input_size']])
        np_pre_cell_state = np_rng.normal(0, 1, [config['batch_size'], config['cell_size']])

        inputs = tf.constant(np_input, dtype=tf.float32)
        pre_cell_state = tf.constant(np_pre_cell_state, dtype=tf.float32)

        output, cell_state = lstm._lstm_layer(inputs, pre_cell_state, 0)

        session.run(tf.global_variables_initializer())
        output, cell_state = session.run([output, cell_state])

        assert output.shape == (config['batch_size'], config['cell_size'])
        assert cell_state.shape == (config['batch_size'], config['cell_size'])

    def test_lnlstm_cell(self, session, lstm_config):
        lstm, config = lstm_config

        np_inputs = np_rng.normal(0, 2, [config['batch_size'], config['input_size']])
        np_pre_cell_state = np_rng.normal(0, 1, [config['batch_size'], config['cell_size']])

        np_weights = np_rng.normal(0, 1, [config['input_size'], 4 * config['cell_size']])
        np_bias = np_rng.normal(0, 1, [4 * config['cell_size']])

        inputs = tf.constant(np_inputs, dtype=tf.float32)
        pre_cell_state = tf.constant(np_pre_cell_state, dtype=tf.float32)
        weights = tf.constant(np_weights, dtype=tf.float32)
        bias = tf.constant(np_bias, dtype=tf.float32)

        cell_size = config['cell_size']

        output, cell_state = lstm._lnlstm_cell(inputs, pre_cell_state, cell_size, weights, bias)
        session.run(tf.global_variables_initializer())
        output, cell_state = session.run([output, cell_state])

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def ln_test(x):
            _eps = 1e-6
            mean = np.mean(x, axis=1, keepdims=True)
            var = np.var(x, axis=1, keepdims=True)
            return (x - mean) / np.sqrt(var + _eps)

        if config['activation'] == 'tanh':
            act = np.tanh
        elif config['activation'] == 'relu':
            act = np.vectorize(lambda x: np.maximum(x, 0))
        elif config['activation'] == 'elu':
            act = np.vectorize(lambda x: np.exp(x) - 1. if (x < 0) else x)
        elif config['activation'] == 'softsign':
            act = np.vectorize(lambda x: x / (1. + np.abs(x)))

        np_ifco = ln_test(np.matmul(np_inputs, np_weights)) + np_bias
        np_fg = sigmoid(np_ifco[:, : cell_size])
        np_ig = sigmoid(np_ifco[:, cell_size: 2 * cell_size])
        np_og = sigmoid(np_ifco[:, 2 * cell_size: 3 * cell_size])
        np_in = act(np_ifco[:, 3 * cell_size:])

        np_cell_state = np_fg * np_pre_cell_state + np_ig * np_in
        np_output = np_og * act(ln_test(np_cell_state))

        assert output.shape == (config['batch_size'], config['cell_size'])
        assert cell_state.shape == (config['batch_size'], config['cell_size'])
        assert np.allclose(output, np_output, atol=1e-06)
        assert np.allclose(cell_state, np_cell_state, atol=1e-06)

    def test_lnlstm_layer(self, session, lstm_config):
        lstm, config = lstm_config

        np_input = np_rng.normal(0, 2, [config['batch_size'], config['input_size']])
        np_pre_cell_state = np_rng.normal(0, 1, [config['batch_size'], config['cell_size']])

        tf_input = tf.placeholder(tf.float32, [config['batch_size'], config['input_size']], name='x')
        tf_pre_cell_state = tf.placeholder(tf.float32, [config['batch_size'], config['cell_size']], name='c')

        output, cell_state = lstm._lnlstm_layer(tf_input, tf_pre_cell_state, 0)

        session.run(tf.global_variables_initializer())
        output, cell_state = session.run([output, cell_state],
                                         feed_dict={tf_input: np_input, tf_pre_cell_state: np_pre_cell_state})

        assert output.shape == (config['batch_size'], config['cell_size'])
        assert cell_state.shape == (config['batch_size'], config['cell_size'])

    def test_call(self, session, lstm_config):
        lstm, config = lstm_config

        np_inputs = np_rng.normal(0, 1, [config['batch_size'], config['input_size']])
        np_pre_cell_state = np_rng.normal(0, 1, [config['batch_size'], config['cell_size']])

        inputs = tf.constant(np_inputs, dtype=tf.float32)
        pre_cell_state = tf.constant(np_pre_cell_state, dtype=tf.float32)

        input_tuple = inputs
        pre_cell_state_tuple = pre_cell_state

        lstm.zero_state(config['batch_size'])
        output_tuple, state_tuple = lstm(input_tuple, pre_cell_state_tuple)

        session.run(tf.global_variables_initializer())
        outputs, states = session.run([output_tuple, state_tuple])

        assert outputs.shape == (config['batch_size'], config['cell_size'])
        assert states.shape == (config['batch_size'], config['cell_size'])

        if lstm.layer_norm:
            total_signal_size = (1 + config['input_size']) * 4 * config['cell_size'] + 2 * 5 * config['cell_size']
        else:
            total_signal_size = (1 + config['input_size']) * 4 * config['cell_size']
        parameter_amount = lstm.parameter_amount
        assert parameter_amount == total_signal_size
