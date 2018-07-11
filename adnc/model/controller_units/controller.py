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
from tensorflow.contrib.rnn import LayerNormBasicLSTMCell, LSTMCell, LSTMBlockCell

from adnc.model.controller_units.custom_lstm_cell import CustomLSTMCell
from adnc.model.utils import get_activation

"""
A wrapper for the controller units.
"""

def get_rnn_cell_list(config, name, reuse=False, seed=123, dtype=tf.float32):
    cell_list = []
    for i, units in enumerate(config['num_units']):
        cell = None
        if config['cell_type'] == 'clstm':
            cell = CustomLSTMCell(units, layer_norm=config['layer_norm'], activation=config['activation'], seed=seed,
                                  reuse=reuse, dtype=dtype, name='{}_{}'.format(name, i))
        elif config['cell_type'] == 'tflstm':

            act = get_activation(config['activation'])

            if config['layer_norm']:
                cell = LayerNormBasicLSTMCell(num_units=units, activation=act, layer_norm=config['layer_norm'],
                                              reuse=reuse)
            elif config['layer_norm'] == False and config['activation'] != 'tanh':
                cell = LSTMCell(num_units=units, activation=act, reuse=reuse)
            else:
                cell = LSTMBlockCell(num_units=units)
        cell_list.append(cell)

    return cell_list
