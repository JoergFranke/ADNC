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


def oneplus(x):
    return 1 + tf.nn.softplus(x)

def get_activation(activation):

    if activation == 'tanh':
        act = tf.tanh
    elif activation == 'relu':
        act = tf.nn.relu
    elif activation == 'elu':
        act = tf.nn.elu
    elif activation == 'softsign':
        act = tf.nn.softsign
    elif activation == 'oneplus':
        act = oneplus
    else:
        raise UserWarning("Activation function not found.")

    return act