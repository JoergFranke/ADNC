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

from adnc.model.utils import layer_norm


@pytest.fixture()
def session():
    with tf.Session() as sess:
        yield sess
    tf.reset_default_graph()

@pytest.fixture()
def np_rng():
    seed = np.random.randint(1, 999)
    return np.random.RandomState(seed)

def test_layer_norm(session, np_rng):
    np_weights = np_rng.normal(0, 1, [64, 128])

    weights = tf.constant(np_weights, dtype=tf.float32)
    weights_ln = layer_norm(weights, 'test')

    session.run(tf.global_variables_initializer())
    weights_ln = session.run(weights_ln)

    assert weights_ln.shape == (64, 128)
