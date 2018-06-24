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

from adnc.model.utils import unit_simplex_initialization


@pytest.fixture()
def session():
    with tf.Session() as sess:
        yield sess
    tf.reset_default_graph()


@pytest.fixture()
def np_rng():
    seed = np.random.randint(1, 999)
    return np.random.RandomState(seed)


BATCH_SIZE = 4
SHAPE = [2, 3]


def test_unit_simplex_initialization(session, np_rng):
    init_matrix = unit_simplex_initialization(np_rng, BATCH_SIZE, SHAPE, dtype=tf.float32)
    np_init_matrix = init_matrix.eval()
    for b in range(BATCH_SIZE):
        tensor = np_init_matrix[b, :, :]
        assert np.sum(tensor) <= 1
        assert tensor.min() >= 0
        assert tensor.max() <= 1
