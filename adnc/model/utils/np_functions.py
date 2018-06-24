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


def softmax(x):
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)

def weighted_softmax(x, s):
    e_x = np.exp(x * s)
    return e_x / e_x.sum(axis=-1, keepdims=True)
