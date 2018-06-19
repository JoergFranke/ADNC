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
from tensorflow.python.ops.rnn_cell_impl import RNNCell
from tensorflow.python.framework import ops
from tensorflow.python.util import nest
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs



def oneplus(x):
    return 1 + tf.nn.softplus(x)



# class CustomMultiRNNCell(RNNCell):
#
#     def __init__(self, cells, dense=False):
#
#
#         super(CustomMultiRNNCell, self).__init__()
#         if not cells:
#           raise ValueError("Must specify at least one cell for MultiRNNCell.")
#         if not nest.is_sequence(cells):
#           raise TypeError(
#               "cells must be a list or tuple, but saw: %s." % cells)
#
#         self._cells = cells
#         self.dense = dense
#
#     @property
#     def state_size(self):
#         return tuple(cell.state_size for cell in self._cells)
#
#     @property
#     def output_size(self):
#         if self.dense:
#             return sum([cell.output_size for cell in self._cells])
#         else:
#             return self._cells[-1].output_size
#
#     def zero_state(self, batch_size, dtype):
#         with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
#             return tuple(cell.zero_state(batch_size, dtype) for cell in self._cells)
#
#     def call(self, inputs, state):
#         """Run this multi-layer cell on inputs, starting from state."""
#         if  self._cells.__len__() == 1:
#             with vs.variable_scope("cell_0"):
#                 return self._cells[0](inputs, state[0])
#
#         else:
#             with vs.variable_scope("cell_0"):
#                 first_out, first_state = self._cells[0](inputs, state[0])
#             out_list = [first_out,]
#             state_list = [first_state]
#
#             for i, cell in enumerate(self._cells[1:]):
#                 with vs.variable_scope("cell_%d" % i):
#
#                     cur_inp, new_state = cell(cur_inp, cur_state)
#
#                     cur_state = state[i]
#
#                     cur_inp, new_state = cell(cur_inp, cur_state)
#                     new_states.append(new_state)
#
#             new_states = tuple(new_states)
#
#             return cur_inp, new_states