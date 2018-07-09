import tensorflow as tf
from tensorflow.python.ops.rnn_cell_impl import RNNCell
from tensorflow.python.framework import ops
from tensorflow.python.util import nest
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs

'''
Same functionality like MultiRNNCell but outputs a list of the output from each RNN layer and not only the last one.
This is required to concatinate all RNN layers to one output vector.
'''


class HolisticMultiRNNCell(RNNCell):
    """RNN cell composed sequentially of multiple simple cells."""

    def __init__(self, cells, state_is_tuple=True):
        """Create a RNN cell composed sequentially of a number of RNNCells.
        Args:
          cells: list of RNNCells that will be composed in this order.
          state_is_tuple: If True, accepted and returned states are n-tuples, where
            `n = len(cells)`.  If False, the states are all
            concatenated along the column axis.  This latter behavior will soon be
            deprecated.
        Raises:
          ValueError: if cells is empty (not allowed), or at least one of the cells
            returns a state tuple but the flag `state_is_tuple` is `False`.
        """
        super(HolisticMultiRNNCell, self).__init__()
        if not cells:
            raise ValueError("Must specify at least one cell for MultiRNNCell.")
        if not nest.is_sequence(cells):
            raise TypeError(
                "cells must be a list or tuple, but saw: %s." % cells)

        self._cells = cells
        self._state_is_tuple = state_is_tuple
        if not state_is_tuple:
            if any(nest.is_sequence(c.state_size) for c in self._cells):
                raise ValueError("Some cells return tuples of states, but the flag "
                                 "state_is_tuple is not set.  State sizes are: %s"
                                 % str([c.state_size for c in self._cells]))


    def compute_output_shape(self, input_shape):
        sizes = [cell.output_size for cell in self._cells]
        return sum(sizes)

    @property
    def state_size(self):
        if self._state_is_tuple:
            return tuple(cell.state_size for cell in self._cells)
        else:
            return sum([cell.state_size for cell in self._cells])

    @property
    def output_size(self):
        sizes = [cell.output_size for cell in self._cells]
        return sum(sizes)

    def zero_state(self, batch_size, dtype):
        with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
            if self._state_is_tuple:
                return tuple(cell.zero_state(batch_size, dtype) for cell in self._cells)
            else:
                # We know here that state_size of each cell is not a tuple and
                # presumably does not contain TensorArrays or anything else fancy
                return super(HolisticMultiRNNCell, self).zero_state(batch_size, dtype)

    def call(self, inputs, state, scope=None):
        """Run this multi-layer cell on inputs, starting from state."""
        cur_state_pos = 0
        cur_inp = inputs
        new_states = []
        new_outputs = []
        for i, cell in enumerate(self._cells):
            with vs.variable_scope("cell_%d" % i):
                if self._state_is_tuple:
                    if not nest.is_sequence(state):
                        raise ValueError(
                            "Expected state to be a tuple of length %d, but received: %s" %
                            (len(self.state_size), state))
                    cur_state = state[i]
                else:
                    cur_state = array_ops.slice(state, [0, cur_state_pos],
                                                [-1, cell.state_size])
                    cur_state_pos += cell.state_size
                cur_inp, new_state = cell(cur_inp, cur_state)
                new_states.append(new_state)
                new_outputs.append(cur_inp)

        new_states = (tuple(new_states) if self._state_is_tuple else
                      array_ops.concat(new_states, 1))

        new_outputs = tf.concat(new_outputs, axis=-1)

        return new_outputs, new_states
