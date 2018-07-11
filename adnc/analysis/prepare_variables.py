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

from adnc.model.utils import softmax
from adnc.model.utils import weighted_softmax

"""
The bucket prepaires and provide a full samples sequence of the DNC internal states for plotting the functions. 
"""

class Bucket:
    def __init__(self, variables, babi_short=True):

        self.babi_short = babi_short

        analyse_values, prediction, decoded_predictions, data_sample, weights_dict = variables
        data, target, mask, x_word = data_sample
        analyse_outputs, analyse_signals, analyse_states = analyse_values
        controller_states, memory_states = analyse_states

        if memory_states.__len__() == 6:
            self.cell_type = 'dnc'
            memory, usage_vector, write_weightings, precedence_weighting, link_matrix, read_weightings = memory_states
            alloc_gate, free_gates, write_gate, write_keys, write_strengths, write_vector, erase_vector, read_keys, read_strengths, read_modes = analyse_signals

            if link_matrix.shape.__len__() == 4:
                link_matrix = np.expand_dims(link_matrix, 2)
            if precedence_weighting.shape.__len__() == 3:
                precedence_weighting = np.expand_dims(precedence_weighting, 2)

            self.read_mode = read_modes
            self.link_matrix = link_matrix
            self.precedence_weighting = precedence_weighting
        else:
            self.cell_type = 'cbmu'
            memory, usage_vector, write_weightings, read_weightings = memory_states
            alloc_gate, free_gates, write_gate, write_keys, write_strengths, write_vector, erase_vector, read_keys, read_strengths = analyse_signals
            # self.read_mode = np.expand_dims(write_gate, -1)
            # self.link_matrix = np.expand_dims(memory, -1)

        self.weights_dict = weights_dict
        self.batch_size = memory.shape[1]
        self.seq_len = memory.shape[0]
        self.max_read_head = read_weightings.shape[2]
        if write_weightings.shape.__len__() == 3:
            self.max_write_head = 1
        else:
            self.max_write_head = write_weightings.shape[2]

        self.memory_width = memory.shape[-1]
        memory_unit_output = analyse_outputs[:, :, :self.max_read_head * self.memory_width]
        controller_output = analyse_outputs[:, :, self.max_read_head * self.memory_width:]

        self.controller_output = controller_output

        self.output_size = analyse_outputs.shape[-1]
        self.analyse_outputs = analyse_outputs

        # prepare strength
        if write_strengths.shape.__len__() == 3:
            write_strengths = np.expand_dims(write_strengths, 2)

        if read_strengths.shape.__len__() == 3:
            read_strengths = np.expand_dims(read_strengths, 2)

        # prepare keys
        if write_keys.shape.__len__() == 3:
            write_keys = np.expand_dims(write_keys, 2)

        if read_keys.shape.__len__() == 3:
            read_keys = np.expand_dims(read_keys, 2)

        # prepare vectos
        if write_vector.shape.__len__() == 3:
            write_vector = np.expand_dims(write_vector, 2)

        if erase_vector.shape.__len__() == 3:
            erase_vector = np.expand_dims(erase_vector, 2)

        # prepare gates
        if write_gate.shape.__len__() == 3:
            write_gate = np.expand_dims(write_gate, 2)
        write_gate = np.stack([write_gate[:, :, :, 0], 1 - write_gate[:, :, :, 0]], axis=3)

        if alloc_gate.shape.__len__() == 3:
            alloc_gate = np.expand_dims(alloc_gate, 2)
        alloc_gate = np.stack([alloc_gate[:, :, :, 0], 1 - alloc_gate[:, :, :, 0]], axis=3)

        free_gates = np.stack([free_gates[:, :, :, 0], 1 - free_gates[:, :, :, 0]], axis=3)

        # prepare weightings
        if write_weightings.shape.__len__() == 3:
            write_weightings = np.expand_dims(write_weightings, 2)

        self.alloc_gate = alloc_gate
        self.free_gate = free_gates
        self.write_gate = write_gate
        self.write_weighting = write_weightings
        self.read_weighting = read_weightings
        self.data = data
        self.mask = mask
        self.prediction = prediction
        self.target = target
        self.x_word = x_word
        self.decoded_predictions = np.stack(decoded_predictions, axis=1)
        self.alloc_gate = alloc_gate
        self.memory = memory
        self.write_strengths = write_strengths
        self.write_keys = write_keys
        self.usage_vector = usage_vector
        self.write_vector = write_vector
        self.read_strength = read_strengths
        self.read_keys = read_keys
        self.read_vector = np.reshape(memory_unit_output,
                                      [self.seq_len, self.batch_size, self.max_read_head, self.memory_width])
        self.erase_vector = erase_vector

    def get_memory_influence(self, batch=-1):

        memory_unit_mask = np.concatenate([np.ones([self.seq_len, self.max_read_head * self.memory_width]), np.zeros(
            [self.seq_len, self.output_size - (self.max_read_head * self.memory_width)])], axis=-1)
        controller_mask = np.concatenate([np.zeros([self.seq_len, self.max_read_head * self.memory_width]), np.ones(
            [self.seq_len, self.output_size - (self.max_read_head * self.memory_width)])], axis=-1)

        controller_influence = []
        memory_unit_influence = []

        if batch == -1:
            batch_size = self.mask.shape[1]
        else:
            batch_size = 1

        for b in range(batch_size):

            if batch != -1:
                b = batch

            matmul = np.matmul(self.analyse_outputs[:, b, :],
                               self.weights_dict['output_layer/weights_concat:0']['var']) + \
                     self.weights_dict['output_layer/bias_merge:0']['var']

            pred_both = softmax(matmul)

            matmul = np.matmul(self.analyse_outputs[:, b, :] * controller_mask,
                               self.weights_dict['output_layer/weights_concat:0']['var']) + \
                     self.weights_dict['output_layer/bias_merge:0']['var']
            pred_c = softmax(matmul)

            matmul = np.matmul(self.analyse_outputs[:, b, :] * memory_unit_mask,
                               self.weights_dict['output_layer/weights_concat:0']['var']) + \
                     self.weights_dict['output_layer/bias_merge:0']['var']
            pred_mu = softmax(matmul)

            co_inf = np.abs(pred_both - pred_mu).sum(axis=-1)
            me_inf = np.abs(pred_both - pred_c).sum(axis=-1)

            co_inf = (1 / (co_inf + me_inf + 1e-8)) * co_inf
            me_inf = (1 / (co_inf + me_inf + 1e-8)) * me_inf

            controller_influence.append(co_inf)
            memory_unit_influence.append(me_inf)

        if controller_influence.__len__() > 1:
            controller_influence = np.mean(controller_influence, axis=1)
            memory_unit_influence = np.mean(memory_unit_influence, axis=1)
        else:
            controller_influence = controller_influence[0]
            memory_unit_influence = memory_unit_influence[0]

        return controller_influence, memory_unit_influence

    def get_basic_functionality(self, batch):

        if self.babi_short:
            max_loc = np.where(np.asarray(self.x_word[batch]) == '-')[0][1] + 1
        else:
            max_loc = np.where(self.mask[:, batch] == 1)[0].max() + 1

        mask = self.mask[:max_loc, batch]
        correct_prediction = np.asarray([p == t and m for p, t, m in
                                         zip(np.argmax(self.prediction[:max_loc, batch, :], axis=-1),
                                             np.argmax(self.target[:max_loc, batch, :], axis=-1), mask)])
        false_prediction = correct_prediction != mask
        text = self.x_word[batch]
        decoded_predictions = self.decoded_predictions[:max_loc, batch]

        alloc_gate = self.alloc_gate[:max_loc, batch, :, :]
        write_gate = self.write_gate[:max_loc, batch, :, :]
        free_gate = self.free_gate[:max_loc, batch, :, :]
        write_weighting = self.write_weighting[:max_loc, batch, :, :]
        read_weighting = self.read_weighting[:max_loc, batch, :, :]
        read_vector = self.read_vector[:max_loc, batch, :, :]
        controller_output = self.controller_output[:max_loc, batch, :]

        write_vector = self.write_vector[:max_loc, batch, :, :]
        erase_vector = self.erase_vector[:max_loc, batch, :, :]
        new_memory = self.memory[:max_loc, batch, :, :]
        add_memory = np.matmul(np.transpose(write_weighting, (0, 2, 1)), write_vector)
        np_erase_memory = (1 - np.expand_dims(write_weighting, 3) * np.expand_dims(erase_vector, 2))
        erase_memory = np.prod(np_erase_memory, axis=1, keepdims=False)
        old_memory = (new_memory - add_memory) / erase_memory
        read_strength = self.read_strength[:max_loc, batch, :, :]

        read_head_influence = self.calculate_read_head_influence(read_vector, self.weights_dict, controller_output)

        if self.cell_type == 'dnc':
            read_mode = self.read_mode[:max_loc, batch, :, :]
            return correct_prediction, false_prediction, text, decoded_predictions, mask, alloc_gate, free_gate, write_gate, write_weighting, read_mode, read_weighting, read_head_influence, old_memory, new_memory, read_strength, max_loc
        else:
            return correct_prediction, false_prediction, text, decoded_predictions, mask, alloc_gate, free_gate, write_gate, write_weighting, read_weighting, read_head_influence, old_memory, new_memory, read_strength, max_loc

    def get_write_process(self, batch):

        if self.babi_short:
            max_loc = np.where(np.asarray(self.x_word[batch]) == '-')[0][1] + 1
        else:
            max_loc = np.where(self.mask[:, batch] == 1)[0].max() + 1

        mask = self.mask[:max_loc, batch]
        correct_prediction = np.asarray([p == t and m for p, t, m in
                                         zip(np.argmax(self.prediction[:max_loc, batch, :], axis=-1),
                                             np.argmax(self.target[:max_loc, batch, :], axis=-1), mask)])
        false_prediction = correct_prediction != mask
        text = self.x_word[batch]
        decoded_predictions = self.decoded_predictions[:, batch]

        alloc_gate = self.alloc_gate[:max_loc, batch, :, :]
        free_gate = self.free_gate[:max_loc, batch, :, :]
        write_gate = self.write_gate[:max_loc, batch, :, :]
        write_weighting = self.write_weighting[:max_loc, batch, :, :]
        write_strength = self.write_strengths[:max_loc, batch, :, :]
        write_key = self.write_keys[:max_loc, batch, :, :, ]
        usage_vector = self.usage_vector[:max_loc, batch, :]
        write_vector = self.write_vector[:max_loc, batch, :, :]
        memory = self.memory[:max_loc, batch, :, :]

        content_weighting = self.calculate_content_weightings(memory, write_key, write_strength[:, :, 0])
        alloc_weighting = self.calculate_allocation_weightings(usage_vector, write_gate[:, :, 0])

        return correct_prediction, false_prediction, text, decoded_predictions, mask, alloc_gate, free_gate, write_gate, \
               write_weighting, content_weighting, write_strength, alloc_weighting, write_vector, write_key, max_loc

    def get_read_process(self, batch):

        if self.babi_short:
            max_loc = np.where(np.asarray(self.x_word[batch]) == '-')[0][1] + 1
        else:
            max_loc = np.where(self.mask[:, batch] == 1)[0].max() + 1

        mask = self.mask[:max_loc, batch]
        correct_prediction = np.asarray([p == t and m for p, t, m in
                                         zip(np.argmax(self.prediction[:max_loc, batch, :], axis=-1),
                                             np.argmax(self.target[:max_loc, batch, :], axis=-1), mask)])
        false_prediction = correct_prediction != mask
        text = self.x_word[batch]
        decoded_predictions = self.decoded_predictions[:, batch]

        read_weighting = self.read_weighting[:max_loc, batch, :, :]
        read_strength = self.read_strength[:max_loc, batch, :, :]
        read_key = self.read_keys[:max_loc, batch, :, :, ]
        read_vector = self.read_vector[:max_loc, batch, :, :]
        memory = self.memory[:max_loc, batch, :, :]
        controller_output = self.controller_output[:max_loc, batch, :]

        read_content_weighting = self.calculate_content_weightings(memory, read_key, read_strength[:, :, 0])
        read_head_influence = self.calculate_read_head_influence(read_vector, self.weights_dict, controller_output)

        if self.cell_type == 'dnc':
            read_mode = self.read_mode[:max_loc, batch, :, :]
            link_matrix = self.link_matrix[:max_loc, batch, :, :, :]
            forward_weighting, backward_weighting = self.calculate_forward_backward_weightings(link_matrix, read_weighting)
            return correct_prediction, false_prediction, text, decoded_predictions, mask, forward_weighting, backward_weighting, read_content_weighting, read_strength, read_key, read_mode, read_weighting, read_vector, read_head_influence, max_loc
        else:
            return correct_prediction, false_prediction, text, decoded_predictions, mask, read_content_weighting, read_strength, read_key, read_weighting, read_vector, read_head_influence, max_loc

    def get_memory_process(self, batch):

        if self.babi_short:
            max_loc = np.where(np.asarray(self.x_word[batch]) == '-')[0][1] + 1
        else:
            max_loc = np.where(self.mask[:, batch] == 1)[0].max() + 1

        mask = self.mask[:max_loc, batch]
        correct_prediction = np.asarray([p == t and m for p, t, m in
                                         zip(np.argmax(self.prediction[:max_loc, batch, :], axis=-1),
                                             np.argmax(self.target[:max_loc, batch, :], axis=-1), mask)])
        false_prediction = correct_prediction != mask
        text = self.x_word[batch]
        decoded_predictions = self.decoded_predictions[:, batch]

        new_memory = self.memory[:max_loc, batch, :, :]
        write_weighting = self.write_weighting[:max_loc, batch, :, :]
        write_vector = self.write_vector[:max_loc, batch, :, :]
        erase_vector = self.erase_vector[:max_loc, batch, :, :]

        add_memory = np.matmul(np.transpose(write_weighting, (0, 2, 1)), write_vector)

        np_erase_memory = (1 - np.expand_dims(write_weighting, 3) * np.expand_dims(erase_vector, 2))
        erase_memory = np.prod(np_erase_memory, axis=1, keepdims=False)

        old_memory = (new_memory - add_memory) / erase_memory

        return correct_prediction, false_prediction, text, decoded_predictions, mask, old_memory, write_weighting, \
               write_vector, erase_vector, add_memory, erase_memory, new_memory, max_loc

    def get_link_matrix_process(self, batch):

        if self.babi_short:
            max_loc = np.where(np.asarray(self.x_word[batch]) == '-')[0][1] + 1
        else:
            max_loc = np.where(self.mask[:, batch] == 1)[0].max() + 1

        mask = self.mask[:max_loc, batch]
        correct_prediction = np.asarray([p == t and m for p, t, m in
                                         zip(np.argmax(self.prediction[:max_loc, batch, :], axis=-1),
                                             np.argmax(self.target[:max_loc, batch, :], axis=-1), mask)])
        false_prediction = correct_prediction != mask
        text = self.x_word[batch]
        decoded_predictions = self.decoded_predictions[:, batch]

        write_weighting = self.write_weighting[:max_loc, batch, :, :]
        new_precedence_weighting = self.precedence_weighting[:max_loc, batch, :, :]
        new_link_matrix = self.link_matrix[:max_loc, batch, :, :, :]

        memory_length = new_link_matrix.shape[-1]

        old_precedence_weighting = (new_precedence_weighting - write_weighting) / (
            1 - np.sum(write_weighting, axis=2, keepdims=True))
        old_link_matrix = np.zeros([max_loc, self.max_write_head, memory_length, memory_length])

        for t in range(max_loc):
            for w in range(self.max_write_head):
                for i in range(memory_length):
                    for j in range(memory_length):
                        if i == j:
                            old_link_matrix[t, w, i, j] = 0
                        else:
                            old_link_matrix[t, w, i, j] = (new_link_matrix[t, w, i, j] - write_weighting[t, w, i] *
                                                           old_precedence_weighting[t, w, j]) / (
                                                              1 - write_weighting[t, w, i] - write_weighting[t, w, j])

        return correct_prediction, false_prediction, text, decoded_predictions, mask, old_link_matrix, \
               old_precedence_weighting, new_precedence_weighting, write_weighting, new_link_matrix, max_loc

    @staticmethod
    def calculate_content_weightings(memory, keys, strength, ):

        time_scale = memory.shape[0]
        memory_length = memory.shape[1]
        read_heads = keys.shape[1]

        if strength.shape.__len__() == 2:
            strength = np.expand_dims(strength, -1)

        similarity = np.empty([time_scale, read_heads, memory_length])
        for t in range(time_scale):
            for r in range(read_heads):
                for l in range(memory_length):
                    similarity[t, r, l] = np.dot(memory[t, l, :], keys[t, r, :]) / (
                        np.sqrt(np.sum(memory[t, l, :] * memory[t, l, :], axis=-1, keepdims=True))
                        * np.sqrt(np.sum(keys[t, r, :] * keys[t, r, :], axis=-1, keepdims=True)))
        weightings = weighted_softmax(similarity, strength)
        return weightings

    @staticmethod
    def calculate_allocation_weightings(usage_vector, write_gates):

        time_scale = usage_vector.shape[0]
        write_heads = write_gates.shape[1]
        memory_length = usage_vector.shape[1]

        np_alloc_weightings = np.zeros([time_scale, write_heads, memory_length])
        for t in range(time_scale):
            for w in range(write_heads):

                free_list = np.argsort(usage_vector[t, :])

                for j in range(memory_length):
                    np_alloc_weightings[t, w, free_list[j]] = (1 - usage_vector[t, free_list[j]]) * np.prod(
                        [usage_vector[t, free_list[i]] for i in range(j)])

                usage_vector[t, :] += ((1 - usage_vector[t, :]) * write_gates[t, w] * np_alloc_weightings[t, w, :])
        return np_alloc_weightings

    @staticmethod
    def calculate_forward_backward_weightings(link_matrix, read_weightings):

        pre_read_weightings = np.concatenate([np.expand_dims(read_weightings[0], axis=0), read_weightings[:-1]], axis=0)

        time_scale = pre_read_weightings.shape[0]
        read_heads = pre_read_weightings.shape[1]
        write_heads = link_matrix.shape[1]
        memory_length = pre_read_weightings.shape[2]

        np_forward_weightings = np.empty([time_scale, read_heads, write_heads, memory_length])
        np_backward_weightings = np.empty([time_scale, read_heads, write_heads, memory_length])

        for t in range(time_scale):
            for r in range(read_heads):
                for w in range(write_heads):
                    np_forward_weightings[t, r, w, :] = np.matmul(pre_read_weightings[t, r, :], link_matrix[t, w, :, :])
                    np_backward_weightings[t, r, w, :] = np.matmul(pre_read_weightings[t, r, :],
                                                                   np.transpose(link_matrix[t, w, :, :]))

        return np_forward_weightings, np_backward_weightings

    @staticmethod
    def calculate_read_head_influence(read_vector, weights_dict, controller_output):

        read_heads = read_vector.shape[1]
        width = read_vector.shape[2]
        read_head_influence = np.empty([read_vector.shape[0], read_heads])

        for t in range(controller_output.shape[0]):

            mu_output = np.reshape(read_vector[t, :, :], [read_heads * width])
            co_output = np.concatenate([mu_output, controller_output[t, :]], axis=-1)
            matmul = np.matmul(co_output, weights_dict['output_layer/weights_concat:0']['var']) + \
                     weights_dict['output_layer/bias_merge:0']['var']
            pred_full = softmax(matmul)

            for r in range(read_heads):
                zero_head = np.ones([read_heads, width])
                zero_head[r, :] = 0
                memory_unit_output = np.reshape(read_vector[t, :, :] * zero_head, [read_heads * width])

                co_output = np.concatenate([memory_unit_output, controller_output[t, :]], axis=-1)
                matmul_muh = np.matmul(co_output, weights_dict['output_layer/weights_concat:0']['var']) + \
                             weights_dict['output_layer/bias_merge:0']['var']
                pred_head = softmax(matmul_muh)

                read_head_influence[t, r] = np.abs(pred_full - pred_head).sum()

        read_head_influence = softmax(read_head_influence)

        return read_head_influence
