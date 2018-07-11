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
import os
import pickle

import numpy as np

from adnc.analysis.plot_functionality import PlotFunctionality
from adnc.analysis.prepare_variables import Bucket
from adnc.model.utils import softmax


class Analyser():
    """
    The analyzer helps to analyze the functionality of the DNC during training. It is used to calculate the
    memory influence and to plot function plots of the memory usage.
    """
    def __init__(self, record_dir, save_variables=False, save_fig=False):
        """
        Args:
            record_dir:     dir to store the function plots
            save_variables: bool, to save weights, gradients and losses in a numpy list
            save_fig:       bool, save plots
        """
        self.record_dir = record_dir
        self.save_variables = save_variables
        self.save_fig = save_fig

        self.max_batch_plot = 1

        if save_fig:
            self.plot_dir = os.path.join(self.record_dir, "plots")
            while not os.path.isdir(self.plot_dir):
                try:
                    os.mkdir(self.plot_dir)
                except ValueError:
                    pass

    def feed_variables(self, variables, epoch, name='variables'):

        variable_name = name + "_{}".format(epoch)

        buck = Bucket(variables)

        plotter = PlotFunctionality(bucket=buck)

        if self.save_variables:
            self.save_variables_to_file(variables, variable_name)

        if self.save_fig:
            plotter.plot_basic_functionality(batch=0, plot_dir=self.plot_dir, name=variable_name, show=False)

        return self.estimate_memory_usage(variables)

    def feed_variables_two(self, variables, epoch, name='variables', save_plot=1):

        variable_name = name + "_{}".format(epoch)

        buck = Bucket(variables)
        plotter = PlotFunctionality(bucket=buck)

        if save_plot > 0:
            if self.save_variables:
                self.save_variables_to_file(variables, variable_name)

            if self.save_fig:
                plotter.plot_basic_functionality(batch=0, plot_dir=self.plot_dir, name=variable_name, show=False)

        return self.estimate_memory_usage(variables)

    @staticmethod
    def plot_analysis(variables, plot_dir, name='variables'):

        buck = Bucket(variables)
        plotter = PlotFunctionality(bucket=buck)

        plotter.plot_basic_functionality(batch=0, plot_dir=plot_dir, name=name, show=True)
        plotter.plot_advanced_functionality(batch=0, plot_dir=plot_dir, name=name, show=True)

    @staticmethod
    def estimate_memory_usage(variables):

        analyse_values, prediction, decoded_predictions, data_sample, weights_dict = variables
        data, target, mask, x_word = data_sample
        analyse_outputs, analyse_signals, analyse_states = analyse_values
        controller_states, memory_states = analyse_states

        if memory_states.__len__() == 6:
            memory, usage_vector, write_weightings, precedence_weighting, link_matrix, read_weightings = memory_states
        else:
            memory, usage_vector, write_weightings, read_weightings = memory_states
        read_head = read_weightings.shape[2]
        memory_width = memory.shape[-1]
        time_len = memory.shape[0]
        memory_unit_mask = np.concatenate([np.ones([time_len, read_head * memory_width]), np.zeros(
            [time_len, analyse_outputs.shape[-1] - (read_head * memory_width)])], axis=-1)
        controller_mask = np.concatenate([np.zeros([time_len, read_head * memory_width]),
                                          np.ones([time_len, analyse_outputs.shape[-1] - (read_head * memory_width)])],
                                         axis=-1)

        controller_influence = []
        memory_unit_influence = []
        for b in range(mask.shape[1]):
            matmul = np.matmul(analyse_outputs[:, b, :], weights_dict['output_layer/weights_concat:0']['var']) + \
                     weights_dict['output_layer/bias_merge:0']['var']
            pred_both = softmax(matmul)

            matmul = np.matmul(analyse_outputs[:, b, :] * controller_mask,
                               weights_dict['output_layer/weights_concat:0']['var']) + \
                     weights_dict['output_layer/bias_merge:0']['var']
            pred_c = softmax(matmul)

            matmul = np.matmul(analyse_outputs[:, b, :] * memory_unit_mask,
                               weights_dict['output_layer/weights_concat:0']['var']) + \
                     weights_dict['output_layer/bias_merge:0']['var']
            pred_mu = softmax(matmul)

            co_inf = (np.abs(pred_both - pred_mu) * np.expand_dims(mask[:, b], 1)).sum() / mask[:, b].sum()
            me_inf = (np.abs(pred_both - pred_c) * np.expand_dims(mask[:, b], 1)).sum() / mask[:, b].sum()

            co_inf = (1 / (co_inf + me_inf)) * co_inf
            me_inf = (1 / (co_inf + me_inf)) * me_inf

            controller_influence.append(co_inf)
            memory_unit_influence.append(me_inf)

        controller_influence = np.mean(controller_influence)
        memory_unit_influence = np.mean(memory_unit_influence)
        return controller_influence, memory_unit_influence

    def save_variables_to_file(self, variables, name):
        save_file = os.path.join(self.record_dir, "{}.plk".format(name))
        with open(save_file, "wb") as f:
            pickle.dump(variables, f)
