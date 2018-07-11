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
import matplotlib

matplotlib.use('agg')

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import ticker

"""
Principle plot functions for gates, modes, input/output sequences or reading/writings of the DNC
"""

class PlotFunctions():
    def __init__(self, legend=False, title=False, text_size=16):

        self.legend = legend
        self.title = title
        self.text_size = text_size

    def plot_data_plus_prediction(self, correct_prediction, false_prediction, text, decoded_predictions, mask, ax):

        ind = np.arange(mask.shape[0])
        ax.set_ylim([0, 1])
        ax.bar(ind, np.ones(correct_prediction.shape), color='lightgray')
        ax_corr = ax.bar(ind, correct_prediction, color='lawngreen')
        ax_false = ax.bar(ind, false_prediction, color='tomato')
        count = 0
        line_loc = []
        for rect in ax_corr:
            if count >= text.__len__():
                word = '---'
            else:
                word = text[count]
                if mask[count] == 1:
                    word = decoded_predictions[count]
                    line_loc.append(rect.get_x())
                    line_loc.append(rect.get_x() + rect.get_width())

            yloc = 0.5
            xloc = rect.get_x() + 0.4
            ax.text(xloc, yloc, word, horizontalalignment='center',
                    verticalalignment='center', rotation='vertical', color='black', clip_on=True, size=16)
            count += 1

        if self.legend:
            ax.legend((ax_corr, ax_false), ('correct', 'wrong'), loc='center left', bbox_to_anchor=(1, 0.5),
                      prop={'size': self.text_size})

        if self.title:
            ax.set_ylabel('Task', size=self.text_size)

        ax.get_yaxis().set_ticks([])

        return line_loc, rect.get_width()

    def plot_data_and_prediction(self, correct_prediction, false_prediction, text, decoded_predictions, mask, ax):

        ind = np.arange(mask.shape[0])
        ax[0].set_ylim([0, 1])

        ax_aws = ax[0].bar(ind, mask, color='lightgray')
        count = 0
        line_loc = []
        for rect in ax_aws:
            if count >= text.__len__():
                word = '---'
            else:
                word = text[count]
                if mask[count] == 1:
                    line_loc.append(rect.get_x())
                    line_loc.append(rect.get_x() + rect.get_width())

            yloc = 0.5
            xloc = rect.get_x() + 0.4
            ax[0].text(xloc, yloc, word, horizontalalignment='center',
                       verticalalignment='center', rotation='vertical', color='black',
                       clip_on=True, size=16)
            count += 1

        if self.legend:
            ax[0].legend((ax_aws), ('answer',), loc='center left', bbox_to_anchor=(1, 0.5),
                         prop={'size': self.text_size})
        if self.title:
            ax[0].annotate('Questions', xy=(0, 0.8), xytext=(-ax[0].yaxis.labelpad - 150, 0),
                           xycoords=ax[0].yaxis.label, textcoords='offset points', size=self.text_size, ha='left',
                           va='center', ma='left')

        ax[1].set_ylim([0, 1])
        ax_corr = ax[1].bar(ind, correct_prediction, color='lawngreen')
        ax_false = ax[1].bar(ind, false_prediction, color='tomato')
        count = 0
        for rect in ax_corr:
            yloc = 0.5
            xloc = rect.get_x() + 0.4
            ax[1].text(xloc, yloc, decoded_predictions[count], horizontalalignment='center',
                       verticalalignment='center', rotation='vertical', color='black',
                       clip_on=True, size=16)
            count += 1

        if self.legend:
            ax[1].legend((ax_corr, ax_false), ('Correct', 'Wrong'), loc='center left', bbox_to_anchor=(1, 0.5),
                         prop={'size': self.text_size})
        if self.title:
            ax[1].annotate('Predictions', xy=(0, 0.8), xytext=(-ax[1].yaxis.labelpad - 150, 0),
                           xycoords=ax[1].yaxis.label, textcoords='offset points', size=self.text_size, ha='left',
                           va='center', ma='left')

        ax[0].get_yaxis().set_ticks([])
        ax[1].get_yaxis().set_ticks([])

        return line_loc, rect.get_width()

    def plot_weightings(self, weightings, ax, name='Weightings', mode='log', color='YlOrRd'):
        assert weightings.shape.__len__() == 2, "plot weightings: need 2D matrix as data"
        if mode == 'log':
            norm = colors.LogNorm(vmin=1e-3, vmax=1)
        else:
            norm = colors.Normalize(vmin=0, vmax=1)
        img = ax.imshow(np.transpose(weightings), interpolation='nearest', norm=norm, cmap=color,
                        aspect='auto')  # gist_stern
        ax.set_adjustable('box-forced')
        if self.title:
            ax.set_ylabel(name, size=self.text_size)
        if self.legend:
            box = ax.get_position()
            ax.set_position([box.x0 - 0.001, box.y0, box.width, box.height])
            axColor = plt.axes([box.x0 + box.width + 0.005, box.y0, 0.005, box.height])
            cb = plt.colorbar(img, cax=axColor, orientation="vertical")
            for l in cb.ax.yaxis.get_ticklabels():
                l.set_size(self.text_size)

    def plot_modes(self, modes, ax, mode_colors, mode_names, name='Modes'):
        assert modes.shape.__len__() == 2, "plot modes: need 2D matrix as data"
        assert modes.shape[1] == mode_colors.__len__() and modes.shape[
                                                               1] == mode_names.__len__(), "plot modes: not same length"

        ind = np.arange(modes.shape[0])
        ax_list = [ax.bar(ind, modes[:, 0], color=mode_colors[0]), ]

        if modes.shape[1] > 1:
            for m in range(1, modes.shape[1]):
                ax_list.append(ax.bar(ind, modes[:, m], bottom=modes[:, :m].sum(axis=1), color=mode_colors[m]))

        ax.set_yticks([0, 1])
        ax.set_ylim(0, 1)

        if self.title:
            if name == 'Read Mode':
                ax.annotate(name, xy=(0, 0.8), xytext=(-ax.yaxis.labelpad - 150, 0), xycoords=ax.yaxis.label,
                            textcoords='offset points', size=self.text_size, ha='left', va='center', ma='left')
            else:
                ax.annotate(name, xy=(0, 0.8), xytext=(-ax.yaxis.labelpad - 135, 0), xycoords=ax.yaxis.label,
                            textcoords='offset points', size=self.text_size, ha='left', va='center', ma='left')
        if self.legend:
            ax.legend(ax_list, mode_names, loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': self.text_size})

    def plot_multi_modes(self, multi_modes, ax, width, mode_colors, mode_names, name='Multi Modes'):
        modes = multi_modes.shape[1]
        ind = np.arange(multi_modes.shape[0])

        width = width / modes
        for j in range(-1, modes - 1):
            ax_list = [ax.bar(ind + j * width + (width * 0.5), multi_modes[:, j, 0], color=mode_colors[0], width=width,
                              align='center'), ]
            if multi_modes.shape[2] > 1:
                for m in range(1, multi_modes.shape[2]):
                    ax_list.append(ax.bar(ind + j * width + (width * 0.5), multi_modes[:, j, m],
                                          bottom=multi_modes[:, j, :m].sum(axis=1), color=mode_colors[m], width=width,
                                          align='center'))

        if self.title:
            ax.annotate(name, xy=(0, 0.8), xytext=(-ax.yaxis.labelpad - 135, 0), xycoords=ax.yaxis.label,
                        textcoords='offset points', size=self.text_size, ha='left', va='center', ma='left')
        if self.legend:
            ax.legend(ax_list, mode_names, loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': self.text_size})
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['0', '', '', '', '', '1'])
        ax.set_ylim(0, 1)

    def plot_matrix(self, matrix, ax, name='Weightings', mode='norm', color='RdYlBu', zero_width=5, zero_add='zeros'):
        assert matrix.shape.__len__() == 3, "plot weightings: need 3D matrix as data"

        if mode == 'log':
            norm = colors.LogNorm(vmin=1e-8, vmax=0.1)
        elif mode == 'norm1':
            norm = colors.Normalize(vmin=0, vmax=1)
        else:
            norm = colors.Normalize(vmin=-1, vmax=1)

        if zero_add == 'zeros':
            matrix = np.concatenate([matrix, np.zeros([matrix.shape[0], matrix.shape[1], zero_width])], axis=2)
            matrix = np.transpose(matrix, axes=(0, 2, 1))
            flat_matrix = np.reshape(matrix, [-1, matrix.shape[2]])
            flat_matrix = np.concatenate([np.zeros([zero_width, flat_matrix.shape[1]]), flat_matrix], axis=0)
        else:
            matrix = np.concatenate([matrix, np.ones([matrix.shape[0], matrix.shape[1], zero_width])], axis=2)
            matrix = np.transpose(matrix, axes=(0, 2, 1))
            flat_matrix = np.reshape(matrix, [-1, matrix.shape[2]])
            flat_matrix = np.concatenate([np.ones([zero_width, flat_matrix.shape[1]]), flat_matrix], axis=0)

        img = ax.imshow(np.transpose(flat_matrix), aspect='auto', interpolation='nearest', norm=norm, cmap=color)

        ax.set_adjustable('box-forced')
        if self.title:
            ax.set_ylabel(name, size=self.text_size)
        if self.legend:
            box = ax.get_position()
            ax.set_position([box.x0 - 0.001, box.y0, box.width, box.height])
            axColor = plt.axes([box.x0 + box.width + 0.005, box.y0, 0.005, box.height])
            cb = plt.colorbar(img, cax=axColor, orientation="vertical")
            for l in cb.ax.yaxis.get_ticklabels():
                l.set_size(self.text_size)
            tick_locator = ticker.MaxNLocator(nbins=3)
            cb.locator = tick_locator
            cb.update_ticks()

    def plot_vector_as_matrix(self, vector, vertical, repeats, ax, name='Weightings', mode='log', color='YlOrRd',
                              zero_width=5):

        assert vector.shape.__len__() == 2, "plot weightings: need 2D matrix as data"

        if mode == 'log':
            norm = colors.LogNorm(vmin=1e-3, vmax=1)
        elif mode == 'norm1':
            norm = colors.Normalize(vmin=0, vmax=1)
        else:
            norm = colors.Normalize(vmin=-1, vmax=1)

        if vertical:
            matrix = np.repeat(vector, repeats, axis=1)
            matrix = np.reshape(matrix, [vector.shape[0], vector.shape[1], repeats])

        else:

            matrix = np.repeat(vector, repeats, axis=0)
            matrix = np.reshape(matrix, [vector.shape[0], repeats, vector.shape[1]])

        matrix = np.concatenate([matrix, np.zeros([matrix.shape[0], matrix.shape[1], zero_width])], axis=2)
        matrix = np.transpose(matrix, axes=(0, 2, 1))
        flat_matrix = np.reshape(matrix, [-1, matrix.shape[2]])
        flat_matrix = np.concatenate([np.zeros([zero_width, flat_matrix.shape[1]]), flat_matrix], axis=0)

        img = ax.imshow(np.transpose(flat_matrix), aspect='auto', interpolation='nearest', norm=norm, cmap=color)

        ax.set_adjustable('box-forced')
        box = ax.get_position()
        ax.set_position([box.x0 - 0.001, box.y0, box.width, box.height])

        if self.legend:
            axColor = plt.axes([box.x0 + box.width + 0.005, box.y0, 0.005, box.height])
            cb = plt.colorbar(img, cax=axColor, orientation="vertical")
            for l in cb.ax.yaxis.get_ticklabels():
                l.set_size(self.text_size)
        if self.title:
            ax.set_ylabel(name, labelpad=30, size=self.text_size)
        ax.set_yticks([])
