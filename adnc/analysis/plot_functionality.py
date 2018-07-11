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

import matplotlib
import numpy as np

matplotlib.use('agg')

import matplotlib.pyplot as plt

from adnc.analysis.plot_functions import PlotFunctions

"""
Function for plotting different memory units behaviours.  
"""

class PlotFunctionality(PlotFunctions):
    def __init__(self, bucket, legend=False, title=False, text_size=16, data_type='png'):
        self.bucket = bucket
        self.data_type = data_type
        super().__init__(legend, title, text_size)

    def plot_basic_functionality(self, batch, plot_dir, name, show=False):

        if self.bucket.cell_type == 'dnc':
            correct_prediction, false_prediction, text, decoded_predictions, mask, alloc_gate, free_gate, write_gate, \
            write_weighting, read_mode, read_weighting, read_head_influence, old_memory, new_memory, read_strength, max_loc = self.bucket.get_basic_functionality(
                batch=batch)
            read_heads = self.bucket.max_read_head
            write_heads = self.bucket.max_write_head
            f, ax = plt.subplots((8 + 2 * read_heads + write_heads), sharex=True)
        else:
            correct_prediction, false_prediction, text, decoded_predictions, mask, alloc_gate, free_gate, write_gate, \
            write_weighting, read_weighting, read_head_influence, old_memory, new_memory, read_strength, max_loc = self.bucket.get_basic_functionality(
                batch=batch)
            read_heads = self.bucket.max_read_head
            write_heads = self.bucket.max_write_head
            f, ax = plt.subplots((8 + read_heads + write_heads), sharex=True)

        controller_influence, memory_unit_influence = self.bucket.get_memory_influence(batch)

        plt.xlim([-1, max_loc])

        line_loc, width = self.plot_data_and_prediction(correct_prediction, false_prediction, text, decoded_predictions,
                                                        mask, ax[:2])
        self.plot_modes(alloc_gate[:, 0, :], ax[2], ['y', 'b'], ['usage', 'content'], name='Alloc Gate')
        self.plot_multi_modes(free_gate, ax[3], width, ['g', 'r'], ['free', 'not free'], name='Free Gates')

        for i in range(write_heads):
            self.plot_modes(write_gate[:, i, :], ax[4 + i], ['g', 'r'], ['write', 'write not'], name='Write Gate')

        self.plot_matrix(old_memory, ax[4 + write_heads], name='Old Memeory', color='bwr')
        self.plot_matrix(new_memory, ax[5 + write_heads], name='New Memeory', color='bwr')

        self.plot_modes(read_head_influence, ax[6 + write_heads], [None for _ in range(read_heads)],
                        ['head {}'.format(i + 1) for i in range(read_heads)], name='Head Influence')

        for i in range(read_heads):
            if self.bucket.cell_type == 'dnc':
                self.plot_modes(read_strength[:, i, :], ax[7 + write_heads + i * 2], ['k'], ['strength'],
                                name='Read Cont. Stg.')
                self.plot_modes(read_mode[:, i, :], ax[8 + write_heads + i * 2], ['m', 'b', 'c'],
                                ['backward', 'content', 'forward'], name='Read Modes')
            else:
                self.plot_modes(read_strength[:, i, :], ax[7 + write_heads + i], ['k'], ['strength'],
                                name='Read Cont. Stg.')

        influence = np.stack([memory_unit_influence, controller_influence], axis=-1)
        self.plot_modes(influence, ax[-1], ['y', 'b'], ['memory usage', 'controller usage'], name='Memory Usage')

        if max_loc < 80:
            for ax_i in ax:
                for l in line_loc:
                    ax_i.axvline(x=l, c='k')

        for _ax in ax:
            for tick in _ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(16)

        for tick in ax[-1].xaxis.get_major_ticks():
            tick.label.set_fontsize(16)
        plt.xlabel("Time step", size='16')

        f.set_size_inches(2 + max_loc * 0.2, 1 + ax.__len__() * 1.4)
        if show:
            plt.show()
        plt.savefig(os.path.join(plot_dir, 'basic_{}_{}'.format(name, batch)), bbox_inches='tight', dpi=160)
        plt.close(f)
        return f

    def plot_write_process(self, batch, plot_dir, name, show=False):

        correct_prediction, false_prediction, text, decoded_predictions, mask, alloc_gate, free_gate, write_gate, \
        write_weighting, content_weighting, write_strength, alloc_weighting, write_vector, write_key, max_loc = self.bucket.get_write_process(
            batch)
        write_heads = self.bucket.max_write_head
        usage_weightings = self.bucket.usage_vector
        f, ax = plt.subplots((3 + 6 * write_heads), sharex=True, figsize=(12, 18))
        plt.xlim([-1, max_loc])

        line_loc, width = self.plot_data_and_prediction(correct_prediction, false_prediction, text, decoded_predictions,
                                                        mask, ax[:2])
        self.plot_multi_modes(free_gate, ax[2], width, ['g', 'r'], ['Free', 'Free not'], name='Free Gates')
        for i in range(write_heads):
            self.plot_weightings(alloc_weighting[:, i, :], ax[3 + i * 6], name='Allocation\nWeighting')
            self.plot_weightings(usage_weightings[:, i, :], ax[4 + i * 6], name='Usage\nWeighting')
            # plot_modes(write_strength[:,i,:], ax[3+i*8], ['k'], ['strength'], name='Content Stg.')
            # plot_weightings(write_key[:,i,:], ax[4+i*8], name='Content Key', mode='norm', color='jet')
            self.plot_weightings(content_weighting[:, i, :], ax[5 + i * 6], name='Content\nWeighting')
            self.plot_modes(alloc_gate[:, i, :], ax[6 + i * 6], ['y', 'b'], ['Allocation', 'Content'],
                            name='Allocation\nGate')
            self.plot_modes(write_gate[:, i, :], ax[8 + i * 6], ['g', 'r'], ['Write', 'Write not'], name='Write Gate')
            self.plot_weightings(write_weighting[:, i, :], ax[7 + i * 6], name='Write\nWeighting')
            # plot_weightings(write_vector[:,i,:], ax[9+i*8], name='Write Vector', mode='norm', color='jet')

        for ax_i in ax:
            for l in line_loc:
                ax_i.axvline(x=l, c='k')

        for _ax in ax:
            for tick in _ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(16)

        for tick in ax[-1].xaxis.get_major_ticks():
            tick.label.set_fontsize(16)

        f.set_size_inches(2 + max_loc * 0.2, 1 + ax.__len__() * 1.4)
        plt.savefig(os.path.join(plot_dir, '{}_{}'.format(name, batch)), bbox_inches='tight', dpi=160)
        if show:
            plt.show()
        plt.close(f)

    def plot_read_process(self, batch, plot_dir, name, show=False):

        correct_prediction, false_prediction, text, decoded_predictions, mask, forward_weighting, backward_weighting, \
        read_content_weighting, read_strength, read_key, read_mode, read_weighting, read_vector, read_head_influence, max_loc = self.bucket.get_read_process(
            batch)
        read_heads = self.bucket.max_read_head
        write_heads = self.bucket.max_write_head

        f, ax = plt.subplots((3 + 5 * read_heads), sharex=True, figsize=(12, 24))
        plt.xlim([-1, max_loc])

        line_loc, width = self.plot_data_and_prediction(correct_prediction, false_prediction, text, decoded_predictions,
                                                        mask, ax[:2])
        for i in range(read_heads):
            for wh in range(write_heads):
                self.plot_weightings(forward_weighting[:, i, wh, :], ax[2 + i * 5],
                                     name='Forward\nWeighting\nHead {}'.format(i + 1))
                self.plot_weightings(backward_weighting[:, i, wh, :], ax[3 + i * 5],
                                     name='Backward\nWeighting\nHead {}'.format(i + 1))
            # plot_modes(read_strength[:,i,:], ax[3+i*8], ['k'], ['strength'], name='Content Stg.')
            # plot_weightings(read_key[:,i,:], ax[4+i*8], name='Content Key', mode='norm', color='jet')
            self.plot_weightings(read_content_weighting[:, i, :], ax[4 + i * 5],
                                 name='Content\nWeighting\nHead {}'.format(i + 1))
            self.plot_modes(read_mode[:, i, :], ax[5 + i * 5], ['m', 'b', 'c'], ['Backward', 'Content', 'Forward'],
                            name='Read Modes\nHead {}'.format(i + 1))
            self.plot_weightings(read_weighting[:, i, :], ax[6 + i * 5], name='Read Wgh. {}'.format(i + 1))
            # plot_weightings(read_vector[:,i,:], ax[8+i*8], name='Read Vector', mode='norm', color='jet')
        self.plot_modes(read_head_influence, ax[-1], [None for _ in range(read_heads)],
                        ['Head {}'.format(i + 1) for i in range(read_heads)], name='Head\nInfluence')

        for ax_i in ax:
            for l in line_loc:
                ax_i.axvline(x=l, c='k')

        for _ax in ax:
            for tick in _ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(16)

        for tick in ax[-1].xaxis.get_major_ticks():
            tick.label.set_fontsize(16)

        f.set_size_inches(2 + max_loc * 0.2, 1 + ax.__len__() * 1.4)
        plt.savefig(os.path.join(plot_dir, '{}_{}'.format(name, batch)), bbox_inches='tight', dpi=160)
        if show:
            plt.show()
        plt.close(f)

    def plot_short_process(self, batch, plot_dir, name, show=False):


        if self.bucket.cell_type == 'dnc':
            correct_prediction, false_prediction, text, decoded_predictions, mask, alloc_gate, free_gate, write_gate, \
            write_weighting, read_mode, read_weighting, read_head_influence, old_memory, new_memory, read_strength, max_loc = self.bucket.get_basic_functionality(
                batch=batch)
        else:
            correct_prediction, false_prediction, text, decoded_predictions, mask, alloc_gate, free_gate, write_gate, \
            write_weighting, read_weighting, read_head_influence, old_memory, new_memory, read_strength, max_loc = self.bucket.get_basic_functionality(
                batch=batch)
        read_heads = self.bucket.max_read_head
        write_heads = self.bucket.max_write_head
        f, ax = plt.subplots((4 + 1 * read_heads + 3 * write_heads - 2), sharex=True, figsize=(12, 18))


        controller_influence, memory_unit_influence = self.bucket.get_memory_influence(batch)
        influence = np.stack([memory_unit_influence, controller_influence], axis=-1)
        influence = influence / influence.sum(axis=1, keepdims=True)


        ax[0].set_title(name, size=33, weight='bold')

        plt.xlim([-1, max_loc])

        line_loc, width = self.plot_data_and_prediction(correct_prediction, false_prediction, text, decoded_predictions,
                                                        mask, ax[:2])

        self.plot_multi_modes(free_gate, ax[2], width, ['g', 'r'], ['Free', 'Free not'], name='Free Gates')
        for i in range(write_heads):
            self.plot_modes(alloc_gate[:, i, :], ax[3 + i * 3], ['y', 'b'], ['Content', 'Usage'], name='Alloc Gate')
            self.plot_modes(write_gate[:, i, :], ax[4 + i * 3], ['g', 'r'], ['Write', 'Write not'], name='Write Gate')

        if self.bucket.cell_type == 'dnc':
            self.plot_modes(read_mode[:,i,:], ax[5+0*1], ['m', 'b','c'], ['Backward', 'Content', 'Forward'], name='Read Mode')
        else:
            self.plot_modes(np.zeros([34, 3]), ax[5 + 0 * 1], ['m', 'b', 'c'], ['Backward', 'Content', 'Forward'],
                            name='Read Mode')

        ax[5 + 0 * 1].set_yticks([])

        self.plot_modes(influence, ax[-1], ['darkorange', 'blueviolet'], ['Memory', 'Controller'],
                        name='Output\nInfluencer')

        for ax_i in ax:
            for l in line_loc:
                ax_i.axvline(x=l, c='k')

        for _ax in ax:
            for tick in _ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(self.text_size)

        for tick in ax[-1].xaxis.get_major_ticks():
            tick.label.set_fontsize(16)
        plt.xlabel("Time steps", size=self.text_size)

        plt.savefig(os.path.join(plot_dir, '{}_{}.{}'.format(name, batch, self.data_type)),
                    bbox_inches='tight', format=self.data_type, dpi=80)
        if show:
            plt.show()
        plt.close(f)

    def plot_memory_process(self, batch, plot_dir, name, show=False, dpi=160):

        correct_prediction, false_prediction, text, decoded_predictions, mask, forward_weighting, backward_weighting, \
        read_content_weighting, read_strength, read_key, read_mode, read_weighting, read_vector, read_head_influence, max_loc = self.bucket.get_read_process(
            batch)

        correct_prediction, false_prediction, text, decoded_predictions, mask, old_memory, write_weighting, \
        write_vector, erase_vector, add_memory, erase_memory, new_memory, max_loc = self.bucket.get_memory_process(
            batch)
        write_heads = self.bucket.max_write_head
        read_heads = self.bucket.max_read_head

        f, ax = plt.subplots((4 + 5 * write_heads + 2 * read_heads), sharex=False,
                             gridspec_kw={'height_ratios': [6, 6, 6, 1, 6, 1, 6, 6, 6, 1, 6, 1, 1]}, figsize=(12, 24))
        plt.xlim([-1, max_loc * new_memory.shape[-1]])
        line_loc, width = self.plot_data_plus_prediction(correct_prediction, false_prediction, text,
                                                         decoded_predictions, mask, ax[0])
        ax[0].axis('tight')
        ax[0].set_xlim(0 - width / 2, max_loc - width / 2)

        self.plot_matrix(old_memory, ax[1], name='Old\nMemeory', color='bwr')
        for i in range(write_heads):
            self.plot_vector_as_matrix(write_weighting[:, i, :], vertical=True, repeats=old_memory.shape[2],
                                       ax=ax[2 + i * 5], name='Write Wgh.', zero_width=5)
            self.plot_vector_as_matrix(write_vector[:, i, :], vertical=False, repeats=old_memory.shape[1],
                                       ax=ax[3 + i * 5], name='Write\nVector\n', zero_width=5, mode='norm', color='bwr',
                                       legend=False)
            self.plot_matrix(add_memory[:, :], ax[4 + i * 5], name='Add\nMatrix', color='bwr')
            self.plot_vector_as_matrix(erase_vector[:, i, :], vertical=False, repeats=old_memory.shape[1],
                                       ax=ax[5 + i * 5], name='Erase\nVector\n', zero_width=5, mode='norm1',
                                       color='YlGnBu', legend=False)
            self.plot_matrix(erase_memory[:, :], ax[6 + i * 5], name='Erase\nMatrix', mode='norm1', color='YlGnBu')
        self.plot_matrix(new_memory, ax[7], name='New\nMemeory', color='bwr')

        for i in range(read_heads):
            self.plot_vector_as_matrix(read_weighting[:, i, :], ax=ax[8 + i * 2], vertical=True,
                                       repeats=old_memory.shape[2], name='Read Wgh.\nHead {}'.format(i + 1),
                                       zero_width=5)
            self.plot_vector_as_matrix(read_vector[:, i, :], vertical=False, repeats=old_memory.shape[1],
                                       ax=ax[9 + i * 2], name='Read\nVector\nHead {}\n'.format(i + 1), zero_width=5,
                                       mode='norm', color='bwr', legend=False)

        for ax_ in ax:
            ax_.set_xticks([])
        ax[-1].axis('off')

        for _ax in ax:
            for tick in _ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(16)

        for tick in ax[-1].xaxis.get_major_ticks():
            tick.label.set_fontsize(16)

        f.set_size_inches(2 + max_loc * 0.4, 1 + ax.__len__() * 1.4)
        plt.savefig(os.path.join(plot_dir, '{}_{}_{}'.format(name, batch, dpi)), bbox_inches='tight', dpi=dpi)
        if show:
            plt.show()
        plt.close(f)

    def plot_link_matrix_process(self, batch, plot_dir, name, show=False):

        correct_prediction, false_prediction, text, decoded_predictions, mask, old_link_matrix, \
        old_precedence_weighting, new_precedence_weighting, write_weighting, new_link_matrix, max_loc = self.bucket.get_link_matrix_process(
            batch)

        correct_prediction, false_prediction, text, decoded_predictions, mask, forward_weighting, backward_weighting, \
        read_content_weighting, read_strength, read_key, read_mode, read_weighting, read_vector, read_head_influence, max_loc = self.bucket.get_read_process(
            batch)

        write_heads = self.bucket.max_write_head
        read_heads = self.bucket.max_read_head

        f, ax = plt.subplots((1 + 3 * write_heads + 2 * read_heads), sharex=False)
        plt.xlim([-1, max_loc * old_link_matrix.shape[-1]])
        line_loc, width = self.plot_data_plus_prediction(correct_prediction, false_prediction, text,
                                                         decoded_predictions, mask, ax[0])
        ax[0].axis('tight')
        ax[0].set_xlim(0 - width / 2, max_loc - width / 2)

        for i in range(write_heads):
            # plot_matrix(old_link_matrix[:,i,:,:], ax[1+i*5], name='Old Link Mat', color='Purples', mode='norm1', zero_add='ones')
            # plot_vector_as_matrix(old_precedence_weighting[:,i,:], vertical=True, repeats=old_link_matrix.shape[2], ax=ax[2+i*5], name='Old Precedence',zero_width=5)
            self.plot_vector_as_matrix(write_weighting[:, i, :], vertical=True, repeats=old_link_matrix.shape[2],
                                       ax=ax[1 + i * 5], name='Write\nWeighting.', zero_width=5)
            self.plot_vector_as_matrix(new_precedence_weighting[:, i, :], vertical=True,
                                       repeats=old_link_matrix.shape[2], ax=ax[2 + i * 5], name='Precedence',
                                       zero_width=5)
            self.plot_matrix(new_link_matrix[:, i, :, :], ax[3 + i * 5], name='Linkage\nMatrix', color='Purples',
                             mode='norm1', zero_add='ones')

        for i in range(read_heads):
            self.plot_vector_as_matrix(forward_weighting[:, i, 0, :], vertical=True, repeats=old_link_matrix.shape[2],
                                       ax=ax[4 + i * 2], name='Forward Wgh.\nHead {}'.format(i + 1), zero_width=5,
                                       color='GnBu')
            self.plot_vector_as_matrix(backward_weighting[:, i, 0, :], vertical=True, repeats=old_link_matrix.shape[2],
                                       ax=ax[5 + i * 2], name='Backward Wgh.\nHead {}'.format(i + 1), zero_width=5,
                                       color='BuGn')

        for _ax in ax:
            for tick in _ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(16)

        for tick in ax[-1].xaxis.get_major_ticks():
            tick.label.set_fontsize(16)
        plt.xlabel("Time step", size='16')

        f.set_size_inches(2 + max_loc * 1.2, 1 + ax.__len__() * 1.4)

        plt.savefig(os.path.join(plot_dir, '{}_{}.{}'.format(name, batch, self.date_type)), bbox_inches='tight',
                    dpi=160)
        if show:
            plt.show()
        plt.close(f)

    def plot_advanced_functionality(self, batch, plot_dir, name, show=False):

        self.plot_write_process(batch, plot_dir, name='{}_write'.format(name), show=show)
        self.plot_read_process(batch, plot_dir, name='{}_read'.format(name), show=show)
        self.plot_memory_process(batch, plot_dir, name='{}_memory'.format(name), show=show)
        self.plot_link_matrix_process(batch, plot_dir, name='{}_link_mat'.format(name), show=show)
