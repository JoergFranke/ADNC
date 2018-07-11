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
import threading
from queue import Queue

import numpy as np
from adnc.data.utils.batch_generator import BatchGenerator

from adnc.data.tasks.repeat_copy import CopyTask
from adnc.data.tasks.cnn_rc import ReadingComprehension
from adnc.data.tasks.babi import bAbI


class DataLoader():
    """
    The data loader loads and process the datasets and provides iterators for training or inference.
    """
    def __init__(self, config, word_dict=None, re_word_dict=None):
        """
        Args:
            config:         dict with the config to pre-process the dataset
            word_dict:      dict with word-feature pairs, optional
            re_word_dict:   dict with feature-word pairs, optional
        """
        self.config = config

        if config['data_set'] == 'copy_task':
            self.dataset = CopyTask(self.config)
        elif config['data_set'] == 'cnn':
            self.dataset = ReadingComprehension(self.config)
        elif config['data_set'] == 'babi':
            self.dataset = bAbI(self.config, word_dict, re_word_dict)

    @property
    def vocabulary_size(self):
        return self.dataset.vocabulary_size

    @property
    def x_size(self):
        return self.dataset.x_size

    @property
    def y_size(self):
        return self.dataset.y_size

    def batch_amount(self, set_name):
        """
        Calculates the batch amount given a batch size
        Args:
            set_name:   str, name of dataset (train, test, valid)

        Returns:        int, number of batches

        """
        if 'max_len' in self.config.keys():
            return np.floor(
                self.dataset.sample_amount(set_name, self.config['max_len']) / self.config['batch_size']).astype(int)
        else:
            return np.floor(self.dataset.sample_amount(set_name) / self.config['batch_size']).astype(int)

    def sample_amount(self, set_name, ):
        return self.dataset.sample_amount(set_name)

    def get_sample(self, set, number):
        return self.dataset.get_sample(set, number)

    def decode_output(self, sample, prediction):
        return self.dataset.decode_output(sample, prediction)

    def get_data_loader(self, set_name, shuffle=True, max_len=False, batch_size=None, get_shuffle_option=False):
        """
        Provides a data iterator of the given dataset.
        Args:
            set_name:           str, name of dataset
            shuffle:            bool, shuffle set or not
            max_len:            int, max length in time of sample
            batch_size:         int, batch size
            get_shuffle_option: bool, returns shuffle function

        Returns:                iter, iterator over dataset

        """

        if batch_size == None:
            batch_size = self.config['batch_size']

        stream_loader_pre = BatchGenerator(self.dataset, set_name, batch_size, shuffle=shuffle, max_len=max_len)
        stream_loader = self._generate_in_background(stream_loader_pre, num_cached=self.config['num_chached'],
                                                     threads=self.config['threads'])
        if get_shuffle_option:
            return stream_loader, stream_loader_pre.shuffle_order
        else:
            return stream_loader

    @staticmethod
    def _generate_in_background(batch_gen, num_cached=10, threads=1):
        """
        Starts threads with parallel batch generator for faster iteration
        Args:
            batch_gen:      func, the batch generator
            num_cached:     int, numb of caches batches
            threads:        int, numb of parallel threads

        Returns:            iter, iterator over dataset

        """

        queue = Queue(maxsize=num_cached)
        sentinel = object()

        def producer():
            for item in batch_gen:
                queue.put(item)
            queue.put(sentinel)

        threads = [threading.Thread(target=producer) for _ in range(threads)]
        for t in threads:
            t.daemon = True
            t.start()

        item = queue.get()
        while item is not sentinel:
            yield item
            item = queue.get()
