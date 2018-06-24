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
import threading


class BatchGenerator():
    def __init__(self, data_set, set, batch_size, shuffle=True, max_len=False):

        self.set = set
        self.data_set = data_set
        self.batch_size = batch_size
        self.sample_amount = self.data_set.sample_amount(self.set)
        self.shuffle = shuffle
        self.max_len = max_len

        self.lock = threading.Lock()

        if self.shuffle:
            self.order = self.data_set.rng.permutation(np.arange(self.sample_amount))
        else:
            self.order = np.arange(self.sample_amount)

        self.sample_count = 0

    def shuffle_order(self):
        self.order = self.data_set.rng.permutation(self.order)

    def increase_sample_count(self):
        with self.lock:
            self.sample_count += 1
            if self.sample_count >= self.sample_amount:
                self.sample_count = 0
                if self.shuffle:
                    self.order = self.data_set.rng.permutation(self.order)

    def __iter__(self):
        return self

    def __next__(self):

        batch_list = []
        for b in range(self.batch_size):

            sample = self.data_set.get_sample(self.set, self.order[self.sample_count])

            while self.max_len and sample['x'].shape[0] > self.max_len:
                self.increase_sample_count()
                sample = self.data_set.get_sample(self.set, self.order[self.sample_count])

            batch_list.append(sample)
            self.increase_sample_count()

        batch = self.data_set.patch_batch(batch_list)
        return batch
