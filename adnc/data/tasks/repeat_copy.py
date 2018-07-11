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
from collections import OrderedDict
from scipy.sparse import csr_matrix

"""
Generates "repeat a input sequences"-samples as described in NTM paper.
"""

class CopyTask():
    def __init__(self, config):

        self.rng = np.random.RandomState(seed=config['seed'])
        self.feature_width = config['feature_width']
        self.tokens = {'rep': 'REP'}

        self.samples = self.create_samples(config['set_list'], self.feature_width)

    def create_samples(self, set_list, feature_width):
        print('### CopyTask: create data')
        samples = {}
        for set, conf in set_list.items():
            samples[set] = []
            for i in range(conf["quantity"]):
                if conf["min_length"] < conf["max_length"]:
                    length = self.rng.randint(conf["min_length"], conf["max_length"])
                else:
                    length = conf["min_length"]
                samples[set].append(self.create_sample(length, self.feature_width))
        return samples

    def create_sample(self, length, feature_width):
        sample = OrderedDict()
        sequence = self.rng.randint(feature_width, size=length)

        x_word = np.concatenate([sequence, [feature_width], [0 for _ in range(length)]])
        y_word = np.concatenate([[0 for _ in range(length + 1)], sequence])

        sample['x_word'] = x_word
        sample['x'] = self._numbers_to_onehot(x_word, feature_width + 1)
        sample['y'] = self._numbers_to_onehot(y_word, feature_width)
        sample['m'] = np.concatenate([[0 for _ in range(length + 1)], [1 for _ in range(length)]])
        return sample

    @staticmethod
    def _numbers_to_onehot(numbers, size):
        length = numbers.__len__()
        row = np.arange(length)
        data = np.ones(length)
        matrix = csr_matrix((data, (row, numbers)), shape=(length, size)).toarray()  # super fast
        return matrix

    @staticmethod
    def _zeros_matrix(len, width):
        row = np.arange(len)
        col = np.zeros(len)
        data = np.zeros(len)
        padding = csr_matrix((data, (row, col)), shape=(len, width)).toarray()
        return padding

    def get_sample(self, set_name, number):
        return self.samples[set_name][number]

    @property
    def vocabulary_size(self):
        return self.feature_width + 1

    @property
    def x_size(self):
        return self.feature_width + 1

    @property
    def y_size(self):
        return self.feature_width

    def sample_amount(self, set_name):
        return self.samples[set_name].__len__()

    def decode_output(self, sample, prediction):
        pass

    def patch_batch(self, list_of_samples):

        batch = {'x': [], 'y': [], 'm': [], 'x_word': []}

        len = []
        for sample in list_of_samples:
            len.append(sample['x'].shape[0])
            batch['x_word'].append(sample['x_word'])

        max_len = np.max(len)

        for sample in list_of_samples:
            cur_len = sample['x'].shape[0]
            if cur_len < max_len:
                add_len = max_len - cur_len
                x_add = self._zeros_matrix(add_len, self.x_size)
                batch['x'].append(np.concatenate([sample['x'], x_add], axis=0))
                y_add = self._zeros_matrix(add_len, self.y_size)
                batch['y'].append(np.concatenate([sample['y'], y_add], axis=0))
                m_add = np.zeros([add_len])
                batch['m'].append(np.concatenate([sample['m'], m_add], axis=0))
            else:
                for key in ['x', 'y', 'm']:
                    batch[key].append(sample[key])

        for key in ['x', 'y', 'm']:
            batch[key] = np.stack(batch[key], axis=0)

        batch['x'] = np.transpose(batch['x'], axes=(1, 0, 2))
        batch['y'] = np.transpose(batch['y'], axes=(1, 0, 2))
        batch['m'] = np.transpose(batch['m'], axes=(1, 0))

        return batch

    @staticmethod
    def decode_output(sample, prediction):
        if prediction.shape.__len__() == 3:
            prediction_decode_list = []
            target_decode_list = []
            for b in range(prediction.shape[1]):
                target_decode_list.append([np.argmax(sample['y'][i, b, :]) for i in range(sample['y'].shape[0])])
                prediction_decode_list.append([np.argmax(prediction[i, b, :]) for i in range(prediction.shape[0])])
            return target_decode_list, prediction_decode_list
        else:
            target_decode = [np.argmax(sample['y'][i, :]) for i in range(sample['y'].shape[0])]
            prediction_decode = [np.argmax(prediction[i, :]) for i in range(prediction.shape[0])]
            return target_decode, prediction_decode


if __name__ == '__main__':
    feature_width = 20
    set_list = {"train": {"quantity": 20, "min_length": 20, "max_length": 50},
                "valid": {"quantity": 20, "min_length": 50, "max_length": 70}}

    config = {'seed': 221, 'feature_width': feature_width, 'set_list': set_list}

    sd = CopyTask(config)

    samples = sd.get_sample('train', 2)

    print("Sample Shape")
    print("Data:   ", samples['x'].shape)
    print("Target: ", samples['y'].shape)
    print("Mask:   ", samples['m'].shape)

    # PRINT COPY TASK SAMPLE
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=False, figsize=(13, 8))
    pad = 5
    length = int((samples['x'].shape[0] - 1) / 2)

    plot_x = np.argmax(samples['x'], axis=1)
    plot_x[length + 1:] = -1
    ax1.plot(plot_x, 's', color='midnightblue')
    respons_flag = np.ones(plot_x.shape) * -1
    respons_flag[length] = feature_width
    ax1.plot(respons_flag, 's', color='deeppink')
    ax1.set_xticklabels([])
    ax1.set_yticks(np.arange(0, feature_width + 1, 1))
    ax1.set_ylim(-0.5, feature_width + 0.5)
    ax1.set_xlim(-0.5, length * 2 + 1.5)
    ax1.annotate('data', xy=(0, 0.5), xytext=(-ax1.yaxis.labelpad - pad, 0), xycoords=ax1.yaxis.label,
                 textcoords='offset points', size='24', ha='right', va='center')

    plot_y = np.argmax(samples['y'], axis=1)
    plot_y[:length + 1] = -1
    ax2.plot(plot_y, 's', color='midnightblue')
    ax2.set_xticklabels([])
    ax2.set_yticks(np.arange(0, feature_width, 1))
    ax2.set_ylim(-0.5, feature_width - 0.5)
    ax2.set_xlim(-0.5, length * 2 + 1.5)
    ax2.annotate('target', xy=(0, 0.5), xytext=(-ax2.yaxis.labelpad - pad, 0), xycoords=ax2.yaxis.label,
                 textcoords='offset points', size='24', ha='right', va='center')

    fig.subplots_adjust(wspace=0, hspace=0)
    fig.tight_layout()
    fig.subplots_adjust(left=0.13)
    plt.show()
