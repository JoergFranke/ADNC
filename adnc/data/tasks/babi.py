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
import copy
import pathlib
import tarfile
from collections import Counter
from urllib.request import Request, urlopen

import numpy as np

"""
Downloads and pre-preocess the 20 bAbI task. It also augmets task 16 as described in paper.
"""

DEFAULT_DATA_FOLDER = "data_babi"
LONGEST_SAMPLE_LENGTH = 1920
bAbI_URL = 'http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz'

class bAbI():
    def __init__(self, config, word_dict=None, re_word_dict=None):

        self.seed = config['seed']
        self.rng = np.random.RandomState(seed=self.seed)

        self.set_type = config['set_type']

        if config['task_selection'][0] == 'all':
            self.task_selection = [i + 1 for i in range(20)]
        else:
            self.task_selection = config['task_selection']

        self.valid_ratio = config['valid_ratio']

        if 'max_len' in config:
            self.max_len = config['max_len']
        else:
            self.max_len = LONGEST_SAMPLE_LENGTH + 1

        if 'augment16' in config:
            self.augment16 = config['augment16']
        else:
            self.augment16 = False

        if 'data_dir' in config:
            data_dir = pathlib.Path(config['data_dir'])
        else:
            data_dir = pathlib.Path(DEFAULT_DATA_FOLDER)

        self.data_dir = self.download_data(data_dir)
        self.samples, self.vocabulary = self.process_samples()

        def one_hot(i, size):
            one_hot = np.zeros(size)
            one_hot[i] = 1
            return one_hot

        if word_dict == None:
            v_size = self.vocabulary.__len__()
            self.word_dict = {k: one_hot(v, v_size) for v, k in enumerate(sorted(self.vocabulary.keys()))}
            self.re_word_dict = {v: k for v, k in enumerate(sorted(self.vocabulary.keys()))}
        else:
            self.word_dict = word_dict
            self.re_word_dict = re_word_dict

    @staticmethod
    def download_data(data_dir):

        folder_name = 'tasks_1-20_v1-2'

        if (data_dir / folder_name).exists():
            data_dir = data_dir / folder_name

        if not data_dir.name == folder_name:
            data_dir.mkdir(parents=True, exist_ok=True)

            print("Download bAbI data")
            req = Request(bAbI_URL, headers={'User-Agent': 'Mozilla/5.0'})

            with urlopen(req) as files:
                with tarfile.open(fileobj=files, mode="r|gz") as tar:
                    tar.extractall(path=DEFAULT_DATA_FOLDER)

            data_dir = data_dir / folder_name

        return data_dir

    def load_data(self, task_selection):
        text_train = []
        text_test = []
        for task_no in task_selection:
            for subset in self.set_type:
                if (self.data_dir / subset).exists():
                    for file_name in (self.data_dir / subset).iterdir():
                        file_name = file_name.name
                        task, task_name, set = file_name.split("_")
                        if task == "qa" + str(task_no):
                            if set == 'test.txt':
                                test_set_location = self.data_dir / subset / file_name
                                if not test_set_location.exists():
                                    raise UserWarning("File not found: {}".format(test_set_location))
                                with open(str(test_set_location), mode='r') as f:
                                    complete_text = f.readlines()
                                    complete_text = [str(task_no) + ' ' + f for f in complete_text]
                                    text_test += complete_text
                            elif set == "train.txt":
                                train_set_location = self.data_dir / subset / file_name
                                if not train_set_location.exists():
                                    raise UserWarning("File not found: {}".format(train_set_location))
                                with open(str(train_set_location), mode='r') as f:
                                    complete_text = f.readlines()
                                    complete_text = [str(task_no) + ' ' + f for f in complete_text]
                                    text_train += complete_text
                            else:
                                raise UserWarning("Inconsistent bAbI data.")
                else:
                    raise UserWarning("Folder of set type not found, incomplete bAbI data folder or wrong set type")
        return text_train, text_test

    def process_samples(self):
        train_text_list, test_text_list = self.load_data(self.task_selection)

        train_samples, train_word_list = self.build_samples(train_text_list, augment16=self.augment16)
        test_samples, test_word_list = self.build_samples(test_text_list)

        vocabulary = Counter(train_word_list + test_word_list)

        train_samples = self.add_mask(train_samples)
        test_samples = self.add_mask(test_samples)

        valid_amount = int(train_samples.__len__() * self.valid_ratio)
        train_samples = self.rng.permutation(train_samples)
        samples = {}
        samples['train'] = train_samples[valid_amount:]
        samples['valid'] = train_samples[:valid_amount]
        samples['test'] = test_samples
        return samples, vocabulary

    def build_samples(self, text_list, augment16=False):

        word_list = []
        samples = []

        tmp_sample_x = []
        tmp_sample_y = []

        numb_ = 0
        sen_x = ''
        task = 0

        for s in text_list:
            # get sentence number
            numb = int(s.split()[1])
            task = int(s.split()[0])
            if numb < numb_:
                word_list += tmp_sample_x + tmp_sample_y
                if tmp_sample_x.__len__() < self.max_len:
                    if augment16 and int(task) == 16:
                        tmp_sample_x, tmp_sample_y = self.augment_task_16(tmp_sample_x, tmp_sample_y)

                    samples.append({'x': tmp_sample_x, 'y': tmp_sample_y, 'task': int(task)})
                tmp_sample_x = []
                tmp_sample_y = []
                numb_ = numb
            else:
                numb_ = numb
            # remove numbers
            sen = ''.join(i for i in s if not i.isdigit())
            # remove \n
            sen = sen.strip()
            sen = sen.replace('.', ' .')
            sen = sen.replace('?', ' ?')
            sen = sen.lower()
            if '\t' in sen:
                # question
                quest = sen.split("\t")
                sen_x_pre = quest[0].split() + ['-' for i in quest[1].split(",")]
                sen_y = ['-' for i in range(quest[0].split().__len__())] + quest[1].split(",")
            else:
                sen_x_pre = sen.split()
                sen_y = ['-' for i in range(sen_x_pre.__len__())]

            if not 'sen_x' in locals():
                sen_x = sen_x_pre
            elif sen_x_pre == sen_x:
                pass
            else:
                sen_x = sen_x_pre
                tmp_sample_x += sen_x_pre
                tmp_sample_y += sen_y

        word_list += tmp_sample_x + tmp_sample_y

        if augment16 and int(task) == 16:
            tmp_sample_x, tmp_sample_y = self.augment_task_16(tmp_sample_x, tmp_sample_y)

        samples.append({'x': tmp_sample_x, 'y': tmp_sample_y, 'task': int(task)})
        return samples, word_list

    @staticmethod
    def add_mask(samples):

        for sample in samples:
            y = sample['y']
            m = [np.where(_y == '-', False, True) for _y in y]
            sample['m'] = np.asarray(m)
        return samples

    def augment_task_16(self, sample_x, sample_y):

        colores = ['white', 'green', 'gray', 'yellow']
        animals = ['lion', 'rhino', 'swan', 'frog']
        first_numb, second_numb = -1, -1
        first_word, second_word, third_word = '', '', ''

        x = sample_x
        x = ' '.join(x)
        x = x.split('.')
        sa = np.asarray(x)
        if sa.shape[0] == 10:  # some samples are longer but even distributed
            sa = np.reshape(sa, (10, 1))
            s_list = []
            for i in range(10):
                if i != 9:
                    s_list.append([x for x in sa[i, 0].split(' ') if x != ''] + ['.'])
                else:
                    s_list.append([x for x in sa[i, 0].split(' ') if x != ''])
            vec = np.zeros((10)).astype(int)
            quest_word = s_list[-1][3]
            for i in range(9):
                if quest_word in s_list[i]:
                    first_word = s_list[i][3]
                    vec[i] = 1
                    first_numb = i
            for i in range(9):
                if first_word == s_list[i][3] and i != first_numb:
                    second_word = s_list[i][0]
                    vec[i] = 2
                    second_numb = i
            for i in range(9):
                if second_word == s_list[i][0] and i != second_numb:
                    third_word = s_list[i][2]
                    vec[i] = 3

            used_animal_list = []
            used_animal_list.append(first_word)
            for i in range(9):
                word = s_list[i][3]
                if vec[i] == 0 and word in animals:
                    if word in used_animal_list:
                        available_animals = [x for x in animals if x not in used_animal_list]
                        sub_animal = self.rng.choice(available_animals)
                        s_list[i][3] = sub_animal
                        used_animal_list.append(sub_animal)
                    else:
                        used_animal_list.append(word)

            # replace double color
            used_color_list = []
            used_color_list.append(third_word)
            for i in range(9):
                word = s_list[i][2]
                if vec[i] == 0 and word in colores:
                    if word in used_color_list:
                        available_colors = [x for x in colores if x not in used_color_list]
                        sub_color = self.rng.choice(available_colors)
                        s_list[i][2] = sub_color
                        used_color_list.append(sub_color)
                    else:
                        used_color_list.append(word)
            sample_x = [item for sublist in s_list for item in sublist]

        return sample_x, sample_y

    def get_sample(self, set, number):

        sample = copy.deepcopy(self.samples[set][number])
        sample['x_word'] = copy.deepcopy(sample['x'])

        new_x = []
        new_y = []

        for x_word, y_word in zip(sample['x'], sample['y']):
            new_x.append(self.word_dict[x_word])
            new_y.append(self.word_dict[y_word])

        sample['x'] = np.stack(new_x, axis=0)
        sample['y'] = np.stack(new_y, axis=0)

        return sample

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
                x_add = np.zeros([add_len, self.x_size])
                batch['x'].append(np.concatenate([sample['x'], x_add], axis=0))
                y_add = np.zeros([add_len, self.y_size])
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

    def decode_output(self, sample, prediction):
        if prediction.shape.__len__() == 3:
            prediction_decode_list = []
            target_decode_list = []
            for b in range(prediction.shape[1]):
                target_decode_list.append(
                    [self.re_word_dict[np.argmax(sample['y'][i, b, :])] for i in range(sample['y'].shape[0])])
                prediction_decode_list.append(
                    [self.re_word_dict[np.argmax(prediction[i, b, :])] for i in range(prediction.shape[0])])
            return target_decode_list, prediction_decode_list
        else:
            target_decode = [self.re_word_dict[np.argmax(sample['y'][i, :])] for i in range(sample['y'].shape[0])]
            prediction_decode = [self.re_word_dict[np.argmax(prediction[i, :])] for i in range(prediction.shape[0])]
            return target_decode, prediction_decode

    @property
    def vocabulary_size(self):
        return self.word_dict.__len__()

    @property
    def x_size(self):
        return self.vocabulary_size

    @property
    def y_size(self):
        return self.vocabulary_size

    def sample_amount(self, set, max_len=False):
        if max_len != False:
            lengths = [sample['x'].__len__() for sample in self.samples[set]]
            return sum(np.asarray(lengths) <= max_len)
        else:
            return self.samples[set].__len__()


if __name__ == '__main__':

    config = {'set_type': ['en-10k'], 'task_selection': ['all'], 'valid_ratio': 0, 'seed': 211}

    print("                                           20 bAbI Tasks - Statistics")
    print(
        "________________________________________________________________________________________________________________")
    total_len = []
    total_sum = 0

    for s in range(20):
        config['task_selection'] = [s + 1]

        sd = bAbI(config)
        samples = [sd.get_sample('train', i) for i in range(sd.sample_amount('train'))]
        len = [sample['x'].__len__() for sample in samples]
        len = np.asarray(len)
        quest_per_sample = [sample['m'].sum() for sample in samples]
        quest_per_sample = np.mean(quest_per_sample)
        vocab_size = sd.vocabulary_size
        print("\033[96m task: {:3}\033[0m, samples: {:5}, quest_per_sample {:5.3f}, vocab_size: {:3}, min len: {:3.0f},"
              " mean len: {:3.0f}, max len: {:4.0f}".format(s + 1, len.shape[0], quest_per_sample, vocab_size,
                                                            len.min(),
                                                            len.mean(), len.max()))
    print(
        "________________________________________________________________________________________________________________")
    config['task_selection'] = ['all']
    sd = bAbI(config)
    word_dict = sd.word_dict
    re_word_dict = sd.re_word_dict
    samples = [sd.get_sample('train', i) for i in range(sd.sample_amount('train'))]
    len = [sample['x'].__len__() for sample in samples]
    len = np.asarray(len)
    quest_per_sample = [sample['m'].sum() for sample in samples]
    quest_per_sample = np.mean(quest_per_sample)
    vocab_size = sd.vocabulary_size
    print("\033[96m task: {}\033[0m, samples: {:5}, quest_per_sample {:5.3f}, vocab_size: {:3}, min len: " \
          "{:3.0f}, mean len: {:3.0f}, max len: {:4.0f}".format('all', len.shape[0], quest_per_sample,
                                                                vocab_size, len.min(), len.mean(), len.max()))

    print("\nbAbI Tasks 16 - Example without augmentation")
    print("____________________________________________")
    config = {'set_type': ['en-10k'], 'task_selection': [16], 'valid_ratio': 0, 'seed': 211, 'max_len': 2500}
    config['augment16'] = False
    sd = bAbI(config)
    sample = sd.get_sample('train', 0)
    print(' '.join(sample['x_word']))
    print("### bernhard and greg are both gray")

    print("\nbAbI Tasks 16 - Example with augmentation")
    print("_________________________________________")
    config = {'set_type': ['en-10k'], 'task_selection': [16], 'valid_ratio': 0, 'seed': 211, 'max_len': 2500}
    config['augment16'] = True
    sd = bAbI(config)
    sample = sd.get_sample('train', 0)
    print(' '.join(sample['x_word']))
    print("### no color is double")
