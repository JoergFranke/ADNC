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
import operator
import pathlib
import tarfile
import pickle
from collections import Counter, OrderedDict
from urllib.request import Request, urlopen

import numpy as np

from adnc.data.utils.data_memorizer import DataMemorizer

"""
Downloads and pre-preocess the CNN RC task. 
"""

DEFAULT_DATA_FOLDER = 'data_cnn'
DEFAULT_TMP_FOLDER = 'data_tmp'
CNN_DATA_URL = 'http://cs.stanford.edu/~danqi/data/cnn.tar.gz'

class ReadingComprehension():
    def __init__(self, config, save=True, debug_max_load=None):

        max_words = 50000

        self.seed = config['seed']
        self.rng = np.random.RandomState(seed=self.seed)

        if 'data_dir' in config:
            data_dir = pathlib.Path(config['data_dir'])
        else:
            data_dir = pathlib.Path(DEFAULT_DATA_FOLDER)

        if 'data_dir' in config:
            tmp_dir = pathlib.Path(config['data_dir'])
        else:
            tmp_dir = pathlib.Path(DEFAULT_TMP_FOLDER)

        if 'answer_first' in config:
            self.answer_first = config['answer_first']
        else:
            self.answer_first = False

        if 'max_len' in config.keys():
            self.max_len = config['max_len']
        else:
            self.max_len = False

        hash_config = {'seed': self.seed, 'answer_first': self.answer_first}
        data_memory = DataMemorizer(hash_config, tmp_dir)

        self.data_dir = self.download_data(data_dir)

        ######################### old

        if data_memory():
            print('### RC: restore data')
            self.samples, self.word_idx_dict, self.entity_dict = data_memory.load_data()

        else:
            print('### RC: create data')
            (train_documents, train_questions, train_answers), word_counter_train = self.load_data(
                self.data_dir / 'train.txt', debug_max_load)
            (dev_documents, dev_questions, dev_answers), word_counter_dev = self.load_data(self.data_dir / 'dev.txt',
                                                                                           debug_max_load)
            (test_documents, test_questions, test_answers), word_counter_test = self.load_data(
                self.data_dir / 'test.txt', debug_max_load)

            word_count = word_counter_train + word_counter_dev + word_counter_test
            word_count['<QST>'] = 100000
            word_count['<MASK>'] = 100000

            word_idx_dict = self.consistent_most_common(word_count, max_words)

            entity_markers = list(set([w for w in word_idx_dict.keys() if
                                       w.startswith('@entity')] + train_answers + dev_answers + test_answers))
            entity_markers = ['<unk_entity>'] + entity_markers
            entity_dict = {w: index for (index, w) in enumerate(sorted(entity_markers))}

            self.samples = {'train': [], 'valid': [], 'test': []}
            self.samples['train'] = self.vectorize_samples(train_documents, train_questions, train_answers,
                                                           word_idx_dict, entity_dict)
            self.samples['valid'] = self.vectorize_samples(dev_documents, dev_questions, dev_answers, word_idx_dict,
                                                           entity_dict)
            self.samples['test'] = self.vectorize_samples(test_documents, test_questions, test_answers, word_idx_dict,
                                                          entity_dict)

            self.word_idx_dict = word_idx_dict
            self.entity_dict = entity_dict

            if save:
                print('### RC: store data')
                data_memory.dump_data([self.samples, self.word_idx_dict, self.entity_dict])

        self.re_entity_dict = {v: k for k, v in self.entity_dict.items()}
        self.idx_word_dict = {v: k for k, v in self.word_idx_dict.items()}

    @staticmethod
    def download_data(data_dir):

        folder_name = 'cnn'

        if (data_dir / folder_name).exists():
            data_dir = data_dir / folder_name

        if not data_dir.name == folder_name:
            data_dir.mkdir(parents=True, exist_ok=True)

            print("### Download CNN data")
            req = Request(CNN_DATA_URL, headers={'User-Agent': 'Mozilla/5.0'})
            print("### Extract CNN data")
            with urlopen(req) as files:
                with tarfile.open(fileobj=files, mode="r|gz") as tar:
                    tar.extractall(path=DEFAULT_DATA_FOLDER)
            print("### CNN data complete")
            data_dir = data_dir / folder_name

        return data_dir

    @staticmethod
    def consistent_most_common(dict, max_keys):

        sorted_dict_list = sorted(dict.items(), key=operator.itemgetter(1))
        most_common_dict = OrderedDict()

        pre_v = sorted_dict_list[-1][1]
        for i, (k, v) in enumerate(sorted_dict_list[::-1]):
            if i >= max_keys and pre_v != v:
                break
            else:
                most_common_dict[k] = i
                pre_v = v

        return most_common_dict

    @staticmethod
    def load_data(in_file, max_example=None, relabeling=True):
        """
            load CNN / Daily Mail data from {train | dev | test}.txt
            relabeling: relabel the entities by their first occurence if it is True.
        """
        word_counter = Counter()
        documents = []
        questions = []
        answers = []
        num_examples = 0
        f = open(str(in_file), 'r', encoding='utf-8')
        while True:
            line = f.readline()
            if not line:
                break
            question = line.strip().lower()
            answer = f.readline().strip()
            document = f.readline().strip().lower()

            if relabeling:
                q_words = question.split(' ')
                d_words = document.split(' ')
                assert answer in d_words

                entity_dict = {}
                entity_id = 0
                for word in d_words + q_words:
                    if (word.startswith('@entity')) and (word not in entity_dict):
                        entity_dict[word] = '@entity' + str(entity_id)
                        entity_id += 1

                q_words = [entity_dict[w] if w in entity_dict else w for w in q_words]
                d_words = [entity_dict[w] if w in entity_dict else w for w in d_words]
                answer = entity_dict[answer]

                for word in q_words:
                    word_counter[word] += 1
                for word in d_words:
                    word_counter[word] += 1

            questions.append(q_words)
            answers.append(answer)
            documents.append(d_words)
            num_examples += 1

            f.readline()
            if (max_example is not None) and (num_examples >= max_example):
                break
        f.close()
        return (documents, questions, answers), word_counter

    def vectorize_samples(self, documents, questions, answers, word_dict, entity_dict):

        samples = []
        for idx, (d_words, q_words, a) in enumerate(zip(documents, questions, answers)):

            assert (a in d_words)
            doc = [word_dict[w] if w in word_dict else 0 for w in d_words]
            quest = [word_dict[w] if w in word_dict else 0 for w in q_words]
            if (len(doc) > 0) and (len(quest) > 0):
                if self.answer_first:
                    x_in = quest + [word_dict['<QST>'], ] + doc
                else:
                    x_in = doc + [word_dict['<QST>'], ] + quest
                x_in = np.asarray(x_in)
                candidates = np.zeros((len(entity_dict)), dtype=bool)
                candidates[[entity_dict[w] for w in d_words if w in entity_dict]] = True
                answer_idx = entity_dict[a] if a in entity_dict else 0
                answer_idx = np.asarray(answer_idx)
                samples.append({'x': x_in, 'candidates': candidates, 'answer_idx': answer_idx})

        return samples

    def get_sample(self, set, number):

        return_sample = self.samples[set][number]
        return return_sample

    def patch_batch(self, list_of_samples):

        max_len = [sample['x'].shape[0] for sample in list_of_samples]
        max_len = max(max_len)

        batch = OrderedDict()
        batch['x'] = np.ones([max_len, list_of_samples.__len__()]) * self.word_idx_dict['<MASK>']
        batch['m'] = np.zeros([max_len, list_of_samples.__len__()])
        batch['answer_idx'] = np.empty([list_of_samples.__len__()])
        batch['candidates'] = np.zeros([list_of_samples.__len__(), self.entity_dict.__len__()])

        for idx, sample in enumerate(list_of_samples):
            start_index = max_len - sample['x'].shape[0]
            batch['x'][start_index:, idx] = sample['x']
            batch['m'][start_index:, idx] = 1
            batch['answer_idx'][idx] = sample['answer_idx']
            batch['candidates'][idx, :] = sample['candidates']
        return batch

    def decode_output(self, sample, prediction):
        pred_argmax = np.argmax(prediction, axis=-1)
        if pred_argmax is list:
            prediction_decode = [self.re_entity_dict[ent] for ent in list(pred_argmax)]
        else:
            prediction_decode = self.re_entity_dict[pred_argmax]
        x_word = [self.idx_word_dict[idx] for idx in sample['x']]

        return x_word, prediction_decode

    def save_dictionary(self, dir):
        with open(os.path.join(dir, 'word_idx_dict.pkl'), 'wb') as outfile:
            pickle.dump(self.word_idx_dict, outfile)

    def load_dictionary(self, dir):
        with open(os.path.join(dir, 'word_idx_dict.pkl'), 'rb') as outfile:
            self.word_idx_dict = pickle.load(outfile)
            self.idx_word_dict = {v:k for k,v in self.word_idx_dict.items()}

    @property
    def vocabulary_size(self):
        return self.word_idx_dict.__len__()

    @property
    def x_size(self):
        return 1

    @property
    def y_size(self):
        return self.entity_dict.__len__()

    def sample_amount(self, set, max_len=False):
        if max_len != False:
            lengths = [sample['x'].__len__() for sample in self.samples[set]]
            return sum(np.asarray(lengths) <= max_len)
        else:
            return self.samples[set].__len__()


if __name__ == '__main__':
    config = {'seed': 223, 'max_len': 20000, 'answer_first': False}

    sd = ReadingComprehension(config, save=True, debug_max_load=None)
    samples = sd.samples

    print("                               CNN task Statistics")
    print("_____________________________________________________________________________________")
    total_len = []
    total_sum = 0

    len = [sample['x'].__len__() for sample in samples['train']]
    len = np.asarray(len)
    vocab_size = sd.vocabulary_size
    print("train samples: {:5}, vocab_size: {:3}, min len: {:3.0f}, mean len: {:3.0f}, max len: {:4.0f}".format(
        len.shape[0], vocab_size, len.min(), len.mean(), len.max()))
    print("max len amount: {}".format(sd.sample_amount('train', max_len=1400)))

    len = [sample['x'].__len__() for sample in samples['valid']]
    len = np.asarray(len)
    vocab_size = sd.vocabulary_size
    print("valid samples: {:5}, vocab_size: {:3}, min len: {:3.0f}, mean len: {:3.0f}, max len: {:4.0f}".format(
        len.shape[0], vocab_size, len.min(), len.mean(), len.max()))
    print("max len amount: {}".format(sd.sample_amount('valid', max_len=1400)))

    len = [sample['x'].__len__() for sample in samples['test']]
    len = np.asarray(len)
    vocab_size = sd.y_size
    print("test samples: {:5}, vocab_size: {:3}, min len: {:3.0f}, mean len: {:3.0f}, max len: {:4.0f}".format(
        len.shape[0], vocab_size, len.min(), len.mean(), len.max()))
    print("max len amount: {}".format(sd.sample_amount('test', max_len=1400)))
