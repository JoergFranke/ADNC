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
import os
import pickle
import urllib.request
import zipfile
import hashlib
from collections import OrderedDict
import tensorflow as tf

"""
Downloads and process glove word embeddings, applies them to a given vocabulary of a dataset. 
"""

class WordEmbedding():
    def __init__(self, embedding_size, vocabulary_size=None, word_idx_dict=None, initialization='uniform', tmp_dir='.',
                 dtype=tf.float32, seed=123):

        self.rng = np.random.RandomState(seed)

        if vocabulary_size == None:
            vocabulary_size = word_idx_dict.__len__()

        if initialization == 'uniform':
            init_tensor = self.initialize_random(vocabulary_size, embedding_size, dtype)
        elif initialization == 'glove':
            init_tensor = self.initialize_with_glove(word_idx_dict, embedding_size, tmp_dir, dtype)

        self.embeddings = tf.Variable(init_tensor, dtype=dtype, name='word_embedding')

    def embed(self, word_idx):
        embed = tf.nn.embedding_lookup(self.embeddings, word_idx, name='embedding_lookup')
        return embed

    @staticmethod
    def initialize_random(vocabulary_size, embedding_size, dtype):
        return tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0, dtype=dtype)

    def initialize_with_glove(self, word_idx_dict, embedding_size, tmp_dir, dtype):

        if embedding_size == 100:
            glove_type = '6B'
        elif embedding_size == 300:
            glove_type = '42B'
        else:
            raise UserWarning('embedding size incompatible to glove word representations')

        embeddings = self.get_glove_embeddings(tmp_dir, glove_type, embedding_size, word_idx_dict)
        return embeddings

    @staticmethod
    def make_dict_hash(dictionary):
        pre = sorted(((k, v) for k, v in dictionary.items()))
        sort_dict = OrderedDict()
        for element in pre:
            if type(element[1]) == dict:
                element_sort = OrderedDict(sorted(element[1].items()))
                for key, value in element_sort.items():
                    sort_value = sorted(value.items())
                    sort_dict[key] = sort_value
            else:
                sort_dict[element[0]] = element[1]

        hash_object = hashlib.md5(str(sort_dict).encode())
        hash = str(hash_object.hexdigest())
        return hash

    def get_glove_embeddings(self, glove_path, glove_set, glove_dim, word_idx_dict):

        dict_hash = self.make_dict_hash(word_idx_dict)
        embeddings_file = os.path.join(glove_path,
                                       'glove_embeddings_{}_{}_{}.plk'.format(dict_hash, glove_set, glove_dim))

        if os.path.isfile(embeddings_file):
            with open(embeddings_file, 'rb') as dict_file:
                embeddings = pickle.load(dict_file)
        else:
            embeddings = self.prepare_glove_embeddings(glove_path, glove_set, glove_dim, word_idx_dict)
            self.save_glove_embeddings(embeddings_file, embeddings)
        return embeddings

    @staticmethod
    def save_glove_embeddings(embeddings_file, embeddings):
        with open(embeddings_file, 'wb') as dict_file:
            pickle.dump(embeddings, dict_file)

    def prepare_glove_embeddings(self, glove_path, glove_set, glove_dim, word_idx_dict):

        glove_embeddings_file = os.path.join(glove_path, "glove.{}.{}d.txt".format(glove_set, glove_dim))

        if not os.path.isfile(glove_embeddings_file):
            print("### Download GloVe Word Representation")
            if glove_set == '6B':
                url = 'http://nlp.stanford.edu/data/glove.6B.zip'
            elif glove_set == '42B':
                url = 'http://nlp.stanford.edu/data/glove.42B.300d.zip'
            elif glove_set == '840B':
                url = 'http://nlp.stanford.edu/data/glove.840B.300d.zip'

            file_name = url.split('/')[-1]
            zip_path = os.path.join(glove_path, file_name)
            filehandle, _ = urllib.request.urlretrieve(url, zip_path)
            zip_file = zipfile.ZipFile(zip_path)
            zip_file.extractall(glove_path)
            zip_file.close()
            os.remove(zip_path)

        glove_dict = OrderedDict()
        with open(glove_embeddings_file, 'r', encoding='utf-8') as gl:
            for line in gl:
                array = line.lstrip().rstrip().split(" ")
                word = array[0]
                vector = list(map(float, array[1:]))
                vector = np.asarray(vector)
                glove_dict[word.lower()] = vector

        # aline vocabulary with GloVe vectors
        embeddings = np.empty([word_idx_dict.__len__(), glove_dim])
        OOV = []
        for word, idx in word_idx_dict.items():
            try:
                vec = glove_dict[word]
                embeddings[idx, :] = vec
            except:
                OOV.append(word)
                rand_vec = self.rng.uniform(-1, 1, glove_dim)
                embeddings[idx, :] = rand_vec
        del (glove_dict)

        print('### Word Embeddings: Out of vocabulary words: {}'.format(OOV.__len__()))
        return embeddings
