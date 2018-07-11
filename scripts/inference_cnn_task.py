#!/usr/bin/env python
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

import tensorflow as tf
import yaml
from tqdm import tqdm

from adnc.data.loader import DataLoader
from adnc.model.mann import MANN

"""
This script performs a inference with the given models of this repository on the CNN RC task.
"""

expt_dir = "experiments/pre_trained/cnn_rc_task/adnc"

# load config file
with open(os.path.join(expt_dir, 'config.yml'), 'r') as f:
    configs = yaml.load(f)
data_set_config = configs['cnn']
model_config = configs['mann']

# set batch size to 1 for inference and max_len to maximal value
BATCH_SIZE = 1
data_set_config['batch_size'] = BATCH_SIZE
data_set_config['max_len'] = 20000
model_config['batch_size'] = BATCH_SIZE

# load data loader and restore word to index dictionary
dl = DataLoader(data_set_config)
valid_loader = dl.get_data_loader('valid', max_len=20000)
test_loader = dl.get_data_loader('test')
dl.dataset.load_dictionary(expt_dir)

# set model config for inference
model_config['input_size'] = dl.x_size
model_config['output_size'] = dl.y_size
model_config['memory_unit_config']['bypass_dropout'] = False
model_config['input_embedding'] = {'word_idx_dict': dl.dataset.word_idx_dict, 'embedding_size': 100}

# init MANN model
model = MANN(model_config)

# print model/data values
print("vocabulary size: {}".format(dl.vocabulary_size))
print("test set length: {}".format(dl.sample_amount('test')))
print("test batch amount: {}".format(dl.batch_amount('test')))
print("valid set length: {}".format(dl.sample_amount('valid')))
print("valid batch amount: {}".format(dl.batch_amount('valid')))
print("model parameter amount: {}".format(model.parameter_amount))
print("x size: {}".format(dl.y_size))

# calculate error
data = model._data
answer_idx = tf.placeholder(tf.int64, [data_set_config['batch_size']], name='labels')
candidates_mask = tf.placeholder(tf.float32, [data_set_config['batch_size'], dl.y_size], name='candidates')
outputs = model.outputs
last_output = outputs[-1, :, :]
masked_output = candidates_mask * last_output
predictions = tf.nn.softmax(masked_output, dim=-1)
arg_predictions = tf.arg_max(predictions, dimension=-1)
equal = tf.cast(tf.equal(arg_predictions, answer_idx), tf.float32)
error_rate = tf.reduce_mean(equal)

# init saver
saver = tf.train.Saver()

# init session
conf = tf.ConfigProto()
conf.gpu_options.allocator_type = 'BFC'
conf.gpu_options.allow_growth = True

with tf.Session(config=conf) as sess:
    saver.restore(sess, os.path.join(expt_dir, "model_dump.ckpt"))

    # calculate validation error
    valid_cost = 0
    valid_error = 0
    valid_count = 0
    for _ in tqdm(range(int(dl.batch_amount('valid')))):
        sample = next(valid_loader)
        verror = sess.run([error_rate], feed_dict={data: sample['x'], answer_idx: sample['answer_idx'],
                                                   candidates_mask: sample['candidates']})
        valid_error += verror
        valid_count += 1
    valid_error = valid_error / valid_count
    print("valid: tf mean acc: {}".format(valid_error))

    # calculate test error
    test_cost = 0
    test_error = 0
    test_count = 0
    for _ in tqdm(range(int(dl.batch_amount('test')))):
        sample = next(test_loader)
        terror = sess.run([error_rate], feed_dict={data: sample['x'], answer_idx: sample['answer_idx'],
                                                   candidates_mask: sample['candidates']})
        test_error += terror
        test_count += 1
    test_error = test_error / test_count
    print("test: tf mean acc: {}".format(test_error))
