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

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # gpu not required for inference

import argparse
import yaml
import numpy as np
import tensorflow as tf

from adnc.data.loader import DataLoader
from adnc.model.mann import MANN

"""
This script performs a inference with the given models of this repository on the bAbI task 1 or on 1-20. Please add the 
model name when calling the script. (dnc, adnc, biadnc, biadnc-all, biadnc-aug16-all) 
"""

parser = argparse.ArgumentParser(description='Load model')
parser.add_argument('model', type=str, default=False, help='model name')
model_name = parser.parse_args().model

# Choose a pre trained model by uncomment
if model_name == 'dnc':
    expt_dir = "experiments/pre_trained/babi_task_1/dnc"  # DNC trained on bAbI tasks 1
elif model_name == 'adnc':
    expt_dir = "experiments/pre_trained/babi_task_1/adnc"  # ADNC trained on bAbI tasks 1
elif model_name == 'biadnc':
    expt_dir = "experiments/pre_trained/babi_task_1/biadnc"  # BiADNC trained on bAbI tasks 1
elif model_name == 'biadnc-all':
    expt_dir = "experiments/pre_trained/babi_task_all/biadnc"  # BiADNC trained on all bAbI tasks
else:
    expt_dir = "experiments/pre_trained/babi_task_all/biadnc_aug16"  # BiADNC trained on all bAbI tasks with task 16 augmentation

config_file = 'config.yml'
with open(os.path.join(expt_dir, config_file), 'r') as f:
    configs = yaml.load(f)  # load config from file

dataset_config = configs['babi_task']
model_config = configs['mann']

dataset_config['batch_size'] = 1
model_config['batch_size'] = 1

dataset_config['threads'] = 1  # only one thread for data loading
dataset_config['max_len'] = 1921  # set max length to maximal
dataset_config['augment16'] = False  # disable augmentation for inference

if dataset_config['task_selection'] == ['all']:
    task_list = [i + 1 for i in range(20)]
else:
    task_list = [int(i) for i in dataset_config['task_selection']]

dl = DataLoader(dataset_config)  # load data loader by config

model_config['input_size'] = dl.x_size  # add data size to model config
model_config['output_size'] = dl.y_size
model_config['memory_unit_config']['bypass_dropout'] = False  # no dropout during inference

model = MANN(model_config)  # load memory augmented neural network  model

data, target, mask = model.feed  # create data feed for session run

word_dict = dl.dataset.word_dict  # full dictionary of all tasks
re_word_dict = dl.dataset.re_word_dict  # reverse dictionary

print("vocabulary size: {}".format(dl.vocabulary_size))
print("train set length: {}".format(dl.sample_amount('train')))
print("valid set length: {}".format(dl.sample_amount('valid')))
print("model parameter amount: {}".format(model.parameter_amount))

saver = tf.train.Saver()
conf = tf.ConfigProto()
conf.gpu_options.per_process_gpu_memory_fraction = 0.8
conf.gpu_options.allocator_type = 'BFC'
conf.gpu_options.allow_growth = True

with tf.Session(config=conf) as sess:
    saver.restore(sess, os.path.join(expt_dir, "model_dump.ckpt"))
    mean_error = []
    for task in task_list:

        # load data loader for task
        dataset_config['task_selection'] = [task]
        dl = DataLoader(dataset_config, word_dict, re_word_dict)
        valid_loader = dl.get_data_loader('test')

        predictions, targets, masks = [], [], []
        all_corrects, all_overall = 0, 0

        # infer model
        for v in range(int(dl.batch_amount('test'))):
            sample = next(valid_loader)
            prediction = sess.run([model.prediction, ], feed_dict={data: sample['x'],
                                                                   target: sample['y'],
                                                                   mask: sample['m']})
            predictions.append(prediction)
            targets.append(sample['y'])
            masks.append(sample['m'])

        # calculate mean error rate for task
        for p, t, m in zip(predictions, targets, masks):
            tm = np.argmax(t, axis=-1)
            pm = np.argmax(p, axis=-1)
            corrects = np.equal(tm, pm)
            all_corrects += np.sum(corrects * m)
            all_overall += np.sum(m)

        word_error_rate = 1 - (all_corrects / all_overall)
        mean_error.append(word_error_rate)

        print("word error rate task {:2}: {:0.3}".format(task, word_error_rate))
    print("mean word error rate   : {:0.3}".format(np.mean(mean_error)))
