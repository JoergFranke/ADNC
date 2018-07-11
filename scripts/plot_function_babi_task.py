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
import yaml
import argparse
import numpy as np
import tensorflow as tf

from adnc.data import DataLoader
from adnc.model import MANN, Optimizer
from adnc.analysis import Bucket, PlotFunctionality

tf.reset_default_graph()

"""
This script plot the memory unit functionality with the given models of this repository on the bAbI task 1 or on 1-20. 
Please add the model name when calling the script. (dnc, adnc, biadnc, biadnc-all, biadnc-aug16-all) 
"""

parser = argparse.ArgumentParser(description='Load model')
parser.add_argument('model', type=str, default=False, help='model name')
model_name = parser.parse_args().model

# Choose a pre trained model by uncomment
if model_name == 'dnc':
    model_dir = "experiments/pre_trained/babi_task_1/dnc"  # DNC trained on bAbI tasks 1
elif model_name == 'adnc':
    model_dir = "experiments/pre_trained/babi_task_1/adnc"  # ADNC trained on bAbI tasks 1
elif model_name == 'biadnc':
    model_dir = "experiments/pre_trained/babi_task_1/biadnc"  # BiADNC trained on bAbI tasks 1
elif model_name == 'biadnc-all':
    model_dir = "experiments/pre_trained/babi_task_all/biadnc"  # BiADNC trained on all bAbI tasks
else:
    model_dir = "experiments/pre_trained/babi_task_all/biadnc_aug16"  # BiADNC trained on all bAbI tasks with task 16 augmentation

plot_dir = "experiments/"


analyse = True
BATCH_SIZE = 10
BATCH_SAMPLE = 2


# load config from file
with open(os.path.join(model_dir, 'config.yml'), 'r') as f:
    configs = yaml.load(f)
dataset_config = configs['babi_task']
trainer_config = configs['training']
model_config = configs['mann']


dataset_config['batch_size'] = BATCH_SIZE
model_config['batch_size'] = BATCH_SIZE

dl = DataLoader(dataset_config)

model_config['input_size'] = dl.x_size
model_config['output_size'] = dl.y_size

word_dict = dl.dataset.word_dict
re_word_dict = dl.dataset.re_word_dict
dataset_config['task_selection'] = [1]
dl2 = DataLoader(dataset_config, word_dict, re_word_dict)
valid_loader = dl2.get_data_loader('valid')


model = MANN(model_config, analyse=True)

data, target, mask = model.feed

trainer = Optimizer(trainer_config, model.loss, model.trainable_variables)
saver = tf.train.Saver()


conf = tf.ConfigProto()
conf.gpu_options.allocator_type = 'BFC'
conf.gpu_options.allow_growth = True
with tf.Session(config=conf) as sess:

    saver.restore(sess, os.path.join(model_dir, "model_dump.ckpt"))

    vsample = next(valid_loader)

    analyse_values, prediction, gradients = sess.run([model.analyse, model.prediction, trainer.gradients],
                                                           feed_dict={data: vsample['x'], target: vsample['y'], mask: vsample['m']})
    weights = {v.name: {'var':g[1], 'grad':g[0], 'shape':g[0].shape } for v, g in zip(model.trainable_variables, gradients)}
    if 'x_word' not in vsample.keys():
        vsample['x_word'] = np.transpose(np.argmax(vsample['x'], axis=-1),(1,0))
    data_sample = [vsample['x'], vsample['y'], vsample['m'], vsample['x_word'],]

    decoded_targets, decoded_predictions = dl.decode_output(vsample, prediction)

    save_list = [analyse_values, prediction, decoded_predictions, data_sample, weights ]
    babi_bucket = Bucket(save_list, babi_short=True)

    plotter = PlotFunctionality(babi_bucket, title=True, legend=True, text_size=22)
    plotter.plot_short_process(batch=BATCH_SAMPLE, plot_dir=plot_dir, name='function plot {}'.format(model_name))
    # plot_advanced_functionality(batch=BATCH_SAMPLE, plot_dir=plot_dir, name='extended function plot {}'.format(model_name))




