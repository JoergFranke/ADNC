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
import argparse
import os
import sys
import time

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from adnc.analysis import Analyser
from adnc.data import DataLoader
from adnc.model import MANN, Optimizer, Supporter
from adnc.model.utils import EarlyStop

"""
This script performs starts a training run on the bAbI task. The training can be fully configured in the config.yml
file. To restore a session use the --sess and --check flag.
"""

tf.reset_default_graph()

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--sess', type=int, default=False, help='session number')
parser.add_argument('--check', type=int, default=False, help='restore checkpoint')
args = parser.parse_args()

session_no = args.sess  # allows to restore a specific session
if not session_no:
    session_no = False

restore_checkpoint = args.check  # allows to restore a specific checkpoint
if not restore_checkpoint:
    restore_checkpoint = False

dataset_name = 'babi_task'  # defines the dataset choosen from config
model_type = 'mann'  # type of model, currently only 'mann'

experiment_name = 'github_example'  # name of the experiment

project_dir = 'experiments/'  # folder to save experiments
config_file = 'config.yml'  # name of config file

early_stop = EarlyStop(10)  # initialize early stopping after 10 higher losses in a row

analyse = True  # allows a closer analysis of the training progress, like memory influence
plot_process = True  # plots a function plot after each epoch

sp = Supporter(project_dir, config_file, experiment_name, dataset_name, model_type,
               session_no)  # initializes supporter class for experiment handling

dl = DataLoader(sp.config(dataset_name))  # initializes data loader class
valid_loader = dl.get_data_loader('valid')  # gets a valid data iterator
train_loader = dl.get_data_loader('train')  # gets a train data iterator

if analyse:
    ana = Analyser(sp.session_dir, save_fig=plot_process,
                   save_variables=True)  # initilizes a analyzer class

sp.config(model_type)['input_size'] = dl.x_size  # after the data loader is initilized, the input size
sp.config(model_type)['output_size'] = dl.y_size  # and output size is known and used for the model
model = MANN(sp.config('mann'), analyse)  # initilizes the model class

data, target, mask = model.feed  # TF data, target and mask placeholders for training

trainer = Optimizer(sp.config('training'), model.loss,
                    model.trainable_variables)  # initilizes a trainer class with the optimizer
optimizer = trainer.optimizer  # the optimizer for training, similar to TF

init_op = tf.global_variables_initializer()
saver = tf.train.Saver(max_to_keep=30)

summary_train_loss = tf.summary.scalar("train_loss", model.loss)
summary_valid_loss = tf.summary.scalar("valid_loss", model.loss)
lstm_scale = tf.summary.scalar("lstm_scale", tf.reduce_mean(model.trainable_variables[2]))
lstm_beta = tf.summary.scalar("lstm_beta", tf.reduce_mean(model.trainable_variables[3]))

sp.pub("vocabulary size: {}".format(dl.vocabulary_size))  # prints values and logs it to a log file
sp.pub("train set length: {}".format(dl.sample_amount('train')))
sp.pub("train batch amount: {}".format(dl.batch_amount('train')))
sp.pub("valid set length: {}".format(dl.sample_amount('valid')))
sp.pub("valid batch amount: {}".format(dl.batch_amount('valid')))
sp.pub("model parameter amount: {}".format(model.parameter_amount))

conf = tf.ConfigProto()  # TF session config for optimal GPU usage
conf.gpu_options.per_process_gpu_memory_fraction = 0.8
conf.gpu_options.allocator_type = 'BFC'
conf.gpu_options.allow_growth = True
conf.allow_soft_placement = True

with tf.Session(config=conf) as sess:
    if sp.restore and restore_checkpoint:  # restores model dumps after a crash or to continiue training
        saver.restore(sess, os.path.join(sp.session_dir, "model_dump_{}.ckpt".format(restore_checkpoint)))
        epoch_start = restore_checkpoint + 1
        sp.pub("restart training with checkpoint {}".format(epoch_start - 1))
    elif sp.restore and not restore_checkpoint:
        if tf.train.latest_checkpoint(sp.session_dir) == None:
            sess.run(init_op)
            epoch_start = 0
            sp.pub("start new training")
        else:
            saver.restore(sess, tf.train.latest_checkpoint(sp.session_dir))
            epoch_start = int(tf.train.latest_checkpoint(sp.session_dir).split('_')[-1].split('.')[0]) + 1
            sp.pub("restart training with checkpoint {}".format(epoch_start - 1))
    else:
        sess.run(init_op)
        epoch_start = 0
        sp.pub("start new training")

    writer = tf.summary.FileWriter(os.path.join(sp.session_dir, "summary"), sess.graph)

    for e in range(epoch_start, sp.config('training')['epochs']):  # loop over all training epochs

        train_cost = 0
        train_count = 0
        all_corrects = 0
        all_overall = 0
        time_e = time.time()
        time_0 = time.time()

        for step in tqdm(range(int(dl.batch_amount('train')))):  # loop over all training samples

            sample = next(train_loader)  # new training sample from train iterator

            _, c, summary, lb, ls = sess.run([optimizer, model.loss, summary_train_loss, lstm_beta, lstm_scale],
                                             feed_dict={data: sample['x'], target: sample['y'], mask: sample['m']})
            train_cost += c
            train_count += 1
            writer.add_summary(summary, e * dl.batch_amount('train') + step)
            writer.add_summary(lb, e * dl.batch_amount('train') + step)
            writer.add_summary(ls, e * dl.batch_amount('train') + step)

        valid_cost = 0
        valid_count = 0

        for v in range(int(dl.batch_amount('valid'))):  # loop over all validation samples
            vsample = next(valid_loader)
            vcost, vpred, summary = sess.run([model.loss, model.prediction, summary_valid_loss],
                                             feed_dict={data: vsample['x'], target: vsample['y'], mask: vsample['m']})
            valid_cost += vcost
            valid_count += 1
            writer.add_summary(summary, e * dl.batch_amount('valid') + v)
            tm = np.argmax(vsample['y'], axis=-1)  # calculates the word error rate
            pm = np.argmax(vpred, axis=-1)
            corrects = np.equal(tm, pm)
            all_corrects += np.sum(corrects * vsample['m'])
            all_overall += np.sum(vsample['m'])

        valid_cost = valid_cost / valid_count
        train_cost = train_cost / train_count
        word_error_rate = 1 - (all_corrects / all_overall)

        if not np.isnan(valid_cost):  # checks NAN

            save_path = saver.save(sess,
                                   os.path.join(sp.session_dir, "model_dump_{}.ckpt".format(e)))  # dumps model weights

            if analyse:  # if analysis, it logs memory influence and plots functionality

                controller_inf = []
                memory_inf = []
                all_corrects = 0
                all_overall = 0

                for vstep in range(
                        10):  # makes ten valid inferneces to get the gradiens for memory influence calculation
                    vsample = next(valid_loader)

                    analyse_values, prediction, gradients = sess.run(
                        [model.analyse, model.prediction, trainer.gradients],
                        feed_dict={data: vsample['x'], target: vsample['y'], mask: vsample['m']})
                    weights = {v.name: {'var': g[1], 'grad': g[0], 'shape': g[0].shape} for v, g in
                               zip(model.trainable_variables, gradients)}
                    if 'x_word' not in vsample.keys():
                        vsample['x_word'] = np.transpose(np.argmax(vsample['x'], axis=-1), (1, 0))
                    data_sample = [vsample['x'], vsample['y'], vsample['m'], vsample['x_word'], ]

                    decoded_targets, decoded_predictions = dl.decode_output(vsample, prediction)

                    save_list = [analyse_values, prediction, decoded_predictions, data_sample, weights]

                    co_inf, mu_inf = ana.feed_variables_two(save_list, e, name="states_epoch",
                                                            save_plot=vstep)  # calculates the memory influence
                    controller_inf.append(co_inf)
                    memory_inf.append(mu_inf)

                controller_inf = np.mean(controller_inf)
                memory_inf = np.mean(memory_inf)

                writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='wer', simple_value=word_error_rate)]),
                                   e * dl.batch_amount('train') + step)
                writer.add_summary(
                    tf.Summary(value=[tf.Summary.Value(tag='controller_inf', simple_value=controller_inf)]),
                    e * dl.batch_amount('train') + step)
                writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='memory_inf', simple_value=memory_inf)]),
                                   e * dl.batch_amount('train') + step)

                sp.pub(
                    "epoch {:3}, step {:5}, train cost {:4.3f}, valid cost {:4.3f}, wer {:4.3f}, controller influence {:4.3f}, "
                    "memory influence {:4.3f}, duration {:5.1f}sec, time: {}, Model saved in {}".format(
                        e, step, train_cost, valid_cost, word_error_rate, controller_inf, memory_inf,
                        time.time() - time_0, sp.time_stamp(), save_path))
                sp.monitor(["epoch", "step", "train cost", "valid cost", "duration", "controller influence",
                            "memory influence", "wer"],
                           [e, step, train_cost, valid_cost, time.time() - time_0, controller_inf, memory_inf,
                            word_error_rate])  # saves the values in an numpy array for later analysis
            else:
                sp.pub(
                    "epoch {:3}, step {:5}, train cost {:4.3f}, valid cost {:4.3f}, duration {:5.1f}sec, time: {}, Model saved in {}".format(
                        e, step, train_cost, valid_cost, time.time() - time_0, sp.time_stamp(), save_path))
                sp.monitor(["epoch", "step", "train cost", "valid cost", "duration"],
                           [e, step, train_cost, valid_cost, time.time() - time_0])
        else:
            sp.pub("ERROR: nan in training")
            sys.exit("NAN")  # end training in case of NAN

        if early_stop(valid_cost):
            sp.pub("EARLYSTOP: valid error increase")
            sys.exit("EARLYSTOP")  # end training when valid loss increases, early stopping
