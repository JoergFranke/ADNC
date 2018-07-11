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
import tensorflow as tf

"""
The Optimizer clas is a wrapper for the TF optimizer and performs gradient clipping and weight decay
"""

class Optimizer:
    def __init__(self, config, loss, variables, use_locking=False):

        self.epochs = config["epochs"]
        self.use_locking = use_locking
        self.learn_rate = config["learn_rate"]
        self.optimizer = config["optimizer"]
        self.optimizer_config = config["optimizer_config"]
        self.gradient_clipping = config["gradient_clipping"]
        self.weight_decay = config["weight_decay"]

        if not isinstance(self.learn_rate, float):
            self.learn_rate = tf.placeholder(tf.float32, shape=[])

        if self.weight_decay:
            with tf.variable_scope('weight_decay') as scope:
                weight_decay = tf.reduce_sum(self.weight_decay * tf.stack([tf.nn.l2_loss(var) for var in variables]))
                loss = loss + weight_decay

        self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0),
                                           trainable=False)

        self.optimizer = self.get_optimizer()

        self.gradients = self.optimizer.compute_gradients(loss, variables)

        if self.gradient_clipping:
            self.gradients = [(tf.clip_by_value(grad, -1 * self.gradient_clipping, self.gradient_clipping), var) for
                              grad, var in self.gradients]
        self.optimizer = self.optimizer.apply_gradients(self.gradients, self.global_step)

    def get_optimizer(self):

        if self.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learn_rate, use_locking=self.use_locking,
                                               epsilon=0.000001)
        elif self.optimizer == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learn_rate,
                                                  momentum=self.optimizer_config['momentum'],
                                                  use_locking=self.use_locking)
        elif self.optimizer == 'momentum':
            optimizer = tf.train.MomentumOptimizer(learning_rate=self.learn_rate,
                                                   momentum=self.optimizer_config['momentum'],
                                                   use_nesterov=self.optimizer_config['nesterov'],
                                                   use_locking=self.use_locking)
        elif self.optimizer == 'adadelta':
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.learn_rate, use_locking=self.use_locking)
        elif self.optimizer == 'adagrad':
            optimizer = tf.train.AdagradOptimizer(learning_rate=self.learn_rate, use_locking=self.use_locking)
        elif self.optimizer == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learn_rate, use_locking=self.use_locking)
        else:
            raise UserWarning("Unknown optimizer")
        return optimizer
