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
import pickle
import socket
import time
from collections import OrderedDict
from shutil import copyfile

import numpy as np
import yaml

"""
The supporter class creates for each training run a folder, logs the prints in a file and saves the weights, 
gradients and losses. 
"""

class ColorCode:
    def __init__(self):
        self.bold = '\033[1m'
        self.underline = '\033[4m'
        self.blue = '\033[94m'
        self.darkcyan = '\033[36m'
        self.green = '\033[92m'
        self.red = '\033[91m'
        self.yellow = '\033[93m'
        self.end = '\033[0m'


class Supporter():
    def __init__(self, experiment_dir, configs, experiment_name, data_set, model_type, session_no=False, task_id=False,
                 session_dir=None, log=True):

        self.log = log

        self.experiment_name = experiment_name.strip()
        self.data_set = data_set.strip()
        self.model_type = model_type.strip()

        self.host_name = socket.gethostname()
        if self.host_name[:3] == 'i13':
            self.host_name = 'cluster'

        if session_dir == None:
            session_dir, session_name = self._make_experiment_dir(experiment_dir, experiment_name, session_no)
            self.session_dir = session_dir
            self.session_name = session_name
        else:
            self.session_dir = session_dir
            self.session_name = session_dir.split('/')[-2]

        # define log file name
        if task_id:
            if task_id == 0:
                self.log_file = "sess_progress_ps.log"
            else:
                self.log_file = "sess_progress_{}.log".format(task_id)
        else:
            self.log_file = "sess_progress.log"

        # load config
        self.restore = False
        if session_no:
            config_file = False
            for file in os.listdir(session_dir):
                if file.endswith("config.yml"):
                    config_file = os.path.join(self.session_dir, file)
                    self.restore = True
                    break
            if not config_file:
                config_file = configs
        elif isinstance(configs, str):
            config_file = configs
        else:
            raise UserWarning('supporter: no config found')

        self.pub('load config file ' + ColorCode().darkcyan + '{}'.format(config_file) + ColorCode().end)
        with open(config_file, 'r') as f:
            self.configs = yaml.load(f)

        # copy config
        if not self.restore:
            copyfile(config_file, os.path.join(self.session_dir, os.path.basename(config_file)))

        if not task_id:
            self.pub('### task id: {}'.format(task_id))
        self.pub('##############################################')
        self.pub(ColorCode().bold + 'SUPPORTER LOG' + ColorCode().end)
        self.pub("model                  : {}".format(self.model_type))
        self.pub("data set               : {}".format(self.data_set))
        self.pub("experiment name        : {}".format(self.experiment_name))
        self.pub("session                : {}".format(self.session_name))
        self.pub("date                   : {}".format(time.strftime("%d-%m-%y")))
        self.pub("time                   : {}".format(time.strftime("%H:%M:%S")))
        self.pub("host                   : {}".format(self.host_name))
        self.pub('##############################################')
        self.pub(ColorCode().bold + 'DATA CONFIG' + ColorCode().end)
        self.pub(self.configs[self.data_set])
        self.pub('##############################################')
        self.pub(ColorCode().bold + 'MODEL CONFIG' + ColorCode().end)
        self.pub(self.configs[self.model_type])
        self.pub('##############################################')
        self.pub(ColorCode().bold + 'TRAIN CONFIG' + ColorCode().end)
        self.pub(self.config('training'))
        self.pub('##############################################')

    def config(self, name):
        return self.configs[name]

    def monitor(self, names, value):
        monitor_file = os.path.join(self.session_dir, "training_log")

        if os.path.isfile(monitor_file + ".npy"):
            # print('load')
            log = np.load(monitor_file + ".npy")[()]

            for n, v in zip(names, value):
                log[n].append(v)
            np.save(monitor_file, log)
        else:
            # print('new')
            log = OrderedDict()
            for n, v in zip(names, value):
                log[n] = [v]
            np.save(monitor_file, log)

    def load_monitor(self):
        monitor_file = os.path.join(self.session_dir, "training_log")
        log = np.load(monitor_file + ".npy")[()]
        return log

    def pub(self, text):
        if type(text) == dict:
            for n, o in sorted(text.items()):
                self.pub("{:<22} : {}".format(n, o))
        else:
            print(text)
            if self.log:
                fobj = open(os.path.join(self.session_dir, self.log_file), "a")
                fobj.write(str(text) + "\n")
                fobj.close()

    def _make_experiment_dir(self, project_dir, experiment_name, session_no):
        # make project dir
        project_dir = project_dir.strip()
        while not os.path.isdir(project_dir):
            try:
                os.mkdir(project_dir)
            except ValueError:
                pass

        # make sub folder name and dir
        sub_folder_name = '{}_{}_{}'.format(self.data_set, self.model_type, self.host_name)

        sub_folder_dir = os.path.join(project_dir, sub_folder_name)
        while not os.path.isdir(sub_folder_dir):
            try:
                os.mkdir(sub_folder_dir)
            except ValueError:
                pass

        # make experiment name and dir
        experiment_dir = os.path.join(sub_folder_dir, experiment_name)
        while not os.path.isdir(experiment_dir):
            try:
                os.mkdir(experiment_dir)
            except ValueError:
                pass

        # make session name and dir or reload
        if session_no == False:
            number = 0
            for dir in os.listdir(experiment_dir):
                if "{}".format('session') in dir:
                    numb = int(dir.split('_')[-1])
                    if numb > number:
                        number = numb
            number += 1
            session_name = "session_{}".format(number)
        else:
            session_name = "session_{}".format(session_no)

        session_dir = os.path.join(experiment_dir, session_name)

        while not os.path.isdir(session_dir):
            try:
                os.mkdir(session_dir)
            except ValueError:
                pass
        return session_dir, session_name

    def save_analyse_stack(self, scan_output, name='scan_output_0'):
        save_file = os.path.join(self.session_dir, name)
        with open(save_file, "wb") as f:
            pickle.dump(scan_output, f)

    @staticmethod
    def load_analyse_stack(session_dir, name='scan_output_0'):
        load_file = os.path.join(session_dir, name)
        with open(load_file, "rb") as f:
            scan_output = pickle.load(f)
        return scan_output

    @staticmethod
    def time_stamp():
        ts = time.time()
        return time.strftime("%d-%m-%y_%H:%M")

    @staticmethod
    def date_stamp():
        ts = time.time()
        return time.strftime("%d-%m-%y")

    def plot_functionality(self, scan_output):
        pass
