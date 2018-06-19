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
import hashlib
from collections import OrderedDict


class DataMemorizer():
    def __init__(self, config, tmp_dir):

        self.hash = self.make_config_hash(config)
        self.tmp_dir = tmp_dir

    def __call__(self, *args, **kwargs):
        return self.check_existent()

    def check_existent(self):
        file_name = os.path.join(self.tmp_dir, self.hash + '.pkl')
        return os.path.isfile(file_name)

    def load_data(self):
        with open(os.path.join(self.tmp_dir, self.hash + '.pkl'), 'rb') as outfile:
            data = pickle.load(outfile)
        return data

    def dump_data(self, data_to_save):
        with open(os.path.join(self.tmp_dir, self.hash + '.pkl'), 'wb') as outfile:
            pickle.dump(data_to_save, outfile)

    def purge_data(self):
        file_name = os.path.join(self.tmp_dir, self.hash + '.pkl')
        if os.path.isfile(file_name):
            os.remove(file_name)

    @staticmethod
    def make_config_hash(dict):
        pre = sorted(((k, v) for k, v in dict.items() if k not in ['batch_size', 'num_chached', 'threads']))
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
