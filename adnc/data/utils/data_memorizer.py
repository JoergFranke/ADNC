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
import pathlib
import pickle
import hashlib
from collections import OrderedDict


class DataMemorizer():
    """
    Given a config, it saves the pre-processed data in a pickle dump.
    """
    def __init__(self, config, tmp_dir):
        """
        Args:
            config:     dict, config of dataset
            tmp_dir:    str, dir to save dataset dump
        """
        self.hash_name = self.make_config_hash(config)
        if isinstance(tmp_dir, pathlib.Path):
            self.tmp_dir = tmp_dir
        else:
            self.tmp_dir = pathlib.Path(tmp_dir)

        if not self.tmp_dir.is_dir():
            self.tmp_dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, *args, **kwargs):
        return self.check_existent()

    def check_existent(self):
        """
        Returns:    bool, if the dataset dump exists
        """
        file_name = self.tmp_dir / self.hash_name
        return file_name.exists()

    def load_data(self):
        """
        Returns:    dataset, pickle load of dataset
        """
        with open(str(self.tmp_dir / self.hash_name), 'rb') as outfile:
            data = pickle.load(outfile)
        return data

    def dump_data(self, data_to_save):
        """
        Args:
            data_to_save:   object, what to save
        """
        with open(str(self.tmp_dir / self.hash_name), 'wb') as outfile:
            pickle.dump(data_to_save, outfile)

    def purge_data(self):
        """
        removes data dump
        """
        file_name = str(self.tmp_dir / self.hash_name)
        if os.path.isfile(file_name):
            os.remove(file_name)

    @staticmethod
    def make_config_hash(dict):
        """
        computes a hash string to name the dataset dump uniquely
        Args:
            dict:   dict, config which describes the dataset

        Returns:    str, hash tag of dataset

        """
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
        return hash + '.pkl'
